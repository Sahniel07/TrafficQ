import json
import math
import numpy as np
from datetime import datetime
from fastdtw import fastdtw
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
import networkx as nx
# Set matplotlib to use non-interactive backend, must be set before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from matplotlib.collections import LineCollection
import contextily as ctx  # Import contextily library for adding map background

# Set global optimization options
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 0.5
matplotlib.rcParams['agg.path.chunksize'] = 10000

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two latitude-longitude points using Haversine formula (in meters).
    """
    R = 6371000  # Earth radius (meters)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_vehicle_trajectories(filename):
    """
    Parse vehicle trajectory data from JSON file,
    return two dictionaries:
      trajectories: {vehicle_id: trajectory_array, ...}
         each trajectory_array is an N×2 numpy array with columns [lat, lon] (spatial information only).
      vehicle_times: {vehicle_id: {"start": start_time_str, "end": end_time_str}, ...}
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    trajectories = {}
    vehicle_times = {}
    for edge in data['data']['entities']['edges']:
        node = edge['node']
        vehicle_id = node['id']
        traj = []
        times = []
        for pt in node['trajectory']:
            t = datetime.fromisoformat(pt['time'])
            times.append(t)
            lon, lat = pt['coordinateLongLat']
            traj.append([lat, lon])
        if traj:
            traj = np.array(traj)
            trajectories[vehicle_id] = traj
            start_time = times[0].strftime("%Y-%m-%d %H:%M:%S")
            end_time = times[-1].strftime("%Y-%m-%d %H:%M:%S")
            vehicle_times[vehicle_id] = {"start": start_time, "end": end_time}
    return trajectories, vehicle_times

def dtw_distance(traj1, traj2):
    """
    Calculate DTW distance between two trajectories (in meters).
    Implemented using fastdtw, with haversine distance as the distance function.
    """
    distance, _ = fastdtw(traj1, traj2, dist=lambda a, b: haversine(a[1], a[0], b[1], b[0]))
    return distance

def compute_distance_matrix(trajectories):
    """
    Calculate pairwise DTW distances for all vehicle trajectories, return distance matrix and vehicle id list.
    """
    vehicle_ids = list(trajectories.keys())
    n = len(vehicle_ids)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = dtw_distance(trajectories[vehicle_ids[i]], trajectories[vehicle_ids[j]])
            D[i, j] = d
            D[j, i] = d
    return D, vehicle_ids

def downsample_trajectory(traj, max_points):
    """
    Efficiently downsample a trajectory, keeping at most max_points points.
    Uses integer division to directly calculate step size, faster than using linspace or similar methods.
    """
    if len(traj) <= max_points:
        return traj
    # Use integer division to directly calculate step size
    step = max(1, len(traj) // max_points)
    return traj[::step]

def parallel_downsample(args):
    """
    Helper function for parallel downsampling.
    """
    vid, traj, max_points = args
    return vid, downsample_trajectory(traj, max_points)

def process_pair(args):
    """
    Function for parallel computation, parameters are (veh1, veh2, traj1, traj2).
    Returns (veh1, veh2, distance).
    """
    veh1, veh2, traj1, traj2 = args
    # Downsample trajectories: take every 5th point
    if len(traj1) > 5:
        traj1 = traj1[::5]
    if len(traj2) > 5:
        traj2 = traj2[::5]
    d = dtw_distance(traj1, traj2)
    return (veh1, veh2, d)

def cluster_similar_trajectories(trajectories, similarity_threshold=100.0, centroid_threshold=200.0):
    """
    Cluster vehicle trajectories.
    First use centroid pre-filtering, then calculate similar vehicle pairs using DTW distances.
    Returns a list of clustering results, each cluster is a list of vehicle IDs.
    """
    print("Calculating vehicle trajectory centroids...")
    # Calculate centroid for each vehicle (simple average, not considering time)
    centroids = {}
    for vid, traj in tqdm(trajectories.items(), desc="Calculating centroids"):
        if len(traj) == 0:
            centroids[vid] = (None, None)
        else:
            avg_lat = np.mean(traj[:, 0])
            avg_lon = np.mean(traj[:, 1])
            centroids[vid] = (avg_lat, avg_lon)
    
    print("Filtering candidate vehicle pairs...")
    pairs = []
    vehicle_ids = list(trajectories.keys())
    total_pairs = len(vehicle_ids) * (len(vehicle_ids) - 1) // 2
    with tqdm(total=total_pairs, desc="Filtering vehicle pairs") as pbar:
        for i in range(len(vehicle_ids)):
            for j in range(i+1, len(vehicle_ids)):
                veh1 = vehicle_ids[i]
                veh2 = vehicle_ids[j]
                traj1 = trajectories[veh1]
                traj2 = trajectories[veh2]
                if traj1.size == 0 or traj2.size == 0:
                    pbar.update(1)
                    continue
                c1 = centroids[veh1]
                c2 = centroids[veh2]
                if c1[0] is None or c2[0] is None:
                    pbar.update(1)
                    continue
                centroid_dist = haversine(c1[1], c1[0], c2[1], c2[0])
                if centroid_dist > centroid_threshold:
                    pbar.update(1)
                    continue
                pairs.append((veh1, veh2, traj1, traj2))
                pbar.update(1)
    
    print(f"Total {len(pairs)} candidate vehicle pairs need DTW distance calculation")
    
    # Use progress bar to monitor parallel computation progress
    print("Calculating vehicle trajectory DTW distances...")
    # Process in batches for better progress visualization
    batch_size = max(1, len(pairs) // 10)  # Split into 10 batches
    similar_pairs = []
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/10...")
        pool = Pool()
        for result in tqdm(pool.imap(process_pair, batch), total=len(batch), desc="Calculating DTW distances"):
            veh1, veh2, d = result
            if d < similarity_threshold:
                similar_pairs.append((veh1, veh2, d))
        pool.close()
        pool.join()
    
    print(f"Found {len(similar_pairs)} similar vehicle pairs")
    
    # Construct clustering graph: vehicles as nodes, edges between similar vehicle pairs
    G = nx.Graph()
    G.add_nodes_from(trajectories.keys())
    for veh1, veh2, d in similar_pairs:
        G.add_edge(veh1, veh2, distance=d)
    
    clusters = list(nx.connected_components(G))
    return clusters

def prepare_cluster_lines(cluster, sampled_trajectories):
    """
    Prepare line collections and start points for plotting clusters, optimizing memory allocation.
    """
    lines = []
    start_points = []
    for vid in cluster:
        traj = sampled_trajectories.get(vid)
        if traj is not None and len(traj) > 1:
            # Directly create point arrays, avoid column_stack
            points = np.zeros((len(traj), 2))
            points[:, 0] = traj[:, 1]  # lon
            points[:, 1] = traj[:, 0]  # lat
            lines.append(points)
            start_points.append((traj[0, 1], traj[0, 0]))
    return lines, start_points

def plot_all_multi_vehicle_clusters(trajectories, clusters):
    """
    Generate separate high-precision visualization images for each multi-vehicle cluster.
    Uses English labels, higher precision and aesthetically pleasing parameter settings.
    """
    print("Generating visualization for each multi-vehicle cluster...")
    
    # Find all multi-vehicle clusters
    multi_vehicle_clusters = [c for c in clusters if len(c) > 1]
    
    if not multi_vehicle_clusters:
        print("No multi-vehicle clusters found, skipping visualization.")
        return
    
    # Set an elegant style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Preprocessing: parallel downsample all trajectories, using more points for higher precision
    print("Downsampling trajectories in parallel...")
    max_points = 50  # Increase downsampling points for higher precision
    
    # Collect all vehicle IDs to be processed
    vehicle_ids = set()
    for cluster in multi_vehicle_clusters:
        vehicle_ids.update(cluster)
    
    # Prepare parallel processing parameters
    sample_args = []
    for vid in vehicle_ids:
        traj = trajectories.get(vid)
        if traj is not None and len(traj) > 0:
            sample_args.append((vid, traj, max_points))
    
    # Parallel downsampling
    sampled_trajectories = {}
    if sample_args:
        pool = Pool()
        for vid, sampled_traj in tqdm(pool.imap(parallel_downsample, sample_args), 
                                      total=len(sample_args), 
                                      desc="Downsampling trajectories"):
            sampled_trajectories[vid] = sampled_traj
        pool.close()
        pool.join()
    
    # Get color list (use more colors)
    colors = plt.cm.tab20.colors  # Use 20-color palette for more colors
    
    # Generate separate images for each multi-vehicle cluster
    for idx, cluster in enumerate(tqdm(multi_vehicle_clusters, desc="Generating cluster visualizations")):
        print(f"Processing cluster {idx+1} with {len(cluster)} vehicles...")
        
        # Create high-resolution plot object
        plt.figure(figsize=(12, 10), dpi=200)
        ax = plt.gca()
        
        # Use better background
        ax.set_facecolor('#f8f9fa')
        
        # Process trajectory data
        lines = []
        start_points_x = []
        start_points_y = []
        end_points_x = []
        end_points_y = []
        
        for i, vid in enumerate(cluster):
            traj = sampled_trajectories.get(vid)
            if traj is not None and len(traj) > 1:
                # Prepare line segment data
                points = np.zeros((len(traj), 2))
                points[:, 0] = traj[:, 1]  # lon
                points[:, 1] = traj[:, 0]  # lat
                lines.append(points)
                
                # Record start and end points
                start_points_x.append(traj[0, 1])
                start_points_y.append(traj[0, 0])
                end_points_x.append(traj[-1, 1])
                end_points_y.append(traj[-1, 0])
        
        # Calculate boundaries
        if lines:
            # Find all points
            all_points = np.vstack(lines)
            min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            
            # Set boundaries, slightly enlarged
            margin_x = (max_x - min_x) * 0.1
            margin_y = (max_y - min_y) * 0.1
            plt.xlim(min_x - margin_x, max_x + margin_x)
            plt.ylim(min_y - margin_y, max_y + margin_y)
            
            # Use color gradients to set colors for each trajectory
            num_vehicles = len(cluster)
            if num_vehicles <= 10:
                # Use fixed color set
                line_colors = [colors[i % len(colors)] for i in range(num_vehicles)]
                
                # Draw all trajectories at once
                lc = LineCollection(lines, colors=line_colors, linewidths=2.0, alpha=0.7)
                ax.add_collection(lc)
            else:
                # Use a single color when there are many vehicles to avoid confusion
                lc = LineCollection(lines, colors='red', linewidths=1.5, alpha=0.6)
                ax.add_collection(lc)
            
            # Draw start and end points
            plt.scatter(start_points_x, start_points_y, c='blue', s=50, marker='o', 
                        edgecolors='white', linewidths=1, alpha=0.8, label='Start Points')
            plt.scatter(end_points_x, end_points_y, c='green', s=50, marker='s', 
                        edgecolors='white', linewidths=1, alpha=0.8, label='End Points')
        
        # Set chart title and labels
        plt.title(f'Cluster {idx+1} Trajectory Visualization ({len(cluster)} Vehicles)', fontsize=16, pad=20)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        
        # Show grid with slight transparency
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        plt.legend(loc='upper right', frameon=True, framealpha=0.9)
        
        # Add scale bar (simplified version)
        from matplotlib.lines import Line2D
        scale_bar_length_km = 0.1  # 100 meter scale bar
        if max_x - min_x > 0.01:  # Only add when map range is large enough
            x_pos = min_x + margin_x * 0.5
            y_pos = min_y + margin_y * 0.5
            dx = scale_bar_length_km / 111.32  # Approximate degrees in longitude (near equator)
            plt.plot([x_pos, x_pos + dx], [y_pos, y_pos], 'k-', linewidth=2)
            plt.text(x_pos + dx/2, y_pos + margin_y * 0.05, f'{scale_bar_length_km * 1000}m', 
                     ha='center', va='bottom', fontsize=8)
        
        # Save image (using higher DPI and simplified compatible parameters)
        filename = f'cluster_{idx+1}_trajectories.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        print(f"✅ Visualization for cluster {idx+1} saved as {filename}")
        plt.close()
    
    print("✅ All cluster visualizations completed successfully.")

def plot_all_clusters_in_one_image(trajectories, clusters):
    """
    Draw all clusters in one high-definition image, showing complete trajectory points for vehicles.
    Uses English labels and elegant styles, with map as background.
    """
    print("Generating visualization with all clusters in one image...")
    
    # Find all multi-vehicle clusters
    multi_vehicle_clusters = [c for c in clusters if len(c) > 1]
    
    if not multi_vehicle_clusters:
        print("No multi-vehicle clusters found, skipping visualization.")
        return
    
    # Set elegant plotting style, but don't use grid which would conflict with map
    plt.style.use('default')
    
    # Create high-resolution image
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    
    # Get more colors
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    
    # Sort by cluster size, so large clusters are drawn first (will be partially covered by small clusters)
    sorted_clusters = sorted(multi_vehicle_clusters, key=len)
    
    print(f"Processing {len(sorted_clusters)} multi-vehicle clusters...")
    
    # Collect all trajectory points for boundary calculation
    all_points = []
    
    # Prepare legend items
    legend_handles = []
    legend_labels = []
    
    # Draw trajectories for each cluster
    for i, cluster in enumerate(tqdm(sorted_clusters, desc="Drawing clusters")):
        color = colors[i % len(colors)]
        
        # Add legend item
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
        legend_labels.append(f'Cluster {i+1} ({len(cluster)} vehicles)')
        
        # Draw trajectory for each vehicle in the cluster
        for vid in cluster:
            traj = trajectories.get(vid)
            if traj is not None and len(traj) > 1:
                # Save complete trajectory points
                all_points.append(traj)
                
                # Draw trajectory line
                ax.plot(traj[:, 1], traj[:, 0], '-', color=color, linewidth=1.5, alpha=0.8, zorder=5)
                
                # Mark start point
                ax.plot(traj[0, 1], traj[0, 0], 'o', color=color, markersize=6, alpha=0.9, zorder=6)
                
                # Mark end point
                ax.plot(traj[-1, 1], traj[-1, 0], 's', color=color, markersize=6, alpha=0.9, zorder=6)
    
    # Calculate and set image boundaries
    if all_points:
        all_points_array = np.vstack(all_points)
        min_lat, max_lat = np.min(all_points_array[:, 0]), np.max(all_points_array[:, 0])
        min_lon, max_lon = np.min(all_points_array[:, 1]), np.max(all_points_array[:, 1])
        
        # Add margins
        margin_x = (max_lon - min_lon) * 0.1  # Increase margin to ensure enough space for map
        margin_y = (max_lat - min_lat) * 0.1
        ax.set_xlim(min_lon - margin_x, max_lon + margin_x)
        ax.set_ylim(min_lat - margin_y, max_lat + margin_y)
        
        # Draw direction arrows after boundary calculation
        for i, cluster in enumerate(sorted_clusters):
            color = colors[i % len(colors)]
            
            for vid in cluster:
                traj = trajectories.get(vid)
                if traj is not None and len(traj) >= 5:  # Ensure trajectory has enough points
                    # Add end point arrow
                    end_idx = len(traj) - 1
                    start_idx = max(0, end_idx - 1)
                    
                    # Calculate direction vector
                    dx = traj[end_idx, 1] - traj[start_idx, 1]  # Longitude difference
                    dy = traj[end_idx, 0] - traj[start_idx, 0]  # Latitude difference
                    
                    # Normalize direction vector
                    magnitude = np.sqrt(dx**2 + dy**2)
                    if magnitude > 0:  # Avoid division by zero
                        dx = dx / magnitude
                        dy = dy / magnitude
                        
                        # Add end point direction arrow
                        ax.quiver(traj[end_idx, 1], traj[end_idx, 0], 
                                 dx, dy, color=color, scale=10, 
                                 scale_units='inches', width=0.003,
                                 headwidth=4, headlength=5, alpha=0.9, zorder=7)
                    
                    # Add direction arrows in the middle of trajectory segments
                    if len(traj) >= 10:  # Add middle arrows for longer trajectories
                        # Determine number of arrows based on trajectory length, maximum 5
                        num_arrows = min(5, len(traj) // 10)
                        
                        if num_arrows > 0:
                            # Calculate interval
                            step = len(traj) // (num_arrows + 1)
                            
                            # Add arrows at middle positions of trajectory
                            for arrow_idx in range(1, num_arrows + 1):
                                # Calculate arrow position
                                idx = step * arrow_idx
                                if idx >= len(traj) - 1:
                                    continue
                                
                                # Use points before and after to calculate direction
                                prev_idx = max(0, idx - 3)  # Take a few points ahead for smoother direction
                                
                                # Calculate direction vector
                                dx = traj[idx, 1] - traj[prev_idx, 1]
                                dy = traj[idx, 0] - traj[prev_idx, 0]
                                
                                # Normalize direction vector
                                magnitude = np.sqrt(dx**2 + dy**2)
                                if magnitude > 0:
                                    dx = dx / magnitude
                                    dy = dy / magnitude
                                    
                                    # Draw middle arrows, slightly smaller
                                    ax.quiver(traj[idx, 1], traj[idx, 0], 
                                             dx, dy, color=color, scale=12,
                                             scale_units='inches', width=0.002,
                                             headwidth=3, headlength=4, alpha=0.7, zorder=7)
    
        # Add map background
        try:
            print("Adding high-resolution map background...")
            # Use higher resolution map source and appropriate zoom level
            # Try other high-definition map sources
            # Higher zoom value means more detailed map, typically 16-18 is city street level
            ctx.add_basemap(ax, crs='EPSG:4326', 
                           source=ctx.providers.CartoDB.Positron,  # Try this clearer map source
                           zoom=17,  # Increase zoom level for more details
                           alpha=1.0)  # Make map completely opaque
            print("High-resolution map background added successfully.")
        except Exception as e:
            print(f"Error adding map background: {e}")
            # If first map source fails, try alternative map sources
            try:
                print("Trying alternative map source...")
                ctx.add_basemap(ax, crs='EPSG:4326', 
                               source=ctx.providers.Stamen.Terrain,  
                               zoom=16, alpha=1.0)
                print("Alternative map background added successfully.")
            except Exception as e2:
                print(f"Error adding alternative map background: {e2}")
                print("Continuing without map background...")
                
    # Set chart title and labels
    plt.title('Vehicle Trajectory Clusters Visualization', fontsize=18, pad=20)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    
    # Add legend (use small size to avoid being too large)
    if len(legend_handles) > 20:
        # If too many clusters, only show first 20
        legend_handles = legend_handles[:20]
        legend_labels = legend_labels[:20]
        legend_labels.append('... and more clusters')
    
    ax.legend(legend_handles, legend_labels, loc='upper right', 
               fontsize='small', frameon=True, framealpha=0.9)
    
    # Save image - increase DPI for overall clarity
    output_file = 'all_trajectory_clusters_with_map.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    
    print(f"✅ All clusters visualization with map background saved as {output_file}")
    plt.close()

def main():
    filename = 'data.json'
    trajectories, vehicle_times = parse_vehicle_trajectories(filename)
    
    similarity_threshold = 100.0  # Hausdorff (here using DTW) distance threshold (meters)
    centroid_threshold = 200.0    # Centroid pre-filtering threshold (meters)
    
    clusters = cluster_similar_trajectories(trajectories, similarity_threshold, centroid_threshold)
    
    print("\nClusters of vehicles with similar trajectories:")
    output_clusters = []
    for idx, cluster in enumerate(clusters, 1):
        cluster_list = []
        for vid in cluster:
            time_info = vehicle_times.get(vid, {"start": "N/A", "end": "N/A"})
            cluster_list.append({
                "vehicle_id": vid,
                "start_time": time_info["start"],
                "end_time": time_info["end"]
            })
        print(f"Cluster {idx}:")
        for item in cluster_list:
            print(f"  Vehicle {item['vehicle_id']} (from {item['start_time']} to {item['end_time']})")
        output_clusters.append({
            "cluster_id": idx,
            "vehicles": cluster_list
        })
    
    # Save clustering results to JSON file
    with open("trajectory_clusters.json", "w", encoding="utf-8") as f:
        json.dump(output_clusters, f, ensure_ascii=False, indent=4)
    
    print("\n✅ All clustering results have been saved to trajectory_clusters.json.")
    
    # Draw all clusters in one image
    plot_all_clusters_in_one_image(trajectories, clusters)

if __name__ == '__main__':
    main()