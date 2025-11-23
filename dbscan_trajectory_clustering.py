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
import os  # 添加os模块用于创建目录

# Set global optimization options
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 0.5
matplotlib.rcParams['agg.path.chunksize'] = 10000

SCALE_BAR_LENGTH_M = 200  # Fixed scale bar length in meters for consistent visualization
MAP_ZOOM_LEVEL = 17  # 统一的地图缩放级别，确保所有图使用相同的缩放
FIGURE_DPI = 300  # 统一的图像DPI
FIGURE_SIZE = (12, 10)  # 统一的图像尺寸
MARGIN_FACTOR = 0.15  # 统一的边距因子
MAX_TRAJECTORY_POINTS = 50  # 统一的轨迹点采样数量


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

def cluster_similar_trajectories_dbscan(trajectories, similarity_threshold=100.0, centroid_threshold=200.0, min_samples=None):
    """
    Use DBSCAN to cluster vehicle trajectories, including distance matrix optimization and dynamic parameter setting.
    First use centroid pre-filtering, then calculate DTW distances for similar vehicle pairs, and finally apply DBSCAN.
    
    Parameters:
    trajectories -- trajectory dictionary {vehicle_id: trajectory_array, ...}
    similarity_threshold -- DTW distance threshold (meters)
    centroid_threshold -- centroid pre-filtering threshold (meters)
    min_samples -- DBSCAN min_samples parameter, automatically calculated if None
    
    Returns:
    clusters -- clustering result list, each cluster is a list of vehicle IDs
    """
    print("Calculating vehicle trajectory centroids...")
    # Calculate the centroid for each vehicle (simple average, ignoring time)
    centroids = {}
    for vid, traj in tqdm(trajectories.items(), desc="Calculating centroids"):
        if len(traj) == 0:
            centroids[vid] = (None, None)
        else:
            avg_lat = np.mean(traj[:, 0])
            avg_lon = np.mean(traj[:, 1])
            centroids[vid] = (avg_lat, avg_lon)
    
    # 记录所有车辆ID
    all_vehicle_ids = list(trajectories.keys())
    
    print("Filtering candidate vehicle pairs...")
    pairs = []
    vehicle_ids = list(trajectories.keys())
    total_pairs = len(vehicle_ids) * (len(vehicle_ids) - 1) // 2
    
    # Pre-filter using centroids as before
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
    
    # 生成一个集合记录所有被处理过的车辆ID
    processed_vehicles = set()
    
    print(f"Total of {len(pairs)} candidate vehicle pairs need DTW distance calculation")
    
    # Calculate DTW distance matrix
    print("Calculating vehicle trajectory DTW distances...")
    batch_size = max(1, len(pairs) // 10)
    similar_pairs = []
    distance_dict = {}  # For storing calculated distances
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/10...")
        pool = Pool()
        for result in tqdm(pool.imap(process_pair, batch), total=len(batch), desc="Calculating DTW distances"):
            veh1, veh2, d = result
            # 记录处理过的车辆ID
            processed_vehicles.add(veh1)
            processed_vehicles.add(veh2)
            # Store calculated distances
            distance_dict[(veh1, veh2)] = d
            distance_dict[(veh2, veh1)] = d
            if d < similarity_threshold:
                similar_pairs.append((veh1, veh2, d))
        pool.close()
        pool.join()
    
    print(f"Found {len(similar_pairs)} similar vehicle pairs")
    
    # Prepare distance matrix for DBSCAN
    # First determine the set of vehicle IDs participating in clustering
    unique_vehicles = set()
    for veh1, veh2, _ in similar_pairs:
        unique_vehicles.add(veh1)
        unique_vehicles.add(veh2)
    
    vehicle_list = list(unique_vehicles)
    n = len(vehicle_list)
    
    # Improvement 1: Dynamically calculate default distance value instead of simply using a multiple of the threshold
    if distance_dict:
        # Calculate the maximum of known distances, adding a safety margin
        actual_max_distance = max(distance_dict.values()) * 1.5
        # Ensure default distance is at least twice the similarity threshold
        max_distance = max(actual_max_distance, similarity_threshold * 2)
        print(f"Using dynamically calculated default distance: {max_distance:.2f} meters (actual maximum distance: {max(distance_dict.values()):.2f} meters)")
    else:
        max_distance = similarity_threshold * 2
    
    # Create distance matrix
    distance_matrix = np.zeros((n, n))
    
    # Fill matrix with default large distance value
    distance_matrix.fill(max_distance)
    
    # Fill in calculated distances
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i, j] = 0  # Diagonal is 0
            else:
                veh1 = vehicle_list[i]
                veh2 = vehicle_list[j]
                if (veh1, veh2) in distance_dict:
                    distance_matrix[i, j] = distance_dict[(veh1, veh2)]
    
    # Improvement 2: Dynamically calculate appropriate min_samples value
    if min_samples is None:
        # Dynamically set min_samples based on data scale
        # Use smaller values for smaller datasets, larger values for larger datasets
        if n < 10:
            min_samples = 2  # Very small dataset
        elif n < 50:
            min_samples = 3  # Small dataset
        elif n < 100:
            min_samples = 4  # Medium dataset
        else:
            # For large datasets, use approximately 3% of total samples, but not less than 5
            min_samples = max(5, int(n * 0.03))
    
    print(f"Using DBSCAN parameters: eps={similarity_threshold} meters, min_samples={min_samples}")
    
    print("Applying DBSCAN clustering algorithm...")
    # Apply DBSCAN. eps is the similarity threshold, min_samples is the minimum number of samples required to form a core point
    dbscan = DBSCAN(eps=similarity_threshold, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(distance_matrix)
    
    # Convert results to cluster list
    clusters = {}
    
    # 首先处理已聚类的点
    for i, label in enumerate(labels):
        if label == -1:  # Noise point
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(vehicle_list[i])
    
    # 将噪声点作为单独的聚类处理
    for i, label in enumerate(labels):
        if label == -1:  # 处理噪声点，为每个噪声点创建单独的聚类
            # 为每个噪声点创建一个新的聚类标签
            new_label = len(clusters) + (i - sum(1 for l in labels if l != -1))
            clusters[new_label] = [vehicle_list[i]]
    
    # 将未处理的车辆（未被纳入DBSCAN）也单独创建聚类
    unprocessed_vehicles = set(all_vehicle_ids) - processed_vehicles
    print(f"Adding {len(unprocessed_vehicles)} additional vehicles that were not processed by DBSCAN")
    
    # 为每个未处理的车辆创建单独的聚类
    for vid in unprocessed_vehicles:
        if trajectories.get(vid) is not None and len(trajectories[vid]) > 0:
            new_label = len(clusters)
            clusters[new_label] = [vid]
    
    # Count and output noise points and cluster information
    noise_count = np.sum(labels == -1)
    cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"DBSCAN identified {noise_count} noise points (not belonging to any cluster), forming {cluster_count} clusters")
    
    # 计算包含单聚类的数量
    single_vehicle_clusters = sum(1 for c in clusters.values() if len(c) == 1)
    print(f"Total clusters after processing: {len(clusters)}, including {single_vehicle_clusters} single-vehicle clusters")
    
    # Calculate cluster size distribution
    cluster_sizes = [len(c) for c in clusters.values()]
    if cluster_sizes:
        print(f"Cluster size statistics: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
              f"average={np.mean(cluster_sizes):.1f}, median={np.median(cluster_sizes)}")
        
        # Output cluster size distribution
        size_counts = {}
        for size in cluster_sizes:
            if size not in size_counts:
                size_counts[size] = 0
            size_counts[size] += 1
        
        print("Cluster size distribution:")
        for size in sorted(size_counts.keys()):
            print(f"  Size {size}: {size_counts[size]} clusters")
    
    return list(clusters.values())

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

def draw_exact_scale_bar(ax, scale_bar_length_m=SCALE_BAR_LENGTH_M):
    """
    在地图上绘制精确长度的比例尺，不管图表显示的地理范围如何，
    该比例尺在所有图表中都表示相同的地理距离
    
    Parameters:
    ax -- matplotlib轴对象
    scale_bar_length_m -- 比例尺长度(米)，默认为SCALE_BAR_LENGTH_M
    """
    # 获取当前图表的显示范围
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # 获取中心点经纬度
    center_lon = (x_min + x_max) / 2
    center_lat = (y_min + y_max) / 2
    
    # 计算当前纬度下的经度尺度
    # 在纬度lat处，1度经度约等于111320*cos(lat)米
    lon_scale = scale_bar_length_m / (111320 * np.cos(np.radians(center_lat)))
    
    # 计算比例尺的起始位置和结束位置
    # 统一放在左下角，距离左边10%，距离底部5%的位置
    x_start = x_min + (x_max - x_min) * 0.1
    y_pos = y_min + (y_max - y_min) * 0.05
    x_end = x_start + lon_scale  # 精确的经度距离
    
    # 绘制比例尺线条
    ax.plot([x_start, x_end], [y_pos, y_pos], 'k-', linewidth=2.5, zorder=20)
    
    # 添加标签
    scale_label = f'{int(scale_bar_length_m)} m'
    
    # 使用相对于图表尺寸的偏移量
    y_offset = (y_max - y_min) * 0.02
    
    # 确保文本有白色背景，更容易阅读
    ax.text((x_start + x_end) / 2, y_pos + y_offset, scale_label, 
            ha='center', va='bottom', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
            zorder=20)

# 重新添加 set_geo_aspect_ratio 函数
def set_geo_aspect_ratio(ax, center_lat):
    """
    设置精确的地理比例，使经度和纬度在视觉上显示相同的地理距离
    
    Parameters:
    ax -- matplotlib轴对象
    center_lat -- 中心点纬度
    """
    # 在给定纬度下，1度经度对应的距离与1度纬度对应的距离之比
    # 计算当前中心纬度下的经度/纬度比例
    # 这是因为在不同纬度，1度经度对应的距离会变化，而1度纬度对应的距离基本固定
    # 赤道上1度经度和1度纬度约等于111km，但在纬度phi处，1度经度约等于111*cos(phi)km
    lon_lat_ratio = np.cos(np.radians(center_lat))
    
    # 设置长宽比，考虑纬度影响
    # 这确保了在地图上，1度经度和1度纬度所表示的实际地理距离是正确的比例
    ax.set_aspect(1 / lon_lat_ratio) # 修正：应该是1/cos(lat)来匹配Mercator投影的视觉效果

# 修改plot_all_clusters函数，使用精确地理比例
def plot_all_clusters(trajectories, clusters, output_dir, min_lon, max_lon, min_lat, max_lat):
    """
    Generate separate high-precision visualization images for each cluster.
    Uses English labels, higher precision and aesthetically pleasing parameter settings.
    
    Parameters:
    trajectories -- trajectory dictionary {vehicle_id: trajectory_array, ...}
    clusters -- list of clusters, each cluster is a list of vehicle IDs
    output_dir -- directory to save visualization images
    min_lon, max_lon, min_lat, max_lat -- global geographic boundaries
    """
    print(f"Generating separate visualization images for each cluster in '{output_dir}' using global bounds...")
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用所有聚类，不再过滤单车聚类
    all_clusters = clusters
    
    if not all_clusters:
        print("No clusters found, skipping visualization.")
        return
    
    # 为了与地图背景兼容，使用默认样式而不是带网格的样式
    plt.style.use('default')
    
    # Preprocessing: parallel downsample all trajectories, using more points for higher precision
    print("Parallel downsampling trajectories...")
    max_points = MAX_TRAJECTORY_POINTS  # 使用统一的轨迹点数量
    
    # Collect all vehicle IDs to be processed
    vehicle_ids = set()
    for cluster in all_clusters:
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
    
    # 计算全局边界的边距
    global_margin_x = (max_lon - min_lon) * MARGIN_FACTOR
    global_margin_y = (max_lat - min_lat) * MARGIN_FACTOR
    
    # 生成单独的图像
    for idx, cluster in enumerate(tqdm(all_clusters, desc="Generating cluster visualizations")):
        print(f"Processing cluster {idx+1}, containing {len(cluster)} vehicles...")
        
        # 使用统一的图表尺寸和DPI
        plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
        ax = plt.gca()
        
        # 使用透明背景以便地图可见
        ax.set_facecolor('none')
        
        # Process trajectory data for this cluster
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
        
        # 强制使用全局边界设置X和Y轴的范围
        plt.xlim(min_lon - global_margin_x, max_lon + global_margin_x)
        plt.ylim(min_lat - global_margin_y, max_lat + global_margin_y)
        
        # 使用精确的地理比例设置
        center_lat = (min_lat + max_lat) / 2  # 使用全局中心纬度
        set_geo_aspect_ratio(ax, center_lat)
        
        # 添加地图背景 - 必须在设置边界和比例之后
        try:
            print(f"  Adding map background for cluster {idx+1}...")
            ctx.add_basemap(ax, crs='EPSG:4326', 
                           source=ctx.providers.CartoDB.Positron,
                           zoom=MAP_ZOOM_LEVEL,  # 使用固定的缩放级别
                           alpha=1.0)
            print(f"  Map background added successfully for cluster {idx+1}")
        except Exception as e:
            print(f"  Error adding primary map background for cluster {idx+1}: {e}")
            try:
                print(f"  Trying alternative map source for cluster {idx+1}...")
                ctx.add_basemap(ax, crs='EPSG:4326', 
                              source=ctx.providers.Stamen.Terrain,
                              zoom=MAP_ZOOM_LEVEL,  # 使用固定的缩放级别
                              alpha=1.0)
                print(f"  Alternative map background added successfully for cluster {idx+1}")
            except Exception as e2:
                print(f"  Error adding alternative map background for cluster {idx+1}: {e2}")
                print(f"  Continuing without map background for cluster {idx+1}")
        
        # 在地图背景之上绘制轨迹
        if lines:
            num_vehicles = len(cluster)
            if num_vehicles <= 10:
                line_colors = [colors[i % len(colors)] for i in range(num_vehicles)]
                lc = LineCollection(lines, colors=line_colors, linewidths=2.5, alpha=0.8, zorder=10)
                ax.add_collection(lc)
            else:
                lc = LineCollection(lines, colors='red', linewidths=2.0, alpha=0.7, zorder=10)
                ax.add_collection(lc)
            
            # 绘制起点和终点
            plt.scatter(start_points_x, start_points_y, c='blue', s=60, marker='o', 
                        edgecolors='white', linewidths=1.5, alpha=0.9, label='Start Point', zorder=11)
            plt.scatter(end_points_x, end_points_y, c='green', s=60, marker='s', 
                        edgecolors='white', linewidths=1.5, alpha=0.9, label='End Point', zorder=11)
            
            # 添加方向箭头
            for i, vid in enumerate(cluster):
                traj = sampled_trajectories.get(vid)
                if traj is not None and len(traj) >= 5:
                    end_idx = len(traj) - 1
                    start_idx = max(0, end_idx - 1)
                    dx = traj[end_idx, 1] - traj[start_idx, 1]
                    dy = traj[end_idx, 0] - traj[start_idx, 0]
                    magnitude = np.sqrt(dx**2 + dy**2)
                    if magnitude > 0:
                        dx = dx / magnitude
                        dy = dy / magnitude
                        color = colors[i % len(colors)] if num_vehicles <= 10 else 'red'
                        ax.quiver(traj[end_idx, 1], traj[end_idx, 0], 
                                dx, dy, color=color, scale=10, 
                                scale_units='inches', width=0.003,
                                headwidth=4, headlength=5, alpha=0.9, zorder=12)
        
        # Set chart title and labels
        plt.title(f'Cluster {idx+1} Trajectory Visualization ({len(cluster)} Vehicles)', fontsize=16, pad=20)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        
        # 不在地图上使用网格
        plt.grid(False)
        
        # Add legend with white background for better visibility on map
        legend = plt.legend(loc='upper right', frameon=True, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        
        # 使用新的比例尺函数
        draw_exact_scale_bar(ax, SCALE_BAR_LENGTH_M)
        
        # 应用tight_layout确保所有元素都能正确显示
        plt.tight_layout()
        
        # Save image (using higher DPI and simplified compatible parameters)
        filename = os.path.join(output_dir, f'cluster_{idx+1}_trajectories.png')
        plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
        
        print(f"✅ Visualization for cluster {idx+1} has been saved as {filename}")
        plt.close()

def plot_all_clusters_in_one_image(trajectories, clusters, output_dir, min_lon, max_lon, min_lat, max_lat):
    """
    Draw all clusters in one high-definition image, showing complete trajectory points for vehicles.
    Uses English labels and elegant styles, with map as background.
    
    Parameters:
    trajectories -- trajectory dictionary {vehicle_id: trajectory_array, ...}
    clusters -- list of clusters, each cluster is a list of vehicle IDs
    output_dir -- directory to save visualization image
    min_lon, max_lon, min_lat, max_lat -- global geographic boundaries
    """
    print(f"Generating overall visualization image containing all clusters in '{output_dir}' using global bounds...")
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用所有聚类，不再过滤单车聚类
    all_clusters = clusters
    
    if not all_clusters:
        print("No clusters found, skipping visualization.")
        return
    
    # Set elegant plotting style, but don't use grid which would conflict with map
    plt.style.use('default')
    
    # 使用统一的图表尺寸和DPI
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    
    # Get more colors
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    
    # Sort by cluster size, so large clusters are drawn first (will be partially covered by small clusters)
    sorted_clusters = sorted(all_clusters, key=len)
    
    print(f"Processing {len(sorted_clusters)} clusters...")
    
    # Collect all trajectory points for boundary calculation - NO LONGER NEEDED FOR BOUNDS
    # all_points = []
    
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
                # Save complete trajectory points - NO LONGER NEEDED FOR BOUNDS
                # all_points.append(traj)
                
                # Draw trajectory line
                ax.plot(traj[:, 1], traj[:, 0], '-', color=color, linewidth=1.5, alpha=0.8, zorder=5)
                
                # Mark start point
                ax.plot(traj[0, 1], traj[0, 0], 'o', color=color, markersize=6, alpha=0.9, zorder=6)
                
                # Mark end point
                ax.plot(traj[-1, 1], traj[-1, 0], 's', color=color, markersize=6, alpha=0.9, zorder=6)
    
    # 强制使用全局边界和边距
    global_margin_x = (max_lon - min_lon) * MARGIN_FACTOR
    global_margin_y = (max_lat - min_lat) * MARGIN_FACTOR
    ax.set_xlim(min_lon - global_margin_x, max_lon + global_margin_x)
    ax.set_ylim(min_lat - global_margin_y, max_lat + global_margin_y)
    
    # 使用精确的地理比例设置
    center_lat = (min_lat + max_lat) / 2
    set_geo_aspect_ratio(ax, center_lat)
    
    # 添加地图背景 - 必须在设置边界和比例之后
    try:
        print("Adding high-resolution map background...")
        ctx.add_basemap(ax, crs='EPSG:4326', 
                       source=ctx.providers.CartoDB.Positron,
                       zoom=MAP_ZOOM_LEVEL,  # 使用固定的缩放级别
                       alpha=1.0)
        print("High-resolution map background added successfully.")
    except Exception as e:
        print(f"Error adding map background: {e}")
        try:
            print("Trying alternative map source...")
            ctx.add_basemap(ax, crs='EPSG:4326', 
                           source=ctx.providers.Stamen.Terrain,
                           zoom=MAP_ZOOM_LEVEL,  # 使用固定的缩放级别
                           alpha=1.0)
            print("Alternative map background added successfully.")
        except Exception as e2:
            print(f"Error adding alternative map background: {e2}")
            print("Continuing execution without map background...")
    
    # Draw direction arrows after setting boundaries and aspect ratio
    for i, cluster in enumerate(sorted_clusters):
        color = colors[i % len(colors)]
        for vid in cluster:
            traj = trajectories.get(vid)
            if traj is not None and len(traj) >= 5:
                # End point arrow
                end_idx = len(traj) - 1
                start_idx = max(0, end_idx - 1)
                dx = traj[end_idx, 1] - traj[start_idx, 1]
                dy = traj[end_idx, 0] - traj[start_idx, 0]
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > 0:
                    dx = dx / magnitude
                    dy = dy / magnitude
                    ax.quiver(traj[end_idx, 1], traj[end_idx, 0], 
                             dx, dy, color=color, scale=10, 
                             scale_units='inches', width=0.003,
                             headwidth=4, headlength=5, alpha=0.9, zorder=7)
                # Middle arrows
                if len(traj) >= 10:
                    num_arrows = min(5, len(traj) // 10)
                    if num_arrows > 0:
                        step = len(traj) // (num_arrows + 1)
                        for arrow_idx in range(1, num_arrows + 1):
                            idx = step * arrow_idx
                            if idx >= len(traj) - 1:
                                continue
                            prev_idx = max(0, idx - 3)
                            dx = traj[idx, 1] - traj[prev_idx, 1]
                            dy = traj[idx, 0] - traj[prev_idx, 0]
                            magnitude = np.sqrt(dx**2 + dy**2)
                            if magnitude > 0:
                                dx = dx / magnitude
                                dy = dy / magnitude
                                ax.quiver(traj[idx, 1], traj[idx, 0], 
                                         dx, dy, color=color, scale=12,
                                         scale_units='inches', width=0.002,
                                         headwidth=3, headlength=4, alpha=0.7, zorder=7)
    
    # 使用新的比例尺函数
    draw_exact_scale_bar(ax, SCALE_BAR_LENGTH_M)
    
    # Set chart title and labels
    plt.title('Vehicle Trajectory Cluster Visualization', fontsize=18, pad=20)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    
    # Add legend (use small size to avoid being too large)
    if len(legend_handles) > 20:
        legend_handles = legend_handles[:20]
        legend_labels = legend_labels[:20]
        legend_labels.append('... and more clusters')
    
    ax.legend(legend_handles, legend_labels, loc='upper right', 
               fontsize='small', frameon=True, framealpha=0.9)
    
    # 应用tight_layout确保所有元素都能正确显示
    plt.tight_layout()
    
    # 保存图表时也使用相同的DPI
    output_file = os.path.join(output_dir, 'all_trajectory_clusters_with_map.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    
    print(f"✅ Visualization of all clusters with map background has been saved as {output_file}")
    plt.close()

def main():
    filename = 'data.json'
    trajectories, vehicle_times = parse_vehicle_trajectories(filename)
    
    print(f"总车辆数: {len(trajectories)}")
    
    # 计算所有轨迹的全局边界
    all_lats = []
    all_lons = []
    for traj in trajectories.values():
        if traj.size > 0:
            all_lats.extend(traj[:, 0])
            all_lons.extend(traj[:, 1])
    
    if not all_lats or not all_lons:
        print("错误：未能从轨迹数据中提取有效的坐标。")
        return
        
    global_min_lat, global_max_lat = min(all_lats), max(all_lats)
    global_min_lon, global_max_lon = min(all_lons), max(all_lons)
    
    print(f"全局地理边界: Lat({global_min_lat:.4f} - {global_max_lat:.4f}), Lon({global_min_lon:.4f} - {global_max_lon:.4f})")
    
    # 使用改进的参数进行聚类
    print("\n=== 使用改进参数进行聚类 ===")
    similarity_threshold = 250.0   # DTW距离阈值(米)
    centroid_threshold = 600.0     # 质心预过滤阈值(米)
    min_samples = 2                # DBSCAN参数:最小样本数
    
    clusters = cluster_similar_trajectories_dbscan(
        trajectories, similarity_threshold, centroid_threshold, min_samples)
    
    # 收集所有出现在多车聚类中的车辆ID
    vehicles_in_multi_clusters = set()
    multi_vehicle_clusters = []
    
    for cluster in clusters:
        if len(cluster) > 1:  # 多车聚类
            multi_vehicle_clusters.append(cluster)
            for vid in cluster:
                vehicles_in_multi_clusters.add(vid)
    
    # 找出所有未被归入多车聚类的车辆ID
    all_vehicle_ids = set(trajectories.keys())
    single_vehicles = all_vehicle_ids - vehicles_in_multi_clusters
    
    print(f"\n统计信息:")
    print(f"总车辆数: {len(all_vehicle_ids)}")
    print(f"参与多车聚类的车辆数: {len(vehicles_in_multi_clusters)}")
    print(f"未参与聚类的单车数: {len(single_vehicles)}")
    print(f"多车聚类数: {len(multi_vehicle_clusters)}")
    
    # 显示多车聚类结果
    print("\n多车聚类结果:")
    output_clusters = []
    for idx, cluster in enumerate(multi_vehicle_clusters, 1):
        cluster_list = []
        for vid in cluster:
            time_info = vehicle_times.get(vid, {"start": "N/A", "end": "N/A"})
            cluster_list.append({
                "vehicle_id": vid,
                "start_time": time_info["start"],
                "end_time": time_info["end"]
            })
        print(f"Cluster {idx} ({len(cluster)} vehicles):")
        for item in cluster_list:
            print(f"  Vehicle {item['vehicle_id']} (from {item['start_time']} to {item['end_time']})")
        output_clusters.append({
            "cluster_id": idx,
            "vehicles": cluster_list
        })
    
    # 输出单车列表
    print("\n单个车辆ID列表:")
    single_output = []
    for vid in single_vehicles:
        time_info = vehicle_times.get(vid, {"start": "N/A", "end": "N/A"})
        vehicle_info = {
            "vehicle_id": vid,
            "start_time": time_info["start"],
            "end_time": time_info["end"]
        }
        single_output.append(vehicle_info)
        print(f"  Vehicle {vid} (from {time_info['start']} to {time_info['end']})")
    
    # 保存到JSON文件
    with open("trajectory_clusters.json", "w", encoding="utf-8") as f:
        summary = {
            "total_vehicles": len(all_vehicle_ids),
            "vehicles_in_clusters": len(vehicles_in_multi_clusters),
            "single_vehicles": len(single_vehicles),
            "multi_vehicle_clusters": len(multi_vehicle_clusters),
            "clusters": output_clusters
        }
        json.dump(summary, f, ensure_ascii=False, indent=4)
    
    # 单独保存单车数据到新文件
    # 更新：在循环中收集单车信息用于汇总文件
    # with open("single_vehicles.json", "w", encoding="utf-8") as f:
    #     single_summary = {
    #         "total_single_vehicles": len(single_vehicles),
    #         "single_vehicles_list": single_output # 使用之前收集的 single_output
    #     }
    #     json.dump(single_summary, f, ensure_ascii=False, indent=4)
    
    print("\n✅ 多车聚类汇总信息已保存到 trajectory_clusters.json")
    # print("✅ 单车列表汇总信息已保存到 single_vehicles.json") # 更新：稍后保存
    
    # 创建输出目录
    multi_vehicle_dir = "multi_vehicle_clusters"
    single_vehicle_dir = "single_vehicle_clusters"
    os.makedirs(multi_vehicle_dir, exist_ok=True) # 确保目录存在
    os.makedirs(single_vehicle_dir, exist_ok=True) # 确保目录存在

    # --- 新增：为每个多车聚类保存单独的JSON文件 ---
    print("\n正在为多车聚类保存单独的JSON文件...")
    for idx, cluster in enumerate(tqdm(multi_vehicle_clusters, desc="Saving multi-vehicle JSONs"), 1):
        cluster_list = []
        for vid in cluster:
            time_info = vehicle_times.get(vid, {"start": "N/A", "end": "N/A"})
            cluster_list.append({
                "vehicle_id": vid,
                "start_time": time_info["start"],
                "end_time": time_info["end"]
            })
        
        cluster_data = {
            "cluster_id": idx,
            "num_vehicles": len(cluster),
            "vehicles": cluster_list
        }
        
        json_filename = os.path.join(multi_vehicle_dir, f'cluster_{idx}_trajectories.json')
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(cluster_data, f, ensure_ascii=False, indent=4)
    # ----------------------------------------------

    # 可视化多车聚类 - 保存到多车聚类目录
    # 传递全局边界
    plot_all_clusters_in_one_image(trajectories, multi_vehicle_clusters, multi_vehicle_dir, 
                                   global_min_lon, global_max_lon, global_min_lat, global_max_lat)
    plot_all_clusters(trajectories, multi_vehicle_clusters, multi_vehicle_dir, 
                      global_min_lon, global_max_lon, global_min_lat, global_max_lat)
    
    # 生成单车轨迹的可视化 - 为每辆单车创建一个单独的聚类，保存到单车目录
    single_vehicle_clusters = [[vid] for vid in single_vehicles]
    
    # --- 新增：为每个单车聚类保存单独的JSON文件 ---
    print(f"\n正在为{len(single_vehicle_clusters)}辆单车生成轨迹可视化和单独的JSON文件...")
    single_output_list_for_summary = [] # 用于汇总文件的列表
    for idx, cluster in enumerate(tqdm(single_vehicle_clusters, desc="Processing single vehicles"), 1):
        vid = cluster[0] # 获取单车ID
        time_info = vehicle_times.get(vid, {"start": "N/A", "end": "N/A"})
        vehicle_info = {
            "vehicle_id": vid,
            "start_time": time_info["start"],
            "end_time": time_info["end"]
        }
        single_output_list_for_summary.append(vehicle_info) # 添加到汇总列表

        # 创建单车JSON数据
        single_vehicle_data = {
            "cluster_id": idx, # 使用与图像文件一致的索引
            "num_vehicles": 1,
            "vehicles": [vehicle_info]
        }

        # 定义JSON文件名并保存
        json_filename = os.path.join(single_vehicle_dir, f'cluster_{idx}_trajectories.json')
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(single_vehicle_data, f, ensure_ascii=False, indent=4)
    # -------------------------------------------------

    # --- 更新：保存单车汇总JSON文件 ---
    with open("single_vehicles.json", "w", encoding="utf-8") as f:
        single_summary = {
            "total_single_vehicles": len(single_vehicles),
            "single_vehicles_list": single_output_list_for_summary # 使用新收集的列表
        }
        json.dump(single_summary, f, ensure_ascii=False, indent=4)
    print("\n✅ 单车列表汇总信息已保存到 single_vehicles.json")
    # ---------------------------------

    # 可视化单车 - 传递全局边界 (这部分不变，但在上面的循环之后执行)
    plot_all_clusters(trajectories, single_vehicle_clusters, single_vehicle_dir, 
                      global_min_lon, global_max_lon, global_min_lat, global_max_lat)

if __name__ == '__main__':
    main()