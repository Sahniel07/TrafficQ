import json
import math
import numpy as np
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import contextily as ctx
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import argparse
import glob
import pathlib

# Global parameters
MAP_ZOOM_LEVEL = 17  # Map zoom level
MARGIN_FACTOR = 0.15  # Boundary margin factor
VIDEO_DPI = 120  # Video DPI, lower to reduce file size
FIGURE_SIZE = (10, 10)  # Square figure size for 1:1 aspect ratio
FPS = 24  # Video frame rate
SCALE_BAR_LENGTH_M = 200  # Scale bar length (meters)
VEHICLE_MARKER_SIZE = 80  # Vehicle marker size
TRAIL_LENGTH = 40  # Vehicle trail length (increase for longer trails)
INTERPOLATION_FACTOR = 5  # Interpolation factor for smoothness (increase for smoother trails)
COLLISION_HIGHLIGHT_SECONDS = 2.0  # Time to highlight before and after collision point (seconds)

# Add a global default FPS, but allow override via args
DEFAULT_FPS = 24

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two geographic points (in meters) using Haversine formula.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_vehicle_trajectories(filename):
    """
    Parse vehicle trajectory data from JSON file.
    Returns three dictionaries:
      trajectories: {vehicle_id: trajectory_array, ...}
         Each trajectory_array is an Nx3 numpy array containing [lat, lon, timestamp].
      vehicle_times: {vehicle_id: {"start": start_time_str, "end": end_time_str}, ...}
      vehicle_classes: {vehicle_id: {"class": class_name, "color": color_hex}, ...}
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    trajectories = {}
    vehicle_times = {}
    vehicle_classes = {}
    
    for edge in data['data']['entities']['edges']:
        node = edge['node']
        vehicle_id = node['id']
        traj = []
        times = []
        timestamps = []
        
        # Record vehicle type and color information
        vehicle_class = node['class']['name']
        vehicle_color = f"#{node['class']['color']}" if node['class']['color'] else "#72deee"
        vehicle_classes[vehicle_id] = {
            "class": vehicle_class,
            "color": vehicle_color
        }
        
        # Parse trajectory points
        for pt in node['trajectory']:
            t = datetime.fromisoformat(pt['time'])
            times.append(t)
            # Convert time to float timestamp (seconds)
            timestamp = t.timestamp()
            timestamps.append(timestamp)
            lon, lat = pt['coordinateLongLat']
            traj.append([lat, lon, timestamp])
        
        if traj:
            traj = np.array(traj)
            trajectories[vehicle_id] = traj
            start_time = times[0].strftime("%Y-%m-%d %H:%M:%S")
            end_time = times[-1].strftime("%Y-%m-%d %H:%M:%S")
            vehicle_times[vehicle_id] = {"start": start_time, "end": end_time}
    
    return trajectories, vehicle_times, vehicle_classes

def interpolate_trajectory(traj, factor=INTERPOLATION_FACTOR):
    """
    Interpolate trajectory points to make animation smoother.
    
    Parameters:
    traj -- Original trajectory points array [lat, lon, timestamp]
    factor -- Interpolation factor, representing the number of points to insert
    
    Returns:
    Interpolated trajectory points array
    """
    if len(traj) <= 1 or factor <= 1:
        return traj
    
    result = []
    for i in range(len(traj) - 1):
        start_lat, start_lon, start_time = traj[i]
        end_lat, end_lon, end_time = traj[i + 1]
        
        result.append(traj[i])
        
        # Calculate interpolation points
        for j in range(1, factor):
            ratio = j / factor
            lat = start_lat + (end_lat - start_lat) * ratio
            lon = start_lon + (end_lon - start_lon) * ratio
            timestamp = start_time + (end_time - start_time) * ratio
            result.append([lat, lon, timestamp])
    
    # Add the last point
    result.append(traj[-1])
    
    return np.array(result)

def parallel_interpolate(args):
    """
    Helper function for parallel interpolation processing.
    """
    vid, traj, factor = args
    return vid, interpolate_trajectory(traj, factor)

def set_geo_aspect_ratio(ax, center_lat):
    """
    Set accurate geographic aspect ratio so longitude and latitude display with the same 
    geographic distance visually.
    
    Parameters:
    ax -- matplotlib axis object
    center_lat -- center latitude
    """
    # Calculate longitude/latitude ratio at the given latitude
    # At typical latitude, 1 degree of longitude is approximately 111,320 * cos(latitude) meters
    # 1 degree of latitude is approximately 111,320 meters
    # We want to ensure 1 degree of longitude and 1 degree of latitude cover the same distance visually
    
    # Force a perfect 1:1 aspect ratio (fixed ratio)
    ax.set_aspect('equal')
    
    # Remove scientific notation from axis labels
    ax.ticklabel_format(useOffset=False, style='plain')

def draw_exact_scale_bar(ax, scale_bar_length_m=SCALE_BAR_LENGTH_M):
    """
    Draw a scale bar on the map with exact length.
    
    Parameters:
    ax -- matplotlib axis object
    scale_bar_length_m -- Scale bar length (meters)
    """
    # Get current figure display range
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Get center point coordinates
    center_lon = (x_min + x_max) / 2
    center_lat = (y_min + y_max) / 2
    
    # Calculate longitude scale at current latitude
    lon_scale = scale_bar_length_m / (111320 * np.cos(np.radians(center_lat)))
    
    # Calculate scale bar start position
    x_start = x_min + (x_max - x_min) * 0.1
    y_pos = y_min + (y_max - y_min) * 0.05
    x_end = x_start + lon_scale
    
    # Draw scale bar line
    ax.plot([x_start, x_end], [y_pos, y_pos], 'k-', linewidth=2.5, zorder=20)
    
    # Add label
    scale_label = f'{int(scale_bar_length_m)} m'
    y_offset = (y_max - y_min) * 0.02
    
    ax.text((x_start + x_end) / 2, y_pos + y_offset, scale_label, 
            ha='center', va='bottom', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
            zorder=20)

def create_trajectory_animation(trajectories, vehicle_classes, output_file,
                            duration_seconds=None, collision_info=None, time_range=None, fps=DEFAULT_FPS):
    """
    Create video animation from vehicle trajectory data.
    
    Parameters:
    trajectories -- Trajectory dictionary {vehicle_id: trajectory_array, ...}
    vehicle_classes -- Vehicle type dictionary {vehicle_id: {"class": class_name, "color": color_hex}, ...}
    output_file -- Output video file path
    duration_seconds -- Video duration (seconds), if None uses actual data duration
    collision_info -- Collision-related information, including time, vehicle IDs, positions, etc.
    time_range -- Optional tuple (min_time, max_time) specifying the exact time range for the animation.
    fps -- Frames per second for the output video.
    """
    print("Preparing to generate vehicle trajectory animation video...")
    
    # Create output directory (if it doesn't exist)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
    
    # Calculate global boundaries and time range
    all_lats = []
    all_lons = []
    all_times = []
    
    for traj in trajectories.values():
        if traj.size > 0:
            all_lats.extend(traj[:, 0])
            all_lons.extend(traj[:, 1])
            all_times.extend(traj[:, 2])
    
    if not all_lats or not all_lons:
        print("Error: Could not extract valid coordinates from trajectory data.")
        return
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    # Determine time range for the animation frames
    if time_range:
        min_time, max_time = time_range
        print(f"Using provided time range for animation: {datetime.fromtimestamp(min_time)} - {datetime.fromtimestamp(max_time)}")
    else:
        # Fallback: Calculate from all trajectories if no range provided
        all_times_calc = []
        for traj in trajectories.values():
            if traj.size > 0:
                all_times_calc.extend(traj[:, 2])
        if not all_times_calc:
            print("Error: No valid timestamps found in trajectory data.")
            return
        min_time, max_time = min(all_times_calc), max(all_times_calc)
        print(f"Calculated time range from data: {datetime.fromtimestamp(min_time)} - {datetime.fromtimestamp(max_time)}")
    
    # Calculate map center point
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Ensure perfectly square map area by making the degrees covered in longitude 
    # and latitude equal, taking the larger of the two
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    max_span = max(lon_span, lat_span)
    
    # Recalculate boundaries to make a perfect square in terms of degrees
    # This ensures the map area is square in coordinate space
    min_lon = center_lon - max_span/2 * (1 + MARGIN_FACTOR)
    max_lon = center_lon + max_span/2 * (1 + MARGIN_FACTOR)
    min_lat = center_lat - max_span/2 * (1 + MARGIN_FACTOR)
    max_lat = center_lat + max_span/2 * (1 + MARGIN_FACTOR)
    
    # Recalculate margins based on new boundaries
    lon_margin = (max_lon - min_lon) * MARGIN_FACTOR
    lat_margin = (max_lat - min_lat) * MARGIN_FACTOR
    
    print(f"Geographic boundaries: Lat({min_lat:.6f} - {max_lat:.6f}), Lon({min_lon:.6f} - {max_lon:.6f})")
    print(f"Time range: {datetime.fromtimestamp(min_time)} - {datetime.fromtimestamp(max_time)}")
    
    # Parallel interpolation processing for trajectory data
    print("Interpolating trajectories to make animation smoother...")
    interpolation_args = [(vid, traj, INTERPOLATION_FACTOR) for vid, traj in trajectories.items()]
    
    # Use parallel processing for interpolation
    with Pool() as pool:
        interpolated_results = list(tqdm(
            pool.imap(parallel_interpolate, interpolation_args),
            total=len(interpolation_args),
            desc="Interpolation"
        ))
    
    # Process interpolation results
    interpolated_trajectories = {}
    for vid, interp_traj in interpolated_results:
        interpolated_trajectories[vid] = interp_traj
    
    # Create time index data structure
    print("Building time index data structure...")
    # Convert all trajectory points to pandas DataFrame for processing
    trajectory_data = []
    
    for vid, traj in interpolated_trajectories.items():
        vehicle_color = vehicle_classes.get(vid, {}).get("color", "#72deee")
        vehicle_class = vehicle_classes.get(vid, {}).get("class", "car")
        
        for i, (lat, lon, timestamp) in enumerate(traj):
            trajectory_data.append({
                'vehicle_id': vid,
                'lat': lat,
                'lon': lon,
                'timestamp': timestamp,
                'color': vehicle_color,
                'class': vehicle_class
            })
    
    df = pd.DataFrame(trajectory_data)
    
    # Determine video frame time interval
    real_duration = max_time - min_time  # Actual data duration (seconds) of the potentially focused range

    if duration_seconds is None:
        # If duration is not specified (shouldn't happen if main/caller calculates it), use default
        duration_seconds = min(max(10, real_duration), 120) # At least 10 seconds, at most 120 seconds
        print(f"Warning: Duration not specified, defaulting to {duration_seconds:.2f} seconds")
    elif duration_seconds <= 0:
         print(f"Warning: Calculated duration is {duration_seconds:.2f} seconds. Setting to a minimum of 0.1s for animation.")
         duration_seconds = 0.1 # Avoid division by zero later, ensure at least a few frames

    print(f"Actual data time range duration: {real_duration:.2f} seconds")
    print(f"Target video duration: {duration_seconds:.2f} seconds")

    # Calculate total frames
    total_frames = max(1, int(duration_seconds * fps)) # Ensure at least 1 frame, use passed fps
    print(f"Total frames: {total_frames}, Frame rate: {fps} fps")

    # Create equally spaced time points array for each video frame
    frame_times = np.linspace(min_time, max_time, total_frames)
    
    # Create figure and axis
    # Create a square figure with equal aspect ratio
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=VIDEO_DPI)
    
    # Turn off axes to make it look cleaner, similar to the reference images
    ax.axis('on')  # You can change to 'off' if you want no axes at all
    
    # Set specific limits for longitude (x-axis) and latitude (y-axis)
    # to ensure they cover exactly the same distance in degrees
    ax.set_xlim(min_lon - lon_margin, max_lon + lon_margin)
    ax.set_ylim(min_lat - lat_margin, max_lat + lat_margin)
    
    # Force a strict 1:1 aspect ratio
    # This is crucial for making sure longitude and latitude are visually equal
    set_geo_aspect_ratio(ax, center_lat)
    
    # Additional aspect ratio enforcement for perfect square map
    ax.set_aspect('equal', adjustable='box')
    
    # Get the figure dimensions and make them perfectly square if needed
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    # If aspect ratio is not 1:1, force it
    if abs(width - height) > 0.01:  # Allow small difference due to floating point
        # Make both equal to the larger dimension
        max_dim = max(width, height)
        fig.set_figwidth(max_dim * fig.get_dpi() / VIDEO_DPI)
        fig.set_figheight(max_dim * fig.get_dpi() / VIDEO_DPI)
        # Force redraw
        fig.canvas.draw()
    
    # Add map background
    try:
        print(f"Adding map background with zoom level {MAP_ZOOM_LEVEL}...")
        ctx.add_basemap(ax, crs='EPSG:4326', 
                      source=ctx.providers.CartoDB.Positron,
                      zoom=MAP_ZOOM_LEVEL,
                      alpha=1.0)
    except Exception as e:
        print(f"Failed to add main map background: {e}")
        try:
            print("Trying backup map source...")
            ctx.add_basemap(ax, crs='EPSG:4326', 
                          source=ctx.providers.Stamen.Terrain,
                          zoom=MAP_ZOOM_LEVEL,
                          alpha=1.0)
        except Exception as e2:
            print(f"Failed to add backup map background: {e2}")
            print("Continuing execution without map background...")
    
    # Add scale bar
    draw_exact_scale_bar(ax, SCALE_BAR_LENGTH_M)
    
    # Create vehicle markers
    scats = {}  # Store scatter plot objects for each vehicle
    trails = {}  # Store trail line objects for each vehicle
    
    # Add time display text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    # Make axes labels similar to the reference image
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # If collision information is available, create collision markers
    collision_marker = None
    collision_text = None
    if collision_info:
        # 不再创建碰撞点标记（五角星）
        # 只创建一个空的散点对象作为占位符，但不会显示
        collision_marker = ax.scatter([], [], s=0, alpha=0)
        
        # Create collision information text (initially invisible)
        severity_color = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'blue'}.get(collision_info['severity'], 'red')
        collision_text = ax.text(0.5, 0.1, '', transform=ax.transAxes, fontsize=14,
                                horizontalalignment='center', verticalalignment='bottom',
                                bbox=dict(facecolor='white', alpha=0, edgecolor=severity_color, linewidth=2),
                                color=severity_color, alpha=0, zorder=15)
    
    # Initialize scatter plots and trail lines for all vehicles
    for vid in interpolated_trajectories.keys():
        color = vehicle_classes.get(vid, {}).get("color", "#72deee")
        # Use bold or special effects for collision vehicles
        is_collision_vehicle = collision_info and vid in [collision_info['vehicle_a'], collision_info['vehicle_b']]
        marker_size = VEHICLE_MARKER_SIZE * (1.5 if is_collision_vehicle else 1.0)
        
        # Create empty scatter plot object
        scat = ax.scatter([], [], s=marker_size, color=color, 
                          marker='o', edgecolor='black', linewidth=1, alpha=0.9, zorder=10)
        scats[vid] = scat
        
        # Create empty trail line object
        line_width = 3 if is_collision_vehicle else 2
        trail, = ax.plot([], [], '-', color=color, linewidth=line_width, alpha=0.7, zorder=9)
        trails[vid] = trail
    
    # Create title
    title_text = 'Vehicle Trajectory Animation'
    if collision_info:
        title_text = f'Vehicle Collision Event - Severity: {collision_info["severity"]}'
    # Make title text larger and more prominent
    title = ax.set_title(title_text, fontsize=16, fontweight='bold')
    
    # Add legend
    if len(set(vc["class"] for vc in vehicle_classes.values())) > 1:
        # If there are multiple vehicle types, add type legend
        unique_classes = {}
        for vid, info in vehicle_classes.items():
            class_name = info["class"]
            class_color = info["color"]
            if class_name not in unique_classes:
                unique_classes[class_name] = class_color
        
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=8, label=class_name) 
                          for class_name, color in unique_classes.items()]
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Initialization function
    def init():
        for scat in scats.values():
            scat.set_offsets(np.empty((0, 2)))
        for trail in trails.values():
            trail.set_data([], [])
        time_text.set_text('')
        
        # Initialize collision marker and text
        if collision_marker:
            collision_marker.set_offsets(np.empty((0, 2)))
        if collision_text:
            collision_text.set_text('')
            collision_text.set_alpha(0)
            
        return [*scats.values(), *trails.values(), time_text] + \
               ([collision_marker, collision_text] if collision_info else [])
    
    # Update function
    def update(frame):
        # Current frame timestamp
        current_time = frame_times[frame]
        
        # Update time text
        time_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        time_text.set_text(f'Time: {time_str}')
        
        # Update position and trail for each vehicle
        visible_vehicles = 0
        
        # Check if close to collision time
        near_collision = False
        if collision_info:
            collision_time = collision_info['time']
            time_diff = abs(current_time - collision_time)
            near_collision = time_diff < COLLISION_HIGHLIGHT_SECONDS
            
            # Update collision marker
            if near_collision:
                # 不再显示碰撞点标记（五角星）
                collision_marker.set_alpha(0)
                
                # Display collision information text
                distance = collision_info['distance']
                severity = collision_info['severity']
                collision_text.set_text(f"Collision Distance: {distance}m | Severity: {severity}")
                collision_text.set_alpha(1.0)
                collision_text.get_bbox_patch().set_alpha(0.8)
            else:
                collision_marker.set_offsets(np.empty((0, 2)))
                collision_marker.set_alpha(0)
                collision_text.set_alpha(0)
                collision_text.get_bbox_patch().set_alpha(0)
        
        for vid, scat in scats.items():
            trail = trails[vid]
            traj = interpolated_trajectories[vid]
            
            # Find all trajectory points before current time
            mask = traj[:, 2] <= current_time
            past_traj = traj[mask]
            
            # Check if in valid time range (has appeared and has not disappeared)
            if len(past_traj) > 0 and current_time <= traj[-1, 2]:
                visible_vehicles += 1
                # Update vehicle current position (last trajectory point)
                current_pos = past_traj[-1, :2]
                # Convert to [lon, lat] format for scatter
                scat.set_offsets([(past_traj[-1, 1], past_traj[-1, 0])])
                
                # Update trail line - show only the most recent TRAIL_LENGTH points
                trail_points = past_traj[-TRAIL_LENGTH:] if len(past_traj) > TRAIL_LENGTH else past_traj
                trail.set_data(trail_points[:, 1], trail_points[:, 0])
                
                # If collision vehicle and close to collision time, highlight
                is_collision_vehicle = collision_info and vid in [collision_info['vehicle_a'], collision_info['vehicle_b']]
                if is_collision_vehicle and near_collision:
                    # 保持边框颜色稳定，不要闪烁
                    scat.set_edgecolor('black')
                    scat.set_linewidth(1.5)
                    # Enlarge vehicle marker
                    current_size = scat.get_sizes()[0]
                    # 固定大小，不要变化
                    original_size = VEHICLE_MARKER_SIZE * 1.5 if is_collision_vehicle else VEHICLE_MARKER_SIZE
                    scat.set_sizes([original_size])
                else:
                    scat.set_edgecolor('black')
                    scat.set_linewidth(1)
                    # Restore original size
                    original_size = VEHICLE_MARKER_SIZE * (1.5 if is_collision_vehicle else 1.0)
                    scat.set_sizes([original_size])
            else:
                # If vehicle has not appeared or has disappeared, set to empty
                scat.set_offsets(np.empty((0, 2)))
                trail.set_data([], [])
        
        # Update title to show current visible vehicles count
        if collision_info and near_collision:
            # 保持标题稳定，不要改变颜色
            title.set_text(f'⚠️ Vehicle Collision Event - Severity: {collision_info["severity"]} ⚠️')
            title.set_color('#FF0000')  # 固定红色
        else:
            title.set_text(f'Vehicle Trajectory Animation (Visible Vehicles: {visible_vehicles})')
            title.set_color('#000000')  # 固定黑色
        
        return [*scats.values(), *trails.values(), time_text, title] + \
               ([collision_marker, collision_text] if collision_info else [])
    
    # Create animation
    print(f"Generating animation with {total_frames} frames...")
    anim = animation.FuncAnimation(fig, update, frames=total_frames, 
                                  init_func=init, blit=True, 
                                  interval=1000/fps)
    
    # Use constrained_layout instead of tight_layout for better handling of aspect ratio
    fig.set_constrained_layout(True)
    
    # Save as video file
    print(f"Saving video to {output_file}...")
    # Use passed fps in writer
    writer = animation.FFMpegWriter(fps=fps, bitrate=5000,
                                    metadata=dict(title="Vehicle Trajectory Collision Animation",
                                                 comment="Generated from vehicle GPS trajectory and collision detection data"))
    
    # Force final check of aspect ratio before saving
    fig.set_constrained_layout(True)
    plt.tight_layout()
    
    # Final adjustments to ensure square aspect ratio in the figure
    fig.set_figwidth(FIGURE_SIZE[0])
    fig.set_figheight(FIGURE_SIZE[1])
    # Redraw the canvas with the adjusted size
    fig.canvas.draw()
    
    anim.save(output_file, writer=writer)
    plt.close()
    
    print(f"✅ Video successfully saved to: {output_file}")
    return output_file

def parse_collision_data(collision_file):
    """
    Parse collision detection report JSON data, extract the most severe collision event.
    
    Parameters:
    collision_file -- Collision report JSON file path
    
    Returns:
    The most severe collision event data
    """
    print(f"Reading collision data file: {collision_file}")
    
    try:
        with open(collision_file, 'r', encoding='utf-8') as f:
            collision_data = json.load(f)
            
        # Check if there is collision data
        if not collision_data.get('collisions', []):
            print("Warning: No collision events found in the report")
            return None
            
        # Find the most severe collision (collision with smallest distance)
        collisions = collision_data['collisions']
        most_severe = min(collisions, key=lambda x: x['distance'])
        
        severity = most_severe.get('severity', 'UNKNOWN')
        distance = most_severe.get('distance', 0)
        
        # Extract related vehicle IDs
        vehicle_a_id = most_severe['vehicle_a']['id']
        vehicle_b_id = most_severe['vehicle_b']['id']
        
        # Extract collision time
        collision_time = most_severe.get('average_time', 0)
        
        print(f"Found the most severe collision event:")
        print(f"  • Vehicles: {vehicle_a_id} and {vehicle_b_id}")
        print(f"  • Distance: {distance} meters")
        print(f"  • Severity: {severity}")
        print(f"  • Time: {datetime.fromtimestamp(collision_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        return most_severe
    except Exception as e:
        print(f"Error parsing collision data: {e}")
        return None

def parse_all_collision_events(collision_file):
    """
    Parse collision detection report JSON data and return all collision events.

    Parameters:
    collision_file -- Collision report JSON file path

    Returns:
    A list of all collision event dictionaries found in the report, or an empty list if none.
    """
    print(f"Reading collision data file: {collision_file}")

    try:
        with open(collision_file, 'r', encoding='utf-8') as f:
            collision_data = json.load(f)

        # Check if there is collision data
        collisions = collision_data.get('collisions', [])
        if not collisions:
            print("Warning: No collision events found in the report")
            return []

        print(f"Found {len(collisions)} collision events in the report.")
        return collisions # Return the list of all events

    except Exception as e:
        print(f"Error parsing collision data file {collision_file}: {e}")
        return [] # Return empty list on error

def generate_single_video(trajectories, vehicle_classes, # Now takes data directly
                         output_file, fps=DEFAULT_FPS,
                         duration=None, time_scale=1.0,
                         focus_on_collision=False, # Whether to adjust time window around the event
                         specific_collision_event=None): # The specific event object to visualize
    """
    Generates a single trajectory animation video based on the provided trajectory data,
    focusing on a specific collision event if provided.

    Args:
        trajectories (dict): Dictionary of {vehicle_id: trajectory_array}.
        vehicle_classes (dict): Dictionary of vehicle class info {vehicle_id: {"class": ..., "color": ...}}.
        output_file (str): Path for the output video file.
        fps (int): Frames per second for the video.
        duration (float, optional): Desired video duration in seconds. Defaults to None (auto-calculated).
        time_scale (float): Time scale factor. Defaults to 1.0 (real-time).
        focus_on_collision (bool): If True and specific_collision_event is provided, adjust time range around the event.
        specific_collision_event (dict, optional): The specific collision event dictionary to focus on.
    """
    event_id_str = f" (Event ID: {specific_collision_event['id']})" if specific_collision_event else ""
    print(f"\n--- Generating Video: {pathlib.Path(output_file).name}{event_id_str} ---")
    print(f"Using trajectory data for {len(trajectories)} vehicles.")

    # --- Collision Handling based on specific_collision_event ---
    collision_info_for_animation = None
    vehicle_a_id = None
    vehicle_b_id = None

    if specific_collision_event:
        print(f"Using specific collision event ID: {specific_collision_event.get('id', 'N/A')}")
        # Prepare collision_info dictionary for the animation function
        # Ensure the event has the necessary structure
        if ('vehicle_a' in specific_collision_event and 'id' in specific_collision_event['vehicle_a'] and 'position' in specific_collision_event['vehicle_a'] and
            'vehicle_b' in specific_collision_event and 'id' in specific_collision_event['vehicle_b'] and 'position' in specific_collision_event['vehicle_b']):
            collision_info_for_animation = {
                'time': specific_collision_event['average_time'],
                'vehicle_a': specific_collision_event['vehicle_a']['id'],
                'vehicle_b': specific_collision_event['vehicle_b']['id'],
                'position_a': (specific_collision_event['vehicle_a']['position']['longitude'],
                              specific_collision_event['vehicle_a']['position']['latitude']),
                'position_b': (specific_collision_event['vehicle_b']['position']['longitude'],
                              specific_collision_event['vehicle_b']['position']['latitude']),
                'distance': specific_collision_event['distance'],
                'severity': specific_collision_event['severity']
            }
            vehicle_a_id = specific_collision_event['vehicle_a']['id']
            vehicle_b_id = specific_collision_event['vehicle_b']['id']
        else:
            print("Warning: Provided specific_collision_event is missing required keys. Cannot use for animation focus.")
            specific_collision_event = None # Invalidate if malformed
            collision_info_for_animation = None

    # Vehicle filtering is now done *before* calling this function.

    # Determine duration, time range
    duration_seconds = duration
    time_range = None

    if trajectories: # Ensure there's data to process
        all_times = [p[2] for traj in trajectories.values() for p in traj if len(traj) > 0]
        if all_times:
            global_min_time = min(all_times)
            global_max_time = max(all_times)
            real_duration = global_max_time - global_min_time

            # --- Logic to determine duration and time_range ---
            if focus_on_collision and specific_collision_event and collision_info_for_animation:
                # Focus on the specific collision event time
                collision_time = collision_info_for_animation['time']
                buffer_seconds = 10 # Time window around the collision
                start_time = max(global_min_time, collision_time - buffer_seconds)
                end_time = min(global_max_time, collision_time + buffer_seconds)

                if end_time <= start_time:
                    print(f"Warning: Cannot focus on collision time ({datetime.fromtimestamp(collision_time)}). It might be outside the data time range plus buffer.")
                    # Fallback: Use default duration and full time range
                    duration_seconds = min(max(10, real_duration), 120)
                    time_range = (global_min_time, global_max_time)
                    print(f"Falling back to default duration: {duration_seconds:.2f} seconds over full time range.")
                else:
                    duration_seconds = end_time - start_time
                    time_range = (start_time, end_time)
                    print(f"Focusing on collision event, video duration adjusted to {duration_seconds:.2f} seconds")

            elif duration is None: # Auto-calculate duration if not focused or specified
                if time_scale != 1.0:
                    # Use time scaling
                    if real_duration > 0 and time_scale > 0:
                        duration_seconds = real_duration / time_scale
                        print(f"Applying time scale factor {time_scale}, video duration set to {duration_seconds:.2f} seconds")
                        time_range = (global_min_time, global_max_time) # Use full range for scaled video
                    else:
                        print(f"Warning: Cannot apply time scale factor {time_scale} due to zero real duration or invalid scale. Using default duration.")
                        duration_seconds = min(max(10, real_duration), 120)
                        time_range = (global_min_time, global_max_time)
                else:
                    # Default duration calculation (no collision focus, no scaling)
                    duration_seconds = min(max(10, real_duration), 120) # At least 10s, at most 120s
                    print(f"Using default video duration: {duration_seconds:.2f} seconds")
                    time_range = (global_min_time, global_max_time) # Use full range
            else:
                # Duration specified by user
                duration_seconds = duration
                time_range = (global_min_time, global_max_time) # Use full data time range
                print(f"Using user-specified duration: {duration_seconds:.2f} seconds")

        else: # No valid timestamps found
            print("Error: No valid timestamps found in trajectory data. Cannot determine time range or duration.")
            return None # Cannot proceed without time info
    else: # No trajectories left after filtering or originally empty
        print("Error: No trajectory data available to generate animation.")
        return None

    # Final checks before calling animation function
    if time_range is None:
        print("Error: Could not determine a valid time range for the animation.")
        return None
    if duration_seconds is None or duration_seconds <= 0:
        print(f"Warning: Final duration is invalid ({duration_seconds}). Setting to 10 seconds.")
        duration_seconds = 10

    # Generate video
    try:
        output_path = create_trajectory_animation(trajectories, vehicle_classes, output_file,
                                  duration_seconds=duration_seconds,
                                  collision_info=collision_info_for_animation,
                                  time_range=time_range, # Pass the calculated time range
                                  fps=fps) # Pass fps
        print(f"--- Finished: {pathlib.Path(output_file).name} ---")
        return output_path
    except Exception as e:
        print(f"Error during animation generation for {output_file} (Event: {specific_collision_event.get('id', 'N/A') if specific_collision_event else 'None'}): {e}")
        return None

def process_collision_report(collision_report_file, data_file, output_dir, target_severity="MEDIUM", fps=DEFAULT_FPS):
    """
    Processes a specific collision report file against a specific data file.
    Iterates through each collision event in the report, and for events
    matching the target severity, generates a focused video using only the involved vehicles'
    trajectories from the specified data file. Saves videos to the output directory.

    Args:
        collision_report_file (str): Path to the collision report JSON file.
        data_file (str): Path to the trajectory data JSON file (e.g., data_2.json).
        output_dir (str): Directory where matching videos will be saved.
        target_severity (str): The collision severity level to filter for (case-insensitive).
        fps (int): Frames per second for the generated videos.
    """
    print(f"\n=== Processing Report: {collision_report_file} against Data: {data_file} ===")
    print(f"Output Directory: {output_dir}, Target Severity: {target_severity}, FPS: {fps}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    target_severity_upper = target_severity.upper()
    generated_count = 0

    # 1. Parse the specified collision report file
    all_events = parse_all_collision_events(collision_report_file)
    if not all_events:
        print(f"No valid collision events found in {collision_report_file}. Exiting.")
        return

    # 2. Parse the specified data file ONCE
    print(f"\nParsing data file: {data_file}")
    try:
        all_trajectories, _, all_vehicle_classes = parse_vehicle_trajectories(data_file)
        if not all_trajectories:
            print(f"Error: No trajectory data found in {data_file}. Cannot generate videos.")
            return
        print(f"Parsed trajectory data for {len(all_trajectories)} vehicles.")
    except Exception as e:
        print(f"Error parsing data file {data_file}: {e}. Cannot generate videos.")
        return

    # 3. Iterate through each event found in the collision report
    print(f"\nIterating through {len(all_events)} collision events...")
    for event in tqdm(all_events, desc="Processing Events"):
        print(f"---")
        event_id = event.get('id', 'N/A')
        event_severity = event.get('severity', 'UNKNOWN').upper()
        print(f"Checking Event ID: {event_id} (Severity: {event.get('severity', 'UNKNOWN')})")

        # 4. Filter by target severity for THIS event
        if event_severity == target_severity_upper:
            print(f"   Severity matches target ({target_severity}). Processing event...")

            # 5. Extract vehicle IDs for this event
            vehicle_a_id = event.get('vehicle_a', {}).get('id')
            vehicle_b_id = event.get('vehicle_b', {}).get('id')

            if not vehicle_a_id or not vehicle_b_id:
                print(f"   Warning: Event {event_id} is missing vehicle IDs. Skipping video generation.")
                continue

            print(f"   Event Vehicles: {vehicle_a_id}, {vehicle_b_id}")

            # 6. Filter trajectories and classes for ONLY these two vehicles
            filtered_trajectories = {vid: traj for vid, traj in all_trajectories.items() if vid in [vehicle_a_id, vehicle_b_id]}
            filtered_classes = {vid: cls for vid, cls in all_vehicle_classes.items() if vid in [vehicle_a_id, vehicle_b_id]}
            
            # 强制设置两辆车使用不同颜色
            # 为碰撞事件中的车辆分配不同的颜色
            colors = ["#e74c3c", "#3498db"]  # 红色和蓝色，鲜明的对比色
            
            # 确保两个车辆ID按顺序获取对应的颜色
            if vehicle_a_id in filtered_classes and vehicle_b_id in filtered_classes:
                filtered_classes[vehicle_a_id]["color"] = colors[0]  # 第一辆车使用红色
                filtered_classes[vehicle_b_id]["color"] = colors[1]  # 第二辆车使用蓝色
                print(f"   已分配不同颜色给两辆车: {vehicle_a_id} → {colors[0]}, {vehicle_b_id} → {colors[1]}")

            # 7. Check if both vehicles were found in the data file
            if vehicle_a_id not in filtered_trajectories:
                print(f"   Warning: Vehicle {vehicle_a_id} (from event {event_id}) not found in {data_file}. Skipping video generation.")
                continue
            if vehicle_b_id not in filtered_trajectories:
                print(f"   Warning: Vehicle {vehicle_b_id} (from event {event_id}) not found in {data_file}. Skipping video generation.")
                continue

            print(f"   Found trajectories for both vehicles in {data_file}.")

            # 8. Generate video for this specific event using only the two vehicles
            output_video_name = f"{target_severity.lower()}_event_{event_id}.mp4"
            output_video_path = os.path.join(output_dir, output_video_name)

            print(f"   Output video: {output_video_path}")
            generate_single_video(
                trajectories=filtered_trajectories, # Pass filtered data
                vehicle_classes=filtered_classes,   # Pass filtered data
                output_file=output_video_path,
                fps=fps,
                specific_collision_event=event, # Pass the current event
                focus_on_collision=True,      # Force focus
            )
            generated_count += 1
        else:
            print(f"   Severity {event_severity} does not match target {target_severity_upper}. Skipping video generation for this event.")

    print(f"\n=== Processing Complete ===")
    print(f"Total events checked in {collision_report_file}: {len(all_events)}")
    print(f"Videos generated ({target_severity} severity): {generated_count}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate videos for specific collision events based on a report and a data file.')

    parser.add_argument('collision_report_file', type=str,
                        help='Path to the input collision report JSON file.')
    parser.add_argument('data_file', type=str,
                        help='Path to the input trajectory data JSON file (e.g., data_2.json).')
    parser.add_argument('--output', '-o', type=str,
                        default='medium_collision_videos',
                        help='Output directory for generated videos (default: medium_collision_videos).')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                        help=f'Video frame rate (default: {DEFAULT_FPS})')
    parser.add_argument('--target-severity', type=str, default='MEDIUM',
                        help='Target collision severity to generate videos for (e.g., HIGH, MEDIUM, LOW, case-insensitive, default: MEDIUM).')

    # Parse command line arguments
    args = parser.parse_args()

    # Call the main processing function
    process_collision_report(
        collision_report_file=args.collision_report_file,
        data_file=args.data_file,
        output_dir=args.output,
        target_severity=args.target_severity,
        fps=args.fps
    )

    print("\nProcessing complete!")

if __name__ == '__main__':
    main() 