import json
import math
from datetime import datetime
import numpy as np
from scipy.spatial import cKDTree
import os
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init()

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two geographic points (in meters) using Haversine formula.
    """
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def time_to_float(t):
    """Convert datetime object to a floating-point timestamp in seconds."""
    return t.timestamp()

def parse_data(filename):
    """
    Parse trajectory data of all vehicles from JSON file, returns a list of:
    [(timestamp, lon, lat, vehicle_id), ...]
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    points = []
    for edge in data['data']['entities']['edges']:
        node = edge['node']
        vehicle_id = node['id']
        for pt in node['trajectory']:
            t = datetime.fromisoformat(pt['time'])
            ts = time_to_float(t)
            lon, lat = pt['coordinateLongLat']
            points.append((ts, lon, lat, vehicle_id))
    return points

def detect_collisions_optimized(points, time_tol=0.1, distance_threshold=2.0, time_bin_size=0.1):
    """
    Optimized collision detection function:
      - points: list of (timestamp, lon, lat, vehicle_id)
      - time_tol: time tolerance for matching points (seconds)
      - distance_threshold: distance threshold for collision (meters)
      - time_bin_size: time window size for binning (seconds)
    Returns list of collision events, each as (point1, point2, distance)
    """
    # Convert data to NumPy structured array
    dtype = [('time', 'f8'), ('lon', 'f8'), ('lat', 'f8'), ('id', 'U50')]
    points_np = np.array(points, dtype=dtype)
    # Sort by time
    points_np.sort(order='time')
    
    # Create time bins
    min_time = points_np['time'][0]
    max_time = points_np['time'][-1]
    bins = np.arange(min_time, max_time + time_bin_size, time_bin_size)
    indices = np.digitize(points_np['time'], bins)
    
    collisions = []
    unique_bins = np.unique(indices)
    
    # For progress tracking
    total_bins = len(unique_bins)
    print(f"\n{Fore.CYAN}⏳ Processing {total_bins} time bins for collision detection...{Style.RESET_ALL}")
    
    # Process in batches for progress reporting
    batch_size = max(1, total_bins // 10)
    for i, b in enumerate(unique_bins):
        # Progress indicator every batch_size bins
        if i % batch_size == 0 or i == total_bins - 1:
            progress = min(100, int(100 * (i + 1) / total_bins))
            progress_bar = '■' * (progress // 5) + '□' * (20 - (progress // 5))
            print(f"\r{Fore.YELLOW}[{progress_bar}] {progress}% ({i+1}/{total_bins}){Style.RESET_ALL}", end='')
            
        # Get data from current bin and adjacent bins to handle edge cases
        bin_mask = (indices == b) | (indices == b - 1) | (indices == b + 1)
        subset = points_np[bin_mask]
        if len(subset) < 2:
            continue
        # Build KD-tree using longitude and latitude (coarse filtering)
        coords = np.column_stack((subset['lon'], subset['lat']))
        tree = cKDTree(coords)
        candidate_pairs = tree.query_pairs(distance_threshold)
        for i, j in candidate_pairs:
            p1 = subset[i]
            p2 = subset[j]
            # Skip if same vehicle
            if p1['id'] == p2['id']:
                continue
            # Check if time difference is within tolerance
            if abs(p1['time'] - p2['time']) <= time_tol:
                # Calculate precise distance using Haversine
                d = haversine(p1['lon'], p1['lat'], p2['lon'], p2['lat'])
                if d < distance_threshold:
                    collisions.append((p1, p2, d))
    
    print(f"\n{Fore.GREEN}✓ Collision detection complete! Found {len(collisions)} potential collisions.{Style.RESET_ALL}")
    return collisions

def deduplicate_collisions(collisions, merge_time_tol=0.5):
    """
    Deduplicate collision events:
      - First group by vehicle pair (sorted ids)
      - For each vehicle pair, sort by average time of the two points
        If adjacent events have time difference less than merge_time_tol (seconds),
        consider them as the same collision and keep only the one with smallest distance.
    Returns deduplicated list of collision events.
    """
    if not collisions:
        return []
        
    print(f"{Fore.CYAN}⏳ Deduplicating collision events...{Style.RESET_ALL}")
    
    # Grouping: key is vehicle pair, value is list of [ (avg_time, p1, p2, distance) ]
    groups = {}
    for p1, p2, d in collisions:
        vehicle_pair = tuple(sorted([p1['id'], p2['id']]))
        avg_time = (p1['time'] + p2['time']) / 2
        groups.setdefault(vehicle_pair, []).append((avg_time, p1, p2, d))
    
    deduped = []
    for vehicle_pair, events in groups.items():
        # Sort by average time
        events.sort(key=lambda x: x[0])
        cluster = []
        merged_events = []
        for event in events:
            if not cluster:
                cluster.append(event)
            else:
                # If time difference with last event is within merge_time_tol, add to current cluster
                if event[0] - cluster[-1][0] <= merge_time_tol:
                    cluster.append(event)
                else:
                    # For current cluster, keep only the event with smallest distance
                    merged = min(cluster, key=lambda x: x[3])
                    merged_events.append(merged)
                    cluster = [event]
        if cluster:
            merged = min(cluster, key=lambda x: x[3])
            merged_events.append(merged)
        # Add representative events to results
        for avg_time, p1, p2, d in merged_events:
            deduped.append((p1, p2, d))
    
    reduction = len(collisions) - len(deduped)
    print(f"{Fore.GREEN}✓ Deduplication complete! Removed {reduction} duplicates ({len(deduped)} unique collisions).{Style.RESET_ALL}")
    return deduped

def classify_collision_severity(distance):
    """Classify collision severity based on distance"""
    if distance < 0.5:
        return "HIGH", Fore.RED
    elif distance < 1.0:
        return "MEDIUM", Fore.YELLOW
    else:
        return "LOW", Fore.CYAN

def format_datetime(timestamp):
    """Format timestamp with milliseconds"""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def main():
    """Main function to detect and report vehicle collisions"""
    # Create header with box
    print("\n" + "=" * 80)
    print(f"{Fore.MAGENTA}█▓▒░ VEHICLE COLLISION DETECTION SYSTEM ░▒▓█{Style.RESET_ALL}".center(80))
    print("=" * 80 + "\n")
    
    # Get input file
    default_file = 'check.json'
    filename = default_file
    if not os.path.exists(filename):
        print(f"{Fore.RED}Warning: Default file '{default_file}' not found.{Style.RESET_ALL}")
        filename = input(f"{Fore.CYAN}Enter path to JSON data file: {Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}⏳ Loading trajectory data from '{filename}'...{Style.RESET_ALL}")
    points = parse_data(filename)
    vehicle_count = len(set(p[3] for p in points))
    point_count = len(points)
    print(f"{Fore.GREEN}✓ Loaded {point_count} data points from {vehicle_count} vehicles.{Style.RESET_ALL}")
    
    # Detection parameters
    time_tol = 0.1
    distance_threshold = 3.0
    time_bin_size = 0.1
    merge_time_tol = 0.5
    
    print(f"\n{Fore.CYAN}▶ Parameters:{Style.RESET_ALL}")
    print(f"  • Time tolerance: {time_tol} seconds")
    print(f"  • Distance threshold: {distance_threshold} meters")
    print(f"  • Time bin size: {time_bin_size} seconds")
    print(f"  • Merge time tolerance: {merge_time_tol} seconds")
    
    # Detect collisions
    collisions = detect_collisions_optimized(
        points, 
        time_tol=time_tol, 
        distance_threshold=distance_threshold, 
        time_bin_size=time_bin_size
    )
    
    # Deduplicate collision events
    deduped_collisions = deduplicate_collisions(collisions, merge_time_tol=merge_time_tol)
    
    # Export results
    output_file = "collision_report.txt"
    json_output_file = "collision_report.json"
    collision_count = len(deduped_collisions)
    
    print("\n" + "-" * 80)
    print(f"{Fore.MAGENTA}▶ COLLISION DETECTION RESULTS{Style.RESET_ALL}")
    print("-" * 80)
    
    # Prepare JSON data structure
    json_result = {
        "metadata": {
            "total_vehicles": vehicle_count,
            "total_data_points": point_count,
            "collisions_detected": collision_count,
            "parameters": {
                "time_tolerance": time_tol,
                "distance_threshold": distance_threshold,
                "time_bin_size": time_bin_size,
                "merge_time_tolerance": merge_time_tol
            }
        },
        "collisions": []
    }
    
    if deduped_collisions:
        # Sort by distance (closest first)
        deduped_collisions.sort(key=lambda x: x[2])
        
        print(f"\n{Fore.GREEN}Found {collision_count} potential collision events:{Style.RESET_ALL}\n")
        
        # Table header
        print(f"{Fore.WHITE}{'ID':^5} | {'VEHICLE A':^15} | {'VEHICLE B':^15} | {'DISTANCE':^10} | {'SEVERITY':^10} | {'TIMESTAMP':^26}{Style.RESET_ALL}")
        print("-" * 90)
        
        # Write to file and console simultaneously
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("VEHICLE COLLISION DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total vehicles analyzed: {vehicle_count}\n")
            f.write(f"Total data points: {point_count}\n")
            f.write(f"Collision events detected: {collision_count}\n\n")
            f.write(f"{'ID':<5} | {'VEHICLE A':<15} | {'VEHICLE B':<15} | {'DISTANCE':<10} | {'SEVERITY':<10} | {'TIMESTAMP':<26}\n")
            f.write("-" * 90 + "\n")
            
            for idx, (p1, p2, d) in enumerate(deduped_collisions, 1):
                severity, color = classify_collision_severity(d)
                avg_time = (p1['time'] + p2['time']) / 2
                formatted_time = format_datetime(avg_time)
                
                # Write to console with color
                print(f"{idx:^5} | {p1['id']:^15} | {p2['id']:^15} | {d:.2f} m     | {color}{severity:^10}{Style.RESET_ALL} | {formatted_time:^26}")
                
                # Write to file (without color)
                f.write(f"{idx:<5} | {p1['id']:<15} | {p2['id']:<15} | {d:.2f} m     | {severity:<10} | {formatted_time:<26}\n")
                
                # Detailed information for each collision
                detailed = (
                    f"\nDetailed information for collision #{idx}:\n"
                    f"  Vehicle A: {p1['id']}\n"
                    f"    Time: {format_datetime(p1['time'])}\n"
                    f"    Position: ({p1['lon']:.6f}, {p1['lat']:.6f})\n"
                    f"  Vehicle B: {p2['id']}\n"
                    f"    Time: {format_datetime(p2['time'])}\n"
                    f"    Position: ({p2['lon']:.6f}, {p2['lat']:.6f})\n"
                    f"  Separation: {d:.2f} meters\n"
                    f"  Time difference: {abs(p1['time'] - p2['time']):.3f} seconds\n"
                    f"  Severity: {severity}\n"
                )
                f.write(detailed)
                
                # Add to JSON structure
                collision_data = {
                    "id": idx,
                    "vehicle_a": {
                        "id": p1['id'],
                        "time": p1['time'],
                        "time_formatted": format_datetime(p1['time']),
                        "position": {
                            "longitude": p1['lon'],
                            "latitude": p1['lat']
                        }
                    },
                    "vehicle_b": {
                        "id": p2['id'],
                        "time": p2['time'],
                        "time_formatted": format_datetime(p2['time']),
                        "position": {
                            "longitude": p2['lon'],
                            "latitude": p2['lat']
                        }
                    },
                    "distance": round(d, 2),
                    "time_difference": round(abs(p1['time'] - p2['time']), 3),
                    "average_time": round(avg_time, 3),
                    "average_time_formatted": formatted_time,
                    "severity": severity
                }
                json_result["collisions"].append(collision_data)
        
        # Save JSON data
        with open(json_output_file, 'w', encoding='utf-8') as json_file:
            json.dump(json_result, json_file, indent=2)
        
        print("\n" + "-" * 80)
        print(f"{Fore.GREEN}✓ Full report saved to '{output_file}'{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ JSON report saved to '{json_output_file}'{Style.RESET_ALL}")
        print("-" * 80)
    else:
        print(f"\n{Fore.GREEN}✓ No collision events detected.{Style.RESET_ALL}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("VEHICLE COLLISION DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total vehicles analyzed: {vehicle_count}\n")
            f.write(f"Total data points: {point_count}\n")
            f.write(f"Result: No collision events detected.\n")
        
        # Save empty JSON result
        with open(json_output_file, 'w', encoding='utf-8') as json_file:
            json.dump(json_result, json_file, indent=2)
        
        print(f"\n{Fore.GREEN}✓ Report saved to '{output_file}'{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ JSON report saved to '{json_output_file}'{Style.RESET_ALL}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()