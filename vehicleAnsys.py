import json
import math
import numpy as np
from datetime import datetime

##########################
# Basic functions: Calculate bearing, circular mean, Haversine distance
##########################
def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing angle from (lat1, lon1) to (lat2, lon2) in degrees, range 0-360°.
    """
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

def circular_mean(angles):
    """
    Calculate the circular mean of a set of angles (in degrees), handling angle wrap-around issues.
    """
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    mean_angle = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
    return mean_angle

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two lat/lon points using the Haversine formula (in meters).
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

##########################
# Data parsing functions
##########################
def parse_vehicle_trajectories(filename):
    """
    Parse vehicle trajectory data from a JSON file,
    returns dictionary: {vehicle_id: [(timestamp, lon, lat), ...], ...}
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    trajectories = {}
    for edge in data['data']['entities']['edges']:
        node = edge['node']
        vehicle_id = node['id']
        traj = []
        for pt in node['trajectory']:
            t = datetime.fromisoformat(pt['time'])
            ts = t.timestamp()
            lon, lat = pt['coordinateLongLat']
            traj.append((ts, lon, lat))
        traj.sort(key=lambda x: x[0])
        trajectories[vehicle_id] = traj
    return trajectories

##########################
# Final turn analysis functions
##########################
def analyze_final_turn(traj, window_size=3, min_disp=0.1):
    """
    Analyze the final turn of a trajectory.
    
    Parameters:
      traj: [(timestamp, lon, lat), ...], already sorted by time
      window_size: number of consecutive segments used to calculate initial and final average bearing
      min_disp: minimum displacement (meters) between points, segments below this will be ignored
      
    Returns:
      (initial_mean, final_mean, net_turn, initial_time, final_time)
      returns (None, None, None, None, None) if insufficient data
    """
    # Initial segment
    initial_bearings = []
    initial_times = []
    i = 0
    while i < len(traj) - 1 and len(initial_bearings) < window_size:
        t1, lon1, lat1 = traj[i]
        t2, lon2, lat2 = traj[i+1]
        disp = haversine(lon1, lat1, lon2, lat2)
        if disp >= min_disp:
            bearing = compute_bearing(lat1, lon1, lat2, lon2)
            initial_bearings.append(bearing)
            initial_times.append(t1)
        i += 1
    if len(initial_bearings) < window_size:
        return None, None, None, None, None
    initial_mean = circular_mean(initial_bearings)
    initial_time_avg = sum(initial_times) / len(initial_times)
    
    # Final segment
    final_bearings = []
    final_times = []
    i = len(traj) - 2
    while i >= 0 and len(final_bearings) < window_size:
        t1, lon1, lat1 = traj[i]
        t2, lon2, lat2 = traj[i+1]
        disp = haversine(lon1, lat1, lon2, lat2)
        if disp >= min_disp:
            bearing = compute_bearing(lat1, lon1, lat2, lon2)
            final_bearings.append(bearing)
            final_times.append(t2)
        i -= 1
    if len(final_bearings) < window_size:
        return None, None, None, None, None
    final_bearings.reverse()
    final_times.reverse()
    final_mean = circular_mean(final_bearings)
    final_time_avg = sum(final_times) / len(final_times)
    
    net_turn = (final_mean - initial_mean + 180) % 360 - 180
    return initial_mean, final_mean, net_turn, initial_time_avg, final_time_avg

def angle_to_arrow(angle):
    """
    Convert angle to direction arrow:
      0-45° or 315-360°: ↑ (North)
      45-135°: → (East)
      135-225°: ↓ (South)
      225-315°: ← (West)
    """
    if angle >= 315 or angle < 45:
        return "↑"  # North
    elif angle < 135:
        return "→"  # East
    elif angle < 225:
        return "↓"  # South
    else:
        return "←"  # West

##########################
# Speed, acceleration and stop detection functions
##########################
def compute_speeds(traj):
    """
    Calculate speed for each segment (in m/s) based on trajectory data,
    returns a list of speeds and corresponding start times.
    """
    speeds = []
    times = []
    for i in range(len(traj) - 1):
        t1, lon1, lat1 = traj[i]
        t2, lon2, lat2 = traj[i+1]
        dt = t2 - t1
        if dt <= 0:
            continue
        dist = haversine(lon1, lat1, lon2, lat2)
        speed = dist / dt
        speeds.append(speed)
        times.append(t1)
    return speeds, times

def resample_speed_data(times, speeds, target_interval=0.5):
    """
    Resample speed data. Merge consecutive points so that each sample has a time interval 
    of approximately target_interval (seconds).
    Returns new time series and average speed list.
    """
    new_times = []
    new_speeds = []
    n = len(times)
    i = 0
    while i < n:
        start_time = times[i]
        window_speeds = []
        window_times = []
        while i < n and times[i] - start_time < target_interval:
            window_speeds.append(speeds[i])
            window_times.append(times[i])
            i += 1
        if window_speeds:
            new_speeds.append(np.mean(window_speeds))
            new_times.append(np.mean(window_times))
    return new_times, new_speeds

def compute_accelerations_interval(times, speeds, target_interval=0.5):
    """
    First resample speed data at target_interval, then calculate acceleration between 
    adjacent resampled points (in m/s²).
    """
    new_times, new_speeds = resample_speed_data(times, speeds, target_interval)
    accelerations = []
    for i in range(len(new_speeds) - 1):
        dt = new_times[i+1] - new_times[i]
        if dt <= 0:
            continue
        a = (new_speeds[i+1] - new_speeds[i]) / dt
        accelerations.append(a)
    return accelerations

def detect_stops(traj, speed_threshold=0.2, merge_gap=1.0):
    """
    Detect stop events in a trajectory.
    A vehicle is considered stopped when speed in consecutive segments is below speed_threshold (m/s).
    merge_gap: if gap between adjacent stop events is less than this value (seconds), they are merged.
    Returns a list, each element is (start_time, end_time, duration).
    """
    stops = []
    current_stop_start = None
    current_stop_end = None
    for i in range(len(traj) - 1):
        t1, lon1, lat1 = traj[i]
        t2, lon2, lat2 = traj[i+1]
        dt = t2 - t1
        if dt <= 0:
            continue
        speed = haversine(lon1, lat1, lon2, lat2) / dt
        if speed < speed_threshold:
            if current_stop_start is None:
                current_stop_start = t1
            current_stop_end = t2
        else:
            if current_stop_start is not None:
                stops.append((current_stop_start, current_stop_end, current_stop_end - current_stop_start))
                current_stop_start = None
                current_stop_end = None
    if current_stop_start is not None:
        stops.append((current_stop_start, current_stop_end, current_stop_end - current_stop_start))
    
    merged_stops = []
    if stops:
        current_stop = stops[0]
        for next_stop in stops[1:]:
            gap = next_stop[0] - current_stop[1]
            if gap <= merge_gap:
                current_stop = (current_stop[0], next_stop[1], next_stop[1] - current_stop[0])
            else:
                merged_stops.append(current_stop)
                current_stop = next_stop
        merged_stops.append(current_stop)
    return merged_stops

##########################
# Main function: Comprehensive analysis of final turn, speed, acceleration, and stops for each vehicle
##########################
def main():
    filename = 'data_2.json'
    trajectories = parse_vehicle_trajectories(filename)
    
    output_lines = []
    # Create JSON output data structure
    json_results = {}
    
    for vehicle_id, traj in trajectories.items():
        # Create entry for each vehicle in JSON structure
        json_results[vehicle_id] = {}
        
        # Add decorative header for each vehicle
        separator = "=" * 80
        header = f"\n{separator}\n"
        header += f"  ANALYSIS REPORT FOR VEHICLE: {vehicle_id}  ".center(80, "*")
        header += f"\n{separator}"
        print(header)
        output_lines.append(header)
        
        # Final turn analysis section
        section_header = "\n📍 FINAL TURN ANALYSIS"
        print(section_header)
        output_lines.append(section_header)
        
        ft_result = analyze_final_turn(traj, window_size=3, min_disp=0.1)
        if ft_result[0] is None:
            line = "   ❌ Insufficient data or movement amplitude too low."
            json_results[vehicle_id]['final_turn'] = {
                'status': 'insufficient_data'
            }
        else:
            initial, final, net_turn, t_initial, t_final = ft_result
            initial_time = datetime.fromtimestamp(t_initial)
            final_time = datetime.fromtimestamp(t_final)
            initial_dir = angle_to_arrow(initial)
            final_dir = angle_to_arrow(final)
            
            if abs(net_turn) >= 170:
                turn_type = "U-turn"
                turn_icon = "↺"
            elif abs(net_turn) < 70:
                turn_type = "mostly straight"
                turn_icon = "↑"
            elif net_turn > 0:
                turn_type = "left turn"
                turn_icon = "↰"
            else:
                turn_type = "right turn"
                turn_icon = "↱"
                
            line = f"   • Initial direction: {initial:.1f}° {initial_dir} at {initial_time}\n" \
                   f"   • Final direction:   {final:.1f}° {final_dir} at {final_time}\n" \
                   f"   • Net turn:          {net_turn:.1f}° {turn_icon} ({turn_type})"
                   
            # Add to JSON results
            json_results[vehicle_id]['final_turn'] = {
                'status': 'success',
                'initial_direction': {
                    'degrees': round(initial, 1),
                    'arrow': initial_dir,
                    'timestamp': t_initial,
                    'formatted_time': initial_time.isoformat()
                },
                'final_direction': {
                    'degrees': round(final, 1),
                    'arrow': final_dir,
                    'timestamp': t_final,
                    'formatted_time': final_time.isoformat()
                },
                'net_turn': {
                    'degrees': round(net_turn, 1),
                    'type': turn_type,
                    'icon': turn_icon
                }
            }
        print(line)
        output_lines.append(line)
        
        # Speed statistics section
        section_header = "\n🚀 SPEED STATISTICS"
        print(section_header)
        output_lines.append(section_header)
        
        speeds, speed_times = compute_speeds(traj)
        if not speeds:
            line = "   ❌ Insufficient speed data, cannot calculate speed statistics."
            json_results[vehicle_id]['speed_stats'] = {
                'status': 'insufficient_data'
            }
            print(line)
            output_lines.append(line)
        else:
            avg_speed = np.mean(speeds)
            moving_speeds = [s for s in speeds if s >= 0.2]
            avg_moving_speed = np.mean(moving_speeds) if moving_speeds else 0
            min_speed = np.min(speeds)
            max_speed = np.max(speeds)
            
            line = f"   • Overall average:   {avg_speed:.2f} m/s ({avg_speed*3.6:.2f} km/h)\n" \
                   f"   • Moving average:    {avg_moving_speed:.2f} m/s ({avg_moving_speed*3.6:.2f} km/h)\n" \
                   f"   • Minimum speed:     {min_speed:.2f} m/s ({min_speed*3.6:.2f} km/h)\n" \
                   f"   • Maximum speed:     {max_speed:.2f} m/s ({max_speed*3.6:.2f} km/h)"
            
            # Add to JSON results
            json_results[vehicle_id]['speed_stats'] = {
                'status': 'success',
                'overall_avg': {
                    'mps': round(avg_speed, 2),
                    'kmh': round(avg_speed * 3.6, 2)
                },
                'moving_avg': {
                    'mps': round(avg_moving_speed, 2),
                    'kmh': round(avg_moving_speed * 3.6, 2)
                },
                'min_speed': {
                    'mps': round(min_speed, 2),
                    'kmh': round(min_speed * 3.6, 2)
                },
                'max_speed': {
                    'mps': round(max_speed, 2),
                    'kmh': round(max_speed * 3.6, 2)
                }
            }
            print(line)
            output_lines.append(line)
            
            # Acceleration calculation section
            section_header = "\n⚡ ACCELERATION ANALYSIS"
            print(section_header)
            output_lines.append(section_header)
            
            accelerations = compute_accelerations_interval(speed_times, speeds, target_interval=0.5)
            if accelerations:
                max_acc = max(accelerations)
                min_acc = min(accelerations)
                line = f"   • Maximum acceleration:    {max_acc:.2f} m/s²\n" \
                       f"   • Maximum deceleration:    {abs(min_acc):.2f} m/s²"
                       
                # Add to JSON results
                json_results[vehicle_id]['acceleration'] = {
                    'status': 'success',
                    'max_acceleration': round(max_acc, 2),
                    'max_deceleration': round(abs(min_acc), 2)
                }
            else:
                line = "   ❌ Unable to calculate acceleration."
                json_results[vehicle_id]['acceleration'] = {
                    'status': 'insufficient_data'
                }
            print(line)
            output_lines.append(line)
            
            # Stop event detection section
            section_header = "\n🛑 STOP EVENTS"
            print(section_header)
            output_lines.append(section_header)
            
            stops = detect_stops(traj, speed_threshold=0.2, merge_gap=1.0)
            line = f"   • Total stop events: {len(stops)}"
            print(line)
            output_lines.append(line)
            
            # Add to JSON results
            json_results[vehicle_id]['stop_events'] = {
                'count': len(stops),
                'events': []
            }
            
            if stops:
                for idx, stop in enumerate(stops):
                    start, end, duration = stop
                    start_time = datetime.fromtimestamp(start)
                    end_time = datetime.fromtimestamp(end)
                    line = f"     └─ Stop #{idx+1}: {start_time} → {end_time} ({duration:.1f} seconds)"
                    print(line)
                    output_lines.append(line)
                    
                    # Add stop event to JSON results
                    json_results[vehicle_id]['stop_events']['events'].append({
                        'id': idx+1,
                        'start_timestamp': start,
                        'end_timestamp': end,
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'duration': round(duration, 1)
                    })
            else:
                line = "     └─ No stop events detected"
                print(line)
                output_lines.append(line)
                
        # Add footer
        footer = f"\n{separator}\n"
        print(footer)
        output_lines.append(footer)
    
    # Save text results
    with open("output_results_2.txt", "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")
    
    # Save JSON results
    with open("output_results_2.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ All results have been saved to:")
    print("   - output_results_2.txt (text format)")
    print("   - output_results_2.json (JSON format)")

if __name__ == '__main__':
    main()