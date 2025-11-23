import json
import math
from datetime import datetime

def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing (in degrees, range 0-360) from (lat1, lon1) to (lat2, lon2).
    """
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

def circular_mean(angles):
    """
    Calculate the circular mean of angles (in degrees), handling angle wrapping.
    """
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    mean_angle = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
    return mean_angle

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate distance between two geographic points (in meters) using Haversine formula.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_vehicle_trajectories(filename):
    """
    Parse vehicle trajectory data from JSON file, returning a dictionary:
      keys: vehicle ids
      values: time-sorted trajectory lists [(timestamp, lon, lat), ...]
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

def analyze_final_turn(traj, window_size=3, min_disp=0.1):
    """
    Analyze the final turn of a trajectory.
    
    Parameters:
      traj: [(timestamp, lon, lat), ...], time-sorted
      window_size: number of consecutive segments for bearing calculation
      min_disp: minimum displacement between points (meters)
      
    Returns:
      (initial_mean, final_mean, net_turn, initial_time, final_time)
      Returns (None, None, None, None, None) if insufficient data.
    """
    # Initial segments
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
    
    # Final segments
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
    Convert angle to cardinal direction arrow:
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

def main():
    filename = 'data.json'
    trajectories = parse_vehicle_trajectories(filename)
    
    for vehicle_id, traj in trajectories.items():
        result = analyze_final_turn(traj, window_size=3, min_disp=0.1)
        if result[0] is None:
            print(f"Vehicle {vehicle_id}: Insufficient data or movement too small to analyze final turn.")
        else:
            initial, final, net_turn, t_initial, t_final = result
            
            if abs(net_turn) >= 170:  # 
                turn_type = "U-turn"
            elif abs(net_turn) < 70:
                turn_type = "mostly straight"
            elif net_turn > 0:
                turn_type = "left turn"
            else:
                turn_type = "right turn"
            
            initial_time = datetime.fromtimestamp(t_initial)
            final_time = datetime.fromtimestamp(t_final)
            initial_dir = angle_to_arrow(initial)
            final_dir = angle_to_arrow(final)
            
            print(f"Vehicle {vehicle_id}: Initial direction {initial:.1f}° {initial_dir} at {initial_time}, "
                  f"final direction {final:.1f}° {final_dir} at {final_time}, "
                  f"net turn {net_turn:.1f}° (from {initial_dir} to {final_dir}, {turn_type}).")

if __name__ == '__main__':
    main()