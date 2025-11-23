import json
import os
import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，确保不显示图形窗口
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import contextily as ctx
from datetime import datetime
from matplotlib.patheffects import withStroke
import matplotlib.dates as mdates
import pytz

# 全局参数
MAP_ZOOM_LEVEL = 18  # 提高地图缩放级别以获得更多细节
MARGIN_FACTOR = 0.1  # 减少边界边距因子
SCALE_BAR_LENGTH_M = 100  # 比例尺长度（米）
FIGURE_SIZE = (10, 10)  # 降低图像尺寸
DPI = 300  # 降低DPI以避免尺寸过大
POINT_SIZE = 80  # 增加点的大小
POINT_ALPHA = 0.9  # 增加点的不透明度
EDGE_WIDTH = 0.8  # 点的边框宽度
COLORMAP = 'plasma'  # 使用对比度更高的颜色映射
LINE_WIDTH = 1.5  # 轨迹线宽度
MAX_DPI = 300  # 最大DPI限制，防止过大
# 时区设置 - 使用中国时区
LOCAL_TIMEZONE = pytz.timezone('Asia/Shanghai')  # 使用中国时区

def haversine(lon1, lat1, lon2, lat2):
    """
    使用Haversine公式计算两个地理点之间的距离（米）
    """
    R = 6371000  # 地球半径，单位米
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def set_geo_aspect_ratio(ax, center_lat):
    """
    设置准确的地理纵横比，使经度和纬度在视觉上具有相同的地理距离。
    """
    # 在1:1的纵横比下强制执行
    ax.set_aspect('equal')
    
    # 移除坐标轴标签的科学计数法
    ax.ticklabel_format(useOffset=False, style='plain')

def draw_exact_scale_bar(ax, scale_bar_length_m=SCALE_BAR_LENGTH_M):
    """
    在地图上绘制精确长度的比例尺。
    """
    # 获取当前图像显示范围
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # 获取中心点坐标
    center_lon = (x_min + x_max) / 2
    center_lat = (y_min + y_max) / 2
    
    # 计算当前纬度下的经度比例
    lon_scale = scale_bar_length_m / (111320 * np.cos(np.radians(center_lat)))
    
    # 计算比例尺起始位置
    x_start = x_min + (x_max - x_min) * 0.1
    y_pos = y_min + (y_max - y_min) * 0.05
    x_end = x_start + lon_scale
    
    # 绘制比例尺线
    ax.plot([x_start, x_end], [y_pos, y_pos], 'k-', linewidth=3, zorder=20)
    
    # 添加标签
    scale_label = f'{int(scale_bar_length_m)} m'
    y_offset = (y_max - y_min) * 0.02
    
    ax.text((x_start + x_end) / 2, y_pos + y_offset, scale_label, 
            ha='center', va='bottom', fontsize=11, 
            bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.5', edgecolor='black'),
            zorder=20, fontweight='bold')

def parse_extracted_vehicle_data(json_file):
    """
    解析由extract_vehicle_data.py生成的提取车辆数据的JSON文件。
    
    返回:
    - traj_data: 轨迹数据的numpy数组 [lat, lon, timestamp]
    - vehicle_info: 包含车辆ID、类别、颜色等信息的字典
    """
    print(f"读取提取的车辆数据文件: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据结构
        if ('data' not in data or 'entities' not in data['data'] or 
            'edges' not in data['data']['entities'] or 
            len(data['data']['entities']['edges']) == 0):
            print(f"错误: 数据文件 {json_file} 的结构不正确或为空。")
            return None, None
        
        # 提取车辆节点数据
        node = data['data']['entities']['edges'][0]['node']
        vehicle_id = node['id']
        
        # 提取车辆类型和颜色信息
        vehicle_class = node['class']['name']
        vehicle_color = f"#{node['class']['color']}" if node['class']['color'] else "#72deee"
        
        # 存储车辆信息
        vehicle_info = {
            "id": vehicle_id,
            "class": vehicle_class,
            "color": vehicle_color
        }
        
        # 解析轨迹点
        traj = []
        for pt in node['trajectory']:
            # 正确解析ISO格式时间，包括时区信息
            t_str = pt['time']
            try:
                # 如果时间包含时区信息，直接解析
                t = datetime.fromisoformat(t_str)
                # 转换为UTC时间
                if t.tzinfo is not None:
                    t = t.astimezone(pytz.UTC)
                else:
                    # 如果没有时区信息，假定为UTC
                    t = pytz.UTC.localize(t)
            except ValueError:
                # 旧版Python可能不支持完整的ISO格式，尝试手动解析
                if 'T' in t_str and '+' in t_str:
                    dt_part, tz_part = t_str.split('+')
                    if 'T' in dt_part:
                        dt_part = dt_part.replace('T', ' ')
                    t = datetime.strptime(dt_part, '%Y-%m-%d %H:%M:%S.%f')
                    t = pytz.UTC.localize(t)
                else:
                    # 无法解析，使用当前时间
                    print(f"警告: 无法解析时间格式: {t_str}，使用默认时间")
                    t = datetime.now(pytz.UTC)
            
            # 转换时间为浮点时间戳（秒）
            timestamp = t.timestamp()
            lon, lat = pt['coordinateLongLat']
            traj.append([lat, lon, timestamp])
        
        if not traj:
            print(f"警告: 车辆ID {vehicle_id} 没有轨迹数据。")
            return None, vehicle_info
        
        # 确保轨迹点按时间戳排序
        traj_data = np.array(traj)
        if len(traj_data) > 1:
            # 检查时间是否已经有序
            if not np.all(np.diff(traj_data[:, 2]) >= 0):
                print(f"警告: 轨迹时间不是按顺序排列的，正在按时间戳排序...")
                # 按时间戳排序
                time_order = np.argsort(traj_data[:, 2])
                traj_data = traj_data[time_order]
                print(f"轨迹已按时间顺序重排，从 {datetime.fromtimestamp(traj_data[0, 2], tz=pytz.UTC).astimezone(LOCAL_TIMEZONE)} 到 {datetime.fromtimestamp(traj_data[-1, 2], tz=pytz.UTC).astimezone(LOCAL_TIMEZONE)}")
        
        return traj_data, vehicle_info
        
    except Exception as e:
        print(f"解析数据文件 {json_file} 时出错: {e}")
        return None, None

def create_custom_colormap():
    """创建高对比度自定义颜色映射"""
    # 从紫色到黄色的渐变
    colors = [(0.4, 0, 0.6), (0, 0.6, 0.8), (0.1, 0.9, 0.5), (0.9, 0.9, 0)]
    return LinearSegmentedColormap.from_list("custom_cmap", colors)

def plot_time_colored_points(traj_data, vehicle_info, output_file=None, colormap=COLORMAP, point_size=POINT_SIZE, show_lines=True, percent=100):
    """
    在地图上绘制带有时间颜色渐变的轨迹点。
    
    参数:
        traj_data: 轨迹数据的numpy数组 [lat, lon, timestamp]
        vehicle_info: 包含车辆ID、类别、颜色等信息的字典
        output_file: 输出图像文件路径
        colormap: 颜色映射名称
        point_size: 点的大小
        show_lines: 是否在点之间绘制连线
        percent: 要绘制的轨迹百分比(1-100)，默认100表示绘制全部轨迹
    
    返回:
        生成的图像文件路径
    """
    if traj_data is None or traj_data.size == 0:
        print("错误: 没有有效的轨迹数据用于绘图。")
        return None
    
    # 如果没有提供输出文件路径，生成默认路径
    if output_file is None:
        output_dir = "point_plots"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"vehicle_{vehicle_info['id']}_points{'_'+str(percent)+'pct' if percent < 100 else ''}.png")
    else:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # 根据percent参数限制轨迹点数量
    if percent < 100:
        points_to_use = max(1, int(len(traj_data) * percent / 100))
        traj_data = traj_data[:points_to_use]
        print(f"\n只使用前{percent}%的轨迹点（{points_to_use}个点）生成高质量时间颜色点图...")
    else:
        print(f"\n正在为车辆ID: {vehicle_info['id']} 生成高质量时间颜色点图...")
    
    # 提取所有轨迹点的经纬度和时间戳
    all_lats = traj_data[:, 0]
    all_lons = traj_data[:, 1]
    all_times = traj_data[:, 2]
    
    # 计算边界
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    
    # 计算地图中心点
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # 确保地图区域是完全正方形
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    max_span = max(lon_span, lat_span)
    
    # 增加边界裁剪逻辑，防止地图区域过大
    # 如果区域异常大（比如点分布很分散），增加边距因子使图像更紧凑
    if max_span > 0.1:  # 超过约11km的范围
        print(f"警告: 轨迹区域跨度较大 ({max_span:.6f}°)，调整边距以优化图像大小")
        adjusted_margin = 0.05  # 使用更小的边距
    else:
        adjusted_margin = MARGIN_FACTOR
    
    # 重新计算边界使坐标空间为正方形
    min_lon = center_lon - max_span/2 * (1 + adjusted_margin)
    max_lon = center_lon + max_span/2 * (1 + adjusted_margin)
    min_lat = center_lat - max_span/2 * (1 + adjusted_margin)
    max_lat = center_lat + max_span/2 * (1 + adjusted_margin)
    
    # 创建图形和轴
    # 计算并检查图像尺寸是否会超出限制
    current_dpi = DPI
    current_figsize = list(FIGURE_SIZE)
    
    # 计算像素大小并确保不超过matplotlib限制(65536x65536)
    MAX_PIXEL_DIM = 65000  # 略小于2^16以留出余量
    
    # 计算当前设置下的像素尺寸
    pixel_width = current_figsize[0] * current_dpi
    pixel_height = current_figsize[1] * current_dpi
    
    # 如果超出限制，则调整DPI或图像尺寸
    if pixel_width > MAX_PIXEL_DIM or pixel_height > MAX_PIXEL_DIM:
        print(f"警告: 图像尺寸({pixel_width}x{pixel_height}像素)过大，调整参数...")
        
        # 计算安全的DPI值
        safe_dpi = min(int(MAX_PIXEL_DIM / max(current_figsize)), MAX_DPI)
        
        # 如果调整DPI后仍然过大，还需要调整图像尺寸
        if safe_dpi < 72:  # 如果DPI太小，调整图像尺寸而不是DPI
            safe_dpi = 100  # 使用合理的最小DPI
            max_figsize = MAX_PIXEL_DIM / safe_dpi
            scale_factor = max_figsize / max(current_figsize)
            current_figsize = [dim * scale_factor for dim in current_figsize]
            print(f"图像尺寸调整为: {current_figsize}, DPI: {safe_dpi}")
        else:
            current_dpi = safe_dpi
            print(f"DPI调整为: {current_dpi}")
    
    # 创建具有调整后大小和DPI的图形
    fig, ax = plt.subplots(figsize=current_figsize, dpi=current_dpi)
    
    # 设置经度(x轴)和纬度(y轴)的特定限制
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # 强制1:1的纵横比
    set_geo_aspect_ratio(ax, center_lat)
    
    # 添加地图背景
    try:
        print(f"添加高分辨率地图背景，缩放级别 {MAP_ZOOM_LEVEL}...")
        ctx.add_basemap(ax, crs='EPSG:4326', 
                      source=ctx.providers.CartoDB.Voyager,  # 使用更详细的地图源
                      zoom=MAP_ZOOM_LEVEL,
                      alpha=1.0)
    except Exception as e:
        print(f"添加主要地图背景失败: {e}")
        try:
            print("尝试备用地图源...")
            ctx.add_basemap(ax, crs='EPSG:4326', 
                          source=ctx.providers.OpenStreetMap.Mapnik,  # 备用高质量地图源
                          zoom=MAP_ZOOM_LEVEL,
                          alpha=1.0)
        except Exception as e2:
            try:
                print("尝试第三种地图源...")
                ctx.add_basemap(ax, crs='EPSG:4326', 
                              source=ctx.providers.Stamen.TonerLite,  # 第三备选地图源
                              zoom=MAP_ZOOM_LEVEL,
                              alpha=1.0)
            except Exception as e3:
                print(f"添加所有备用地图背景都失败: {e3}")
                print("继续执行，不使用地图背景...")
    
    # 添加比例尺
    draw_exact_scale_bar(ax, SCALE_BAR_LENGTH_M)
    
    # 计算路径总长度（米）
    total_distance = 0
    for i in range(len(traj_data) - 1):
        lat1, lon1 = traj_data[i, 0], traj_data[i, 1]
        lat2, lon2 = traj_data[i + 1, 0], traj_data[i + 1, 1]
        segment_distance = haversine(lon1, lat1, lon2, lat2)
        total_distance += segment_distance
    
    # 使用自定义颜色映射或指定的颜色映射
    if colormap == 'custom':
        cmap = create_custom_colormap()
    else:
        cmap = plt.get_cmap(colormap)
    
    # 标准化时间戳以用于颜色映射
    norm = Normalize(vmin=min(all_times), vmax=max(all_times))
    
    # 首先绘制轨迹线（如果启用）
    if show_lines:
        # 为轨迹线创建颜色数组
        points = np.array([all_lons, all_lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 为每个线段计算平均时间戳作为颜色
        line_colors = np.array([(all_times[i] + all_times[i+1])/2 for i in range(len(all_times)-1)])
        
        # 创建着色线集合
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=LINE_WIDTH, alpha=0.7, zorder=9)
        lc.set_array(line_colors)
        ax.add_collection(lc)
    
    # 绘制颜色随时间变化的点
    sc = ax.scatter(all_lons, all_lats, c=all_times, cmap=cmap, norm=norm,
                   s=point_size, alpha=POINT_ALPHA, edgecolor='black', linewidth=EDGE_WIDTH, zorder=10)
    
    # 标记起点和终点
    ax.scatter(traj_data[0, 1], traj_data[0, 0], c='white', s=point_size*3, marker='^', 
              edgecolor='black', linewidth=2, zorder=12, label='Start')
    ax.scatter(traj_data[-1, 1], traj_data[-1, 0], c='white', s=point_size*3, marker='s', 
              edgecolor='black', linewidth=2, zorder=12, label='End')
    
    # 获取开始和结束时间并转换为本地时区
    start_time = datetime.fromtimestamp(traj_data[0, 2], tz=pytz.UTC).astimezone(LOCAL_TIMEZONE)
    end_time = datetime.fromtimestamp(traj_data[-1, 2], tz=pytz.UTC).astimezone(LOCAL_TIMEZONE)
    duration_seconds = traj_data[-1, 2] - traj_data[0, 2]
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 添加颜色条，显示时间映射
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, shrink=0.8)
    cbar.set_label('Time Progression', fontsize=12, fontweight='bold')
    
    # 设置颜色条刻度标签为日期时间格式
    # 获取更有意义的时间点（不仅仅是均匀分布）
    # 计算总时间长度
    time_duration = max(all_times) - min(all_times)
    
    # 根据总时间长度选择合适的时间格式和间隔点数
    if time_duration < 60:  # 小于1分钟
        time_format = '%H:%M:%S.%f'  # 包含毫秒
        # 为短时间范围使用更多刻度
        num_ticks = min(len(all_times), 5) if len(all_times) > 1 else 2
    elif time_duration < 3600:  # 小于1小时
        time_format = '%H:%M:%S'
        num_ticks = 5
    else:  # 大于等于1小时
        time_format = '%H:%M:%S'
        num_ticks = 5
    
    # 生成均匀分布的时间点
    if len(all_times) > 1:
        time_ticks = np.linspace(min(all_times), max(all_times), num_ticks)
        # 确保包含第一个和最后一个时间点
        if num_ticks > 2 and len(all_times) > 2:
            time_ticks[0] = min(all_times)
            time_ticks[-1] = max(all_times)
    else:
        # 如果只有一个时间点，创建一个有意义的范围
        time_ticks = np.array([min(all_times)])
    
    # 转换时间戳为本地时区的时间字符串，并截断毫秒部分为3位
    time_labels = []
    for t in time_ticks:
        dt = datetime.fromtimestamp(t, tz=pytz.UTC).astimezone(LOCAL_TIMEZONE)
        if time_format == '%H:%M:%S.%f':
            # 截断毫秒到3位
            time_str = dt.strftime('%H:%M:%S.%f')[:-3]
        else:
            time_str = dt.strftime(time_format)
        time_labels.append(time_str)
    
    # 设置颜色条刻度和标签
    cbar.set_ticks(time_ticks)
    cbar.set_ticklabels(time_labels)
    cbar.ax.tick_params(labelsize=10, labelrotation=45 if time_duration > 3600 else 0)
    
    # 添加信息标题
    title = (f"Vehicle Trajectory: {vehicle_info['class']} (ID: {vehicle_info['id']})\n"
             f"Total Distance: {total_distance:.1f}m, Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # 添加时间信息，带白色背景以提高可读性，使用本地时区
    # 为短时间轨迹增加毫秒显示
    if duration_seconds < 60:
        time_format = '%Y-%m-%d %H:%M:%S.%f'
        time_info = (f"Start: {start_time.strftime(time_format)[:-3]}\n"  # 截断毫秒到3位
                     f"End: {end_time.strftime(time_format)[:-3]}")
    else:
        time_format = '%Y-%m-%d %H:%M:%S'
        time_info = (f"Start: {start_time.strftime(time_format)}\n"
                     f"End: {end_time.strftime(time_format)}")
        
    ax.text(0.02, 0.02, time_info, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.5', edgecolor='black'),
            zorder=15, fontweight='bold')
    
    # 添加点数信息
    points_info = f"Trajectory Points: {len(traj_data)}"
    ax.text(0.98, 0.02, points_info, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.5', edgecolor='black'),
            zorder=15, fontweight='bold')
    
    # 添加坐标轴标签
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    
    # 使坐标轴刻度更清晰
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 添加图例，使其更加明显
    leg = ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    leg.get_frame().set_linewidth(1.5)
    
    # 显示网格
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_file, dpi=current_dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 高质量时间颜色点图已保存至: {output_file}")
    return output_file

def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='根据提取的车辆JSON数据生成高质量时间颜色点图')
    
    parser.add_argument('json_file', type=str,
                        help='提取的车辆JSON文件路径 (由extract_vehicle_data.py生成)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出图像文件路径 (默认: ./point_plots/vehicle_ID_points.png)')
    parser.add_argument('--colormap', '-c', type=str, default=COLORMAP,
                        help=f'颜色映射名称 (默认: {COLORMAP}, 可用: viridis, plasma, inferno, custom等)')
    parser.add_argument('--point-size', '-p', type=int, default=POINT_SIZE,
                        help=f'点的大小 (默认: {POINT_SIZE})')
    parser.add_argument('--no-lines', action='store_true',
                        help='不绘制轨迹连线，只显示点')
    parser.add_argument('--percent', '-pct', type=int, default=100,
                        help='要绘制的轨迹百分比(1-100)，默认100表示绘制全部轨迹')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析车辆数据
    traj_data, vehicle_info = parse_extracted_vehicle_data(args.json_file)
    
    if traj_data is None or vehicle_info is None:
        print("错误: 无法从JSON文件中提取有效的轨迹数据。")
        return
    
    # 绘制时间颜色点图
    output_path = plot_time_colored_points(
        traj_data=traj_data,
        vehicle_info=vehicle_info,
        output_file=args.output,
        colormap=args.colormap,
        point_size=args.point_size,
        show_lines=not args.no_lines,
        percent=args.percent
    )
    
    # 输出结果
    if output_path:
        print(f"\n绘图完成! 高质量时间颜色点图已保存至: {output_path}")
    else:
        print("\n绘图失败! 未能生成时间颜色点图。")

if __name__ == '__main__':
    main() 