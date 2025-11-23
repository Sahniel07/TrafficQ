import json
import os
import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，确保不显示图形窗口
import matplotlib.pyplot as plt
import contextily as ctx
from datetime import datetime

# 全局参数
MAP_ZOOM_LEVEL = 17  # 地图缩放级别
MARGIN_FACTOR = 0.15  # 边界边距因子
SCALE_BAR_LENGTH_M = 200  # 比例尺长度（米）
FIGURE_SIZE = (10, 10)  # 正方形图像尺寸，以保持1:1纵横比
DPI = 300  # 输出图像DPI

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
    ax.plot([x_start, x_end], [y_pos, y_pos], 'k-', linewidth=2.5, zorder=20)
    
    # 添加标签
    scale_label = f'{int(scale_bar_length_m)} m'
    y_offset = (y_max - y_min) * 0.02
    
    ax.text((x_start + x_end) / 2, y_pos + y_offset, scale_label, 
            ha='center', va='bottom', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
            zorder=20)

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
            t = datetime.fromisoformat(pt['time'])
            # 转换时间为浮点时间戳（秒）
            timestamp = t.timestamp()
            lon, lat = pt['coordinateLongLat']
            traj.append([lat, lon, timestamp])
        
        if not traj:
            print(f"警告: 车辆ID {vehicle_id} 没有轨迹数据。")
            return None, vehicle_info
            
        traj_data = np.array(traj)
        return traj_data, vehicle_info
        
    except Exception as e:
        print(f"解析数据文件 {json_file} 时出错: {e}")
        return None, None

def plot_trajectory(traj_data, vehicle_info, output_file=None):
    """
    绘制车辆轨迹图。
    
    参数:
        traj_data: 轨迹数据的numpy数组 [lat, lon, timestamp]
        vehicle_info: 包含车辆ID、类别、颜色等信息的字典
        output_file: 输出图像文件路径
    
    返回:
        生成的图像文件路径
    """
    if traj_data is None or traj_data.size == 0:
        print("错误: 没有有效的轨迹数据用于绘图。")
        return None
    
    # 如果没有提供输出文件路径，生成默认路径
    if output_file is None:
        output_dir = "trajectory_plots"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"vehicle_{vehicle_info['id']}_trajectory.png")
    else:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n正在为车辆ID: {vehicle_info['id']} 生成轨迹图...")
    
    # 提取所有轨迹点的经纬度
    all_lats = traj_data[:, 0]
    all_lons = traj_data[:, 1]
    
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
    
    # 重新计算边界使坐标空间为正方形
    min_lon = center_lon - max_span/2 * (1 + MARGIN_FACTOR)
    max_lon = center_lon + max_span/2 * (1 + MARGIN_FACTOR)
    min_lat = center_lat - max_span/2 * (1 + MARGIN_FACTOR)
    max_lat = center_lat + max_span/2 * (1 + MARGIN_FACTOR)
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    # 设置经度(x轴)和纬度(y轴)的特定限制
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # 强制1:1的纵横比
    set_geo_aspect_ratio(ax, center_lat)
    
    # 添加地图背景
    try:
        print(f"添加地图背景，缩放级别 {MAP_ZOOM_LEVEL}...")
        ctx.add_basemap(ax, crs='EPSG:4326', 
                      source=ctx.providers.CartoDB.Positron,
                      zoom=MAP_ZOOM_LEVEL,
                      alpha=1.0)
    except Exception as e:
        print(f"添加主要地图背景失败: {e}")
        try:
            print("尝试备用地图源...")
            ctx.add_basemap(ax, crs='EPSG:4326', 
                          source=ctx.providers.Stamen.Terrain,
                          zoom=MAP_ZOOM_LEVEL,
                          alpha=1.0)
        except Exception as e2:
            print(f"添加备用地图背景失败: {e2}")
            print("继续执行，不使用地图背景...")
    
    # 添加比例尺
    draw_exact_scale_bar(ax, SCALE_BAR_LENGTH_M)
    
    # 获取车辆颜色
    color = vehicle_info.get('color', "#72deee")
    
    # 计算路径总长度（米）
    total_distance = 0
    for i in range(len(traj_data) - 1):
        lat1, lon1 = traj_data[i, 0], traj_data[i, 1]
        lat2, lon2 = traj_data[i + 1, 0], traj_data[i + 1, 1]
        segment_distance = haversine(lon1, lat1, lon2, lat2)
        total_distance += segment_distance
    
    # 绘制轨迹线
    ax.plot(all_lons, all_lats, '-', color=color, linewidth=2.5, 
           label=f"{vehicle_info['class']} (ID: {vehicle_info['id']})")
    
    # 标记起点和终点
    ax.scatter(traj_data[0, 1], traj_data[0, 0], color=color, s=100, marker='^', 
              edgecolor='black', linewidth=1.5, zorder=10, label='起点')
    ax.scatter(traj_data[-1, 1], traj_data[-1, 0], color=color, s=100, marker='s', 
              edgecolor='black', linewidth=1.5, zorder=10, label='终点')
    
    # 获取开始和结束时间
    start_time = datetime.fromtimestamp(traj_data[0, 2])
    end_time = datetime.fromtimestamp(traj_data[-1, 2])
    duration_seconds = traj_data[-1, 2] - traj_data[0, 2]
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 添加信息标题
    title = (f"车辆轨迹: {vehicle_info['class']} (ID: {vehicle_info['id']})\n"
             f"总距离: {total_distance:.1f}米, 时长: {int(hours)}小时{int(minutes)}分{int(seconds)}秒")
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加时间信息
    time_info = (f"开始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                 f"结束: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    ax.text(0.02, 0.02, time_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # 添加坐标轴标签
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 显示网格
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 轨迹图已保存至: {output_file}")
    return output_file

def plot_trajectory_segments(traj_data, vehicle_info, output_dir=None, num_segments=10):
    """
    将轨迹分成指定数量的段，并为每一段生成单独的图像。
    
    参数:
        traj_data: 轨迹数据的numpy数组 [lat, lon, timestamp]
        vehicle_info: 包含车辆ID、类别、颜色等信息的字典
        output_dir: 输出目录
        num_segments: 分段数量
    
    返回:
        生成的图像文件路径列表
    """
    if traj_data is None or traj_data.size == 0:
        print("错误: 没有有效的轨迹数据用于分段绘图。")
        return None
    
    # 如果没有提供输出目录，生成默认目录
    if output_dir is None:
        output_dir = os.path.join("trajectory_plots", f"vehicle_{vehicle_info['id']}_segments")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n正在为车辆ID: {vehicle_info['id']} 生成{num_segments}段轨迹图...")
    
    # 提取所有轨迹点的经纬度
    all_lats = traj_data[:, 0]
    all_lons = traj_data[:, 1]
    
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
    
    # 重新计算边界使坐标空间为正方形
    min_lon = center_lon - max_span/2 * (1 + MARGIN_FACTOR)
    max_lon = center_lon + max_span/2 * (1 + MARGIN_FACTOR)
    min_lat = center_lat - max_span/2 * (1 + MARGIN_FACTOR)
    max_lat = center_lat + max_span/2 * (1 + MARGIN_FACTOR)
    
    # 根据时间戳将轨迹分段
    start_time = traj_data[0, 2]
    end_time = traj_data[-1, 2]
    time_span = end_time - start_time
    
    # 计算每段的时间范围
    segment_time_spans = []
    for i in range(num_segments):
        seg_start = start_time + (time_span * i / num_segments)
        seg_end = start_time + (time_span * (i + 1) / num_segments)
        segment_time_spans.append((seg_start, seg_end))
    
    # 计算全程总距离（米）
    total_distance = 0
    for i in range(len(traj_data) - 1):
        lat1, lon1 = traj_data[i, 0], traj_data[i, 1]
        lat2, lon2 = traj_data[i + 1, 0], traj_data[i + 1, 1]
        segment_distance = haversine(lon1, lat1, lon2, lat2)
        total_distance += segment_distance
    
    # 生成的文件路径列表
    output_files = []
    
    # 获取车辆颜色
    color = vehicle_info.get('color', "#72deee")
    
    # 为每段生成图像
    for segment_idx, (seg_start_time, seg_end_time) in enumerate(segment_time_spans):
        segment_num = segment_idx + 1
        print(f"\n处理第 {segment_num}/{num_segments} 段轨迹...")
        
        # 创建输出文件路径
        output_file = os.path.join(output_dir, f"vehicle_{vehicle_info['id']}_segment_{segment_num}_of_{num_segments}.png")
        output_files.append(output_file)
        
        # 提取当前段的轨迹点
        segment_mask = (traj_data[:, 2] >= seg_start_time) & (traj_data[:, 2] <= seg_end_time)
        segment_data = traj_data[segment_mask]
        
        # 如果当前段没有数据点，跳过
        if len(segment_data) == 0:
            print(f"警告: 第 {segment_num} 段没有轨迹点，跳过该段。")
            continue
        
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
        
        # 设置经度(x轴)和纬度(y轴)的特定限制
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        
        # 强制1:1的纵横比
        set_geo_aspect_ratio(ax, center_lat)
        
        # 添加地图背景
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', 
                          source=ctx.providers.CartoDB.Positron,
                          zoom=MAP_ZOOM_LEVEL,
                          alpha=1.0)
        except Exception as e:
            try:
                ctx.add_basemap(ax, crs='EPSG:4326', 
                              source=ctx.providers.Stamen.Terrain,
                              zoom=MAP_ZOOM_LEVEL,
                              alpha=1.0)
            except Exception as e2:
                print("继续执行，不使用地图背景...")
        
        # 添加比例尺
        draw_exact_scale_bar(ax, SCALE_BAR_LENGTH_M)
        
        # 首先绘制完整轨迹作为背景（淡色）
        ax.plot(all_lons, all_lats, '-', color=color, linewidth=1.5, alpha=0.3, 
               label=f"完整轨迹")
        
        # 计算当前段的距离
        segment_distance = 0
        for i in range(len(segment_data) - 1):
            lat1, lon1 = segment_data[i, 0], segment_data[i, 1]
            lat2, lon2 = segment_data[i + 1, 0], segment_data[i + 1, 1]
            seg_dist = haversine(lon1, lat1, lon2, lat2)
            segment_distance += seg_dist
        
        # 然后高亮绘制当前段轨迹
        ax.plot(segment_data[:, 1], segment_data[:, 0], '-', color=color, linewidth=3.5, 
               label=f"第{segment_num}段")
        
        # 标记当前段的起点和终点
        ax.scatter(segment_data[0, 1], segment_data[0, 0], color=color, s=100, marker='^', 
                  edgecolor='black', linewidth=1.5, zorder=10, label=f'段起点')
        ax.scatter(segment_data[-1, 1], segment_data[-1, 0], color=color, s=100, marker='s', 
                  edgecolor='black', linewidth=1.5, zorder=10, label=f'段终点')
        
        # 获取当前段的开始和结束时间
        seg_start_datetime = datetime.fromtimestamp(seg_start_time)
        seg_end_datetime = datetime.fromtimestamp(seg_end_time)
        segment_duration = seg_end_time - seg_start_time
        hours, remainder = divmod(segment_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # 添加信息标题
        title = (f"车辆轨迹 (第{segment_num}/{num_segments}段): {vehicle_info['class']} (ID: {vehicle_info['id']})\n"
                 f"段距离: {segment_distance:.1f}米, 总距离: {total_distance:.1f}米, 段时长: {int(hours)}时{int(minutes)}分{int(seconds)}秒")
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加时间信息
        time_info = (f"段开始: {seg_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"段结束: {seg_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        ax.text(0.02, 0.02, time_info, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # 添加坐标轴标签
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        
        # 添加进度条
        progress_text = f"进度: {segment_num}/{num_segments}"
        progress_percent = segment_num / num_segments
        ax.text(0.98, 0.98, progress_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        ax.axhline(y=min_lat + lat_span * 0.03, xmin=0.1, xmax=0.9, color='lightgray', linewidth=5)
        ax.axhline(y=min_lat + lat_span * 0.03, xmin=0.1, xmax=0.1 + 0.8 * progress_percent, color=color, linewidth=5)
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 显示网格
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 第 {segment_num}/{num_segments} 段轨迹图已保存至: {output_file}")
    
    print(f"\n✅ 所有{num_segments}段轨迹图已保存至: {output_dir}")
    return output_files

def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='根据提取的车辆JSON数据生成轨迹图')
    
    parser.add_argument('json_file', type=str,
                        help='提取的车辆JSON文件路径 (由extract_vehicle_data.py生成)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出图像文件路径 (默认: ./trajectory_plots/vehicle_ID_trajectory.png)')
    parser.add_argument('--segments', '-s', type=int, default=0,
                        help='将轨迹分成多少段绘制 (0表示不分段，默认为0)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析车辆数据
    traj_data, vehicle_info = parse_extracted_vehicle_data(args.json_file)
    
    if traj_data is None or vehicle_info is None:
        print("错误: 无法从JSON文件中提取有效的轨迹数据。")
        return
    
    # 根据是否需要分段绘制选择不同的处理方法
    if args.segments > 0:
        # 分段绘制轨迹
        output_dir = args.output if args.output else None
        output_paths = plot_trajectory_segments(
            traj_data=traj_data,
            vehicle_info=vehicle_info,
            output_dir=output_dir,
            num_segments=args.segments
        )
        
        # 输出结果
        if output_paths:
            print(f"\n分段绘图完成! 已生成 {len(output_paths)} 个图像文件。")
        else:
            print("\n分段绘图失败! 未能生成轨迹图。")
    else:
        # 绘制单个完整轨迹图
        output_path = plot_trajectory(
            traj_data=traj_data,
            vehicle_info=vehicle_info,
            output_file=args.output
        )
        
        # 输出结果
        if output_path:
            print(f"\n绘图完成! 轨迹图已保存至: {output_path}")
        else:
            print("\n绘图失败! 未能生成轨迹图。")

if __name__ == '__main__':
    main() 