import json
import os
import argparse
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import contextily as ctx
from vehicle_trajectory_video import (
    parse_vehicle_trajectories,
    generate_single_video,
    DEFAULT_FPS,
    set_geo_aspect_ratio,
    draw_exact_scale_bar,
    SCALE_BAR_LENGTH_M,
    MAP_ZOOM_LEVEL,
    MARGIN_FACTOR
)

def generate_trajectory_image(trajectories, vehicle_classes, output_file):
    """
    生成车辆轨迹静态图像。
    
    参数:
        trajectories -- 轨迹字典 {vehicle_id: trajectory_array, ...}
        vehicle_classes -- 车辆类型字典 {vehicle_id: {"class": class_name, "color": color_hex}, ...}
        output_file -- 输出图像文件路径
    
    返回:
        str: 生成的图像文件路径
    """
    print(f"\n生成静态轨迹图: {output_file}")
    
    # 创建输出目录(如果不存在)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
    
    # 计算地理边界
    all_lats = []
    all_lons = []
    
    for traj in trajectories.values():
        if traj.size > 0:
            all_lats.extend(traj[:, 0])
            all_lons.extend(traj[:, 1])
    
    if not all_lats or not all_lons:
        print("错误: 无法从轨迹数据中提取有效坐标。")
        return None
    
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
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    
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
    
    # 绘制每个车辆的轨迹
    for vid, traj in trajectories.items():
        if traj.size > 0:
            color = vehicle_classes.get(vid, {}).get("color", "#72deee")
            vehicle_class = vehicle_classes.get(vid, {}).get("class", "car")
            
            # 绘制轨迹线
            ax.plot(traj[:, 1], traj[:, 0], '-', color=color, linewidth=2.5, 
                   label=f"{vehicle_class} (ID: {vid})")
            
            # 标记起点和终点
            ax.scatter(traj[0, 1], traj[0, 0], color=color, s=100, marker='^', 
                      edgecolor='black', linewidth=1, zorder=10)  # 起点三角形
            ax.scatter(traj[-1, 1], traj[-1, 0], color=color, s=100, marker='s', 
                      edgecolor='black', linewidth=1, zorder=10)  # 终点方形
    
    # 添加标题和图例
    plt.title('车辆轨迹图', fontsize=16, fontweight='bold')
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    
    # 如果有多个车辆，添加图例
    if len(trajectories) > 1:
        ax.legend(loc='upper right')
    
    # 显示网格
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 静态轨迹图已保存至: {output_file}")
    return output_file

def generate_vehicle_id_video(data_file, vehicle_id, output_dir, fps=DEFAULT_FPS, duration=None, generate_image=True):
    """
    为指定的车辆ID生成轨迹视频和静态轨迹图。
    
    参数:
        data_file (str): 轨迹数据JSON文件路径
        vehicle_id (str): 要生成视频的车辆ID
        output_dir (str): 输出视频目录
        fps (int): 视频帧率
        duration (float, optional): 视频时长（秒）。如果为None，将自动计算。
        generate_image (bool): 是否同时生成静态轨迹图
    
    返回:
        tuple: (视频文件路径, 图像文件路径)，如果生成失败则对应项为None
    """
    print(f"\n=== 为车辆ID: {vehicle_id} 生成轨迹视频 ===")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析轨迹数据
    print(f"解析数据文件: {data_file}")
    try:
        all_trajectories, _, all_vehicle_classes = parse_vehicle_trajectories(data_file)
        if not all_trajectories:
            print(f"错误: 在{data_file}中未找到轨迹数据。无法生成视频。")
            return None, None
        print(f"已解析{len(all_trajectories)}个车辆的轨迹数据。")
    except Exception as e:
        print(f"解析数据文件{data_file}时出错: {e}。无法生成视频。")
        return None, None
    
    # 检查指定的车辆ID是否存在于数据中
    if vehicle_id not in all_trajectories:
        print(f"错误: 车辆ID {vehicle_id} 在数据文件中未找到。")
        return None, None
    
    # 只保留指定车辆的轨迹和类别信息
    filtered_trajectories = {vehicle_id: all_trajectories[vehicle_id]}
    filtered_classes = {vehicle_id: all_vehicle_classes.get(vehicle_id, {"class": "car", "color": "#72deee"})}
    
    # 生成输出文件路径
    output_video_name = f"vehicle_{vehicle_id}.mp4"
    output_video_path = os.path.join(output_dir, output_video_name)
    
    output_image_path = None
    if generate_image:
        output_image_name = f"vehicle_{vehicle_id}.png"
        output_image_path = os.path.join(output_dir, output_image_name)
    
    # 生成视频
    video_path = None
    try:
        print(f"生成视频: {output_video_path}")
        video_path = generate_single_video(
            trajectories=filtered_trajectories,
            vehicle_classes=filtered_classes,
            output_file=output_video_path,
            fps=fps,
            duration=duration,
            focus_on_collision=False,
            specific_collision_event=None
        )
        print(f"视频生成完成: {pathlib.Path(output_video_path).name}")
    except Exception as e:
        print(f"生成车辆ID {vehicle_id} 的视频时出错: {e}")
    
    # 生成静态轨迹图
    image_path = None
    if generate_image and output_image_path:
        try:
            image_path = generate_trajectory_image(
                trajectories=filtered_trajectories,
                vehicle_classes=filtered_classes,
                output_file=output_image_path
            )
        except Exception as e:
            print(f"生成车辆ID {vehicle_id} 的静态轨迹图时出错: {e}")
    
    return video_path, image_path

def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='根据指定的车辆ID生成轨迹视频和静态轨迹图')
    
    parser.add_argument('data_file', type=str,
                        help='轨迹数据JSON文件路径')
    parser.add_argument('vehicle_id', type=str,
                        help='要生成视频的车辆ID')
    parser.add_argument('--output', '-o', type=str,
                        default='vehicle_videos',
                        help='输出文件目录 (默认: vehicle_videos)')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                        help=f'视频帧率 (默认: {DEFAULT_FPS})')
    parser.add_argument('--duration', type=float, default=None,
                        help='视频时长（秒），如不指定则自动计算')
    parser.add_argument('--no-image', action='store_true',
                        help='不生成静态轨迹图，只生成视频')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 生成视频和图像
    video_path, image_path = generate_vehicle_id_video(
        data_file=args.data_file,
        vehicle_id=args.vehicle_id,
        output_dir=args.output,
        fps=args.fps,
        duration=args.duration,
        generate_image=not args.no_image
    )
    
    # 输出结果摘要
    print("\n=== 处理完成! ===")
    if video_path:
        print(f"✅ 视频已保存至: {video_path}")
    else:
        print("❌ 视频生成失败")
    
    if not args.no_image:
        if image_path:
            print(f"✅ 静态轨迹图已保存至: {image_path}")
        else:
            print("❌ 静态轨迹图生成失败")

if __name__ == '__main__':
    main() 