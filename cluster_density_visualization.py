import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from tqdm import tqdm
import os
import contextily as ctx
from sklearn.neighbors import KernelDensity
import pandas as pd
from scipy.spatial import ConvexHull
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def load_clusters_and_trajectories(cluster_json_file, trajectory_data_file):
    """
    加载聚类结果和轨迹数据
    """
    # 加载聚类结果
    with open(cluster_json_file, 'r', encoding='utf-8') as f:
        clusters_data = json.load(f)
    
    # 从原始代码中提取轨迹数据
    from parse_vehicle_trajectories import parse_vehicle_trajectories
    trajectories, _ = parse_vehicle_trajectories(trajectory_data_file)
    
    # 将聚类数据转换为我们需要的格式
    clusters = {}
    for cluster_info in clusters_data:
        cluster_id = cluster_info['cluster_id']
        vehicle_ids = [v['vehicle_id'] for v in cluster_info['vehicles']]
        clusters[cluster_id] = vehicle_ids
        
    return clusters, trajectories

def calculate_cluster_density_metrics(clusters, trajectories):
    """
    计算每个聚类的各种密度指标
    """
    density_metrics = {}
    
    for cluster_id, vehicle_ids in tqdm(clusters.items(), desc="计算聚类密度指标"):
        # 筛选出当前聚类的轨迹
        cluster_trajectories = {vid: trajectories[vid] for vid in vehicle_ids if vid in trajectories}
        
        if not cluster_trajectories:
            continue
            
        # 计算该聚类的所有轨迹点
        all_points = []
        for traj in cluster_trajectories.values():
            all_points.extend(traj)
        
        all_points = np.array(all_points)
        
        # 指标1: 轨迹数量
        num_trajectories = len(cluster_trajectories)
        
        # 指标2: 点的数量
        num_points = len(all_points)
        
        # 指标3: 计算点密度 (点数/凸包面积)
        density = 0
        try:
            if len(all_points) >= 3:  # 凸包至少需要3个点
                # 计算凸包
                hull = ConvexHull(all_points)
                # 计算面积 (平方度)
                area = hull.volume
                # 密度 = 点数/面积
                if area > 0:
                    density = num_points / area
        except Exception as e:
            print(f"计算聚类 {cluster_id} 密度时出错: {e}")
        
        # 指标4: 平均轨迹间距离
        avg_distance = 0
        if num_trajectories > 1:
            # 这里可以使用原始代码中的DTW距离计算，但为简化起见，使用点的平均位置
            centroid = np.mean(all_points, axis=0)
            distances = np.sqrt(np.sum((all_points - centroid)**2, axis=1))
            avg_distance = np.mean(distances)
        
        # 存储指标
        density_metrics[cluster_id] = {
            'num_trajectories': num_trajectories,
            'num_points': num_points,
            'density': density,
            'avg_distance': avg_distance,
            'points': all_points,
            'bounds': {
                'min_lat': np.min(all_points[:, 0]) if len(all_points) > 0 else 0,
                'max_lat': np.max(all_points[:, 0]) if len(all_points) > 0 else 0,
                'min_lon': np.min(all_points[:, 1]) if len(all_points) > 0 else 0,
                'max_lon': np.max(all_points[:, 1]) if len(all_points) > 0 else 0
            }
        }
    
    return density_metrics

def create_density_heatmap(clusters, trajectories, density_metrics, output_dir="density_maps"):
    """
    为每个聚类创建密度热力图
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置风格
    plt.style.use('default')
    
    # 统计指标，用于生成汇总报告
    density_summary = []
    
    # 只处理多车辆聚类
    multi_vehicle_clusters = {cid: vids for cid, vids in clusters.items() if len(vids) > 1}
    
    for cluster_id, vehicle_ids in tqdm(multi_vehicle_clusters.items(), desc="生成密度热力图"):
        if int(cluster_id) not in density_metrics:
            continue
            
        metrics = density_metrics[int(cluster_id)]
        points = metrics['points']
        
        if len(points) < 10:  # 跳过点太少的聚类
            continue
            
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
        
        # 提取坐标
        lats = points[:, 0]
        lons = points[:, 1]
        
        # 1. 第一张图：显示原始轨迹和热力图
        
        # 首先绘制轨迹线
        for vid in vehicle_ids:
            if vid in trajectories:
                traj = trajectories[vid]
                ax1.plot(traj[:, 1], traj[:, 0], '-', linewidth=1.0, alpha=0.6)
                # 标记起点和终点
                ax1.plot(traj[0, 1], traj[0, 0], 'o', markersize=4, alpha=0.8)
                ax1.plot(traj[-1, 1], traj[-1, 0], 's', markersize=4, alpha=0.8)
        
        # 计算核密度估计
        if len(lats) > 1:
            # 创建网格
            x_min, x_max = np.min(lons) - 0.0005, np.max(lons) + 0.0005
            y_min, y_max = np.min(lats) - 0.0005, np.max(lats) + 0.0005
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([lons, lats])
            
            try:
                # 使用高斯核密度估计
                kernel = gaussian_kde(values)
                z = np.reshape(kernel(positions), xx.shape)
                
                # 绘制热力图
                im = ax1.imshow(z.T, extent=[x_min, x_max, y_min, y_max], 
                              origin='lower', cmap='hot', alpha=0.7)
                plt.colorbar(im, ax=ax1, label='Density')
            except Exception as e:
                print(f"绘制聚类 {cluster_id} 热力图时出错: {e}")
        
        # 设置标题和标签
        ax1.set_title(f'Cluster {cluster_id} Density Heatmap\n{metrics["num_trajectories"]} vehicles, {metrics["num_points"]} points', 
                     fontsize=12)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        # 尝试添加地图背景
        try:
            ctx.add_basemap(ax1, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, 
                          zoom=17, alpha=0.8)
        except Exception as e:
            print(f"添加地图背景失败: {e}")
        
        # 2. 第二张图：密度统计和分析
        
        # 使用2D直方图表示点的分布
        h = ax2.hist2d(lons, lats, bins=20, cmap='viridis')
        plt.colorbar(h[3], ax=ax2, label='Point Count')
        
        # 尝试绘制凸包
        try:
            if len(points) >= 3:
                hull = ConvexHull(points[:, :2])
                hull_points = points[hull.vertices, :2]
                # 添加凸包多边形
                ax2.add_patch(Polygon(hull_points[:, [1, 0]], alpha=0.3, color='r', fill=True))
        except Exception as e:
            print(f"绘制凸包失败: {e}")
        
        ax2.set_title(f'Density Analysis\nDensity: {metrics["density"]:.6f} points/area', fontsize=12)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_density.png'), dpi=300)
        plt.close()
        
        # 添加到汇总指标
        density_summary.append({
            'cluster_id': cluster_id,
            'vehicles': metrics['num_trajectories'],
            'points': metrics['num_points'],
            'density': metrics['density'],
            'avg_distance': metrics['avg_distance']
        })
    
    # 创建汇总密度图
    create_density_summary(density_summary, output_dir)
    
    return density_summary

def create_density_summary(density_summary, output_dir):
    """
    创建密度汇总报告和可视化
    """
    if not density_summary:
        print("没有足够的数据创建密度汇总")
        return
        
    # 转换为DataFrame
    df = pd.DataFrame(density_summary)
    
    # 按密度排序
    df_sorted = df.sort_values('density', ascending=False)
    
    # 保存汇总CSV
    df_sorted.to_csv(os.path.join(output_dir, 'density_summary.csv'), index=False)
    
    # 创建密度排名图
    plt.figure(figsize=(10, 8), dpi=200)
    
    # 只取前20个最高密度的聚类
    top_clusters = df_sorted.head(20)
    
    # 创建柱状图
    sns.barplot(x='cluster_id', y='density', data=top_clusters)
    plt.title('Top 20 Clusters by Density')
    plt.xticks(rotation=45)
    plt.xlabel('Cluster ID')
    plt.ylabel('Density (points/area)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'density_ranking.png'), dpi=300)
    plt.close()
    
    # 创建散点图，比较不同指标
    plt.figure(figsize=(10, 8), dpi=200)
    plt.scatter(df['vehicles'], df['density'], alpha=0.7, s=df['points']/50)
    plt.title('Cluster Density vs. Number of Vehicles')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Density (points/area)')
    plt.grid(True, alpha=0.3)
    
    # 添加标签标注
    for i, row in df.iterrows():
        plt.annotate(str(row['cluster_id']), 
                    (row['vehicles'], row['density']),
                    textcoords="offset points", 
                    xytext=(0,5), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'density_vs_vehicles.png'), dpi=300)
    plt.close()
    
    print(f"✅ 密度汇总报告和可视化已保存到 {output_dir} 目录")

def main():
    # 输入文件
    cluster_json_file = 'trajectory_clusters.json'
    trajectory_data_file = 'data.json'
    
    # 加载数据
    print("加载聚类和轨迹数据...")
    clusters, trajectories = load_clusters_and_trajectories(cluster_json_file, trajectory_data_file)
    
    # 计算每个聚类的密度指标
    print("计算聚类密度指标...")
    density_metrics = calculate_cluster_density_metrics(clusters, trajectories)
    
    # 创建密度可视化
    print("生成密度热力图和分析...")
    create_density_heatmap(clusters, trajectories, density_metrics)
    
    print("所有密度分析已完成!")

if __name__ == "__main__":
    main() 