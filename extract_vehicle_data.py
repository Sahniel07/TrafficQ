import json
import os
import argparse
import pathlib
from datetime import datetime

def extract_vehicle_data(data_file, vehicle_id, output_file=None):
    """
    从原始数据文件中提取指定车辆ID的数据并保存为新的JSON文件。
    
    参数:
        data_file (str): 原始轨迹数据JSON文件路径
        vehicle_id (str): 要提取的车辆ID
        output_file (str, optional): 输出JSON文件路径。如果不提供，将自动生成。
    
    返回:
        str: 生成的JSON文件路径，如果提取失败则返回None
    """
    print(f"\n=== 提取车辆ID: {vehicle_id} 的数据 ===")
    
    # 读取原始数据文件
    print(f"读取数据文件: {data_file}")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # 检查数据结构
        if 'data' not in original_data or 'entities' not in original_data['data'] or 'edges' not in original_data['data']['entities']:
            print(f"错误: 数据文件 {data_file} 的结构不正确。无法提取数据。")
            return None
            
        print(f"成功读取原始数据文件。")
    except Exception as e:
        print(f"读取数据文件 {data_file} 时出错: {e}")
        return None
    
    # 提取指定车辆ID的数据
    vehicle_data = None
    edges = original_data['data']['entities']['edges']
    for edge in edges:
        if 'node' in edge and 'id' in edge['node'] and edge['node']['id'] == vehicle_id:
            vehicle_data = edge
            break
    
    if vehicle_data is None:
        print(f"错误: 车辆ID {vehicle_id} 在数据文件中未找到。")
        return None
    
    # 创建要保存的数据结构（保持与原始数据一致的结构）
    extracted_data = {
        "data": {
            "entities": {
                "edges": [vehicle_data]
            }
        }
    }
    
    # 如果没有提供输出文件路径，生成默认路径
    if output_file is None:
        output_dir = "extracted_data"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"vehicle_{vehicle_id}.json")
    else:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # 保存提取的数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 成功提取车辆ID {vehicle_id} 的数据并保存至: {output_file}")
        return output_file
    except Exception as e:
        print(f"保存提取的数据时出错: {e}")
        return None

def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='从原始数据文件中提取指定车辆ID的数据')
    
    parser.add_argument('data_file', type=str,
                        help='原始轨迹数据JSON文件路径')
    parser.add_argument('vehicle_id', type=str,
                        help='要提取的车辆ID')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出JSON文件路径 (默认: ./extracted_data/vehicle_ID.json)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 提取数据
    output_path = extract_vehicle_data(
        data_file=args.data_file,
        vehicle_id=args.vehicle_id,
        output_file=args.output
    )
    
    # 输出结果
    if output_path:
        print(f"\n提取完成! 数据已保存至: {output_path}")
    else:
        print("\n提取失败! 未能保存数据文件。")

if __name__ == '__main__':
    main() 