#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import os
from datetime import datetime

def sort_trajectory_by_time(input_file, output_file=None):
    """
    读取JSON文件，对trajectory字段按time升序排序，并保存到新文件
    
    参数:
    input_file (str): 输入JSON文件路径
    output_file (str, optional): 输出JSON文件路径，如果不提供则自动生成
    
    返回:
    str: 输出文件的路径
    """
    # 如果未提供输出文件路径，则生成一个
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_sorted.json"
    
    # 读取JSON文件
    print(f"正在读取文件: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None
    
    # 检查数据结构并定位trajectory
    if 'data' not in data or 'entities' not in data['data'] or 'edges' not in data['data']['entities']:
        print("错误: JSON结构不匹配预期格式")
        return None
    
    # 遍历所有edges，对每个node中的trajectory进行排序
    edges_count = 0
    trajectories_sorted = 0
    
    for edge in data['data']['entities']['edges']:
        if 'node' in edge and 'trajectory' in edge['node']:
            edges_count += 1
            try:
                # 按time字段排序
                edge['node']['trajectory'] = sorted(edge['node']['trajectory'], 
                                                   key=lambda x: x.get('time', ''))
                trajectories_sorted += 1
            except Exception as e:
                print(f"排序过程出错: {e}，跳过此轨迹")
    
    if trajectories_sorted == 0:
        print("警告: 未找到任何可排序的trajectory字段")
        return None
    
    print(f"排序完成，共处理 {edges_count} 个entities，成功排序 {trajectories_sorted} 个trajectory")
    
    # 写入新的JSON文件
    print(f"正在保存到文件: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"文件已保存: {output_file}")
    except Exception as e:
        print(f"保存文件出错: {e}")
        return None
    
    return output_file

if __name__ == "__main__":
    # 命令行参数处理
    if len(sys.argv) < 2:
        print("用法: python sort_trajectory.py <输入JSON文件> [输出JSON文件]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = sort_trajectory_by_time(input_file, output_file)
    
    if result:
        print(f"处理成功，已保存到 {result}")
    else:
        print("处理失败")
        sys.exit(1) 