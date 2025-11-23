#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def count_vehicles(file_path):
    """
    统计分析报告中的车辆总数
    
    Args:
        file_path: 分析报告的文件路径
    
    Returns:
        int: 车辆总数
    """
    # 用于匹配车辆ID的正则表达式
    vehicle_pattern = re.compile(r'\*\*\*\*\*\s+ANALYSIS REPORT FOR VEHICLE:\s+([0-9a-f\-]+)\s+\*\*\*\*\*\*')
    
    # 存储找到的所有车辆ID
    vehicle_ids = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = vehicle_pattern.search(line)
                if match:
                    vehicle_id = match.group(1)
                    vehicle_ids.add(vehicle_id)
        
        return len(vehicle_ids)
    
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return 0

if __name__ == "__main__":
    file_path = "output_results.txt"
    vehicle_count = count_vehicles(file_path)
    print(f"总共发现 {vehicle_count} 辆车") 