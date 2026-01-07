#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本功能：
1. 识别指定文件夹下所有json文件
2. 提取score为1的数据
3. 只保留raw_response字段，改名为raw_problem
4. 合并所有数据到merge_data.json
"""

import json
import os
import argparse
from pathlib import Path


def extract_score_one_data(folder_path):
    """
    从指定文件夹的所有json文件中提取score为1的数据
    
    Args:
        folder_path: 要处理的文件夹路径
        
    Returns:
        list: 包含所有提取的数据（每个数据只有raw_problem字段）
    """
    folder = Path(folder_path)
    merged_data = []
    
    # 获取文件夹下所有json文件
    json_files = list(folder.glob("*.json"))
    
    # 排除输出文件本身
    json_files = [f for f in json_files if f.name != "merge_data.json"]
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for json_file in json_files:
        print(f"正在处理: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 确保data是列表
            if not isinstance(data, list):
                print(f"警告: {json_file.name} 不是数组格式，跳过")
                continue
            
            # 提取score为1的数据
            count = 0
            for item in data:
                if isinstance(item, dict) and item.get("score") == 1:
                    # 检查是否有raw_response字段
                    if "raw_response" in item:
                        # 创建新对象，只包含raw_problem字段
                        new_item = {
                            "raw_problem": item["raw_response"]
                        }
                        merged_data.append(new_item)
                        count += 1
            
            print(f"  - 从 {json_file.name} 提取了 {count} 条score=1的数据")
            
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析 {json_file.name}: {e}")
        except Exception as e:
            print(f"错误: 处理 {json_file.name} 时出错: {e}")
    
    return merged_data


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='从指定文件夹的JSON文件中提取score为1的数据，并合并到merge_data.json'
    )
    parser.add_argument(
        '--folder', '-f',
        type=str,
        required=True,
        help='要处理的文件夹路径（输入和输出都在此文件夹）'
    )
    
    args = parser.parse_args()
    
    # 获取指定的文件夹路径
    folder_path = Path(args.folder).resolve()
    
    # 检查文件夹是否存在
    if not folder_path.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        return
    
    if not folder_path.is_dir():
        print(f"错误: 路径不是文件夹: {folder_path}")
        return
    
    # 从指定文件夹提取数据
    print(f"开始处理文件夹: {folder_path}")
    merged_data = extract_score_one_data(folder_path)
    
    # 保存合并后的数据到同一文件夹
    output_file = folder_path / "merge_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！共提取 {len(merged_data)} 条数据")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

