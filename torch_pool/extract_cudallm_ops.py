#!/usr/bin/env python3
"""
从 cudallm parquet 文件中提取算子组合和对应的 PyTorch 代码
保存到 resources/torch_ops 目录
"""

import json
import os
import sys
from typing import Dict, List, Any
import pandas as pd


def normalize_operator_name(op: str) -> str:
    """
    规范化算子名称：
    - nn.xxx -> torch.nn.xxx
    - F.xxx -> torch.nn.functional.xxx
    - torch.xxx -> torch.xxx (保持不变)
    """
    op = op.strip()
    
    # 已经是 torch. 开头
    if op.startswith("torch."):
        return op
    
    # nn.xxx -> torch.nn.xxx
    if op.startswith("nn."):
        return f"torch.{op}"
    
    # F.xxx -> torch.nn.functional.xxx
    if op.startswith("F."):
        return f"torch.nn.functional.{op[2:]}"
    
    # 如果都不匹配，尝试添加 torch. 前缀
    if "." not in op:
        return f"torch.{op}"
    
    return op


def extract_ops_from_parquet(
    parquet_path: str,
    output_dir: str = "resources/torch_ops"
) -> Dict[str, Any]:
    """
    从 parquet 文件中提取算子组合和代码
    
    Args:
        parquet_path: parquet 文件路径
        output_dir: 输出目录
        
    Returns:
        统计信息字典
    """
    print(f"Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 按算子数量分组
    grouped_data: Dict[int, List[Dict[str, Any]]] = {}
    
    # 统计信息
    stats = {
        "total_samples": len(df),
        "by_operator_count": {},
        "normalized_ops": set(),
    }
    
    # 处理每一行
    for idx, row in df.iterrows():
        # 解析 ops
        ops_raw = row['ops']
        if isinstance(ops_raw, str):
            try:
                ops = json.loads(ops_raw)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse ops at row {idx}: {ops_raw}")
                continue
        elif isinstance(ops_raw, list):
            ops = ops_raw
        else:
            print(f"Warning: Unexpected ops type at row {idx}: {type(ops_raw)}")
            continue
        
        # 规范化算子名称
        normalized_ops = [normalize_operator_name(op) for op in ops]
        
        # 获取代码
        py_code = row.get('py_code', '')
        if pd.isna(py_code):
            py_code = ''
        else:
            py_code = str(py_code)
        
        # 按算子数量分组
        op_count = len(normalized_ops)
        
        if op_count not in grouped_data:
            grouped_data[op_count] = []
        
        # 创建数据项（只保留算子和代码）
        item = {
            "operators": normalized_ops,
            "code": py_code,
        }
        
        grouped_data[op_count].append(item)
        
        # 更新统计
        stats["normalized_ops"].update(normalized_ops)
    
    # 更新统计信息
    for op_count, items in grouped_data.items():
        stats["by_operator_count"][op_count] = len(items)
    
    # 保存到文件
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    
    for op_count in sorted(grouped_data.keys()):
        output_file = os.path.join(output_dir, f"{op_count}ops_cudallm.json")
        
        # 保存数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(grouped_data[op_count], f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(grouped_data[op_count])} samples with {op_count} operators to {output_file}")
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, "cudallm_stats.json")
    stats_dict = {
        "total_samples": stats["total_samples"],
        "by_operator_count": stats["by_operator_count"],
        "unique_operators": sorted(list(stats["normalized_ops"])),
        "total_unique_operators": len(stats["normalized_ops"]),
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nStatistics saved to {stats_file}")
    
    return stats_dict


def main():
    """主函数"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # 输入文件路径
    parquet_path = "/share/personal/S/kechangxin/datasets/cudallm/rl_cuda_llm_0424.parquet"
    
    # 输出目录
    output_dir = "resources/torch_ops"
    
    # 检查文件是否存在
    if not os.path.exists(parquet_path):
        print(f"Error: File not found: {parquet_path}")
        sys.exit(1)
    
    # 提取数据
    stats = extract_ops_from_parquet(parquet_path, output_dir)
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total unique operators: {stats['total_unique_operators']}")
    print("\nSamples by operator count:")
    for op_count in sorted(stats['by_operator_count'].keys()):
        count = stats['by_operator_count'][op_count]
        print(f"  {op_count} operators: {count} samples")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

