#!/usr/bin/env python3
"""
基于 cudallm 数据构建 SFT 数据集（llamafactory 格式）
使用 question_generate.py 中的 prompt 构建方法
"""

import os
import sys
import json
from typing import List, Dict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from torch_pool.operator_query import PyTorchOperatorQuery
from torch_generate.question_generate import get_operator_docstrings, build_prompt


def load_example_code(example_file: str = None) -> str:
    """
    加载示例代码
    
    Args:
        example_file: 示例代码文件路径
        
    Returns:
        示例代码字符串
    """
    if example_file and os.path.exists(example_file):
        with open(example_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # 使用默认示例
        return """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        x = self.pool1(x)
        x = self.pool2(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]"""


def build_sft_dataset_from_cudallm(
    input_files: List[str],
    output_file: str,
    operator_query: PyTorchOperatorQuery,
    example_code: str,
) -> Dict:
    """
    从 cudallm JSON 文件构建 SFT 数据集
    
    Args:
        input_files: 输入的 cudallm JSON 文件列表（如 1ops_cudallm.json, 2ops_cudallm.json）
        output_file: 输出文件路径
        operator_query: PyTorchOperatorQuery 实例
        example_code: 示例代码
        
    Returns:
        统计信息字典
    """
    all_data = []
    stats = {
        "total_samples": 0,
        "by_file": {},
        "empty_code_count": 0,
        "empty_operators_count": 0,
    }
    
    print("=" * 80)
    print("Building SFT Dataset from cudallm data")
    print("=" * 80)
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: File not found: {input_file}, skipping...")
            continue
        
        print(f"\nProcessing: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        file_count = 0
        file_empty_code = 0
        file_empty_ops = 0
        
        for item in data:
            operators = item.get("operators", [])
            code = item.get("code", "").strip()
            
            # 跳过空数据
            if not operators:
                file_empty_ops += 1
                continue
            
            if not code:
                file_empty_code += 1
                continue
            
            # 获取 docstrings
            try:
                docstrings = get_operator_docstrings(operator_query, operators)
            except Exception as e:
                print(f"Warning: Failed to get docstrings for {operators}: {e}")
                docstrings = {op: f"No documentation available for {op}" for op in operators}
            
            # 构建 prompt
            try:
                prompt = build_prompt(operators, docstrings, example_code)
            except Exception as e:
                print(f"Warning: Failed to build prompt for {operators}: {e}")
                continue
            
            # 构建 llamafactory 格式的数据
            sft_item = {
                "instruction": prompt,
                "input": "",
                "output": code,
            }
            
            all_data.append(sft_item)
            file_count += 1
        
        stats["by_file"][os.path.basename(input_file)] = {
            "total": len(data),
            "valid": file_count,
            "empty_code": file_empty_code,
            "empty_operators": file_empty_ops,
        }
        
        print(f"  Processed: {len(data)} samples")
        print(f"  Valid: {file_count} samples")
        print(f"  Skipped (empty code): {file_empty_code}")
        print(f"  Skipped (empty operators): {file_empty_ops}")
    
    stats["total_samples"] = len(all_data)
    stats["empty_code_count"] = sum(f["empty_code"] for f in stats["by_file"].values())
    stats["empty_operators_count"] = sum(f["empty_operators"] for f in stats["by_file"].values())
    
    # 保存数据集
    print(f"\nSaving dataset to: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_data)} samples to {output_file}")
    
    return stats


def main():
    """主函数"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # 输入文件
    input_dir = "resources/torch_ops"
    input_files = [
        os.path.join(input_dir, "1ops_cudallm.json"),
        os.path.join(input_dir, "2ops_cudallm.json"),
        os.path.join(input_dir, "3ops_cudallm.json"),
    ]
    
    # 输出文件
    output_file = "resources/sft_dataset/cudallm_sft.json"
    
    # 示例代码文件
    example_file = "resources/KernelBench/level2/6_Conv3d_Softmax_MaxPool_MaxPool.py"
    
    # 初始化 operator query
    print("Initializing PyTorchOperatorQuery...")
    operator_query = PyTorchOperatorQuery()
    
    # 加载示例代码
    example_code = load_example_code(example_file)
    print(f"Loaded example code from: {example_file if os.path.exists(example_file) else 'default'}")
    
    # 构建数据集
    stats = build_sft_dataset_from_cudallm(
        input_files=input_files,
        output_file=output_file,
        operator_query=operator_query,
        example_code=example_code,
    )
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total valid samples: {stats['total_samples']:,}")
    print(f"Total skipped (empty code): {stats['empty_code_count']:,}")
    print(f"Total skipped (empty operators): {stats['empty_operators_count']:,}")
    print("\nBy file:")
    for filename, file_stats in stats["by_file"].items():
        print(f"  {filename}:")
        print(f"    Total: {file_stats['total']:,}")
        print(f"    Valid: {file_stats['valid']:,}")
        print(f"    Empty code: {file_stats['empty_code']:,}")
        print(f"    Empty operators: {file_stats['empty_operators']:,}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

