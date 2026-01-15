#!/usr/bin/env python3
"""
通用 SFT 数据集构建脚本
支持从不同格式的数据源（如 cudallm, kernelbook）构建 SFT 数据集
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional
import re

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from torch_pool.operator_query import PyTorchOperatorQuery
from torch_generate.question_generate import get_operator_docstrings, build_prompt


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


def extract_operators_from_code(code: str) -> List[str]:
    """
    从 Python 代码中提取使用的 PyTorch 算子

    Args:
        code: Python 代码字符串

    Returns:
        算子列表
    """
    operators = set()

    # 匹配 torch.xxx 函数调用
    torch_pattern = (
        r"torch\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*[\(\[]"
    )
    for match in re.finditer(torch_pattern, code):
        op = f"torch.{match.group(1)}"
        # 过滤掉一些非算子的调用
        if not any(
            skip in op
            for skip in [
                "torch.tensor",
                "torch.randn",
                "torch.rand",
                "torch.zeros",
                "torch.ones",
                "torch.arange",
                "torch.linspace",
                "torch.empty",
                "torch.full",
                "torch.cuda",
                "torch.device",
                "torch.no_grad",
                "torch.enable_grad",
                "torch.set_grad_enabled",
            ]
        ):
            operators.add(op)

    # 匹配 nn.xxx 模块使用
    nn_pattern = r"nn\.([A-Z][a-zA-Z0-9_]*)\s*\("
    for match in re.finditer(nn_pattern, code):
        op = f"torch.nn.{match.group(1)}"
        operators.add(op)

    # 匹配 F.xxx 函数调用
    f_pattern = r"F\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    for match in re.finditer(f_pattern, code):
        op = f"torch.nn.functional.{match.group(1)}"
        operators.add(op)

    return sorted(list(operators))


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


def build_sft_item(
    operators: List[str],
    code: str,
    operator_query: PyTorchOperatorQuery,
    example_code: str,
) -> Optional[Dict[str, str]]:
    """
    构建单条 SFT 数据

    Args:
        operators: 算子列表
        code: Python 代码
        operator_query: PyTorchOperatorQuery 实例
        example_code: 示例代码

    Returns:
        SFT 数据字典，失败返回 None
    """
    if not operators or not code.strip():
        return None

    # 获取 docstrings
    try:
        docstrings = get_operator_docstrings(operator_query, operators)
    except Exception:
        docstrings = {op: f"No documentation available for {op}" for op in operators}

    # 构建 prompt
    try:
        prompt = build_prompt(operators, docstrings, example_code)
    except Exception:
        return None

    # 构建 llamafactory 格式的数据
    sft_item = {
        "instruction": prompt,
        "input": "",
        "output": code,
    }

    return sft_item


def process_cudallm_format(
    input_files: List[str],
    operator_query: PyTorchOperatorQuery,
    example_code: str,
) -> List[Dict[str, str]]:
    """
    处理 cudallm 格式的数据（带有 operators 和 code 字段）

    Args:
        input_files: 输入文件列表
        operator_query: PyTorchOperatorQuery 实例
        example_code: 示例代码

    Returns:
        SFT 数据列表
    """
    all_data = []

    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: File not found: {input_file}, skipping...")
            continue

        print(f"Processing cudallm format: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            operators = item.get("operators", [])
            code = item.get("code", "").strip()

            sft_item = build_sft_item(operators, code, operator_query, example_code)
            if sft_item:
                all_data.append(sft_item)

        print(f"  Processed {len(data)} items")

    return all_data


def process_kernelbook_format(
    input_file: str,
    operator_query: PyTorchOperatorQuery,
    example_code: str,
    max_samples: int = None,
) -> List[Dict[str, str]]:
    """
    处理 kernelbook 格式的数据（带有 python_code 字段）

    Args:
        input_file: 输入文件路径
        operator_query: PyTorchOperatorQuery 实例
        example_code: 示例代码
        max_samples: 最大样本数

    Returns:
        SFT 数据列表
    """
    all_data = []

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return all_data

    print(f"Processing kernelbook format: {input_file}")
    print("Loading data (this may take a while for large files)...")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items")

    if max_samples:
        data = data[:max_samples]
        print(f"Processing first {max_samples} samples")

    success_count = 0
    empty_ops_count = 0
    empty_code_count = 0

    for idx, item in enumerate(data):
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(data)}")

        # 获取 python_code
        code = item.get("python_code", "").strip()
        if not code:
            empty_code_count += 1
            continue

        # 从代码中提取算子
        operators = extract_operators_from_code(code)
        if not operators:
            empty_ops_count += 1
            continue

        # 规范化算子名称
        operators = [normalize_operator_name(op) for op in operators]

        sft_item = build_sft_item(operators, code, operator_query, example_code)
        if sft_item:
            all_data.append(sft_item)
            success_count += 1

    print(f"  Total items: {len(data)}")
    print(f"  Success: {success_count}")
    print(f"  Empty code: {empty_code_count}")
    print(f"  No operators found: {empty_ops_count}")

    return all_data


def process_generic_json(
    input_file: str,
    operator_query: PyTorchOperatorQuery,
    example_code: str,
    code_field: str = "code",
    operators_field: str = None,
    max_samples: int = None,
) -> List[Dict[str, str]]:
    """
    处理通用 JSON 格式的数据

    Args:
        input_file: 输入文件路径
        operator_query: PyTorchOperatorQuery 实例
        example_code: 示例代码
        code_field: 代码字段名
        operators_field: 算子字段名（如果为None则从代码中提取）
        max_samples: 最大样本数

    Returns:
        SFT 数据列表
    """
    all_data = []

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return all_data

    print(f"Processing generic JSON: {input_file}")
    print(f"  Code field: {code_field}")
    print(f"  Operators field: {operators_field or 'auto-extract'}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items")

    if max_samples:
        data = data[:max_samples]
        print(f"Processing first {max_samples} samples")

    success_count = 0

    for idx, item in enumerate(data):
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(data)}")

        # 获取代码
        code = item.get(code_field, "").strip()
        if not code:
            continue

        # 获取或提取算子
        if operators_field and operators_field in item:
            operators = item[operators_field]
            if isinstance(operators, str):
                operators = [operators]
        else:
            operators = extract_operators_from_code(code)

        if not operators:
            continue

        # 规范化算子名称
        operators = [normalize_operator_name(op) for op in operators]

        sft_item = build_sft_item(operators, code, operator_query, example_code)
        if sft_item:
            all_data.append(sft_item)
            success_count += 1

    print(f"  Success: {success_count}/{len(data)}")

    return all_data


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT dataset from various data sources"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input file or directory path"
    )
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument(
        "--format",
        type=str,
        choices=["cudallm", "kernelbook", "generic"],
        default="kernelbook",
        help="Input data format (default: kernelbook)",
    )
    parser.add_argument(
        "--code-field",
        type=str,
        default="python_code",
        help="Field name for code in generic format (default: python_code)",
    )
    parser.add_argument(
        "--operators-field",
        type=str,
        default=None,
        help="Field name for operators in generic format (default: auto-extract)",
    )
    parser.add_argument(
        "--example-file", type=str, default=None, help="Path to example code file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    args = parser.parse_args()

    # 初始化
    print("=" * 80)
    print("Building SFT Dataset")
    print("=" * 80)

    print("Initializing PyTorchOperatorQuery...")
    operator_query = PyTorchOperatorQuery()

    # 加载示例代码
    example_code = load_example_code(args.example_file)
    print(
        f"Loaded example code from: {args.example_file if args.example_file else 'default'}"
    )

    # 处理数据
    all_data = []

    if args.format == "cudallm":
        # cudallm 格式：支持多文件
        if os.path.isdir(args.input):
            input_files = [
                os.path.join(args.input, f)
                for f in os.listdir(args.input)
                if f.endswith(".json") and "cudallm" in f
            ]
        else:
            input_files = [args.input]
        all_data = process_cudallm_format(input_files, operator_query, example_code)

    elif args.format == "kernelbook":
        all_data = process_kernelbook_format(
            args.input, operator_query, example_code, args.max_samples
        )

    elif args.format == "generic":
        all_data = process_generic_json(
            args.input,
            operator_query,
            example_code,
            code_field=args.code_field,
            operators_field=args.operators_field,
            max_samples=args.max_samples,
        )

    # 保存结果
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_data)} samples to {args.output}")

    # 打印统计信息
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total valid samples: {len(all_data):,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
