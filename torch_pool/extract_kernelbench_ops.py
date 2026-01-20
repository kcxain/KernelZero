#!/usr/bin/env python3
"""
提取 KernelBench level1/level2 模块中使用的 PyTorch 算子，并写入 jsonl。
每行结构: {"filename": "<相对 KernelBench 的路径>", "ops": ["torch.xxx", ...]}
"""

import json
import os
import sys
from typing import Dict, List

# Ensure project root is on sys.path for "utils" imports when run as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.ops_helper import extract_operators_from_code  # noqa: E402


def collect_ops_for_level(level_dir: str, kernelbench_root: str) -> List[Dict[str, List[str]]]:
    """
    遍历某个 level 目录，收集每个 Python 文件的算子。
    """
    results: List[Dict[str, List[str]]] = []
    for entry in sorted(os.listdir(level_dir)):
        if not entry.endswith(".py"):
            continue
        file_path = os.path.join(level_dir, entry)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        ops = extract_operators_from_code(code)
        rel_path = os.path.relpath(file_path, kernelbench_root)
        results.append({"filename": rel_path, "ops": ops})
    return results


def write_jsonl(records: List[Dict[str, List[str]]], output_path: str) -> None:
    """
    将记录写入 jsonl 文件。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    kernelbench_root = os.path.join(PROJECT_ROOT, "resources", "KernelBench")

    level_configs = {
        "level1": os.path.join(kernelbench_root, "level1.jsonl"),
        "level2": os.path.join(kernelbench_root, "level2.jsonl"),
    }

    for level_name, output_file in level_configs.items():
        level_dir = os.path.join(kernelbench_root, level_name)
        if not os.path.isdir(level_dir):
            print(f"[WARN] missing directory: {level_dir}")
            continue

        records = collect_ops_for_level(level_dir, kernelbench_root)
        write_jsonl(records, output_file)

        print(f"{level_name}: wrote {len(records)} entries to {output_file}")


if __name__ == "__main__":
    main()