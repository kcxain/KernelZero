#!/usr/bin/env python3
"""
PyTorch 2.6 算子 docstring 提取脚本
提取 torch, torch.nn, torch.nn.functional 的所有算子及其 docstring
"""

import inspect
import json
import os
from typing import Dict, Set
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTorchOperatorScraper:
    """PyTorch 算子 docstring 提取类"""

    def __init__(self, base_url: str = "https://docs.pytorch.org/docs/2.6/"):
        self.base_url = base_url
        self.operators = {"torch": {}, "torch.nn": {}, "torch.nn.functional": {}}

    def get_torch_operators(self) -> Set[str]:
        """获取 torch 模块的所有算子"""
        operators = set()

        # 获取所有公共函数和类
        for name in dir(torch):
            if not name.startswith("_"):
                obj = getattr(torch, name)
                # 过滤掉模块、类型等
                if callable(obj) or inspect.isclass(obj):
                    operators.add(name)

        return operators

    def get_nn_operators(self) -> Set[str]:
        """获取 torch.nn 模块的所有算子（类）"""
        operators = set()

        for name in dir(nn):
            if not name.startswith("_"):
                obj = getattr(nn, name)
                if inspect.isclass(obj):
                    operators.add(name)

        return operators

    def get_functional_operators(self) -> Set[str]:
        """获取 torch.nn.functional 模块的所有算子（函数）"""
        operators = set()

        for name in dir(F):
            if not name.startswith("_"):
                obj = getattr(F, name)
                if callable(obj):
                    operators.add(name)

        return operators

    def get_doc_url(self, module: str, operator: str) -> str:
        """构建文档 URL

        格式: https://docs.pytorch.org/docs/2.6/generated/torch.sparse_coo_tensor.html#torch.sparse_coo_tensor
        """
        # 构建完整的模块路径
        full_path = f"{module}.{operator}"
        # 生成 URL: base_url/generated/{module}.{operator}.html#{module}.{operator}
        return f"{self.base_url}/generated/{full_path}.html#{full_path}"

    def get_operator_doc(self, module: str, operator: str) -> Dict[str, str]:
        """获取算子的 docstring 信息"""
        try:
            if module == "torch":
                obj = getattr(torch, operator)
            elif module == "torch.nn":
                obj = getattr(nn, operator)
            elif module == "torch.nn.functional":
                obj = getattr(F, operator)
            else:
                return {"signature": "", "description": "", "url": ""}

            doc = inspect.getdoc(obj)

            # 获取签名
            signature = ""
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    signature = str(sig)
                except (ValueError, TypeError):
                    signature = ""

            # 格式化返回（如果没有 docstring，返回 None 作为 description）
            return {
                "signature": f"{operator}{signature}" if signature else operator,
                "description": doc if doc else None,
                "url": self.get_doc_url(module, operator),
            }
        except Exception:
            return {
                "signature": operator,
                "description": None,
                "url": self.get_doc_url(module, operator),
            }

    def scrape_all(self, save_path: str = "torch_operators.json"):
        """提取所有算子的 docstring"""
        print("开始提取 PyTorch 2.6 算子 docstring...")
        print(f"PyTorch 版本: {torch.__version__}\n")

        # 获取所有算子
        print("1. 获取 torch 算子...")
        torch_ops = self.get_torch_operators()
        print(f"   找到 {len(torch_ops)} 个算子")

        print("2. 获取 torch.nn 算子...")
        nn_ops = self.get_nn_operators()
        print(f"   找到 {len(nn_ops)} 个算子")

        print("3. 获取 torch.nn.functional 算子...")
        functional_ops = self.get_functional_operators()
        print(f"   找到 {len(functional_ops)} 个算子\n")

        # 爬取每个算子的信息
        total = len(torch_ops) + len(nn_ops) + len(functional_ops)
        current = 0

        print("4. 提取算子 docstring...\n")

        # torch 算子
        skipped_count = 0
        for op in sorted(torch_ops):
            current += 1
            print(f"   [{current}/{total}] 提取 torch.{op}...", end="\r", flush=True)
            op_info = self.get_operator_doc("torch", op)
            # 只保存有有效 description 的算子（计算相关的）
            if op_info.get("description"):
                self.operators["torch"][op] = op_info
            else:
                skipped_count += 1

        # torch.nn 算子
        for op in sorted(nn_ops):
            current += 1
            print(f"   [{current}/{total}] 提取 torch.nn.{op}...", end="\r", flush=True)
            op_info = self.get_operator_doc("torch.nn", op)
            # 只保存有有效 description 的算子（计算相关的）
            if op_info.get("description"):
                self.operators["torch.nn"][op] = op_info
            else:
                skipped_count += 1

        # torch.nn.functional 算子
        for op in sorted(functional_ops):
            current += 1
            print(
                f"   [{current}/{total}] 提取 torch.nn.functional.{op}...",
                end="\r",
                flush=True,
            )
            op_info = self.get_operator_doc("torch.nn.functional", op)
            # 只保存有有效 description 的算子（计算相关的）
            if op_info.get("description"):
                self.operators["torch.nn.functional"][op] = op_info
            else:
                skipped_count += 1

        valid_count = sum(len(ops) for ops in self.operators.values())
        print(
            f"\n   完成！共提取 {valid_count} 个有效算子（已过滤 {skipped_count} 个无文档的算子）"
        )

        # 保存结果
        self.save_results(save_path)
        print(f"\n结果已保存到: {save_path}")

    def save_results(self, save_path: str):
        """保存结果到文件"""
        # 保存 JSON 格式
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.operators, f, indent=2, ensure_ascii=False)

        # 同时保存人类可读的文本格式
        txt_path = save_path.replace(".json", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PyTorch 2.6 算子文档\n")
            f.write("=" * 80 + "\n\n")

            for module, ops in self.operators.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"模块: {module}\n")
                f.write(f"{'=' * 80}\n\n")

                for op_name, op_info in sorted(ops.items()):
                    f.write(f"算子名: {op_name}\n")
                    f.write(f"签名: {op_info.get('signature', 'N/A')}\n")
                    f.write(f"文档链接: {op_info.get('url', 'N/A')}\n")
                    f.write(f"用法说明:\n{op_info.get('description', 'N/A')}\n")
                    f.write("-" * 80 + "\n\n")

        print(f"结果已保存到: {txt_path}")


def main():
    """主函数"""
    scraper = PyTorchOperatorScraper()

    # 设置保存路径到 resources/torch_operators/ 目录
    # torch_pool/ -> 项目根目录 -> resources/torch_operators/
    project_root = os.path.dirname(os.path.dirname(__file__))
    resources_dir = os.path.join(project_root, "resources", "torch_operators")
    os.makedirs(resources_dir, exist_ok=True)
    save_path = os.path.join(resources_dir, "torch_operators.json")

    # 开始爬取
    scraper.scrape_all(save_path)

    # 打印统计信息
    print("\n" + "=" * 80)
    print("统计信息:")
    print("=" * 80)
    for module, ops in scraper.operators.items():
        print(f"{module}: {len(ops)} 个算子")


if __name__ == "__main__":
    main()
