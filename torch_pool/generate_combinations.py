#!/usr/bin/env python3
"""
生成算子组合
基于 api_sampler 采样 2k 条两个算子的组合和 2k 条三个算子的组合
确保覆盖 transformers_modules.json 中的所有算子，且组合不重复
"""

import json
import os
import random
import sys
from typing import List, Set, Optional

# 添加项目根目录到路径（必须在导入 torch_pool 之前）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from torch_pool.api_sampler import TorchAPISampler


def generate_combinations(
    sampler: TorchAPISampler,
    num_combinations: int = 2000,
    combination_size: int = 2,
    output_file: Optional[str] = None,
) -> List[List[str]]:
    """
    生成算子组合，确保覆盖 transformers_modules.json 中的所有算子且不重复

    Args:
        sampler: TorchAPISampler 实例
        num_combinations: 需要生成的组合数量
        combination_size: 每个组合包含的算子数量（2 或 3）
        output_file: 输出文件路径

    Returns:
        组合列表
    """
    # 只覆盖 transformers_modules.json 中出现的算子
    target_ops = sorted(list(sampler.ops_in_transformers))
    print(f"\n生成 {num_combinations} 个 {combination_size} 算子组合...")
    print(f"目标算子数（transformers 中出现的）: {len(target_ops)}")
    print(f"总可用算子数: {len(sampler.all_ops_list)}")

    combinations: List[List[str]] = []
    seen_combinations: Set[frozenset] = set()  # 用于去重（不考虑顺序）

    # 第一阶段：确保覆盖 transformers 中出现的所有算子
    # 为每个算子至少生成一个组合
    print("\n第一阶段：确保覆盖 transformers 中的所有算子...")
    uncovered_ops = set(target_ops)
    attempts = 0
    max_attempts = len(target_ops) * 100  # 最多尝试次数

    while uncovered_ops and attempts < max_attempts:
        attempts += 1
        # 随机选择一个未覆盖的算子作为种子
        seed_op = random.choice(list(uncovered_ops))

        # 基于共现频率采样组合
        try:
            combo = sampler.sample_api_combination(
                seed_api=seed_op,
                num_apis=combination_size,
                num_samples=1,
                method="cooccurrence_weighted",
                temperature=1.5,  # 稍微增加随机性
            )[0]

            # 确保组合大小正确
            if len(combo) != combination_size:
                continue

            # 转换为 frozenset 用于去重
            combo_set = frozenset(combo)
            if combo_set not in seen_combinations:
                combinations.append(combo)
                seen_combinations.add(combo_set)
                # 更新未覆盖的算子集合
                uncovered_ops -= set(combo)

                if len(combinations) % 100 == 0:
                    print(
                        f"  已生成 {len(combinations)} 个组合，未覆盖算子: {len(uncovered_ops)}"
                    )
        except Exception:
            # 如果采样失败，尝试随机采样
            try:
                combo = sampler._sample_random(seed_op, combination_size)
                combo_set = frozenset(combo)
                if combo_set not in seen_combinations:
                    combinations.append(combo)
                    seen_combinations.add(combo_set)
                    uncovered_ops -= set(combo)
            except Exception:
                pass

    print(f"  第一阶段完成：生成 {len(combinations)} 个组合")
    print(f"  未覆盖的算子数: {len(uncovered_ops)}")
    if uncovered_ops:
        print(f"  未覆盖的算子示例: {list(uncovered_ops)[:5]}")

    # 第二阶段：继续采样直到达到目标数量
    print(f"\n第二阶段：继续采样直到达到 {num_combinations} 个组合...")
    attempts = 0
    max_attempts = num_combinations * 10  # 最多尝试次数

    while len(combinations) < num_combinations and attempts < max_attempts:
        attempts += 1

        # 随机选择一个 transformers 中的算子作为种子
        seed_op = random.choice(target_ops)

        # 基于共现频率采样组合
        try:
            combo = sampler.sample_api_combination(
                seed_api=seed_op,
                num_apis=combination_size,
                num_samples=1,
                method="cooccurrence_weighted",
                temperature=1.5,
            )[0]

            if len(combo) != combination_size:
                continue

            combo_set = frozenset(combo)
            if combo_set not in seen_combinations:
                combinations.append(combo)
                seen_combinations.add(combo_set)

                if len(combinations) % 500 == 0:
                    print(f"  已生成 {len(combinations)} / {num_combinations} 个组合")
        except Exception:
            # 如果采样失败，尝试随机采样
            try:
                combo = sampler._sample_random(seed_op, combination_size)
                combo_set = frozenset(combo)
                if combo_set not in seen_combinations:
                    combinations.append(combo)
                    seen_combinations.add(combo_set)
            except Exception:
                pass

    print(f"\n最终生成 {len(combinations)} 个组合")

    # 统计覆盖情况（只统计 transformers 中的算子）
    covered_ops = set()
    for combo in combinations:
        covered_ops.update(combo)

    # 计算在 transformers 算子中的覆盖率
    covered_in_transformers = covered_ops & set(target_ops)
    coverage = len(covered_in_transformers) / len(target_ops) * 100 if target_ops else 0
    print(
        f"Transformers 算子覆盖情况: {len(covered_in_transformers)} / {len(target_ops)} ({coverage:.1f}%)"
    )

    uncovered = set(target_ops) - covered_in_transformers
    if uncovered:
        print(
            f"未覆盖的 transformers 算子 ({len(uncovered)} 个): {sorted(list(uncovered))[:10]}..."
        )

    # 也统计一下是否使用了 transformers 之外的算子
    ops_outside_transformers = covered_ops - set(target_ops)
    if ops_outside_transformers:
        print(
            f"额外使用的非-transformers 算子 ({len(ops_outside_transformers)} 个): {sorted(list(ops_outside_transformers))[:10]}..."
        )

    # 保存结果
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combinations, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    return combinations


def main():
    """主函数"""
    print("=" * 80)
    print("生成算子组合")
    print("=" * 80)

    # 创建采样器
    sampler = TorchAPISampler(force_recompute=False)

    # 设置输出目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "resources", "operator_combinations")
    os.makedirs(output_dir, exist_ok=True)

    # 生成两个算子的组合
    print("\n" + "=" * 80)
    print("生成两个算子的组合")
    print("=" * 80)
    combinations_2 = generate_combinations(
        sampler=sampler,
        num_combinations=2000,
        combination_size=2,
        output_file=os.path.join(output_dir, "2ops.json"),
    )

    # 生成三个算子的组合
    print("\n" + "=" * 80)
    print("生成三个算子的组合")
    print("=" * 80)
    combinations_3 = generate_combinations(
        sampler=sampler,
        num_combinations=2000,
        combination_size=3,
        output_file=os.path.join(output_dir, "3ops.json"),
    )

    # 生成汇总报告
    report = {
        "2ops": {
            "count": len(combinations_2),
            "file": "2ops.json",
        },
        "3ops": {
            "count": len(combinations_3),
            "file": "3ops.json",
        },
        "total_ops_in_transformers": len(sampler.ops_in_transformers),
        "total_ops_available": len(sampler.all_ops),
    }

    report_file = os.path.join(output_dir, "report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)
    print(f"两个算子组合: {len(combinations_2)} 条")
    print(f"三个算子组合: {len(combinations_3)} 条")
    print(f"报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
