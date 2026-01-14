#!/usr/bin/env python3
"""
Torch API 采样程序
基于 transformers Module 中 API 共现统计信息，实现 API 组合采样

重写版本：
- 基于 torch_pool/__init__.py 中定义的算子列表
- 计算共现频率并应用平滑
- 缓存计算结果到本地文件
"""

import json
import os
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from torch_pool.ops import torch_ops, torch_nn_ops, torch_nn_functional_ops


class TorchAPISampler:
    """基于共现统计的 Torch API 采样器"""

    def __init__(
        self,
        transformers_modules_file: Optional[str] = None,
        cache_file: Optional[str] = None,
        smoothing_alpha: float = 1.0,
        force_recompute: bool = False,
    ):
        """
        初始化采样器

        Args:
            transformers_modules_file: transformers_modules.json 文件路径
            cache_file: 缓存文件路径（存储计算好的共现频率）
            smoothing_alpha: 平滑参数（拉普拉斯平滑，默认1.0）
            force_recompute: 是否强制重新计算（忽略缓存）
        """
        # 合并所有算子列表
        self.all_ops: Set[str] = (
            set(torch_ops) | set(torch_nn_ops) | set(torch_nn_functional_ops)
        )
        self.all_ops_list: List[str] = sorted(self.all_ops)

        print(f"总算子数量: {len(self.all_ops)}")
        print(f"  - torch.*: {len(torch_ops)}")
        print(f"  - torch.nn.*: {len(torch_nn_ops)}")
        print(f"  - torch.nn.functional.*: {len(torch_nn_functional_ops)}")

        # 设置文件路径
        if transformers_modules_file is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            transformers_modules_file = os.path.join(
                project_root,
                "resources",
                "transformers_modules",
                "transformers_modules.json",
            )

        if cache_file is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_file = os.path.join(
                project_root,
                "resources",
                "api_cooccurrence_cache.json",
            )

        self.transformers_modules_file = transformers_modules_file
        self.cache_file = cache_file
        self.smoothing_alpha = smoothing_alpha

        # 统计数据
        self.cooccurrence_matrix: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.api_frequency: Counter = Counter()
        self.ops_in_transformers: Set[str] = set()

        # 加载或计算统计数据
        if not force_recompute and os.path.exists(self.cache_file):
            print(f"从缓存文件加载: {self.cache_file}")
            self._load_cache()
        else:
            print("计算共现频率...")
            self._load_data()
            self._build_statistics()
            self._save_cache()

    def _load_data(self):
        """加载 transformers modules 数据"""
        if not os.path.exists(self.transformers_modules_file):
            raise FileNotFoundError(
                f"Transformers modules 文件不存在: {self.transformers_modules_file}\n"
                "请先运行 transformers_scraper.py 生成数据"
            )

        with open(self.transformers_modules_file, "r", encoding="utf-8") as f:
            self.modules_data = json.load(f)

        print(f"已加载 {len(self.modules_data)} 个 Module 的数据")

    def _build_statistics(self):
        """构建 API 共现统计信息（只统计我们关心的算子）"""
        print("构建 API 共现统计信息...")

        # 统计每个 Module 中的 API 组合（只考虑我们关心的算子）
        for module_name, module_info in self.modules_data.items():
            used_apis = module_info.get("used_apis", [])
            if not used_apis:
                continue

            # 过滤：只保留在我们算子列表中的 API
            filtered_apis = [api for api in used_apis if api in self.all_ops]
            if not filtered_apis:
                continue

            # 更新 API 频率
            for api in filtered_apis:
                self.api_frequency[api] += 1
                self.ops_in_transformers.add(api)

            # 构建共现矩阵（同一 Module 中的 API 两两共现）
            apis_set = set(filtered_apis)
            for api1 in apis_set:
                for api2 in apis_set:
                    if api1 != api2:
                        self.cooccurrence_matrix[api1][api2] += 1

        print(f"  在 transformers 中出现的算子: {len(self.ops_in_transformers)}")
        print(
            f"  未在 transformers 中出现的算子: {len(self.all_ops) - len(self.ops_in_transformers)}"
        )

        # 应用平滑：对所有算子对进行平滑处理
        print(f"应用平滑 (alpha={self.smoothing_alpha})...")
        self._apply_smoothing()

        print(f"  构建了 {len(self.cooccurrence_matrix)} 个算子的共现关系")

    def _apply_smoothing(self):
        """应用拉普拉斯平滑到共现矩阵"""
        # 对于所有算子对，如果不存在则初始化为平滑值
        # 对于已存在的，加上平滑值
        for api1 in self.all_ops_list:
            for api2 in self.all_ops_list:
                if api1 != api2:
                    # 拉普拉斯平滑：count + alpha
                    self.cooccurrence_matrix[api1][api2] += self.smoothing_alpha

    def _save_cache(self):
        """保存计算结果到缓存文件"""
        print(f"保存计算结果到缓存文件: {self.cache_file}")

        # 准备缓存数据
        cache_data = {
            "all_ops": self.all_ops_list,
            "ops_in_transformers": list(self.ops_in_transformers),
            "api_frequency": dict(self.api_frequency),
            "cooccurrence_matrix": {
                api1: dict(api2_dict)
                for api1, api2_dict in self.cooccurrence_matrix.items()
            },
            "smoothing_alpha": self.smoothing_alpha,
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        print("缓存已保存")

    def _load_cache(self):
        """从缓存文件加载计算结果"""
        with open(self.cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        self.all_ops_list = cache_data["all_ops"]
        self.all_ops = set(self.all_ops_list)
        self.ops_in_transformers = set(cache_data["ops_in_transformers"])
        self.api_frequency = Counter(cache_data["api_frequency"])

        # 恢复共现矩阵
        for api1, api2_dict in cache_data["cooccurrence_matrix"].items():
            self.cooccurrence_matrix[api1] = defaultdict(float, api2_dict)

        self.smoothing_alpha = cache_data.get("smoothing_alpha", 1.0)

        print(f"  已加载 {len(self.all_ops)} 个算子的共现统计")
        print(f"  在 transformers 中出现的算子: {len(self.ops_in_transformers)}")

    def get_cooccurrence_stats(self, api: str) -> Dict[str, float]:
        """
        获取指定 API 的共现统计信息

        Args:
            api: API 名称

        Returns:
            与其他 API 的共现频率字典（已平滑）
        """
        if api not in self.all_ops:
            raise ValueError(f"API '{api}' 不在算子列表中")

        return dict(self.cooccurrence_matrix.get(api, {}))

    def get_top_cooccurring_apis(
        self, api: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        获取与指定 API 共现频率最高的 API 列表

        Args:
            api: 输入 API 名称
            top_k: 返回前 k 个

        Returns:
            [(api_name, cooccurrence_count), ...] 列表
        """
        cooccurrence = self.get_cooccurrence_stats(api)
        sorted_apis = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)
        return sorted_apis[:top_k]

    def sample_api_combination(
        self,
        seed_api: str,
        num_apis: int = 5,
        num_samples: int = 10,
        method: str = "cooccurrence_weighted",
        temperature: float = 1.0,
        min_cooccurrence: float = 0.0,
    ) -> List[List[str]]:
        """
        根据种子 API 采样 API 组合

        Args:
            seed_api: 种子 API（输入的 API）
            num_apis: 每个组合包含的 API 数量（包括种子 API）
            num_samples: 生成多少个组合
            method: 采样方法
                - "cooccurrence_weighted": 基于共现频率加权采样
                - "frequency_weighted": 基于 API 频率加权采样
                - "hybrid": 混合方法（共现频率 + API 频率）
                - "random": 随机采样
            temperature: 温度参数（控制采样随机性，越大越随机）
            min_cooccurrence: 最小共现次数阈值（已平滑后的值）

        Returns:
            API 组合列表，每个组合是一个 API 名称列表
        """
        if seed_api not in self.all_ops:
            raise ValueError(f"API '{seed_api}' 不在算子列表中")

        combinations = []

        for _ in range(num_samples):
            if method == "cooccurrence_weighted":
                combination = self._sample_by_cooccurrence(
                    seed_api, num_apis, temperature, min_cooccurrence
                )
            elif method == "frequency_weighted":
                combination = self._sample_by_frequency(seed_api, num_apis, temperature)
            elif method == "hybrid":
                combination = self._sample_by_hybrid(
                    seed_api, num_apis, temperature, min_cooccurrence
                )
            elif method == "random":
                combination = self._sample_random(seed_api, num_apis)
            else:
                raise ValueError(f"未知的采样方法: {method}")

            combinations.append(combination)

        return combinations

    def _sample_by_cooccurrence(
        self, seed_api: str, num_apis: int, temperature: float, min_cooccurrence: float
    ) -> List[str]:
        """基于共现频率加权采样"""
        combination = [seed_api]
        available_apis = [api for api in self.all_ops_list if api != seed_api]

        # 获取与种子 API 共现的 API 及其频率（已平滑）
        cooccurrence = self.get_cooccurrence_stats(seed_api)

        # 过滤：只考虑共现次数 >= min_cooccurrence 的 API
        filtered_cooccurrence = {
            api: count
            for api, count in cooccurrence.items()
            if count >= min_cooccurrence and api in available_apis
        }

        # 如果没有足够的共现 API，从所有 API 中采样（使用平滑后的值）
        if len(filtered_cooccurrence) < num_apis - 1:
            filtered_cooccurrence = {
                api: cooccurrence.get(api, self.smoothing_alpha)
                for api in available_apis
            }

        # 计算采样权重（应用温度参数）
        weight_values = [
            (count ** (1.0 / temperature)) for count in filtered_cooccurrence.values()
        ]
        total_weight = sum(weight_values)
        if total_weight == 0:
            # 如果所有权重为0，使用均匀分布
            weights = [1.0 / len(filtered_cooccurrence)] * len(filtered_cooccurrence)
        else:
            weights = [w / total_weight for w in weight_values]

        # 采样剩余 API
        apis = list(filtered_cooccurrence.keys())
        sampled_indices = np.random.choice(
            len(apis), size=min(num_apis - 1, len(apis)), replace=False, p=weights
        )
        for idx in sampled_indices:
            combination.append(apis[idx])

        return combination

    def _sample_by_frequency(
        self, seed_api: str, num_apis: int, temperature: float
    ) -> List[str]:
        """基于 API 频率加权采样"""
        combination = [seed_api]
        available_apis = [api for api in self.all_ops_list if api != seed_api]

        # 计算频率权重（对于未出现的算子，使用平滑值）
        frequencies = {
            api: self.api_frequency.get(api, self.smoothing_alpha)
            for api in available_apis
        }

        weight_values = [(freq ** (1.0 / temperature)) for freq in frequencies.values()]
        total_weight = sum(weight_values)
        if total_weight == 0:
            weights = [1.0 / len(frequencies)] * len(frequencies)
        else:
            weights = [w / total_weight for w in weight_values]

        # 采样
        apis = list(frequencies.keys())
        sampled_indices = np.random.choice(
            len(apis), size=min(num_apis - 1, len(apis)), replace=False, p=weights
        )
        for idx in sampled_indices:
            combination.append(apis[idx])

        return combination

    def _sample_by_hybrid(
        self,
        seed_api: str,
        num_apis: int,
        temperature: float,
        min_cooccurrence: float,
        cooccurrence_weight: float = 0.7,
    ) -> List[str]:
        """混合方法：结合共现频率和 API 频率"""
        combination = [seed_api]
        available_apis = [api for api in self.all_ops_list if api != seed_api]

        # 获取共现频率（已平滑）
        cooccurrence = self.get_cooccurrence_stats(seed_api)
        filtered_cooccurrence = {
            api: count
            for api, count in cooccurrence.items()
            if count >= min_cooccurrence and api in available_apis
        }

        # 计算混合权重
        scores = {}
        max_cooccurrence = (
            max(filtered_cooccurrence.values())
            if filtered_cooccurrence
            else self.smoothing_alpha
        )
        max_frequency = (
            max(self.api_frequency.values())
            if self.api_frequency
            else self.smoothing_alpha
        )

        for api in available_apis:
            cooccurrence_score = filtered_cooccurrence.get(api, self.smoothing_alpha)
            frequency_score = self.api_frequency.get(api, self.smoothing_alpha)

            # 归一化并混合
            normalized_cooccurrence = cooccurrence_score / max(1, max_cooccurrence)
            normalized_frequency = frequency_score / max(1, max_frequency)

            scores[api] = (
                cooccurrence_weight * normalized_cooccurrence
                + (1 - cooccurrence_weight) * normalized_frequency
            )

        # 应用温度参数
        weight_values = [(score ** (1.0 / temperature)) for score in scores.values()]
        total_weight = sum(weight_values)
        if total_weight == 0:
            weights = [1.0 / len(scores)] * len(scores)
        else:
            weights = [w / total_weight for w in weight_values]

        # 采样
        apis = list(scores.keys())
        sampled_indices = np.random.choice(
            len(apis), size=min(num_apis - 1, len(apis)), replace=False, p=weights
        )
        for idx in sampled_indices:
            combination.append(apis[idx])

        return combination

    def _sample_random(self, seed_api: str, num_apis: int) -> List[str]:
        """随机采样"""
        combination = [seed_api]
        available_apis = [api for api in self.all_ops_list if api != seed_api]

        sampled = random.sample(available_apis, min(num_apis - 1, len(available_apis)))
        combination.extend(sampled)

        return combination

    def get_statistics_summary(self) -> Dict:
        """获取统计信息摘要"""
        total_ops = len(self.all_ops)
        ops_in_transformers = len(self.ops_in_transformers)
        ops_not_in_transformers = total_ops - ops_in_transformers

        # 计算总共现对数（对称矩阵，除以2）
        total_cooccurrences = (
            sum(
                sum(cooccurrence.values())
                for cooccurrence in self.cooccurrence_matrix.values()
            )
            / 2
        )

        top_apis = self.api_frequency.most_common(10)

        return {
            "total_ops": total_ops,
            "ops_in_transformers": ops_in_transformers,
            "ops_not_in_transformers": ops_not_in_transformers,
            "total_cooccurrence_pairs": int(total_cooccurrences),
            "smoothing_alpha": self.smoothing_alpha,
            "top_10_apis": top_apis,
        }


def main():
    """示例用法"""
    print("=" * 80)
    print("Torch API 采样程序示例")
    print("=" * 80)

    # 创建采样器（首次运行会计算并缓存，后续运行会直接加载缓存）
    sampler = TorchAPISampler(force_recompute=True)

    # 打印统计信息
    stats = sampler.get_statistics_summary()
    print("\n统计信息摘要:")
    print(f"  总算子数: {stats['total_ops']}")
    print(f"  在 transformers 中出现的算子: {stats['ops_in_transformers']}")
    print(f"  未在 transformers 中出现的算子: {stats['ops_not_in_transformers']}")
    print(f"  共现对数量: {stats['total_cooccurrence_pairs']}")
    print(f"  平滑参数: {stats['smoothing_alpha']}")
    print("\nTop 10 最常用 API:")
    for api, count in stats["top_10_apis"]:
        print(f"  {api}: {count} 次")

    # 示例：输入一个 API，采样组合
    seed_api = "torch.nn.Linear"

    print(f"\n{'=' * 80}")
    print(f"基于种子 API: {seed_api}")
    print("=" * 80)

    # 获取共现统计
    print(f"\n与 {seed_api} 共现频率最高的 Top 10 API:")
    top_cooccurring = sampler.get_top_cooccurring_apis(seed_api, top_k=10)
    for api, count in top_cooccurring:
        print(f"  {api}: {count:.2f}")

    # 采样组合
    print("\n采样方法 1: 基于共现频率加权 (cooccurrence_weighted)")
    combinations = sampler.sample_api_combination(
        seed_api=seed_api,
        num_apis=2,
        num_samples=5,
        method="cooccurrence_weighted",
        temperature=1.0,
    )
    for i, combo in enumerate(combinations, 1):
        print(f"  组合 {i}: {combo}")

    print("\n采样方法 2: 混合方法 (hybrid)")
    combinations = sampler.sample_api_combination(
        seed_api=seed_api, num_apis=2, num_samples=5, method="hybrid", temperature=1.0
    )
    for i, combo in enumerate(combinations, 1):
        print(f"  组合 {i}: {combo}")


if __name__ == "__main__":
    main()
