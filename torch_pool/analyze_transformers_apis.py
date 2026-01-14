#!/usr/bin/env python3
"""
分析 transformers_modules.json 中使用的 API 统计
包括可视化：最常用的 API、共现频率最高的 API 对等
"""

import json
import os
from collections import Counter, defaultdict
from typing import Set, Dict, List
import itertools

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # 使用非交互式后端
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("警告: matplotlib 或 seaborn 未安装，可视化功能将不可用")


def is_utility_api(api: str) -> bool:
    """
    判断是否为工具类或辅助函数 API
    这些 API 通常用于张量创建、保存/加载、随机数控制、梯度控制等
    """
    utility_patterns = [
        # 张量创建
        "torch.tensor",
        "torch.zeros",
        "torch.ones",
        "torch.empty",
        "torch.rand",
        "torch.randn",
        "torch.rand_like",
        "torch.zeros_like",
        "torch.ones_like",
        "torch.empty_like",
        "torch.randn_like",
        "torch.full",
        "torch.full_like",
        "torch.eye",
        "torch.arange",
        "torch.linspace",
        "torch.logspace",
        "torch.scalar_tensor",
        # 保存/加载
        "torch.load",
        "torch.save",
        # 随机数控制
        "torch.manual_seed",
        "torch.seed",
        "torch.Generator",
        "torch.randperm",
        "torch.randint",
        # 梯度控制
        "torch.no_grad",
        "torch.enable_grad",
        "torch.autocast",
        # 类型检查和转换
        "torch.is_tensor",
        "torch.from_numpy",
        "torch.as_tensor",
        # 工具函数
        "torch.numel",
        "torch.Size",
        "torch.get_default_dtype",
        "torch.set_default_dtype",
        "torch.are_deterministic_algorithms_enabled",
        # 其他工具
        "torch.compile",
        "torch.chunk",
        "torch.concat",
        "torch.concatenate",
        "torch.flatten",
        "torch.tile",
        "torch.broadcast_to",
        "torch.hstack",
        "torch.outer",
        "torch.argwhere",
        "torch.unique_consecutive",
        "torch.cdist",
        "torch.conj",
        "torch.cosine_similarity",
    ]

    # 检查是否匹配任何工具类模式
    for pattern in utility_patterns:
        if api == pattern or api.startswith(pattern + "."):
            return True

    return False


def analyze_transformers_apis(json_file: str = None):
    """分析 transformers modules 中使用的 API"""
    if json_file is None:
        # 默认路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_file = os.path.join(
            project_root,
            "resources",
            "transformers_modules",
            "transformers_modules.json",
        )

    print("=" * 80)
    print("Transformers Modules API 统计分析")
    print("=" * 80)

    # 加载数据
    print(f"\n加载数据文件: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        modules_data = json.load(f)

    print(f"总 Module 数: {len(modules_data)}")

    # 需要过滤掉的 API
    excluded_apis = {"torch.nn.Module", "torch.nn.ModuleList"}

    # 统计所有使用的 API
    all_apis: Set[str] = set()
    api_counter = Counter()
    modules_with_apis = 0
    modules_without_apis = 0
    excluded_count = 0
    utility_count = 0

    # 用于共现分析：记录每个模块使用的 API 集合
    module_api_sets: List[Set[str]] = []

    for module_name, module_info in modules_data.items():
        used_apis = module_info.get("used_apis", [])
        if used_apis:
            modules_with_apis += 1
            module_apis = set()  # 当前模块使用的 API（已过滤）
            for api in used_apis:
                # 过滤掉排除的 API
                if api in excluded_apis:
                    excluded_count += 1
                    continue
                # 过滤掉工具类 API
                if is_utility_api(api):
                    utility_count += 1
                    continue
                all_apis.add(api)
                api_counter[api] += 1
                module_apis.add(api)
            if module_apis:
                module_api_sets.append(module_apis)
        else:
            modules_without_apis += 1

    # 打印统计信息
    print(f"\n{'=' * 80}")
    print("统计结果:")
    print("=" * 80)
    print(f"使用 API 的 Module 数: {modules_with_apis}")
    print(f"未使用 API 的 Module 数: {modules_without_apis}")
    print(
        f"总的不同 API 数量: {len(all_apis)} (已排除 {len(excluded_apis)} 个基础类 + 工具类函数)"
    )
    print(f"总 API 调用次数: {sum(api_counter.values())}")
    print(f"被排除的基础类 API 调用次数: {excluded_count}")
    print(f"被排除的工具类 API 调用次数: {utility_count}")

    # 按模块分类统计
    torch_apis = set()
    nn_apis = set()
    functional_apis = set()
    other_apis = set()

    for api in all_apis:
        if api.startswith("torch.nn.functional"):
            functional_apis.add(api)
        elif api.startswith("torch.nn"):
            nn_apis.add(api)
        elif api.startswith("torch."):
            torch_apis.add(api)
        else:
            other_apis.add(api)

    print(f"\n{'=' * 80}")
    print("API 分类统计:")
    print("=" * 80)
    print(f"torch.* API 数量: {len(torch_apis)}")
    print(f"torch.nn.* API 数量: {len(nn_apis)}")
    print(f"torch.nn.functional.* API 数量: {len(functional_apis)}")
    if other_apis:
        print(f"其他 API 数量: {len(other_apis)}")

    # Top 20 最常用的 API
    print(f"\n{'=' * 80}")
    print("Top 20 最常用的 API:")
    print("=" * 80)
    for api, count in api_counter.most_common(20):
        print(f"  {api}: {count} 次")

    # 按模块统计
    print(f"\n{'=' * 80}")
    print("按模块分类的 Top API:")
    print("=" * 80)

    torch_counter = Counter({
        api: count
        for api, count in api_counter.items()
        if api.startswith("torch.") and not api.startswith("torch.nn")
    })
    nn_counter = Counter({
        api: count
        for api, count in api_counter.items()
        if api.startswith("torch.nn.") and not api.startswith("torch.nn.functional")
    })
    functional_counter = Counter({
        api: count
        for api, count in api_counter.items()
        if api.startswith("torch.nn.functional")
    })

    if torch_counter:
        print("\ntorch.* Top 10:")
        for api, count in torch_counter.most_common(10):
            print(f"  {api}: {count} 次")

    if nn_counter:
        print("\ntorch.nn.* Top 10:")
        for api, count in nn_counter.most_common(10):
            print(f"  {api}: {count} 次")

    if functional_counter:
        print("\ntorch.nn.functional.* Top 10:")
        for api, count in functional_counter.most_common(10):
            print(f"  {api}: {count} 次")

    # 计算共现频率
    print(f"\n{'=' * 80}")
    print("计算 API 共现频率...")
    print("=" * 80)
    cooccurrence = defaultdict(int)
    for module_apis in module_api_sets:
        # 对每个模块中的 API 对进行组合
        for api1, api2 in itertools.combinations(sorted(module_apis), 2):
            cooccurrence[(api1, api2)] += 1

    # Top 20 共现频率最高的 API 对
    top_cooccurrences = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[
        :20
    ]
    print("\nTop 20 共现频率最高的 API 对:")
    for (api1, api2), count in top_cooccurrences:
        print(f"  {api1} <-> {api2}: {count} 次")

    # 准备返回数据
    result = {
        "total_modules": len(modules_data),
        "modules_with_apis": modules_with_apis,
        "modules_without_apis": modules_without_apis,
        "total_unique_apis": len(all_apis),
        "total_api_calls": sum(api_counter.values()),
        "torch_apis": len(torch_apis),
        "nn_apis": len(nn_apis),
        "functional_apis": len(functional_apis),
        "other_apis": len(other_apis),
        "top_apis": api_counter.most_common(50),
        "top_cooccurrences": top_cooccurrences[:50],
        "api_counter": dict(api_counter),
        "cooccurrence": dict(cooccurrence),
    }

    # 可视化
    if VISUALIZATION_AVAILABLE:
        print(f"\n{'=' * 80}")
        print("生成可视化图表...")
        print("=" * 80)
        output_dir = os.path.join(os.path.dirname(json_file), "..", "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        create_visualizations(result, output_dir)
        print(f"可视化图表已保存到: {output_dir}")
    else:
        print("\n提示: 安装 matplotlib 和 seaborn 以启用可视化功能")
        print("  pip install matplotlib seaborn")

    return result


def create_visualizations(result: Dict, output_dir: str):
    """创建可视化图表"""
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1. Top 20 最常用的 API 柱状图
    top_apis = result["top_apis"][:20]
    if top_apis:
        plt.figure(figsize=(14, 8))
        apis, counts = zip(*top_apis)
        # 简化 API 名称显示
        api_names = [
            api.replace("torch.nn.functional.", "F.")
            .replace("torch.nn.", "nn.")
            .replace("torch.", "")
            for api in apis
        ]
        bars = plt.barh(
            range(len(api_names)),
            counts,
            color=sns.color_palette("husl", len(api_names)),
        )
        plt.yticks(range(len(api_names)), api_names)
        plt.xlabel("Usage Count", fontsize=12)
        plt.title("Top 20 Most Used APIs", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(count + max(counts) * 0.01, i, str(count), va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "top_20_apis.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 2. API 分类饼图
    categories = {
        "torch.*": result["torch_apis"],
        "torch.nn.*": result["nn_apis"],
        "torch.nn.functional.*": result["functional_apis"],
    }
    if result["other_apis"] > 0:
        categories["Other"] = result["other_apis"]

    plt.figure(figsize=(10, 8))
    labels = list(categories.keys())
    sizes = list(categories.values())
    colors = sns.color_palette("Set2", len(labels))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.title("API Category Distribution", fontsize=14, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "api_categories.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Top 20 共现频率最高的 API 对热力图（简化版）
    top_cooccurrences = result["top_cooccurrences"][:20]
    if top_cooccurrences:
        # 提取所有涉及的 API
        all_cooccur_apis = set()
        for (api1, api2), _ in top_cooccurrences:
            all_cooccur_apis.add(api1)
            all_cooccur_apis.add(api2)

        # 创建共现矩阵
        api_list = sorted(all_cooccur_apis)
        cooccur_matrix = [[0] * len(api_list) for _ in range(len(api_list))]
        api_to_idx = {api: idx for idx, api in enumerate(api_list)}

        for (api1, api2), count in top_cooccurrences:
            idx1, idx2 = api_to_idx[api1], api_to_idx[api2]
            cooccur_matrix[idx1][idx2] = count
            cooccur_matrix[idx2][idx1] = count

        # 只显示前15个 API 以保持可读性
        max_apis = min(15, len(api_list))
        api_list_short = api_list[:max_apis]
        matrix_short = [row[:max_apis] for row in cooccur_matrix[:max_apis]]

        plt.figure(figsize=(12, 10))
        api_names_short = [
            api.replace("torch.nn.functional.", "F.")
            .replace("torch.nn.", "nn.")
            .replace("torch.", "")
            for api in api_list_short
        ]
        sns.heatmap(
            matrix_short,
            xticklabels=api_names_short,
            yticklabels=api_names_short,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            cbar_kws={"label": "Co-occurrence Count"},
        )
        plt.title(
            "Top API Co-occurrence Heatmap (Top 15)", fontsize=14, fontweight="bold"
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "api_cooccurrence_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 4. 共现频率最高的 API 对柱状图
    if top_cooccurrences:
        plt.figure(figsize=(14, 8))
        pairs = [
            f"{api1.replace('torch.nn.functional.', 'F.').replace('torch.nn.', 'nn.').replace('torch.', '')}\n<->\n{api2.replace('torch.nn.functional.', 'F.').replace('torch.nn.', 'nn.').replace('torch.', '')}"
            for (api1, api2), _ in top_cooccurrences[:15]
        ]
        counts = [count for _, count in top_cooccurrences[:15]]
        bars = plt.barh(
            range(len(pairs)), counts, color=sns.color_palette("viridis", len(pairs))
        )
        plt.yticks(range(len(pairs)), pairs)
        plt.xlabel("Co-occurrence Count", fontsize=12)
        plt.title(
            "Top 15 Most Frequent API Co-occurrence Pairs",
            fontsize=14,
            fontweight="bold",
        )
        plt.gca().invert_yaxis()
        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(count + max(counts) * 0.01, i, str(count), va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "top_cooccurrence_pairs.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("  ✓ top_20_apis.png - Top 20 最常用的 API")
        print("  ✓ api_categories.png - API 分类分布")
        print("  ✓ api_cooccurrence_heatmap.png - API 共现热力图")
        print("  ✓ top_cooccurrence_pairs.png - Top 15 共现频率最高的 API 对")


if __name__ == "__main__":
    analyze_transformers_apis()
