#!/usr/bin/env python3
"""
测试 api_sampler.py 的功能
"""

from torch_pool.api_sampler import TorchAPISampler


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 80)
    print("测试1: 基本功能测试")
    print("=" * 80)

    sampler = TorchAPISampler()

    # 测试统计信息
    stats = sampler.get_statistics_summary()
    print("\n统计信息:")
    print(f"  总 Module 数: {stats['total_modules']}")
    print(f"  唯一 API 数: {stats['total_unique_apis']}")
    print(f"  共现对数量: {stats['total_cooccurrence_pairs']}")

    # 测试获取共现统计
    seed_api = "torch.nn.Linear"
    if seed_api in sampler.api_frequency:
        cooccurrence = sampler.get_cooccurrence_stats(seed_api)
        print(f"\n{seed_api} 的共现统计:")
        print(f"  共现 API 数量: {len(cooccurrence)}")
        if cooccurrence:
            top_5 = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  Top 5 共现 API:")
            for api, count in top_5:
                print(f"    {api}: {count} 次")
    else:
        print(f"\n警告: {seed_api} 不在统计数据中")
        # 使用一个存在的 API
        if sampler.api_list:
            seed_api = sampler.api_list[0]
            print(f"使用替代 API: {seed_api}")

    print("\n✓ 基本功能测试通过")


def test_sampling_methods():
    """测试不同的采样方法"""
    print("\n" + "=" * 80)
    print("测试2: 采样方法测试")
    print("=" * 80)

    sampler = TorchAPISampler()

    # 找一个存在的 API
    if not sampler.api_list:
        print("错误: 没有可用的 API")
        return

    seed_api = sampler.api_list[0]
    print(f"\n使用种子 API: {seed_api}")

    methods = ["cooccurrence_weighted", "frequency_weighted", "hybrid", "random"]

    for method in methods:
        print(f"\n方法: {method}")
        try:
            combinations = sampler.sample_api_combination(
                seed_api=seed_api,
                num_apis=5,
                num_samples=3,
                method=method,
                temperature=1.0,
            )
            for i, combo in enumerate(combinations, 1):
                print(f"  组合 {i}: {combo}")
        except Exception as e:
            print(f"  错误: {e}")

    print("\n✓ 采样方法测试完成")


def test_top_cooccurring():
    """测试获取共现频率最高的 API"""
    print("\n" + "=" * 80)
    print("测试3: 获取共现频率最高的 API")
    print("=" * 80)

    sampler = TorchAPISampler()

    # 找一个存在的 API
    if not sampler.api_list:
        print("错误: 没有可用的 API")
        return

    seed_api = sampler.api_list[0]
    print(f"\n种子 API: {seed_api}")

    top_apis = sampler.get_top_cooccurring_apis(seed_api, top_k=10)
    print("\nTop 10 共现 API:")
    for api, count in top_apis:
        print(f"  {api}: {count} 次")

    print("\n✓ 共现频率测试完成")


def test_different_temperatures():
    """测试不同温度参数的影响"""
    print("\n" + "=" * 80)
    print("测试4: 不同温度参数的影响")
    print("=" * 80)

    sampler = TorchAPISampler()

    if not sampler.api_list:
        print("错误: 没有可用的 API")
        return

    seed_api = sampler.api_list[0]
    print(f"\n种子 API: {seed_api}")

    temperatures = [0.5, 1.0, 2.0]

    for temp in temperatures:
        print(f"\n温度参数: {temp}")
        combinations = sampler.sample_api_combination(
            seed_api=seed_api,
            num_apis=5,
            num_samples=2,
            method="cooccurrence_weighted",
            temperature=temp,
        )
        for i, combo in enumerate(combinations, 1):
            print(f"  组合 {i}: {combo}")

    print("\n✓ 温度参数测试完成")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_sampling_methods()
        test_top_cooccurring()
        test_different_temperatures()
        print("\n" + "=" * 80)
        print("所有测试完成！")
        print("=" * 80)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback

        traceback.print_exc()
