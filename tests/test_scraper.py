#!/usr/bin/env python3
"""
快速测试脚本 - 验证爬虫基本功能
"""

import torch

def test_imports():
    """测试必要的导入"""
    try:
        import requests
        from bs4 import BeautifulSoup
        print("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install beautifulsoup4 requests")
        return False

def test_torch_version():
    """测试 PyTorch 版本"""
    print(f"PyTorch 版本: {torch.__version__}")
    if torch.__version__.startswith('2.6'):
        print("✓ PyTorch 2.6 已安装")
        return True
    else:
        print(f"⚠ 警告: 当前版本是 {torch.__version__}，不是 2.6")
        return True  # 仍然可以运行

def test_operator_detection():
    """测试算子检测功能"""
    from torch_pool.scraper import PyTorchOperatorScraper
    
    scraper = PyTorchOperatorScraper()
    
    # 测试获取算子
    torch_ops = scraper.get_torch_operators()
    nn_ops = scraper.get_nn_operators()
    functional_ops = scraper.get_functional_operators()
    
    print(f"\n检测到的算子数量:")
    print(f"  torch: {len(torch_ops)} 个")
    print(f"  torch.nn: {len(nn_ops)} 个")
    print(f"  torch.nn.functional: {len(functional_ops)} 个")
    
    # 显示一些示例
    print(f"\n示例算子:")
    print(f"  torch: {sorted(list(torch_ops))[:5]}")
    print(f"  torch.nn: {sorted(list(nn_ops))[:5]}")
    print(f"  torch.nn.functional: {sorted(list(functional_ops))[:5]}")
    
    return len(torch_ops) > 0 and len(nn_ops) > 0 and len(functional_ops) > 0

def main():
    """主测试函数"""
    print("=" * 60)
    print("PyTorch 算子爬虫测试")
    print("=" * 60)
    
    all_passed = True
    
    print("\n1. 测试依赖...")
    if not test_imports():
        all_passed = False
        sys.exit(1)
    
    print("\n2. 测试 PyTorch 版本...")
    test_torch_version()
    
    print("\n3. 测试算子检测...")
    if not test_operator_detection():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！可以运行 scraper.py")
    else:
        print("✗ 部分测试失败")
    print("=" * 60)

if __name__ == "__main__":
    main()

