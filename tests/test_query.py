#!/usr/bin/env python3
"""
测试 operator_query.py 的功能
"""

from torch_pool.operator_query import PyTorchOperatorQuery
import torch.nn as nn
import torch.nn.functional as F


def test_query_description():
    """测试查询 description"""
    print("=" * 80)
    print("测试1: 查询算子 description")
    print("=" * 80)
    
    query = PyTorchOperatorQuery()
    
    # 测试完整路径
    desc = query.get_description("torch.add")
    print(f"torch.add:\n{desc[:200]}...\n" if desc and len(desc) > 200 else f"torch.add:\n{desc}\n")
    
    # 测试仅算子名
    desc = query.get_description("Linear")
    print(f"Linear:\n{desc[:200]}...\n" if desc and len(desc) > 200 else f"Linear:\n{desc}\n")
    
    # 测试 functional
    desc = query.get_description("torch.nn.functional.relu")
    print(f"torch.nn.functional.relu:\n{desc[:200]}...\n" if desc and len(desc) > 200 else f"torch.nn.functional.relu:\n{desc}\n")


def test_analyze_module_class():
    """测试分析 Module 类"""
    print("=" * 80)
    print("测试2: 分析 Module 类")
    print("=" * 80)
    
    query = PyTorchOperatorQuery()
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            x = F.dropout(x, p=0.5)
            x = F.softmax(x, dim=1)
            return x
    
    apis = query.analyze_module(SimpleNet)
    print(f"SimpleNet 使用的 API ({len(apis)} 个):")
    for api in apis:
        print(f"  - {api}")
    print()


def test_analyze_module_instance():
    """测试分析 Module 实例"""
    print("=" * 80)
    print("测试3: 分析 Module 实例")
    print("=" * 80)
    
    query = PyTorchOperatorQuery()
    
    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3)
            self.bn = nn.BatchNorm2d(64)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            return x
    
    model = ConvNet()
    apis = query.analyze_module(model)
    print(f"ConvNet 实例使用的 API ({len(apis)} 个):")
    for api in apis:
        print(f"  - {api}")
    print()


def test_analyze_code_string():
    """测试分析代码字符串"""
    print("=" * 80)
    print("测试4: 分析代码字符串")
    print("=" * 80)
    
    query = PyTorchOperatorQuery()
    
    code = """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(128 * 8 * 8, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = F.log_softmax(x, dim=1)
            return x
    """
    
    apis = query.analyze_module(code)
    print(f"代码中使用的 API ({len(apis)} 个):")
    for api in apis:
        print(f"  - {api}")
    print()


def test_get_full_info():
    """测试获取完整信息"""
    print("=" * 80)
    print("测试5: 获取算子完整信息")
    print("=" * 80)
    
    query = PyTorchOperatorQuery()
    
    info = query.get_full_info("torch.add")
    if info:
        print(f"torch.add 的完整信息:")
        print(f"  签名: {info.get('signature', 'N/A')}")
        print(f"  描述: {info.get('description', 'N/A')[:100]}...")
        print(f"  链接: {info.get('url', 'N/A')}")
    print()


if __name__ == "__main__":
    try:
        test_query_description()
        test_analyze_module_class()
        test_analyze_module_instance()
        test_analyze_code_string()
        test_get_full_info()
        print("=" * 80)
        print("所有测试完成！")
        print("=" * 80)
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

