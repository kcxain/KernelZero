#!/usr/bin/env python3
"""
测试 API 提取功能，检查是否能正确提取 nn.xxx 和 F.xxx
"""

from torch_pool.operator_query import PyTorchOperatorQuery

# 测试代码
test_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(3, 64, 3)
    
    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = torch.cat([x, x], dim=1)
        return x
"""

print("=" * 80)
print("测试 API 提取功能")
print("=" * 80)

query = PyTorchOperatorQuery()

print("\n1. 测试代码字符串分析:")
apis = query.analyze_module(test_code)
print(f"提取到的 API ({len(apis)} 个):")
for api in sorted(apis):
    print(f"  {api}")

print("\n2. 检查是否能找到这些 API 的描述:")
test_apis = ["torch.nn.Linear", "torch.nn.ReLU", "torch.nn.functional.relu", "torch.cat"]
for api in test_apis:
    desc = query.get_description(api)
    if desc:
        print(f"  ✓ {api}: 找到描述 ({len(desc)} 字符)")
    else:
        print(f"  ✗ {api}: 未找到描述")

