# PyTorch 算子工具集

这个目录包含两个主要工具：

## 1. 算子提取工具 (scraper.py)

用于提取 PyTorch 2.6 的所有算子信息，包括：
- `torch` 模块的算子
- `torch.nn` 模块的算子
- `torch.nn.functional` 模块的算子

================================================================================
统计信息:
================================================================================
torch: 507 个算子
torch.nn: 166 个算子
torch.nn.functional: 139 个算子

### 使用方法

在 `torch_pool/` 目录下运行：
```bash
cd torch_pool
python scraper.py
```

或者使用便捷脚本（从项目根目录）：
```bash
./scripts/run_scraper.sh
```

脚本会：
1. 自动检测 PyTorch 2.6 中所有可用的算子
2. 提取每个算子的 docstring 和函数签名
3. 将结果保存为 JSON 和文本两种格式

### 输出文件

输出文件保存在 `resources/torch_operators/` 目录下：
- `torch_operators.json`: JSON 格式的算子信息（包含签名、docstring 和文档链接）
- `torch_operators.txt`: 人类可读的文本格式

### 目录结构

```
resources/
  torch_operators/
    torch_operators.json      # 算子数据（JSON 格式）
    torch_operators.txt       # 算子数据（文本格式）
  KernelBench/
    ...
```

## 2. 算子查询和分析工具 (operator_query.py)

用于快速查询算子信息和分析 Module 使用的 API。

### 功能

1. **查询算子 description**：根据算子名快速查找对应的 description
2. **分析 Module API**：分析 Module 类、实例或代码字符串，返回使用的 PyTorch API 列表

### 使用方法

**推荐方式（从项目根目录运行）：**
```python
from torch_pool.operator_query import PyTorchOperatorQuery

# 创建查询器
query = PyTorchOperatorQuery()

# 1. 查询算子 description
description = query.get_description("torch.add")
print(description)

# 2. 获取完整信息
info = query.get_full_info("torch.nn.Linear")
print(info)  # 包含 signature, description, url

# 3. 分析 Module 类
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
    
    def forward(self, x):
        return F.relu(self.linear(x))

apis = query.analyze_module(MyNet)
print(apis)  # ['torch.nn.Linear', 'torch.nn.functional.relu']

# 4. 分析 Module 实例
model = MyNet()
apis = query.analyze_module(model)

# 5. 分析代码字符串
code = """
class MyModel(nn.Module):
    def forward(self, x):
        return F.relu(nn.Linear(10, 20)(x))
"""
apis = query.analyze_module(code)
```

## 3. Transformers Module 爬取工具 (transformers_scraper.py)

用于爬取 transformers 库中所有的 Torch Module，并分析每个 Module 使用的 Torch API。

### 功能

1. **自动发现 Module**：自动扫描 transformers 库中的所有 nn.Module 子类
2. **API 分析**：使用 PyTorchOperatorQuery 分析每个 Module 使用的 Torch API
3. **数据保存**：将结果保存为 JSON 和文本两种格式

### 依赖

需要安装 transformers 库：
```bash
pip install transformers
```

### 使用方法

在 `torch_pool/` 目录下运行：
```bash
cd torch_pool
python transformers_scraper.py
```

或者使用便捷脚本（从项目根目录）：
```bash
./scripts/run_transformers_scraper.sh
```

### 输出文件

输出文件保存在 `resources/transformers_modules/` 目录下：
- `transformers_modules.json`: JSON 格式的 Module 分析结果
- `transformers_modules.txt`: 人类可读的文本格式

### 输出格式

每个 Module 包含以下信息：
- `name`: Module 名称
- `full_name`: 完整路径（如 `transformers.models.bert.BertModel`）
- `module`: 所属模块
- `docstring`: Module 的文档字符串
- `used_apis`: 使用的 Torch API 列表
- `api_count`: 使用的 API 数量


================================================================================
统计信息:
================================================================================
总 Module 数: 4791
使用 API 的 Module 数: 1913
总 API 调用数: 5002
平均每个 Module 使用的 API 数: 1.04

最常用的 Top 10 API:
  torch.cat: 606 次
  torch.arange: 541 次
  torch.zeros: 460 次
  torch.ones: 359 次
  torch.tensor: 207 次
  torch.matmul: 181 次
  torch.no_grad: 170 次
  torch.full: 157 次
  torch.stack: 155 次
  torch.where: 98 次

## 4. Torch API 采样工具 (api_sampler.py)

基于 transformers Module 中 API 共现统计信息，实现从整个 Torch API 中采样 API 组合。

### 功能

1. **共现统计**：分析 transformers Module 中 API 的共现频率
2. **智能采样**：根据种子 API 和共现统计信息，生成合理的 API 组合
3. **多种采样方法**：支持多种采样策略

### 使用方法

**推荐方式（从项目根目录运行）：**
```python
from torch_pool.api_sampler import TorchAPISampler

# 创建采样器
sampler = TorchAPISampler()

# 1. 获取统计信息
stats = sampler.get_statistics_summary()
print(f"总 API 数: {stats['total_unique_apis']}")

# 2. 获取与某个 API 共现频率最高的 API
top_apis = sampler.get_top_cooccurring_apis("torch.nn.Linear", top_k=10)
for api, count in top_apis:
    print(f"{api}: {count} 次")

# 3. 基于种子 API 采样组合
combinations = sampler.sample_api_combination(
    seed_api="torch.nn.Linear",
    num_apis=5,              # 每个组合包含 5 个 API
    num_samples=10,          # 生成 10 个组合
    method="cooccurrence_weighted",  # 采样方法
    temperature=1.0          # 温度参数（控制随机性）
)

for i, combo in enumerate(combinations, 1):
    print(f"组合 {i}: {combo}")
```

### 采样方法

- **`cooccurrence_weighted`**: 基于共现频率加权采样（推荐）
- **`frequency_weighted`**: 基于 API 使用频率加权采样
- **`hybrid`**: 混合方法（结合共现频率和 API 频率）
- **`random`**: 随机采样

### 参数说明

- `seed_api`: 种子 API（输入的 API）
- `num_apis`: 每个组合包含的 API 数量
- `num_samples`: 生成多少个组合
- `method`: 采样方法
- `temperature`: 温度参数（越大越随机，越小越确定）
- `min_cooccurrence`: 最小共现次数阈值

### 目录结构

```
KernelZero/
  torch_pool/                  # 工具脚本目录
    scraper.py                 # PyTorch 算子提取工具
    operator_query.py          # 算子查询和分析工具
    transformers_scraper.py    # Transformers Module 爬取工具
    api_sampler.py             # Torch API 采样工具
    README.md                  # 本文件
  scripts/                     # 运行脚本目录
    run_scraper.sh            # 运行算子提取脚本
    run_transformers_scraper.sh  # 运行 Transformers 爬取脚本
  tests/                       # 测试脚本目录
    test_scraper.py            # 测试算子提取工具
    test_query.py              # 测试查询工具
    test_api_sampler.py        # 测试 API 采样工具
  resources/                   # 数据资源目录
    torch_operators/
      torch_operators.json     # PyTorch 算子数据
      torch_operators.txt
    transformers_modules/
      transformers_modules.json  # Transformers Module 分析结果
      transformers_modules.txt
    KernelBench/
      ...
```