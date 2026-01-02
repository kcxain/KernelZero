# 测试说明

## 运行测试的方式

### 方式 1：使用 pytest（推荐）

从项目根目录运行：
```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_api_sampler.py

# 运行特定测试函数
pytest tests/test_api_sampler.py::test_basic_functionality
```

`conftest.py` 会自动配置 Python 路径，无需手动设置。

### 方式 2：使用 python -m 方式运行

从项目根目录运行：
```bash
# 运行测试文件
python -m tests.test_api_sampler
python -m tests.test_query
python -m tests.test_scraper
```

这种方式会自动将项目根目录添加到 Python 路径中。

### 方式 3：安装项目为可编辑包（最优雅）

从项目根目录运行：
```bash
# 安装项目为可编辑包
pip install -e .

# 之后就可以在任何地方直接导入
python tests/test_api_sampler.py
```

安装后，`torch_pool` 模块可以在任何地方直接导入，无需路径设置。

## 为什么不需要 sys.path.insert？

1. **项目根目录有 `__init__.py`**：使项目根目录成为一个 Python 包
2. **`conftest.py`**：pytest 会自动加载并配置路径
3. **`python -m` 方式**：Python 会自动处理包导入
4. **可编辑安装**：最优雅的方式，一次配置，到处使用

