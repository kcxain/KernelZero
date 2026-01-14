"""
Torch Pool - PyTorch 算子工具集

提供以下功能：
- PyTorch 算子提取和查询
- Transformers Module 分析
- Torch API 采样
"""

# 使用相对导入，避免循环依赖
try:
    from .operator_query import PyTorchOperatorQuery
    from .api_sampler import TorchAPISampler

    __all__ = [
        "PyTorchOperatorQuery",
        "TorchAPISampler",
    ]
except ImportError:
    # 如果相对导入失败（例如直接运行），尝试绝对导入
    import sys
    import os

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)

    from operator_query import PyTorchOperatorQuery
    from api_sampler import TorchAPISampler

    __all__ = [
        "PyTorchOperatorQuery",
        "TorchAPISampler",
    ]

__version__ = "1.0.0"
