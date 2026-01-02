#!/usr/bin/env python3
"""
PyTorch 算子查询和分析工具
- 根据算子名快速查找 description
- 分析 Module 使用的 API
"""

import ast
import inspect
import json
import os
import re
from typing import Dict, List, Optional, Set, Union

# 可选导入 torch（分析代码字符串时不需要）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


class PyTorchOperatorQuery:
    """PyTorch 算子查询和分析类"""

    def __init__(self, operators_file: Optional[str] = None):
        """
        初始化查询器

        Args:
            operators_file: 算子 JSON 文件路径，默认为 resources/torch_operators/torch_operators.json
        """
        if operators_file is None:
            # 默认路径：项目根目录/resources/torch_operators/torch_operators.json
            # torch_pool/ -> 项目根目录 -> resources/torch_operators/
            project_root = os.path.dirname(os.path.dirname(__file__))
            operators_file = os.path.join(
                project_root, "resources", "torch_operators", "torch_operators.json"
            )

        self.operators_file = operators_file
        self.operators: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._load_operators()

        # 构建快速查找索引
        self._operator_index: Dict[str, tuple] = {}  # {operator_name: (module, info)}
        self._build_index()

    def _load_operators(self):
        """加载算子数据"""
        if not os.path.exists(self.operators_file):
            raise FileNotFoundError(
                f"算子文件不存在: {self.operators_file}\n"
                "请先运行 scraper.py 生成算子数据"
            )

        with open(self.operators_file, "r", encoding="utf-8") as f:
            self.operators = json.load(f)

    def _build_index(self):
        """构建算子索引，支持多种查找方式"""
        for module, ops in self.operators.items():
            for op_name, op_info in ops.items():
                # 完整路径索引: torch.add, torch.nn.Linear
                full_name = f"{module}.{op_name}"
                self._operator_index[full_name] = (module, op_info)

                # 仅算子名索引（如果冲突，保留第一个）
                if op_name not in self._operator_index:
                    self._operator_index[op_name] = (module, op_info)

    def get_description(self, operator_name: str) -> Optional[str]:
        """
        根据算子名查找 description

        Args:
            operator_name: 算子名，可以是：
                - 完整路径: "torch.add", "torch.nn.Linear", "torch.nn.functional.relu"
                - 仅算子名: "add", "Linear", "relu"（如果有冲突，返回第一个匹配的）

        Returns:
            description 字符串，如果未找到返回 None
        """
        if operator_name in self._operator_index:
            _, op_info = self._operator_index[operator_name]
            return op_info.get("description")

        # 尝试直接查找
        for module, ops in self.operators.items():
            if operator_name in ops:
                return ops[operator_name].get("description")

        return None

    def get_full_info(self, operator_name: str) -> Optional[Dict[str, str]]:
        """
        获取算子的完整信息（包括 signature, description, url）

        Args:
            operator_name: 算子名

        Returns:
            包含 signature, description, url 的字典，如果未找到返回 None
        """
        if operator_name in self._operator_index:
            _, op_info = self._operator_index[operator_name]
            return op_info

        # 尝试直接查找
        for module, ops in self.operators.items():
            if operator_name in ops:
                return ops[operator_name]

        return None

    def analyze_module(
        self,
        module: Union[type, object, str],
        include_torch: bool = True,
        include_nn: bool = True,
        include_functional: bool = True,
    ) -> List[str]:
        """
        分析 Module 使用的 PyTorch API

        Args:
            module: 可以是：
                - Module 类（如 nn.Linear）
                - Module 实例（如 nn.Linear(10, 20)）
                - Python 代码字符串
            include_torch: 是否包含 torch 模块的 API
            include_nn: 是否包含 torch.nn 模块的 API
            include_functional: 是否包含 torch.nn.functional 模块的 API

        Returns:
            使用的 API 列表，格式为 ["torch.add", "torch.nn.Linear", ...]
        """
        # 获取源代码
        source_code = self._get_source_code(module)

        # 解析代码，提取使用的 API
        used_apis = self._extract_apis_from_code(
            source_code, include_torch, include_nn, include_functional
        )

        # 规范化 API 名称：将 nn.xxx 转换为 torch.nn.xxx，F.xxx 转换为 torch.nn.functional.xxx
        normalized_apis = []
        for api in used_apis:
            if api.startswith("nn."):
                # nn.Linear -> torch.nn.Linear
                normalized_api = f"torch.{api}"
            elif api.startswith("F."):
                # F.relu -> torch.nn.functional.relu
                normalized_api = f"torch.nn.functional.{api[2:]}"
            elif api.startswith("torch."):
                # 已经是完整路径
                normalized_api = api
            else:
                # 其他情况，保持原样
                normalized_api = api
            normalized_apis.append(normalized_api)

        # 过滤：只返回在算子数据中存在的 API
        valid_apis = []
        for api in normalized_apis:
            if self.get_description(api) is not None:
                valid_apis.append(api)

        return sorted(set(valid_apis))

    def _get_source_code(self, module: Union[type, object, str]) -> str:
        """获取模块的源代码（包括所有方法）"""
        if isinstance(module, str):
            # 已经是代码字符串
            return module

        # 获取类或对象的源代码
        module_class = module if isinstance(module, type) else module.__class__
        source_parts = []

        # 1. 尝试获取整个类的源代码
        try:
            full_source = inspect.getsource(module_class)
            source_parts.append(full_source)
        except (OSError, TypeError):
            # 如果获取整个类失败，尝试获取各个方法的源代码
            # 2. 获取 __init__ 方法
            if hasattr(module_class, "__init__"):
                try:
                    init_source = inspect.getsource(module_class.__init__)
                    source_parts.append(init_source)
                except (OSError, TypeError):
                    pass

            # 3. 获取 forward 方法（对于 nn.Module 很重要）
            if hasattr(module_class, "forward"):
                try:
                    forward_source = inspect.getsource(module_class.forward)
                    source_parts.append(forward_source)
                except (OSError, TypeError):
                    pass

            # 4. 获取其他公共方法
            for name, method in inspect.getmembers(module_class, inspect.isfunction):
                if not name.startswith("_") and name not in ["__init__", "forward"]:
                    try:
                        method_source = inspect.getsource(method)
                        source_parts.append(method_source)
                    except (OSError, TypeError):
                        pass

        return "\n".join(source_parts)

    def _extract_apis_from_code(
        self,
        source_code: str,
        include_torch: bool,
        include_nn: bool,
        include_functional: bool,
    ) -> Set[str]:
        """从源代码中提取使用的 PyTorch API"""
        if not source_code:
            return set()

        used_apis = set()

        try:
            # 使用 AST 解析代码
            tree = ast.parse(source_code)
            visitor = PyTorchAPIVisitor(include_torch, include_nn, include_functional)
            visitor.visit(tree)
            used_apis.update(visitor.apis)
        except SyntaxError:
            # 如果 AST 解析失败，使用正则表达式作为备选
            used_apis.update(
                self._extract_apis_with_regex(
                    source_code, include_torch, include_nn, include_functional
                )
            )

        return used_apis

    def _extract_apis_with_regex(
        self,
        source_code: str,
        include_torch: bool,
        include_nn: bool,
        include_functional: bool,
    ) -> Set[str]:
        """使用正则表达式提取 API（备选方案）"""
        apis = set()

        # 匹配 torch.xxx, nn.xxx, F.xxx 等模式
        patterns = []
        if include_torch:
            patterns.append(r"\btorch\.(\w+)\b")
        if include_nn:
            patterns.append(r"\bnn\.(\w+)\b")
            patterns.append(r"\btorch\.nn\.(\w+)\b")
        if include_functional:
            patterns.append(r"\bF\.(\w+)\b")
            patterns.append(r"\btorch\.nn\.functional\.(\w+)\b")

        for pattern in patterns:
            matches = re.findall(pattern, source_code)
            for match in matches:
                # 确定模块前缀
                if pattern.startswith(r"\btorch\.") and "nn" not in pattern:
                    apis.add(f"torch.{match}")
                elif "nn\." in pattern or "torch\.nn\." in pattern:
                    if "functional" in pattern:
                        apis.add(f"torch.nn.functional.{match}")
                    else:
                        apis.add(f"torch.nn.{match}")
                elif pattern.startswith(r"\bF\."):
                    apis.add(f"torch.nn.functional.{match}")

        return apis


class PyTorchAPIVisitor(ast.NodeVisitor):
    """AST 访问器，用于提取 PyTorch API 调用"""

    def __init__(self, include_torch: bool, include_nn: bool, include_functional: bool):
        self.include_torch = include_torch
        self.include_nn = include_nn
        self.include_functional = include_functional
        self.apis: Set[str] = set()

    def visit_Attribute(self, node: ast.Attribute):
        """访问属性节点，提取 torch.xxx, nn.xxx, F.xxx 等"""
        # 递归获取完整的属性链
        full_name = self._get_full_attribute_name(node)

        if full_name:
            # 检查是否是 PyTorch API
            if self.include_torch and full_name.startswith("torch."):
                # 排除 torch.nn 和 torch.nn.functional（它们会被单独处理）
                if not full_name.startswith("torch.nn"):
                    self.apis.add(full_name)
            elif self.include_nn and (
                full_name.startswith("nn.") or full_name.startswith("torch.nn.")
            ):
                if not full_name.startswith("torch.nn.functional"):
                    # 规范化：nn.xxx -> torch.nn.xxx
                    if full_name.startswith("nn."):
                        normalized = f"torch.{full_name}"
                    else:
                        normalized = full_name
                    self.apis.add(normalized)
            elif self.include_functional and (
                full_name.startswith("F.")
                or full_name.startswith("torch.nn.functional.")
            ):
                # 规范化：F.xxx -> torch.nn.functional.xxx
                if full_name.startswith("F."):
                    normalized = f"torch.nn.functional.{full_name[2:]}"
                else:
                    normalized = full_name
                self.apis.add(normalized)

        self.generic_visit(node)

    def _get_full_attribute_name(self, node: ast.Attribute) -> Optional[str]:
        """获取完整的属性名，如 torch.add, nn.Linear"""
        parts = []
        current = node

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        # 处理基础名称（可能是 Name 或 Constant）
        if isinstance(current, ast.Name):
            parts.append(current.id)
        elif isinstance(current, ast.Constant) and isinstance(current.value, str):
            parts.append(current.value)
        else:
            return None

        # 反转列表，因为是从右到左构建的
        parts.reverse()
        return ".".join(parts)

    def visit_Call(self, node: ast.Call):
        """访问函数调用节点"""
        # 如果调用的是属性（如 torch.add()），visit_Attribute 会处理
        # 这里主要处理直接调用的情况
        self.generic_visit(node)


def main():
    """示例用法"""
    # 创建查询器
    query = PyTorchOperatorQuery()

    # 示例1: 查找算子 description
    print("=" * 80)
    print("示例1: 查找算子 description")
    print("=" * 80)

    desc = query.get_description("torch.add")
    print(f"torch.add 的 description:\n{desc}\n")

    desc = query.get_description("Linear")
    print(f"Linear 的 description:\n{desc}\n")

    # 示例2: 分析 Module
    print("=" * 80)
    print("示例2: 分析 Module 使用的 API")
    print("=" * 80)

    # 示例：分析一个简单的 Module
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            x = F.dropout(x, p=0.5)
            return x

    apis = query.analyze_module(SimpleNet)
    print(f"SimpleNet 使用的 API: {apis}\n")

    # 示例3: 分析代码字符串
    print("=" * 80)
    print("示例3: 分析代码字符串")
    print("=" * 80)

    code = """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MyModel(nn.Module):
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
    """

    apis = query.analyze_module(code)
    print(f"代码中使用的 API: {apis}\n")


if __name__ == "__main__":
    main()
