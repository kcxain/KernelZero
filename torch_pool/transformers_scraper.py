#!/usr/bin/env python3
"""
Transformers 库 Module 爬取和分析工具
爬取 transformers 库中所有的 Torch Module，并分析使用的 Torch API
"""

import inspect
import json
import os
import sys
from typing import Dict, List, Optional, Set
import torch.nn as nn

# 导入 PyTorchOperatorQuery
from operator_query import PyTorchOperatorQuery

# 尝试导入 transformers
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers 库未安装，请运行: pip install transformers")


class TransformersModuleScraper:
    """Transformers 库 Module 爬取和分析类"""

    def __init__(self):
        """初始化爬取器"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers 库未安装")
        
        self.query = PyTorchOperatorQuery()
        self.modules: Dict[str, Dict] = {}
        
    def get_all_transformers_modules(self) -> Set[type]:
        """获取 transformers 库中所有的 nn.Module 子类"""
        modules = set()
        visited = set()  # 避免重复
        
        def check_and_add(obj, name_path=""):
            """递归检查并添加 Module"""
            if obj in visited:
                return
            visited.add(obj)
            
            # 检查是否是 nn.Module 的子类
            try:
                if (inspect.isclass(obj) and 
                    issubclass(obj, nn.Module) and 
                    obj != nn.Module and
                    obj.__module__.startswith("transformers")):
                    modules.add(obj)
            except (TypeError, AttributeError):
                pass
        
        # 1. 遍历 transformers 顶层模块
        for name in dir(transformers):
            if name.startswith("_"):
                continue
            try:
                obj = getattr(transformers, name)
                check_and_add(obj, name)
            except (TypeError, AttributeError):
                continue
        
        # 2. 遍历 transformers.models 下的所有模型
        if hasattr(transformers, "models"):
            models_module = transformers.models
            for model_name in dir(models_module):
                if model_name.startswith("_") or model_name in ["utils", "auto"]:
                    continue
                
                try:
                    model_module = getattr(models_module, model_name)
                    if not inspect.ismodule(model_module):
                        continue
                    
                    # 遍历模型模块下的所有类
                    for class_name in dir(model_module):
                        if class_name.startswith("_"):
                            continue
                        
                        try:
                            cls = getattr(model_module, class_name)
                            check_and_add(cls, f"{model_name}.{class_name}")
                        except (TypeError, AttributeError):
                            continue
                except (TypeError, AttributeError):
                    continue
        
        # 3. 遍历 transformers.modeling_* 模块
        for module_name in dir(transformers):
            if module_name.startswith("modeling_") or module_name.startswith("configuration_"):
                try:
                    module_obj = getattr(transformers, module_name)
                    if inspect.ismodule(module_obj):
                        for attr_name in dir(module_obj):
                            if attr_name.startswith("_"):
                                continue
                            try:
                                attr_obj = getattr(module_obj, attr_name)
                                check_and_add(attr_obj, f"{module_name}.{attr_name}")
                            except (TypeError, AttributeError):
                                continue
                except (TypeError, AttributeError):
                    continue
        
        return modules
    
    def get_module_info(self, module_class: type) -> Dict:
        """获取 Module 的信息"""
        module_name = f"{module_class.__module__}.{module_class.__name__}"
        
        # 获取源代码
        try:
            source = inspect.getsource(module_class)
        except (OSError, TypeError):
            source = ""
        
        # 分析使用的 API
        try:
            used_apis = self.query.analyze_module(module_class)
        except Exception as e:
            print(f"警告: 分析 {module_name} 时出错: {e}")
            used_apis = []
        
        # 获取 docstring
        docstring = inspect.getdoc(module_class) or ""
        
        return {
            "name": module_class.__name__,
            "full_name": module_name,
            "module": module_class.__module__,
            "docstring": docstring[:500] if len(docstring) > 500 else docstring,  # 限制长度
            "used_apis": sorted(used_apis),
            "api_count": len(used_apis),
        }
    
    def scrape_all(self, save_path: Optional[str] = None):
        """爬取所有 Module 并保存"""
        if save_path is None:
            # 默认保存路径
            # torch_pool/ -> 项目根目录 -> resources/transformers_modules/
            project_root = os.path.dirname(os.path.dirname(__file__))
            resources_dir = os.path.join(project_root, "resources", "transformers_modules")
            os.makedirs(resources_dir, exist_ok=True)
            save_path = os.path.join(resources_dir, "transformers_modules.json")
        
        print("=" * 80)
        print("Transformers 库 Module 爬取和分析")
        print("=" * 80)
        print(f"Transformers 版本: {transformers.__version__}\n")
        
        # 获取所有 Module
        print("1. 获取所有 transformers Module...")
        all_modules = self.get_all_transformers_modules()
        print(f"   找到 {len(all_modules)} 个 Module\n")
        
        if len(all_modules) == 0:
            print("警告: 未找到任何 Module，尝试使用备用方法...")
            all_modules = self._get_modules_fallback()
            print(f"   使用备用方法找到 {len(all_modules)} 个 Module\n")
        
        # 分析每个 Module
        print("2. 分析每个 Module 使用的 Torch API...")
        total = len(all_modules)
        current = 0
        
        for module_class in sorted(all_modules, key=lambda x: x.__name__):
            current += 1
            module_name = f"{module_class.__module__}.{module_class.__name__}"
            print(f"   [{current}/{total}] 分析 {module_name}...", end="\r", flush=True)
            
            try:
                info = self.get_module_info(module_class)
                self.modules[module_name] = info
            except Exception as e:
                print(f"\n   警告: 分析 {module_name} 失败: {e}")
                self.modules[module_name] = {
                    "name": module_class.__name__,
                    "full_name": module_name,
                    "module": module_class.__module__,
                    "docstring": "",
                    "used_apis": [],
                    "api_count": 0,
                    "error": str(e),
                }
        
        print(f"\n   完成！共分析 {total} 个 Module")
        
        # 保存结果
        self.save_results(save_path)
        print(f"\n结果已保存到: {save_path}")
        
        # 打印统计信息
        self.print_statistics()
    
    def _get_modules_fallback(self) -> Set[type]:
        """备用方法：通过遍历所有 transformers 子模块获取模型"""
        modules = set()
        
        # 尝试从 transformers.modeling_utils 获取
        try:
            from transformers import modeling_utils
            for name in dir(modeling_utils):
                if name.startswith("_"):
                    continue
                try:
                    obj = getattr(modeling_utils, name)
                    if (inspect.isclass(obj) and 
                        issubclass(obj, nn.Module) and 
                        obj != nn.Module):
                        modules.add(obj)
                except (TypeError, AttributeError):
                    continue
        except ImportError:
            pass
        
        # 尝试从 transformers.configuration_utils 获取
        try:
            from transformers import configuration_utils
            for name in dir(configuration_utils):
                if name.startswith("_"):
                    continue
                try:
                    obj = getattr(configuration_utils, name)
                    if (inspect.isclass(obj) and 
                        issubclass(obj, nn.Module) and 
                        obj != nn.Module):
                        modules.add(obj)
                except (TypeError, AttributeError):
                    continue
        except ImportError:
            pass
        
        return modules
    
    def save_results(self, save_path: str):
        """保存结果到文件"""
        # 保存 JSON 格式
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.modules, f, indent=2, ensure_ascii=False)
        
        # 同时保存人类可读的文本格式
        txt_path = save_path.replace(".json", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Transformers 库 Module 分析结果\n")
            f.write("=" * 80 + "\n\n")
            
            # 按 API 数量排序
            sorted_modules = sorted(
                self.modules.items(),
                key=lambda x: x[1].get("api_count", 0),
                reverse=True
            )
            
            for module_name, info in sorted_modules:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Module: {info.get('name', 'N/A')}\n")
                f.write(f"完整路径: {module_name}\n")
                f.write(f"模块: {info.get('module', 'N/A')}\n")
                f.write(f"使用的 API 数量: {info.get('api_count', 0)}\n")
                f.write(f"Docstring: {info.get('docstring', 'N/A')[:200]}...\n" if len(info.get('docstring', '')) > 200 else f"Docstring: {info.get('docstring', 'N/A')}\n")
                f.write(f"使用的 API:\n")
                for api in info.get("used_apis", []):
                    f.write(f"  - {api}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"结果已保存到: {txt_path}")
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 80)
        print("统计信息:")
        print("=" * 80)
        
        total_modules = len(self.modules)
        total_apis = sum(info.get("api_count", 0) for info in self.modules.values())
        modules_with_apis = sum(1 for info in self.modules.values() if info.get("api_count", 0) > 0)
        
        print(f"总 Module 数: {total_modules}")
        print(f"使用 API 的 Module 数: {modules_with_apis}")
        print(f"总 API 调用数: {total_apis}")
        print(f"平均每个 Module 使用的 API 数: {total_apis / total_modules if total_modules > 0 else 0:.2f}")
        
        # 统计最常用的 API
        api_usage = {}
        for info in self.modules.values():
            for api in info.get("used_apis", []):
                api_usage[api] = api_usage.get(api, 0) + 1
        
        if api_usage:
            print(f"\n最常用的 Top 10 API:")
            sorted_apis = sorted(api_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            for api, count in sorted_apis:
                print(f"  {api}: {count} 次")


def main():
    """主函数"""
    if not TRANSFORMERS_AVAILABLE:
        print("错误: transformers 库未安装")
        print("请运行: pip install transformers")
        sys.exit(1)
    
    scraper = TransformersModuleScraper()
    scraper.scrape_all()


if __name__ == "__main__":
    main()

