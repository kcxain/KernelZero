"""
AST 相关工具：重写 load_inline 调用
"""
import ast
from pathlib import Path


def rewrite_load_inline(src_path: str, dst_path: str) -> None:
    """
    把 `op = load_inline(...)` 重写为 `import ext_name as op`

    用途：CUDA 编译阶段和执行阶段解耦
    - 编译阶段（CPU）先执行包含 load_inline 的 model_new.py，完成扩展编译
    - 然后用本函数把 load_inline 语句重写为普通 import，写入 model_new_patch.py
    - 执行阶段只需要 `from model_new_patch import ModelNew`，不会再触发编译
    """
    source = Path(src_path).read_text()
    tree = ast.parse(source)

    for i, node in enumerate(tree.body):
        if not _is_load_inline_assign(node):
            continue

        call = node.value  # type: ignore
        ext_alias = node.targets[0].id  # type: ignore
        ext_name = _extract_extension_name(call)

        # 重写为 `import ext_name as alias`
        tree.body[i] = ast.parse(f"import {ext_name} as {ext_alias}").body[0]

    result = ast.unparse(tree)
    Path(dst_path).write_text(result)


def _is_load_inline_assign(node: ast.AST) -> bool:
    """判断是否是 `xxx = load_inline(...)` 形式"""
    if not isinstance(node, ast.Assign):
        return False
    if not isinstance(node.value, ast.Call):
        return False
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return False

    call = node.value
    # load_inline(...) 或 xxx.load_inline(...)
    if isinstance(call.func, ast.Name) and call.func.id == "load_inline":
        return True
    if isinstance(call.func, ast.Attribute) and call.func.attr == "load_inline":
        return True
    return False


def _extract_extension_name(call: ast.Call) -> str:
    """从 load_inline 调用中提取 name 参数"""
    for kw in call.keywords:
        if kw.arg == "name":
            if isinstance(kw.value, ast.Constant):
                return kw.value.value
    raise RuntimeError("Cannot find 'name' keyword in load_inline call")
