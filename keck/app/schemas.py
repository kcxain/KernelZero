from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CompileResult:
    """编译结果"""

    success: bool
    ext_filename: Optional[str] = None
    ext_content: Optional[bytes] = None
    model_new_patch: Optional[str] = None
    message: str = ""


@dataclass
class EvalResult:
    """执行/验证结果"""

    success: bool
    message: Any = ""  # 可以是 str 或 dict


@dataclass
class ValidationResponse:
    """API 返回的验证结果"""

    formated: bool
    compiled: bool
    passed: bool
    msg: Any  # str | dict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formated": self.formated,
            "compiled": self.compiled,
            "passed": self.passed,
            "msg": self.msg,
        }


def format_error(msg: str) -> ValidationResponse:
    """格式错误（如缺少代码）"""
    return ValidationResponse(formated=False, compiled=False, passed=False, msg=msg)


def compile_error(msg: str) -> ValidationResponse:
    """编译错误"""
    return ValidationResponse(formated=True, compiled=False, passed=False, msg=msg)


def eval_error(msg: str) -> ValidationResponse:
    """执行/验证错误"""
    return ValidationResponse(formated=True, compiled=True, passed=False, msg=msg)


def success(msg: Any) -> ValidationResponse:
    """验证通过"""
    return ValidationResponse(formated=True, compiled=True, passed=True, msg=msg)
