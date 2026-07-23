"""
消息解析工具：从日志中提取关键信息
"""

import re
from typing import Dict, Any

# ============ 预编译正则表达式 ============

# 编译错误关键字
_ERROR_PATTERN = re.compile(
    r"(error:|FAILED:|RuntimeError:|CalledProcessError:|Exception:)", re.IGNORECASE
)

# 文件路径模式（支持常见扩展名）
_PATH_PATTERN = re.compile(
    r"(/[\w\-./]+/)*[\w\-]+\.(?:cu|cpp|c|cc|cxx|h|hpp|py|so|o|d|txt)"
)

# 性能指标
_SPEEDUP_PATTERN = re.compile(
    r"Torch time:\s*([\d.]+)s,\s*(CUDA|Triton) time:\s*([\d.]+)s,\s*Speedup:\s*([\d.]+)x",
    re.IGNORECASE,
)

# Profiler 输出区块（注意 "Troch" 是模板里的拼写，保持兼容）
_TORCH_PROFILE_PATTERN = re.compile(
    r"#### Benchmark Torch Start\s*(.*?)\s*#### Benchmark Troch End", re.DOTALL
)
_CANDIDATE_PROFILE_PATTERN = re.compile(
    r"#### Benchmark (CUDA|Triton) Start\s*(.*?)\s*#### Benchmark \1 End",
    re.DOTALL,
)


def parse_compile_msg(msg: str, keep_levels: int = 5, max_lines: int = 5) -> str:
    """
    解析编译错误信息，提取关键行并简化路径

    Args:
        msg: 编译日志
        keep_levels: 路径保留的目录层数
        max_lines: 最多返回的行数
    """
    if not msg:
        return "empty log"

    # 提取关键错误行
    key_lines = [
        line.strip() for line in msg.splitlines() if _ERROR_PATTERN.search(line)
    ]

    if not key_lines:
        # 没有找到关键错误行，返回前几行
        return "\n".join(msg.splitlines()[:max_lines])

    # 简化路径
    def shorten_path(match: re.Match) -> str:
        path = match.group(0)
        parts = path.strip("/").split("/")
        if len(parts) > keep_levels:
            parts = parts[-keep_levels:]
        return "/".join(parts)

    cleaned = []
    seen = set()

    for line in key_lines:
        # 简化路径
        line = _PATH_PATTERN.sub(shorten_path, line)
        # 清理引号和转义
        line = line.replace("\\", "").replace("'", "").replace('"', "")
        # 压缩空格
        line = " ".join(line.split())

        if line and line not in seen:
            seen.add(line)
            cleaned.append(line)
            if len(cleaned) >= max_lines:
                break

    return "\n".join(cleaned) if cleaned else msg[:500]


def parse_eval_msg(msg: str) -> Dict[str, Any]:
    """
    从测试日志中解析性能指标

    Returns:
        {
            "torch_time": float,
            "candidate_time": float,
            "speedup": float,
            "backend": str,
            "torch_profile": {"torch": str, "cuda": str} | None
        }
    """
    # 解析 speedup 信息（必须存在）
    speedup_match = _SPEEDUP_PATTERN.search(msg)
    if speedup_match is None:
        # 截断日志，避免错误消息过长
        preview = msg[:300] + "..." if len(msg) > 300 else msg
        raise ValueError(f"Cannot find speedup info. Log preview: {preview}")

    result: Dict[str, Any] = {
        "torch_time": float(speedup_match.group(1)),
        "candidate_time": float(speedup_match.group(3)),
        "speedup": float(speedup_match.group(4)),
        "backend": speedup_match.group(2).lower(),
        "torch_profile": None,
    }

    if result["backend"] == "cuda":
        result["cuda_time"] = result["candidate_time"]
    elif result["backend"] == "triton":
        result["triton_time"] = result["candidate_time"]

    # 解析 profiler 输出（可选，profiler 可能被禁用）
    torch_match = _TORCH_PROFILE_PATTERN.search(msg)
    candidate_match = _CANDIDATE_PROFILE_PATTERN.search(msg)

    if torch_match and candidate_match:
        candidate_backend = candidate_match.group(1).lower()
        result["torch_profile"] = {
            "torch": torch_match.group(1).strip(),
            candidate_backend: candidate_match.group(2).strip(),
        }

    return result


def extract_error_summary(msg: str, max_length: int = 200) -> str:
    """
    从错误消息中提取简短摘要

    适用于 API 响应中需要简短错误描述的场景
    """
    if not msg:
        return "unknown error"

    # 尝试找到最后一个 Error/Exception 行
    lines = msg.strip().splitlines()
    for line in reversed(lines):
        if _ERROR_PATTERN.search(line):
            summary = line.strip()
            if len(summary) > max_length:
                return summary[:max_length] + "..."
            return summary

    # 找不到就返回最后几行
    tail = "\n".join(lines[-3:])
    if len(tail) > max_length:
        return tail[:max_length] + "..."
    return tail
