import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Tuple, Any, Dict

try:
    import ray
except ImportError:  # pragma: no cover - optional in local/offline eval mode
    ray = None

from app.config import settings
from app.utils import parse_eval_msg


def _get_subprocess_env() -> Dict[str, str]:
    """
    获取子进程的环境变量，确保 CUDA_VISIBLE_DEVICES 正确设置

    Ray 分配的 GPU 通过 ray.get_gpu_ids() 获取，
    需要显式传递给子进程以确保 GPU 隔离
    """
    env = os.environ.copy()

    # 获取 Ray 分配给当前任务的 GPU，并始终显式设置 CUDA_VISIBLE_DEVICES：
    # - gpu_ids 非空：限定为 Ray 分配的 GPU，确保一卡一程序
    # - gpu_ids 为空（Ray 调度异常）：设为 ""，禁止访问任何 GPU，快速失败
    #   而不是静默地让子进程看到节点上所有 GPU
    try:
        if ray is not None and ray.is_initialized():
            gpu_ids = ray.get_gpu_ids()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    except Exception:
        # 不在 Ray 环境中（如本地测试），保持原有设置
        pass

    return env


def _load_templates() -> tuple[str, str, str]:
    """加载模板文件，返回 (cuda_template, torch_template, triton_template)"""
    template_dir = settings.templates_dir
    cuda_template = Path(f"{template_dir}/test_cuda.py").read_text(encoding="utf-8")
    torch_template = Path(f"{template_dir}/test_torch.py").read_text(encoding="utf-8")
    triton_template = Path(f"{template_dir}/test_triton.py").read_text(
        encoding="utf-8"
    )
    return cuda_template, torch_template, triton_template


# 模块加载时立即读取模板，避免多线程竞争和重复 I/O
_CUDA_TEMPLATE, _TORCH_TEMPLATE, _TRITON_TEMPLATE = _load_templates()

_MAX_CAPTURE_BYTES = 2 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Kill the candidate process and any children it spawned."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            if process.poll() is None:
                process.kill()
        return

    if process.poll() is None:
        process.kill()


def _run_test_process(tmpdir: str, timeout: int) -> tuple[int | None, str, str | None]:
    """Run untrusted candidate code in its own process session.

    Output is drained continuously but only a bounded prefix is retained. This
    prevents generated code from exhausting the Ray worker through logs, while
    ``start_new_session`` keeps process-group signals inside the candidate.
    """
    captured = bytearray()
    truncated = [False]

    try:
        process = subprocess.Popen(
            [sys.executable, "test.py"],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            shell=False,
            cwd=tmpdir,
            env=_get_subprocess_env(),
            start_new_session=os.name == "posix",
            bufsize=0,
        )
    except OSError as exc:
        return None, "", f"failed: could not start test process: {exc}"

    assert process.stdout is not None

    def drain_output() -> None:
        while True:
            try:
                chunk = process.stdout.read(_READ_CHUNK_BYTES)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            remaining = _MAX_CAPTURE_BYTES - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])
            if len(chunk) > max(remaining, 0):
                truncated[0] = True

    reader = threading.Thread(target=drain_output, name="keck-test-log", daemon=True)
    reader.start()

    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        # Also clean up background children left behind after the main test exits.
        _terminate_process_group(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=5)
        process.stdout.close()
        if reader.is_alive():
            reader.join(timeout=1)

    test_log = bytes(captured).decode("utf-8", errors="replace")
    if truncated[0]:
        test_log += "\n[Keck truncated candidate output after 2 MiB]"

    if timed_out:
        detail = _extract_error(test_log) if test_log.strip() else ""
        suffix = f"; {detail}" if detail else ""
        return None, test_log, f"failed: test timed out{suffix}"

    return process.returncode, test_log, None


def exec_cuda_eval(
    ext_filename: str,
    ext_content: bytes,
    model_new_patch: str,
    pytorch_module: str,
) -> Tuple[bool, Any]:
    """
    在 GPU 上执行 CUDA 扩展 vs PyTorch 的正确性验证和性能测试

    Args:
        ext_filename: 编译好的 .so 文件名
        ext_content: .so 文件内容
        model_new_patch: 重写后的 model_new_patch.py 内容
        pytorch_module: PyTorch 基线实现（model.py）

    Returns:
        (success, result_or_error_msg)
    """
    cfg = settings.execute

    with tempfile.TemporaryDirectory() as tmpdir:
        # 写入所有需要的文件
        Path(os.path.join(tmpdir, ext_filename)).write_bytes(ext_content)
        Path(os.path.join(tmpdir, "model_new_patch.py")).write_text(
            model_new_patch, encoding="utf-8"
        )
        Path(os.path.join(tmpdir, "model.py")).write_text(
            pytorch_module, encoding="utf-8"
        )
        Path(os.path.join(tmpdir, "test.py")).write_text(
            _CUDA_TEMPLATE, encoding="utf-8"
        )

        try:
            returncode, test_log, process_error = _run_test_process(
                tmpdir, cfg.exec_timeout
            )
        except Exception as e:
            return False, f"failed: test runner error: {type(e).__name__}: {e}"
        finally:
            _cleanup_gpu()

        if process_error is not None:
            return False, process_error

        if returncode != 0:
            return False, _extract_error(test_log, returncode)

        if "#### Correctness check passed!" not in test_log:
            return (
                False,
                f"failed: correctness check not passed. log: [{test_log[:500]}]",
            )

        return True, parse_eval_msg(test_log)


def exec_torch_check(gt_torch: str, to_check_torch: str) -> Tuple[bool, str]:
    """
    在 GPU 上执行两个 PyTorch 实现的正确性对比

    Args:
        gt_torch: 基准 PyTorch 实现（model.py）
        to_check_torch: 待检查的 PyTorch 实现（model_new.py）

    Returns:
        (success, result_or_error_msg)
    """
    cfg = settings.execute

    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "model.py")).write_text(
            gt_torch, encoding="utf-8"
        )
        Path(os.path.join(tmpdir, "model_new.py")).write_text(
            to_check_torch, encoding="utf-8"
        )
        Path(os.path.join(tmpdir, "test.py")).write_text(
            _TORCH_TEMPLATE, encoding="utf-8"
        )

        try:
            returncode, test_log, process_error = _run_test_process(
                tmpdir, cfg.exec_timeout
            )
        except Exception as e:
            return False, f"failed: test runner error: {type(e).__name__}: {e}"
        finally:
            _cleanup_gpu()

        if process_error is not None:
            return False, process_error

        if returncode != 0:
            return False, _extract_error(test_log, returncode)

        if "#### Correctness check passed!" not in test_log:
            return False, "failed: correctness check not passed"

        return True, test_log


def exec_triton_eval(
    triton_code: str,
    pytorch_module: str,
) -> Tuple[bool, Any]:
    """
    在 GPU 上执行 Triton 实现 vs PyTorch 的正确性验证和性能测试

    Args:
        triton_code: Triton 实现（model_new.py）
        pytorch_module: PyTorch 基线实现（model.py）

    Returns:
        (success, result_or_error_msg)
    """
    cfg = settings.execute

    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "model_new.py")).write_text(
            triton_code, encoding="utf-8"
        )
        Path(os.path.join(tmpdir, "model.py")).write_text(
            pytorch_module, encoding="utf-8"
        )
        Path(os.path.join(tmpdir, "test.py")).write_text(
            _TRITON_TEMPLATE, encoding="utf-8"
        )

        try:
            returncode, test_log, process_error = _run_test_process(
                tmpdir, cfg.exec_timeout
            )
        except Exception as e:
            return False, f"failed: test runner error: {type(e).__name__}: {e}"
        finally:
            _cleanup_gpu()

        if process_error is not None:
            return False, process_error

        if returncode != 0:
            return False, _extract_error(test_log, returncode)

        if "#### Correctness check passed!" not in test_log:
            return (
                False,
                f"failed: correctness check not passed. log: [{test_log[:500]}]",
            )

        return True, parse_eval_msg(test_log)


def _cleanup_gpu():
    """
    GPU 清理占位函数

    注意：实际的 GPU 工作是在子进程中完成的，子进程退出后 GPU 内存会自动释放。
    这里不应该 import torch，否则会导致 Ray Worker 进程初始化 CUDA 并占用 GPU 内存。
    """
    pass


def _extract_error(test_log: str, returncode: int | None = None) -> str:
    """从测试日志中提取错误信息"""
    process_failure = ""
    if returncode is not None and returncode < 0:
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = "UNKNOWN"
        process_failure = (
            f"failed: test process terminated by signal "
            f"{signal_number} ({signal_name})"
        )
    elif returncode not in (None, 0):
        process_failure = f"failed: test process exited with code {returncode}"

    lines = test_log.strip().splitlines()
    filtered = [
        line
        for line in lines
        if "AssertionError" in line
        or "Mismatch" in line
        or "Error" in line
        or "Exception" in line
        or "SyntaxError" in line
    ]
    if filtered:
        detail = chr(10).join(filtered[:5])
    else:
        detail = chr(10).join(lines[:5])

    if process_failure and detail:
        return f"{process_failure}\n{detail}"
    if process_failure:
        return process_failure
    return f"failed: test error: {detail}"
