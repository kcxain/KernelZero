import asyncio
from typing import Dict, Any

import ray

from app.config import settings
from app.schemas import ValidationResponse, compile_error, success
from app.services.compiler import compile_cuda_extension
from app.services.executor import exec_cuda_eval, exec_torch_check, exec_triton_eval
from app.utils import parse_compile_msg


async def _async_get(ref: ray.ObjectRef, poll_interval: float = 0.05, max_interval: float = 1.0):
    """
    非阻塞等待 Ray ObjectRef 完成

    使用 ray.wait 轮询而不是 asyncio.to_thread(ray.get)，
    避免线程池开销，更高效地利用事件循环。
    采用指数退避策略：短任务响应快，长任务（编译）减少空转 CPU。
    """
    interval = poll_interval
    while True:
        ready, _ = ray.wait([ref], timeout=0)
        if ready:
            return ray.get(ref)
        await asyncio.sleep(interval)
        # 指数退避，上限 max_interval
        interval = min(interval * 1.5, max_interval)


def _infrastructure_failure(stage: str, exc: Exception) -> Dict[str, Any]:
    """Convert Ray/worker failures into a normal failed-sample response."""
    detail = " ".join(str(exc).split())[:1000]
    return ValidationResponse(
        formated=True,
        compiled=False,
        passed=False,
        msg=f"failed: {stage} infrastructure error: {type(exc).__name__}: {detail}",
    ).to_dict()


class ValidationPipeline:
    """
    编排 CPU 编译任务和 GPU 执行任务

    - 编译任务：num_cpus=N, num_gpus=0（纯 CPU）
    - 执行任务：num_cpus=1, num_gpus=1（需要 GPU）
    """

    def __init__(self, compiler_cpus: int | None = None):
        cpus = compiler_cpus or settings.compile.compiler_cpus
        gpus = settings.execute.execute_gpus

        # 创建 Ray remote 函数
        self._compile_remote = ray.remote(num_cpus=cpus)(compile_cuda_extension)
        gpu_task_options = {
            "num_gpus": gpus,
            "max_calls": 1,
            "max_retries": 0,
        }
        self._cuda_eval_remote = ray.remote(**gpu_task_options)(exec_cuda_eval)
        self._triton_eval_remote = ray.remote(**gpu_task_options)(exec_triton_eval)
        self._torch_check_remote = ray.remote(**gpu_task_options)(exec_torch_check)

    # ========== 同步 API ==========

    def compile_and_eval(self, cuda_code: str, pytorch_module: str) -> Dict[str, Any]:
        """同步版本：编译 + 执行"""
        # 1. 编译
        compile_ok, compile_result = ray.get(self._compile_remote.remote(cuda_code))

        if not compile_ok:
            msg = parse_compile_msg(compile_result.message)
            return compile_error(msg).to_dict()

        # 2. 执行
        eval_ok, eval_result = ray.get(
            self._cuda_eval_remote.remote(
                ext_filename=compile_result.ext_filename,
                ext_content=compile_result.ext_content,
                model_new_patch=compile_result.model_new_patch,
                pytorch_module=pytorch_module,
            )
        )

        if eval_ok:
            return success(eval_result).to_dict()
        return ValidationResponse(
            formated=True, compiled=True, passed=False, msg=eval_result
        ).to_dict()

    def check_torch(self, gt_torch: str, to_check_torch: str) -> Dict[str, Any]:
        """同步版本：Torch vs Torch 对比"""
        ok, result = ray.get(self._torch_check_remote.remote(gt_torch, to_check_torch))

        if ok:
            return success(result).to_dict()
        return ValidationResponse(
            formated=True, compiled=True, passed=False, msg=result
        ).to_dict()

    def eval_triton(self, triton_code: str, pytorch_module: str) -> Dict[str, Any]:
        """同步版本：Triton vs Torch 对比"""
        try:
            ok, result = ray.get(
                self._triton_eval_remote.remote(
                    triton_code=triton_code,
                    pytorch_module=pytorch_module,
                )
            )
        except Exception as exc:
            return _infrastructure_failure("Triton evaluation", exc)

        if ok:
            return success(result).to_dict()
        return ValidationResponse(
            formated=True, compiled=True, passed=False, msg=result
        ).to_dict()

    async def compile_and_eval_async(
        self, cuda_code: str, pytorch_module: str
    ) -> Dict[str, Any]:
        """
        异步版本：编译 + 执行

        使用 Ray 原生异步轮询，避免阻塞事件循环
        """
        # 1. 编译（CPU 任务）
        compile_future = self._compile_remote.remote(cuda_code)
        compile_ok, compile_result = await _async_get(compile_future)

        if not compile_ok:
            msg = parse_compile_msg(compile_result.message)
            return compile_error(msg).to_dict()

        # 2. 执行（GPU 任务）
        eval_future = self._cuda_eval_remote.remote(
            ext_filename=compile_result.ext_filename,
            ext_content=compile_result.ext_content,
            model_new_patch=compile_result.model_new_patch,
            pytorch_module=pytorch_module,
        )
        eval_ok, eval_result = await _async_get(eval_future)

        if eval_ok:
            return success(eval_result).to_dict()
        return ValidationResponse(
            formated=True, compiled=True, passed=False, msg=eval_result
        ).to_dict()

    async def check_torch_async(
        self, gt_torch: str, to_check_torch: str
    ) -> Dict[str, Any]:
        """异步版本：Torch vs Torch 对比"""
        future = self._torch_check_remote.remote(gt_torch, to_check_torch)
        ok, result = await _async_get(future)

        if ok:
            return success(result).to_dict()
        return ValidationResponse(
            formated=True, compiled=True, passed=False, msg=result
        ).to_dict()

    async def eval_triton_async(
        self, triton_code: str, pytorch_module: str
    ) -> Dict[str, Any]:
        """异步版本：Triton vs Torch 对比"""
        try:
            future = self._triton_eval_remote.remote(triton_code, pytorch_module)
            ok, result = await _async_get(future)
        except Exception as exc:
            return _infrastructure_failure("Triton evaluation", exc)

        if ok:
            return success(result).to_dict()
        return ValidationResponse(
            formated=True, compiled=True, passed=False, msg=result
        ).to_dict()
