"""
FastAPI 入口：只负责路由和请求处理
"""

import asyncio
from typing import Optional

import ray
from fastapi import FastAPI, Form

from app.config import settings
from app.ray_init import init_ray
from app.schemas import ValidationResponse, format_error
from app.services import ValidationPipeline


APP_NAME = "ray-cuda-exec"
app = FastAPI(title=APP_NAME)

# 初始化 Ray 和验证流水线
init_ray()
pipeline = ValidationPipeline()
request_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)


def _request_failure(stage: str, exc: Exception) -> dict:
    """Keep API-level failures local to the current candidate."""
    detail = " ".join(str(exc).split())[:1000]
    return ValidationResponse(
        formated=True,
        compiled=False,
        passed=False,
        msg=f"failed: {stage} service error: {type(exc).__name__}: {detail}",
    ).to_dict()


@app.post("/compute_score")
async def compute_score(
    cuda_code: Optional[str] = Form(None),
    triton_code: Optional[str] = Form(None),
    torch_code: Optional[str] = Form(None),
):
    """
    验证 CUDA / Triton 实现的正确性和性能

    Args:
        cuda_code: 包含 load_inline 的 CUDA 实现代码
        triton_code: Triton 实现代码，包含 ModelNew/Model
        torch_code: PyTorch 基线实现代码
    """
    if cuda_code is not None and triton_code is not None:
        return format_error("Please provide either cuda_code or triton_code").to_dict()

    if cuda_code is None and triton_code is None:
        return format_error("Cannot find cuda_code or triton_code").to_dict()

    async with request_semaphore:
        try:
            if triton_code is not None:
                return await pipeline.eval_triton_async(triton_code, torch_code or "")

            return await pipeline.compile_and_eval_async(cuda_code, torch_code or "")
        except Exception as exc:
            return _request_failure("compute_score", exc)


@app.post("/check_torch")
async def check_torch(
    gt_torch: Optional[str] = Form(None),
    to_check_torch: Optional[str] = Form(None),
):
    """
    对比两个 PyTorch 实现的正确性

    Args:
        gt_torch: 基准 PyTorch 实现
        to_check_torch: 待检查的 PyTorch 实现
    """
    if to_check_torch is None:
        return format_error("Cannot find torch code block").to_dict()

    async with request_semaphore:
        try:
            return await pipeline.check_torch_async(gt_torch or "", to_check_torch)
        except Exception as exc:
            return _request_failure("check_torch", exc)


@app.get("/healthz")
async def healthz():
    """健康检查接口"""
    return {
        "ray_initialized": ray.is_initialized(),
        "cluster_resources": ray.cluster_resources(),
        "available_resources": ray.available_resources(),
        "max_concurrent_requests": settings.max_concurrent_requests,
    }
