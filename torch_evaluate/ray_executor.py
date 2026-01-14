"""
Ray-based GPU resource management for CUDA execution.
This module provides GPU isolation using Ray to prevent CUDA context pollution.
"""

import ray
from typing import List, Dict, Tuple, Optional
from dynamic_checker import execute_model_on_cuda, execute_model_on_cpu


@ray.remote(num_gpus=0.25)
def execute_model_on_cuda_ray_worker(
    code: str, device_id: int, timeout: int = 10
) -> Tuple[bool, Optional[str]]:
    """
    Ray remote function to execute model on CUDA with GPU isolation.
    Each Ray task gets its own GPU, preventing context pollution.

    Args:
        code: Python code string
        device_id: CUDA device ID (Ray will assign this)
        timeout: Maximum execution time

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    # Ray automatically sets CUDA_VISIBLE_DEVICES, so device_id will be 0 in this context
    # But we use the actual device_id for logging
    return execute_model_on_cuda(
        code, timeout=timeout, device_id=0, cleanup_before=True
    )


@ray.remote
def execute_model_on_cpu_ray_worker(
    code: str, timeout: int = 10
) -> Tuple[bool, Optional[str]]:
    """
    Ray remote function to execute model on CPU.
    CPU tasks don't need GPU resources, so they can run in parallel with CUDA tasks.

    Args:
        code: Python code string
        timeout: Maximum execution time

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    return execute_model_on_cpu(code, timeout=timeout)


def init_ray_if_needed(num_gpus: int = None, ray_address: str = None):
    """
    Initialize Ray if not already initialized.

    Args:
        num_gpus: Number of GPUs available. If None, uses all available.
        ray_address: Ray cluster address. If None, starts local Ray.
    """
    import torch

    if not ray.is_initialized():
        if ray_address:
            ray.init(address=ray_address)
        else:
            # Start local Ray with GPU support
            available_gpus = num_gpus if num_gpus else torch.cuda.device_count()
            ray.init(
                num_gpus=available_gpus,
                ignore_reinit_error=True,
            )


def execute_with_ray(
    code: str,
    num_gpus: int = None,
    timeout: int = 10,
    ray_address: str = None,
    use_cpu: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Execute model using Ray for resource management.
    This is a convenience wrapper for single execution.

    Args:
        code: Python code string
        num_gpus: Number of GPUs available. If None, uses all available.
        timeout: Maximum execution time
        ray_address: Ray cluster address. If None, starts local Ray.
        use_cpu: If True, execute on CPU. If False, execute on CUDA.

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    if use_cpu:
        results = batch_execute_cpu_with_ray(
            [code], timeout=timeout, ray_address=ray_address
        )
    else:
        results = batch_execute_with_ray(
            [code], num_gpus=num_gpus, timeout=timeout, ray_address=ray_address
        )
    return results[0] if results else (False, "Ray execution failed")


def batch_execute_with_ray(
    codes: List[str],
    num_gpus: int = None,
    timeout: int = 10,
    ray_address: str = None,
    max_concurrent: int = None,
) -> List[Tuple[bool, Optional[str]]]:
    """
    Batch execute multiple models using Ray with GPU resource management.
    This function properly parallelizes execution across available GPUs.

    Args:
        codes: List of Python code strings
        num_gpus: Number of GPUs available. If None, uses all available.
        timeout: Maximum execution time per model
        ray_address: Ray cluster address
        max_concurrent: Maximum number of concurrent tasks. If None, uses all available GPUs * 2 (since num_gpus=0.5 per task)

    Returns:
        List of (success: bool, error_message: Optional[str]) tuples
    """
    import torch

    if not torch.cuda.is_available():
        return [(False, "CUDA not available")] * len(codes)

    # Initialize Ray if not already initialized
    init_ray_if_needed(num_gpus, ray_address)

    try:
        # Calculate max concurrent tasks
        # Since each task uses 0.5 GPU, we can run 2 tasks per GPU
        if max_concurrent is None:
            available_gpus = num_gpus if num_gpus else torch.cuda.device_count()
            max_concurrent = available_gpus * 2  # 2 tasks per GPU (0.5 GPU each)

        # Submit all tasks (Ray will queue them and execute in parallel)
        futures = [
            execute_model_on_cuda_ray_worker.remote(
                code=code,
                device_id=0,  # Ray handles device assignment
                timeout=timeout,
            )
            for code in codes
        ]

        # Collect results with proper timeout handling
        # Use wait() to get results as they complete, allowing better parallelism
        results = []
        remaining_futures = futures.copy()

        while remaining_futures:
            # Wait for at least one task to complete, with timeout
            ready, not_ready = ray.wait(
                remaining_futures,
                num_returns=min(len(remaining_futures), max_concurrent),
                timeout=timeout + 5,
            )

            if ready:
                # Get results from completed tasks
                for future in ready:
                    try:
                        result = ray.get(future, timeout=1)
                        results.append(result)
                    except Exception as e:
                        results.append((False, f"Ray task error: {str(e)}"))

                remaining_futures = not_ready
            else:
                # Timeout - mark remaining as failed
                for future in remaining_futures:
                    try:
                        ray.cancel(future)
                        results.append((False, "Ray execution timeout"))
                    except:
                        results.append((False, "Ray execution timeout"))
                break

        # Ensure we have results for all codes
        while len(results) < len(codes):
            results.append((False, "Ray execution incomplete"))

        return results

    except Exception as e:
        return [(False, f"Ray batch execution error: {str(e)}")] * len(codes)


def batch_execute_cpu_with_ray(
    codes: List[str],
    timeout: int = 10,
    ray_address: str = None,
    max_concurrent: int = None,
) -> List[Tuple[bool, Optional[str]]]:
    """
    Batch execute multiple models on CPU using Ray.
    CPU tasks can run in parallel without GPU resources.

    Args:
        codes: List of Python code strings
        timeout: Maximum execution time per model
        ray_address: Ray cluster address
        max_concurrent: Maximum number of concurrent tasks. If None, uses CPU count.

    Returns:
        List of (success: bool, error_message: Optional[str]) tuples
    """
    import os

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        if ray_address:
            ray.init(address=ray_address)
        else:
            # Start local Ray without GPU requirement
            ray.init(ignore_reinit_error=True)

    try:
        # Calculate max concurrent tasks
        if max_concurrent is None:
            max_concurrent = os.cpu_count() or 4  # Use CPU count, default to 4

        # Submit all tasks (Ray will queue them and execute in parallel)
        futures = [
            execute_model_on_cpu_ray_worker.remote(
                code=code,
                timeout=timeout,
            )
            for code in codes
        ]

        # Collect results with proper timeout handling
        results = []
        remaining_futures = futures.copy()

        while remaining_futures:
            # Wait for at least one task to complete, with timeout
            ready, not_ready = ray.wait(
                remaining_futures,
                num_returns=min(len(remaining_futures), max_concurrent),
                timeout=timeout + 5,
            )

            if ready:
                # Get results from completed tasks
                for future in ready:
                    try:
                        result = ray.get(future, timeout=1)
                        results.append(result)
                    except Exception as e:
                        results.append((False, f"Ray task error: {str(e)}"))

                remaining_futures = not_ready
            else:
                # Timeout - mark remaining as failed
                for future in remaining_futures:
                    try:
                        ray.cancel(future)
                        results.append((False, "Ray execution timeout"))
                    except Exception:
                        results.append((False, "Ray execution timeout"))
                break

        # Ensure we have results for all codes
        while len(results) < len(codes):
            results.append((False, "Ray execution incomplete"))

        return results

    except Exception as e:
        return [(False, f"Ray CPU batch execution error: {str(e)}")] * len(codes)
