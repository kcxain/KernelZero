"""
Triton vs PyTorch 正确性验证 + 性能测试模板

执行时需要以下文件在同一目录：
- model.py: PyTorch 基线实现，包含 Model, get_inputs, get_init_inputs
- model_new.py: Triton 实现，包含 ModelNew
"""

import gc
import os

import torch

from model import Model, get_inputs, get_init_inputs

import model_new

# 优先使用 ModelNew，如果没有再使用 Model
if hasattr(model_new, "ModelNew"):
    ModelNew = model_new.ModelNew
elif hasattr(model_new, "Model"):
    ModelNew = model_new.Model
else:
    raise ImportError("Neither 'ModelNew' nor 'Model' found in model_new")

# 是否启用详细 profiler（环境变量控制，默认关闭以提升速度）
ENABLE_PROFILER = os.getenv("ENABLE_PROFILER", "0") == "1"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _cleanup_gpu():
    """清理 GPU 缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def transform_tensors(tensors, fn):
    """递归地对 tensor 应用变换函数"""
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, dict):
        return {k: transform_tensors(v, fn) for k, v in tensors.items()}
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(transform_tensors(t, fn) for t in tensors)
    return tensors


def check_equal(actual, expected):
    """检查两个输出是否在容差范围内相等"""
    if not isinstance(actual, (list, tuple)):
        actual = [actual]
    if not isinstance(expected, (list, tuple)):
        expected = [expected]

    assert len(actual) == len(expected), (
        f"Output count mismatch: {len(actual)} vs {len(expected)}"
    )

    for i, (x, y) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(
            x, y, atol=1e-2, rtol=1e-2, msg=lambda m: f"Output {i} mismatch: {m}"
        )


def benchmark_model(model, inputs, warmup_runs=2, benchmark_runs=3):
    """Benchmark 模型执行时间"""
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(*inputs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for _ in range(benchmark_runs):
            _ = model(*inputs)
        end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time = elapsed_ms / 1000.0 / benchmark_runs

    if ENABLE_PROFILER:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            with torch.no_grad():
                _ = model(*inputs)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        print(f"[Profiler disabled] avg_time={avg_time * 1000:.3f}ms")

    return avg_time


def main():
    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]

    set_seed(42)
    torch_model = Model(*init_inputs).cuda()
    set_seed(42)
    triton_model = ModelNew(*init_inputs).cuda()

    try:
        triton_model.load_state_dict(torch_model.state_dict())
    except Exception:
        pass

    inputs = get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    torch_inputs = transform_tensors(inputs, lambda x: x.cuda())
    triton_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

    with torch.no_grad():
        triton_outputs = triton_model(*triton_inputs)
        torch.cuda.synchronize()

        torch_outputs = torch_model(*torch_inputs)
        torch.cuda.synchronize()

        check_equal(triton_outputs, torch_outputs)

    print("#### Correctness check passed!")

    print("#### Benchmark Torch Start")
    torch_time = benchmark_model(torch_model, torch_inputs)
    print("#### Benchmark Troch End")

    print("#### Benchmark Triton Start")
    triton_time = benchmark_model(triton_model, triton_inputs)
    print("#### Benchmark Triton End")

    speedup = torch_time / triton_time if triton_time > 0 else 0
    print(
        f"Torch time: {torch_time:.6f}s, Triton time: {triton_time:.6f}s, Speedup: {speedup:.2f}x"
    )

    del triton_model, torch_model, torch_inputs, triton_inputs
    del triton_outputs, torch_outputs
    gc.collect()
    _cleanup_gpu()


if __name__ == "__main__":
    main()
