softmax_example = """\
Here is an basic example showing how to write a softmax kernel. It is efficient to use shfl sync instructions to speed up warp/block reduction.
===== Example Start =====
Given PyTorch model:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.softmax(input, dim=self.dim)


def get_inputs():
    input = torch.randn(1024, 512).cuda()
    return [input]


def get_init_inputs():
    return [0]

```
You should respond with:
<think>
[Your thinking process]
</think>
Then the final answer:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for softmax
source = \"""
#include <torch/extension.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        v = std::max(v, __shfl_xor_sync(0xffffffff, v, mask));
    }
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    const int num_warps = blockDim.x / WARP_SIZE;
    v = warp_reduce_max(v);
    __shared__ float shm[32];
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        shm[warp_id] = v;
    }
    __syncthreads();
    v = warp_reduce_max((lane_id < num_warps) ? shm[lane_id] : -INFINITY);
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, mask);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    const int num_warps = blockDim.x / WARP_SIZE;
    v = warp_reduce_sum(v);
    __shared__ float shm[32];
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        shm[warp_id] = v;
    }
    __syncthreads();
    v = warp_reduce_sum((lane_id < num_warps) ? shm[lane_id] : 0.f);
    return v;
}

__global__ void softmax_kernel(const float *input, float *output, int N, int S) {
    const int tile_offset = (blockIdx.x - blockIdx.x % S) * N + blockIdx.x % S;
    const float *input_tile = input + tile_offset;
    float *output_tile = output + tile_offset;

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        max_val = std::max(max_val, input_tile[i * S]);
    }
    max_val = block_reduce_max(max_val);

    float sum = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += std::exp(input_tile[i * S] - max_val);
    }
    sum = block_reduce_sum(sum);

    const float inv_sum = 1.f / sum;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output_tile[i * S] = std::exp(input_tile[i * S] - max_val) * inv_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input, int dim) {
    const int N = input.size(dim);
    torch::Tensor output = torch::empty_like(input);
    const int block_size = 256;
    const int grid_size = input.numel() / N;
    softmax_kernel<<<grid_size, block_size>>>(input.const_data_ptr<float>(), output.mutable_data_ptr<float>(), N,
                                              input.stride(dim));
    return output;
}
\"""

cpp_src = \"""
torch::Tensor softmax_cuda(torch::Tensor input, int dim);
\"""

# Compile the inline CUDA code for softmax op
softmax_cuda = load_inline(
    name="softmax_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["softmax_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return softmax_cuda.softmax_cuda(input, self.dim)

```
===== Example End =====
"""

