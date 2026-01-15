distill_optimize_prompt = '''\
You are an expert in PyTorch and CUDA programming. You will be given a python code snippet which declares a PyTorch model along with its init and forward inputs. The model is an instance of class Model. It will be created with arguments from get_init_inputs(). Then its forward function will be called with data from get_inputs(). Your task is to write a custom CUDA extension to accelerate the model forward function. Note that:
- Provide only a single python code block in your final answer.
- Name your optimized model as ModelNew. Keep its __init__ and forward function signature the same as Model. Keep the names of all submodules unchanged. Ensure the keys of model state dict are unchanged. Do not create any extra tensor parameter during model initialization.
- Inline the CUDA code within quotes and assign it to the source variable. Inline the C++ function definition into the cpp_src variable. Compile and load the extension using torch.utils.cpp_extension.load_inline.
- Carefully decide the kernel function signature and pass the correct arguments into your kernel.
- Do not perform extra initialization on parameters of any submodule. Keep them initialized by default.
- Implement all CUDA operators by yourself. Do not call any function from torch namespace except for allocating or initializing tensors. Do not call any function from torch.nn.functional namespace. Do not call the forward function of any submodule. You can only use the parameters and attributes of the submodule. For example, you should pass self.linear.weight and self.linear.bias as arguments to your CUDA kernel instead of directly running self.linear(x).
- You can implement more than one kernel in the CUDA extension. If there are multiple operators within forward function, you must implement all of them no matter how many CUDA kernels are needed.
- Optimize the kernel as much as possible. **DO NOT** use cublas/cudnn library calls to implement any operator.

## Generate CUDA kernels optimized for A100-80GB, below is the Architecture information.

- GPU Architecture: Ampere
- GPU Memory: 80GB
- Memory Bandwidth: 1935 GB/s
- FP64 TFLOPS: 9.7
- FP64 Tensor Core TFLOPS: 19.5
- FP32 TFLOPS: 19.5
- TF32 Tensor Core TFLOPS: 156 (312 with sparsity)
- BFLOAT16 Tensore Core TFLOPS: 312 (624 with sparsity)
- FP16 Tensor Core TFLOPS: 312 (624 with sparsity)
- INT8 Tensor Core TOPS: 624 (1248 with sparsity)
- Register File Size: 64K 32-bit registers per SM
- Maximum number of registers per thread: 255
- Maximum number of thread blocks per SM: 32
- Shared memory capacity per SM: 164 KB
- Maximum shared memory per thread block: 163 KB

## Optimize a GPU kernel by applying the following techniques.

1. **Tiling**: Partition data into tiles that fit in shared memory to reduce global memory accesses and improve memory access efficiency.

===== Tiling and Warp/Block Reduction Example Start =====
Given a PyTorch model:
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
You can use tiling and warp/block reduction to write the following kernel:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for softmax
source = """
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
    softmax_kernel<<<grid_size, block_size>>>(input.const_data_ptr<float>(), output.mutable_data_ptr<float>(), N, input.stride(dim));
    return output;
}
"""

cpp_src = """
torch::Tensor softmax_cuda(torch::Tensor input, int dim);
"""
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
===== Tiling and Warp/Block Reduction Example End =====

2. **Fusion**: Merge consecutive operators into a single kernel to minimize memory read/write operations, especially beneficial for memory-bound workloads.

== Fusion Example Start ==
Given PyTorch model:
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = torch.tanh(x)
        x = torch.tanh(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
```
You can use fusion to write the following kernel:
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <ATen/cuda/CUDAContext.h>

#define TILE_W 32
#define TILE_H 8

// For the fixed configuration: 16 filters, each with 3 (in_channels) x 3 x 3 weights.
__constant__ float const_weight[16 * 27];
__constant__ float const_bias[16];

// This kernel fuses a valid 3x3 convolution (for in_channels=3, out_channels=16)
// with a per–pixel minimum reduction over the 16 filters and two successive tanh activations.
// It uses a 2D tiled shared–memory scheme. Each block loads one tile (with halo) per channel.
__global__ void fused_conv_min_tanh_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int H_in,
                                             int W_in) {
    // Fixed parameters.
    const int in_channels = 3;
    const int ksize = 3;
    // Valid convolution reduces the output dimensions.
    const int H_out = H_in - ksize + 1;
    const int W_out = W_in - ksize + 1;
    // Each block in the z-dimension processes one image from the batch.
    int n = blockIdx.z;
    
    // Compute the top–left corner indices of the tile in the input.
    int tile_start_h = blockIdx.y * TILE_H;
    int tile_start_w = blockIdx.x * TILE_W;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Shared memory tile must hold the tile plus a halo (of size ksize-1).
    const int sh_height = TILE_H + ksize - 1;  // 8+2 = 10
    const int sh_width  = TILE_W + ksize - 1;   // 32+2 = 34
    __shared__ float shmem[in_channels][TILE_H + 2][TILE_W + 2]; // [3][10][34]
    
    // Precompute stride values for the input tensor.
    int batch_stride = in_channels * H_in * W_in;
    int channel_stride = H_in * W_in;
    
    // Cooperative loading of the input tile (with halo) into shared memory.
    // Loop over each channel, and have threads load a contiguous subset of the tile.
    for (int c = 0; c < in_channels; c++) {
        for (int i = ty; i < sh_height; i += blockDim.y) {
            for (int j = tx; j < sh_width; j += blockDim.x) {
                int global_i = tile_start_h + i;
                int global_j = tile_start_w + j;
                float value = 0.f;
                if (global_i < H_in && global_j < W_in) {
                    value = input[n * batch_stride + c * channel_stride +
                                  global_i * W_in + global_j];
                }
                shmem[c][i][j] = value;
            }
        }
    }
    __syncthreads();
    
    // Each thread computes one output pixel if within the output bounds.
    int out_i = tile_start_h + ty;
    int out_j = tile_start_w + tx;
    if (out_i < H_out && out_j < W_out) {
        // Initialize accumulators for each filter with the corresponding bias.
        float s[16];
        #pragma unroll
        for (int f = 0; f < 16; f++) {
            s[f] = const_bias[f];
        }
        
        // For each input channel and kernel element, accumulate weighted sums.
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                #pragma unroll
                for (int j = 0; j < 3; j++) {
                    float v = shmem[c][ty + i][tx + j];
                    int w_offset = c * 9 + i * 3 + j;  // 9 = 3*3 per channel.
                    #pragma unroll
                    for (int f = 0; f < 16; f++) {
                        s[f] = __fmaf_rn(v, const_weight[f * 27 + w_offset], s[f]);
                    }
                }
            }
        }
        
        // Compute the minimum value over the 16 filters.
        float m = s[0];
        #pragma unroll
        for (int f = 1; f < 16; f++) {
            m = fminf(m, s[f]);
        }
        
        // Fuse two tanh activations.
        float t = tanhf(tanhf(m));
        
        int out_index = n * (H_out * W_out) + out_i * W_out + out_j;
        output[out_index] = t;
    }
}

torch::Tensor fused_conv_min_tanh_cuda(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias) {
    int batch = input.size(0);
    int H_in = input.size(2);
    int W_in = input.size(3);
    const int ksize = 3;
    int H_out = H_in - ksize + 1;
    int W_out = W_in - ksize + 1;
    
    auto output = torch::empty({batch, 1, H_out, W_out}, input.options());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // Copy weight and bias into constant memory.
    cudaMemcpyToSymbolAsync(const_weight, weight.data_ptr<float>(), sizeof(float) * 16 * 27, 0, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyToSymbolAsync(const_bias, bias.data_ptr<float>(), sizeof(float) * 16, 0, cudaMemcpyDeviceToDevice, stream);
    
    dim3 blockDim(TILE_W, TILE_H, 1);
    dim3 gridDim((W_out + TILE_W - 1) / TILE_W, (H_out + TILE_H - 1) / TILE_H, batch);
    
    fused_conv_min_tanh_kernel<<<gridDim, blockDim, 0, stream>>>(input.data_ptr<float>(),
                                                                   output.data_ptr<float>(),
                                                                   H_in, W_in);
    return output;
}
"""

cpp_source = r"""
torch::Tensor fused_conv_min_tanh_cuda(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias);
"""

# Compile the CUDA extension with optimization and fast math flags.
fused_conv_module = load_inline(
    name="fused_conv_min_tanh",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_conv_min_tanh_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_ldflags=["-Wl,-rpath,$ORIGIN"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # Use nn.Conv2d to set up the weight, bias, and initialization identically.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        return fused_conv_module.fused_conv_min_tanh_cuda(x, self.conv.weight, self.conv.bias)
```
== Fusion Example Start ==

3. **Pipeline**: Overlap computation with data movement to hide memory latency and maximize hardware utilization.
4. **Reordering**: Swap loop orders to improve memory access locality, enhancing cache efficiency and reducing cache misses.

Apply these strategies together to generate a highly efficient, memory-optimized CUDA kernel.

## Now you are given the below PyTorch model:
```python
__py_code_to_be_fill__
```\
'''
