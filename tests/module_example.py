import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------- CUDA extension source -------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef __device__
#define __device__ __device__
#endif

// Gelu approximation (torch.nn.functional.gelu)
__device__ __forceinline__ float gelu_approx(float x) {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    const float c = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + c * x_cubed);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

// -----------------------------------------------------------
// Kernel: embedding lookup + padding handling + Gelu
// out layout: (batch, emb_dim) contiguous
// -----------------------------------------------------------
__global__ void embed_gelu_kernel(
    const int64_t* __restrict__ indices,
    const float*   __restrict__ weight,
    float*         __restrict__ out,
    const int64_t batch,
    const int64_t emb_dim,
    const int64_t padding_idx)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * emb_dim;
    if (idx >= total) return;

    int64_t row = idx / emb_dim;        // batch index
    int64_t col = idx % emb_dim;        // embedding dimension

    int64_t token = indices[row];
    float val = 0.0f;
    if (token != padding_idx) {
        // weight is (num_embeddings, emb_dim) row-major
        val = weight[token * emb_dim + col];
    }
    // apply gelu (zero stays zero)
    out[row * emb_dim + col] = gelu_approx(val);
}

// -----------------------------------------------------------
torch::Tensor embed_gelu(torch::Tensor indices,
                         torch::Tensor weight,
                         int64_t padding_idx)
{
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(),  "weight must be a CUDA tensor");
    TORCH_CHECK(indices.dtype() == torch::kLong, "indices must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat, "weight must be float");

    const int64_t batch    = indices.size(0);
    const int64_t emb_dim  = weight.size(1);

    auto out = torch::empty({batch, emb_dim},
                            weight.options());

    const int threads = 256;
    const int64_t total = batch * emb_dim;
    const int blocks = (total + threads - 1) / threads;

    embed_gelu_kernel<<<blocks, threads>>>(
        indices.const_data_ptr<int64_t>(),
        weight.const_data_ptr<float>(),
        out.mutable_data_ptr<float>(),
        batch,
        emb_dim,
        padding_idx);

    // In case of launch failure
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed : ", cudaGetErrorString(err));
    }

    return out;
}
"""

cpp_source = r"""
torch::Tensor embed_gelu(torch::Tensor indices,
                         torch::Tensor weight,
                         int64_t padding_idx);
"""

# Compile the inline CUDA extension
embed_gelu_mod = load_inline(
    name="embed_gelu_mod",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["embed_gelu"],
    verbose=False,
)

# ------------------- Optimized model -------------------
class ModelNew(nn.Module):
    def __init__(self, emb_dim, num_embeddings, embedding_bag_idx, padding_idx):
        super().__init__()
        # keep exactly the same submodules and state‑dict keys
        self.embedding = nn.ParameterDict({"emb": nn.Parameter(torch.randn(num_embeddings, emb_dim))})
        self.embed_idxs = embedding_bag_idx
        self.padding_idx = padding_idx

    def forward(self, x):
        """
        x: LongTensor of shape (batch_size,)
        Returns: Tensor of shape (batch_size, 1, emb_dim)
        """
        # Call our custom CUDA kernel: embedding + GELU
        out = embed_gelu_mod.embed_gelu(x, self.embedding["emb"], self.padding_idx)
        # Unsqueeze dimension 1 (batch, 1, emb_dim)
        return out.view(x.size(0), 1, -1)
