import torch.nn.functional as F
import ast


class ModuleAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.used_apis = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'F':
                self.used_apis.add(node.func.attr)
        self.generic_visit(node)




class TorchPool:
    def __init__(self):
        self.all_api = self.get_all_api()
    
    def get_all_api(self):
        return [name for name in dir(F) if callable(getattr(F, name)) and not name.startswith("_")]

    def analyse(self, module_or_code):
        used_apis = set()
        # print(type(module_or_code))
        if isinstance(module_or_code, str):
            try:
                tree = ast.parse(module_or_code)
            except SyntaxError:
                return []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'F':
                        used_apis.add(node.func.attr)
        else:
            for name, obj in module_or_code.__dict__.items():
                if callable(obj):
                    used_apis.add(name)
        
        return [api for api in used_apis if api in self.all_api]

if __name__ == "__main__":
    pool = TorchPool()
    print(pool.get_all_api())
    content = "import torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\n# ------------------- CUDA extension source -------------------\ncuda_source = r\"\"\"\n#include <torch/extension.h>\n#include <cuda_runtime.h>\n#include <math_constants.h>\n\n#ifndef __device__\n#define __device__ __device__\n#endif\n\n// Gelu approximation (torch.nn.functional.gelu)\n__device__ __forceinline__ float gelu_approx(float x) {\n    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))\n    const float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)\n    const float c = 0.044715f;\n    float x_cubed = x * x * x;\n    float inner = sqrt_2_over_pi * (x + c * x_cubed);\n    float tanh_inner = tanhf(inner);\n    return 0.5f * x * (1.0f + tanh_inner);\n}\n\n// -----------------------------------------------------------\n// Kernel: embedding lookup + padding handling + Gelu\n// out layout: (batch, emb_dim) contiguous\n// -----------------------------------------------------------\n__global__ void embed_gelu_kernel(\n    const int64_t* __restrict__ indices,\n    const float*   __restrict__ weight,\n    float*         __restrict__ out,\n    const int64_t batch,\n    const int64_t emb_dim,\n    const int64_t padding_idx)\n{\n    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n    int64_t total = batch * emb_dim;\n    if (idx >= total) return;\n\n    int64_t row = idx / emb_dim;        // batch index\n    int64_t col = idx % emb_dim;        // embedding dimension\n\n    int64_t token = indices[row];\n    float val = 0.0f;\n    if (token != padding_idx) {\n        // weight is (num_embeddings, emb_dim) row-major\n        val = weight[token * emb_dim + col];\n    }\n    // apply gelu (zero stays zero)\n    out[row * emb_dim + col] = gelu_approx(val);\n}\n\n// -----------------------------------------------------------\ntorch::Tensor embed_gelu(torch::Tensor indices,\n                         torch::Tensor weight,\n                         int64_t padding_idx)\n{\n    TORCH_CHECK(indices.is_cuda(), \"indices must be a CUDA tensor\");\n    TORCH_CHECK(weight.is_cuda(),  \"weight must be a CUDA tensor\");\n    TORCH_CHECK(indices.dtype() == torch::kLong, \"indices must be int64\");\n    TORCH_CHECK(weight.dtype() == torch::kFloat, \"weight must be float\");\n\n    const int64_t batch    = indices.size(0);\n    const int64_t emb_dim  = weight.size(1);\n\n    auto out = torch::empty({batch, emb_dim},\n                            weight.options());\n\n    const int threads = 256;\n    const int64_t total = batch * emb_dim;\n    const int blocks = (total + threads - 1) / threads;\n\n    embed_gelu_kernel<<<blocks, threads>>>(\n        indices.const_data_ptr<int64_t>(),\n        weight.const_data_ptr<float>(),\n        out.mutable_data_ptr<float>(),\n        batch,\n        emb_dim,\n        padding_idx);\n\n    // In case of launch failure\n    cudaError_t err = cudaGetLastError();\n    if (err != cudaSuccess) {\n        AT_ERROR(\"CUDA kernel failed : \", cudaGetErrorString(err));\n    }\n\n    return out;\n}\n\"\"\"\n\ncpp_source = r\"\"\"\ntorch::Tensor embed_gelu(torch::Tensor indices,\n                         torch::Tensor weight,\n                         int64_t padding_idx);\n\"\"\"\n\n# Compile the inline CUDA extension\nembed_gelu_mod = load_inline(\n    name=\"embed_gelu_mod\",\n    cpp_sources=cpp_source,\n    cuda_sources=cuda_source,\n    functions=[\"embed_gelu\"],\n    verbose=False,\n)\n\n# ------------------- Optimized model -------------------\nclass ModelNew(nn.Module):\n    def __init__(self, emb_dim, num_embeddings, embedding_bag_idx, padding_idx):\n        super().__init__()\n        # keep exactly the same submodules and state‑dict keys\n        self.embedding = nn.ParameterDict({\"emb\": nn.Parameter(torch.randn(num_embeddings, emb_dim))})\n        self.embed_idxs = embedding_bag_idx\n        self.padding_idx = padding_idx\n\n    def forward(self, x):\n        \"\"\"\n        x: LongTensor of shape (batch_size,)\n        Returns: Tensor of shape (batch_size, 1, emb_dim)\n        \"\"\"\n        # Call our custom CUDA kernel: embedding + GELU\n        out = embed_gelu_mod.embed_gelu(x, self.embedding[\"emb\"], self.padding_idx)\n        # Unsqueeze dimension 1 (batch, 1, emb_dim)\n        return out.view(x.size(0), 1, -1)"                    
    print(pool.analyse(content))