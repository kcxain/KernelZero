import re
from typing import List
from torch_pool.ops import torch_ops, torch_nn_functional_ops, torch_nn_ops

torch_ops = set(torch_ops)
torch_nn_functional_ops = set(torch_nn_functional_ops)
torch_nn_ops = set(torch_nn_ops)
torch_pattern = (
        r"torch\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*[\(\[]"
    )
nn_pattern = r"nn\.([A-Z][a-zA-Z0-9_]*)\s*\("
f_pattern = r"F\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\("


def extract_operators_from_code(code: str) -> List[str]:
    """
    从 Python 代码中提取使用的 PyTorch 算子

    Args:
        code: Python 代码字符串

    Returns:
        算子列表
    """
    operators = set()

    # 匹配 torch.xxx 函数调用
    for match in re.finditer(torch_pattern, code):
        op = f"torch.{match.group(1)}"
        if op in torch_ops or op in torch_nn_ops or op in torch_nn_functional_ops:
            operators.add(op)

    # 匹配 nn.xxx 模块使用
    for match in re.finditer(nn_pattern, code):
        op = f"torch.nn.{match.group(1)}"
        if op in torch_nn_ops:
            operators.add(op)

    # 匹配 F.xxx 函数调用
    for match in re.finditer(f_pattern, code):
        op = f"torch.nn.functional.{match.group(1)}"
        if op in torch_nn_functional_ops:
            operators.add(op)

    return sorted(list(operators))

if __name__ == "__main__":
    code = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))  # Broadcastable bias over channels
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)                            # (B, C, D, H, W)
        x = x.mean(dim=2, keepdim=True)                       # Mean pool over depth dim (D)
        x = x + self.bias                                     # Bias add per channel
        x = torch.softmax(x, dim=1)                           # Softmax over channels
        x = torch.tanh(x)                                     # Nonlinearity
        x = x * self.scaling_factor                           # Scaling
        return x

# === Test config ===
batch_size = 16
in_channels  = 16  
out_channels = 64  
depth = 32; height = width = 128  
kernel_size  = 3
stride       = 1  
padding = 1
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]
"""
    print(extract_operators_from_code(code))
