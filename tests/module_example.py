import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, bias_shape):
        super().__init__()
        self.triu = nn.UninitializedParameter(
            torch.triu(torch.randn(input_size, input_size))
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x + self.triu
        x = x + self.bias
        return x


input_size = 128
bias_shape = (input_size, 1, 1)


def get_inputs():
    return [torch.randn(input_size, input_size)]


def get_init_inputs():
    return [input_size, bias_shape]

init_inputs = get_init_inputs()
init_inputs = [
    inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in init_inputs
]
inputs = get_inputs()
if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
inputs = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]

print(init_inputs)
print(inputs)


model = Model(*init_inputs).cuda()
model.eval()
output = model(*inputs)

print(output)
