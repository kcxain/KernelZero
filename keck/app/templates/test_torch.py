"""
PyTorch vs PyTorch 正确性验证模板

执行时需要以下文件在同一目录：
- model.py: 基准 PyTorch 实现，包含 Model, get_inputs, get_init_inputs
- model_new.py: 待检查的 PyTorch 实现，包含 Model
"""

import gc
import random

import numpy as np
import torch

from model import Model, get_inputs, get_init_inputs
from model_new import Model as ModelNew


def seed_everywhere(seed=0):
    """固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transform_tensors(tensors, fn):
    """递归地对 tensor 应用变换函数"""
    if not isinstance(tensors, (list, tuple)):
        return tensors
    outputs = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor = fn(tensor)
        elif isinstance(tensor, (list, tuple)):
            tensor = transform_tensors(tensor, fn)
        elif isinstance(tensor, dict):
            tensor = {k: transform_tensors(v, fn) for k, v in tensor.items()}
        outputs.append(tensor)
    return outputs


def check_equal(actual, expected):
    """检查两个输出是否在容差范围内相等"""
    assert isinstance(actual, (list, tuple)) == isinstance(expected, (list, tuple))
    if not isinstance(actual, (list, tuple)):
        actual = [actual]
        expected = [expected]
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-2, rtol=1e-2)


def main():
    # 初始化模型
    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]

    seed_everywhere()
    torch_model = Model(*init_inputs).cuda()
    seed_everywhere()
    check_model = ModelNew(*init_inputs).cuda()

    # 准备输入
    torch_inputs = get_inputs()
    if not isinstance(torch_inputs, (list, tuple)):
        torch_inputs = [torch_inputs]

    torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
    check_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

    # 正确性检查
    with torch.no_grad():
        seed_everywhere()
        gt_outputs = torch_model(*torch_inputs)
        seed_everywhere()
        check_outputs = check_model(*check_inputs)
        check_equal(check_outputs, gt_outputs)

    print("#### Correctness check passed!")

    # 清理
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
