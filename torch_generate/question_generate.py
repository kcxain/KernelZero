import os
import sys
import argparse
from typing import List, Dict, Tuple
import json
import regex as re

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from torch_pool.operator_query import PyTorchOperatorQuery
except ImportError:
    # 如果导入失败，在main函数中处理
    PyTorchOperatorQuery = None

STORAGE_PATH = os.getenv("STORAGE_PATH")


def load_operator_combination(combination_file: str, index: int = None) -> List[str]:
    """
    从组合文件中加载算子组合

    Args:
        combination_file: 组合文件路径
        index: 组合索引，如果为None则随机选择一个

    Returns:
        算子组合列表，例如 ["torch.nn.Conv3d", "torch.softmax", "torch.nn.MaxPool3d"]
    """
    with open(combination_file, "r", encoding="utf-8") as f:
        combinations = json.load(f)

    if index is None:
        import random

        index = random.randint(0, len(combinations) - 1)

    if index >= len(combinations):
        raise ValueError(
            f"Index {index} out of range. Total combinations: {len(combinations)}"
        )

    return combinations[index]


def get_operator_docstrings(
    operator_query: PyTorchOperatorQuery, operators: List[str]
) -> Dict[str, str]:
    """
    获取算子的docstring

    Args:
        operator_query: PyTorchOperatorQuery 实例
        operators: 算子列表

    Returns:
        算子名到docstring的映射
    """
    docstrings = {}
    for op in operators:
        desc = operator_query.get_description(op)
        if desc:
            docstrings[op] = desc
        else:
            docstrings[op] = f"No documentation available for {op}"
    return docstrings


def build_prompt(
    operators: List[str], docstrings: Dict[str, str], example_code: str
) -> str:
    """
    构建生成prompt

    Args:
        operators: 算子组合列表
        docstrings: 算子docstring字典
        example_code: 示例代码

    Returns:
        prompt字符串
    """
    operators_str = ", ".join(operators)

    docstrings_text = "\n".join([
        f"**{op}**:\n{doc}\n" for op, doc in docstrings.items()
    ])

    prompt = f"""You are an expert PyTorch developer. Your task is to generate a complete PyTorch nn.Module that uses the following operators: {operators_str}.

## Operator Documentation:

{docstrings_text}

## Example Format:

Please follow this exact format (see the example below):

```python
{example_code}
```

## Requirements:

1. Create a class named `Model` that inherits from `nn.Module`
2. The model MUST use ALL of the following operators: {operators_str}
3. Include proper docstrings for the class and forward method
4. Define `batch_size`, input dimensions, and other necessary parameters
5. Implement `get_inputs()` function that returns a list of input tensors for forward()
6. Implement `get_init_inputs()` function that returns a list of arguments for __init__()
7. Use appropriate dimensions and parameters for each operator
8. The code should be complete and runnable

## Output Format:

Output ONLY the Python code, starting with `import torch` and ending with the `get_init_inputs()` function. Do NOT include any explanations, markdown formatting, or additional text outside the code block.

Generate the complete PyTorch module now:"""

    return prompt


def load_all_operator_combinations(combination_file: str) -> List[List[str]]:
    """
    加载文件中的所有算子组合

    Args:
        combination_file: 组合文件路径

    Returns:
        所有算子组合的列表
    """
    with open(combination_file, "r", encoding="utf-8") as f:
        combinations = json.load(f)
    return combinations


def generate_module_for_combination(
    operators: List[str],
    operator_query: PyTorchOperatorQuery,
    example_code: str,
    model,
    tokenizer,
    num_samples: int = 1,
) -> List[Dict]:
    """
    为单个算子组合生成模块

    Args:
        operators: 算子组合列表
        operator_query: PyTorchOperatorQuery 实例
        example_code: 示例代码
        model: vLLM 模型实例
        tokenizer: tokenizer
        num_samples: 每个组合生成的样本数

    Returns:
        生成结果列表
    """
    # 导入vLLM相关（延迟导入）
    import vllm

    # 获取docstrings
    docstrings = get_operator_docstrings(operator_query, operators)

    # 构建prompt
    user_prompt = build_prompt(operators, docstrings, example_code)

    # 构建chat
    chat = [
        {
            "role": "system",
            "content": (
                "You are an expert PyTorch developer specializing in creating efficient neural network modules. "
                "You write clean, well-documented, and production-ready PyTorch code."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True
        )
    else:
        prompt = "system: " + chat[0]["content"] + "\n" + "user: " + chat[1]["content"]

    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=0.7,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # 生成
    completions = model.generate([prompt] * num_samples, sampling_params=sample_params)

    results = []
    for completion in completions:
        response = completion.outputs[0].text
        try:
            # 提取代码块
            code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", response, re.DOTALL)
            if code_blocks:
                # 使用最后一个代码块
                code = code_blocks[-1].strip()
            else:
                # 如果没有代码块，尝试提取整个响应
                code = response.strip()

            results.append({
                "operators": operators,
                "code": code,
                "raw_response": response,
                "prompt": prompt,  # 新增：保存prompt
            })
        except Exception as e:
            results.append({
                "operators": operators,
                "code": response,
                "raw_response": response,
                "prompt": prompt,  # 新增：保存prompt
            })

    return results


def build_prompts_batch(
    all_combinations: List[List[str]],
    operator_query: PyTorchOperatorQuery,
    example_code: str,
    tokenizer,
    num_samples_per_combination: int = 1,
) -> List[Tuple[List[str], str]]:
    """
    批量构建多个组合的 prompt

    Args:
        all_combinations: 所有算子组合列表
        operator_query: PyTorchOperatorQuery 实例
        example_code: 示例代码
        tokenizer: tokenizer
        num_samples_per_combination: 每个组合生成的样本数

    Returns:
        (operators, prompt) 元组列表
    """
    prompts_with_operators = []

    for operators in all_combinations:
        # 获取docstrings
        docstrings = get_operator_docstrings(operator_query, operators)

        # 构建prompt
        user_prompt = build_prompt(operators, docstrings, example_code)

        # 构建chat
        chat = [
            {
                "role": "system",
                "content": (
                    "You are an expert PyTorch developer specializing in creating efficient neural network modules. "
                    "You write clean, well-documented, and production-ready PyTorch code."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True,
            )
        else:
            prompt = (
                "system: " + chat[0]["content"] + "\n" + "user: " + chat[1]["content"]
            )

        # 为每个组合生成 num_samples_per_combination 个 prompt
        for _ in range(num_samples_per_combination):
            prompts_with_operators.append((operators, prompt))

    return prompts_with_operators


def process_batch_results(
    prompts_with_operators: List[Tuple[List[str], str]],
    completions,
) -> List[Dict]:
    """
    批量处理生成结果

    Args:
        prompts_with_operators: (operators, prompt) 元组列表
        completions: 模型生成结果

    Returns:
        处理后的结果列表
    """
    results = []

    for (operators, prompt), completion in zip(prompts_with_operators, completions):
        response = completion.outputs[0].text
        try:
            # 提取代码块
            code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", response, re.DOTALL)
            if code_blocks:
                # 使用最后一个代码块
                code = code_blocks[-1].strip()
            else:
                # 如果没有代码块，尝试提取整个响应
                code = response.strip()

            results.append({
                "operators": operators,
                "code": code,
                "raw_response": response,
                "prompt": prompt,  # 新增：保存prompt
            })
        except Exception as e:
            results.append({
                "operators": operators,
                "code": response,
                "raw_response": response,
                "prompt": prompt,  # 新增：保存prompt
            })

    return results


def main(args):
    # 延迟导入 vLLM
    import vllm
    from transformers import AutoTokenizer

    # 确保PyTorchOperatorQuery可用
    global PyTorchOperatorQuery
    if PyTorchOperatorQuery is None:
        from torch_pool.operator_query import PyTorchOperatorQuery

    # 直接使用传入的 model_path（已经是合并后的模型）
    model_path = args.model
    tokenizer_path = args.model
    print(f"Using model from: {model_path}")

    # 加载所有算子组合
    if args.combination_file:
        print(f"Loading operator combinations from: {args.combination_file}")
        all_combinations = load_all_operator_combinations(args.combination_file)
        print(f"Total combinations: {len(all_combinations)}")
    else:
        # 如果没有指定，使用默认组合文件
        default_file = os.path.join(
            _project_root,
            "resources",
            "torch_ops",
            "2ops.json",
        )
        if os.path.exists(default_file):
            all_combinations = load_all_operator_combinations(default_file)
            print(f"Using default combination file: {default_file}")
            print(f"Total combinations: {len(all_combinations)}")
        else:
            raise ValueError(
                "No operator combination file specified and default file not found"
            )

    # 如果指定了combination_index，只处理该索引的组合
    if args.combination_index is not None:
        if args.combination_index >= len(all_combinations):
            raise ValueError(
                f"Index {args.combination_index} out of range. Total combinations: {len(all_combinations)}"
            )
        all_combinations = [all_combinations[args.combination_index]]
        print(f"Processing only combination at index {args.combination_index}")

    # 加载算子查询器
    operator_query = PyTorchOperatorQuery()
    print("Operator query initialized")

    # 加载示例代码
    example_file = os.path.join(
        _project_root,
        "resources",
        "KernelBench",
        "level2",
        "6_Conv3d_Softmax_MaxPool_MaxPool.py",
    )
    if os.path.exists(example_file):
        with open(example_file, "r", encoding="utf-8") as f:
            example_code = f.read()
        print(f"Loaded example code from: {example_file}")
    else:
        # 使用默认示例
        example_code = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        x = self.pool1(x)
        x = self.pool2(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]"""
        print("Using default example code")

    # 初始化模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = vllm.LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        gpu_memory_utilization=0.8,
        seed=int(args.suffix) if args.suffix else 42,
    )
    print(f"Model loaded successfully from: {model_path}")

    # 批量构建所有 prompt
    print(f"\n{'=' * 80}")
    print(f"批量构建 {len(all_combinations)} 个算子组合的 prompt...")
    print(f"{'=' * 80}")

    prompts_with_operators = build_prompts_batch(
        all_combinations=all_combinations,
        operator_query=operator_query,
        example_code=example_code,
        tokenizer=tokenizer,
        num_samples_per_combination=args.num_samples,
    )

    total_prompts = len(prompts_with_operators)
    print(f"总共构建了 {total_prompts} 个 prompt")
    print(f"  - 组合数: {len(all_combinations)}")
    print(f"  - 每个组合样本数: {args.num_samples}")

    # 批量生成
    print(f"\n{'=' * 80}")
    print(
        f"开始批量生成（批量大小: {args.batch_size if hasattr(args, 'batch_size') else total_prompts}）..."
    )
    print(f"{'=' * 80}")

    import vllm

    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=0.7,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    all_results = []

    # 直接整体处理所有 prompt
    batch_prompts_with_ops = prompts_with_operators
    batch_prompts = [prompt for _, prompt in batch_prompts_with_ops]

    try:
        # 批量生成
        completions = model.generate(batch_prompts, sampling_params=sample_params)

        # 批量处理结果
        batch_results = process_batch_results(batch_prompts_with_ops, completions)

        # 为每个结果添加组合索引
        for i, result in enumerate(batch_results):
            operators = batch_prompts_with_ops[i][0]
            prompt = batch_prompts_with_ops[i][1]
            # 找到对应的组合索引
            combination_idx = (
                all_combinations.index(operators)
                if operators in all_combinations
                else -1
            )
            result["combination_index"] = combination_idx
            # 确保prompt都加入最终结果，即使process_batch_results返回已包含prompt
            if "prompt" not in result:
                result["prompt"] = prompt
            all_results.append(result)

    except Exception as e:
        print(f"  整体处理错误: {e}")
        # 为失败的批次添加错误结果
        for operators, prompt in batch_prompts_with_ops:
            combination_idx = (
                all_combinations.index(operators)
                if operators in all_combinations
                else -1
            )
            all_results.append({
                "combination_index": combination_idx,
                "operators": operators,
                "code": "",
                "prompt": prompt,
            })

    print(f"\n总体进度: {total_prompts}/{total_prompts} (100.0%)")

    # 保存结果
    output_dir = (
        os.path.join(STORAGE_PATH, "generated_modules")
        if STORAGE_PATH
        else "./generated_modules"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.save_name}_{args.suffix}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PyTorch modules based on operator combinations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model path (should be a merged HuggingFace model)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix to add to the output file"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="generated_module",
        help="Base name for output file",
    )
    parser.add_argument(
        "--combination_file",
        type=str,
        default=None,
        help="Path to operator combination JSON file (2ops.json or 3ops.json)",
    )
    parser.add_argument(
        "--combination_index",
        type=int,
        default=None,
        help="Index of combination to use (if None, processes all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for generation (if None, processes all at once)",
    )
    args = parser.parse_args()

    main(args)


"""输出格式
[
    {
        "operators": ["torch.nn.Conv3d", "torch.softmax", "torch.nn.MaxPool3d"],
        "code": "生成的PyTorch模块代码",
        "raw_response": "模型的原始响应",
        "prompt": "用于生成该样本的完整prompt"
    },
]
"""
