# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

import json
import random
import sys
from pathlib import Path
def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}



def process_image(image: Union[Dict[str, Any], ImageObject, str], min_pixels: int, max_pixels: int) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            self.dataset = load_dataset("parquet", data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            # 支持 JSON 文件格式（用于 torch 生成 CUDA 任务）
            if data_path.endswith('.json') or data_path.endswith('.jsonl'):
                # 从 JSON/JSONL 文件加载数据
                with open(data_path, 'r', encoding='utf-8') as f:
                    if data_path.endswith('.jsonl'):
                        # JSONL 格式：每行一个 JSON 对象
                        data_list = [json.loads(line) for line in f if line.strip()]
                    else:
                        # JSON 格式：可能是一个列表
                        data_list = json.load(f)
                        if not isinstance(data_list, list):
                            data_list = [data_list]
                # 转换为 datasets.Dataset
                from datasets import Dataset
                self.dataset = Dataset.from_list(data_list)
            else:
                # 默认使用 parquet 格式
                self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:
            raise ValueError(
                f"data_path '{data_path}' is neither a valid directory nor a valid file. "
                f"Please check if the path exists and is accessible."
            )

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if "questioner_format_with_persona" in self.format_prompt:
            print("load personas")
            personas_dataset = load_dataset("proj-persona/PersonaHub", "math", split="train")
            self.personas = [item['input persona'] for item in personas_dataset]
            # self.personas = self.personas.select(range(100))
        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(self._filter_overlong_prompts, desc="Filtering overlong prompts")

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 如果使用 kernelbench 模板，从 raw_problem 字段读取数据
        if self.format_prompt and "PyTorch and CUDA" in self.format_prompt:
            # kernelbench 模板使用 raw_problem 字段作为输入，变量名为 py_code
            # 参考 fs-eval/fs_eval/prompt_builder.py 的构造方式
            if "raw_problem" in example:
                py_code = example["raw_problem"]
                format_prompt = Template(self.format_prompt.strip())
                
                # 导入 GPU_SPEC_INFO 和 softmax_example（与 fs-eval 保持一致）
                try:
                    # 获取项目根目录
                    # 从 verl/utils/dataset.py 向上找到项目根（KernelZero）
                    current_file = Path(__file__).resolve()
                    # verl/utils/dataset.py -> verl/utils -> verl -> KernelZero
                    project_root = current_file.parent.parent.parent
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))
                    from examples.format_prompt.hardware.gpu_specs import GPU_SPEC_INFO
                    from examples.format_prompt.examples import softmax_example
                except ImportError as e:
                    # 如果导入失败，使用默认值
                    print(f"Warning: Failed to import GPU_SPEC_INFO or softmax_example: {e}")
                    GPU_SPEC_INFO = {}
                    softmax_example = ""
                
                # 按照 fs-eval 的方式构造模板参数
                prompt_str = format_prompt.render(
                    GPU_name="A100-80GB",
                    GPU_list=GPU_SPEC_INFO,
                    examples=softmax_example,
                    py_code=py_code,
                    GPU_info_format="bullets",
                )
            else:
                # 如果没有 raw_problem，尝试使用 prompt_key
                prompt_str = example.get(self.prompt_key, "")
        else:
            # 非 kernelbench 格式，从 prompt_key 获取 prompt_str
            prompt_str: str = example.get(self.prompt_key, "")

        if self.format_prompt and "questioner_format_with_persona" in self.format_prompt:
            print("load personas")
            return [
                {
                    "role": "system",
                    "content": (
                        f"You are {random.choice(self.personas)}.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                        "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                        "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging reasoning question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        if self.format_prompt and "questioner_format" in self.format_prompt:
            # print('detected questioner_format')
            return [
                {
                    "role": "system",
                    "content": (
                        "You are an expert competition-math problem setter.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                        "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                        "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging reasoning question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        if self.format_prompt and "solver_format" in self.format_prompt:
            return [{"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},{"role": "user", "content": prompt_str}]
        if self.format_prompt:
            # kernelbench 模板（包含 "PyTorch and CUDA"）已经在前面处理过了，这里处理其他模板
            if "PyTorch and CUDA" not in self.format_prompt:
                format_prompt = Template(self.format_prompt.strip())
                prompt_str = format_prompt.render(content=prompt_str)
        
        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        if self.tokenizer.chat_template:
            return (
                len(processing_class.apply_chat_template(messages, add_generation_prompt=True)) <= self.max_prompt_length
            )
        else:
            return (
                len("system: " + messages[0]["content"] + '\n' + "user: " + messages[1]["content"]) <= self.max_prompt_length
            )
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            raw_image_data = example.pop(self.image_key)
            images = [
                process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                for image in raw_image_data
            ]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": raw_image_data}
        else:
            if self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                prompt = "system: " + messages[0]["content"] + '\n' + "user: " + messages[1]["content"]
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        # 保存原始数据集索引，用于全局索引追踪
        example["dataset_index"] = index
        
        # 设置 ground_truth：如果是 kernelbench 格式（包含 "PyTorch and CUDA"），使用 raw_problem；否则使用 answer_key
        if self.format_prompt and "PyTorch and CUDA" in self.format_prompt:
            # 对于 kernelbench，ground_truth 应该是 raw_problem（torch 代码）
            if "raw_problem" in example:
                example["ground_truth"] = example["raw_problem"]
            else:
                # 如果没有 raw_problem，尝试使用 answer_key
                example["ground_truth"] = example.pop(self.answer_key, "")
        else:
            example["ground_truth"] = example.pop(self.answer_key)
        
        return example
