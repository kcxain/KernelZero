import os
import re
from typing import Dict, List

import requests


def extract_code_block(text: str) -> str:
    """
    从模型输出中提取代码块
    """
    # 尝试提取 ```python ... ``` 格式的代码块
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # 返回最后一个代码块（通常是最终答案）
        return matches[-1].strip()
    # 如果没有代码块标记，返回整个文本
    return text.strip()


def compute_score(predicts: List[str], ground_truths: List[str]) -> List[Dict[str, float]]:
    """
    计算生成的 CUDA 代码的得分
    
    参考 fs-eval/fs_eval/testbench/kernel_eval/execution.py 的实现方式
    调用 API 评估，如果 passed 则 reward=1.0，否则 reward=0.0
    
    Args:
        predicts: 模型生成的代码列表（包含 CUDA 代码）
        ground_truths: PyTorch 参考代码列表（raw_problem 字段，即 torch_code）
        
    Returns:
        List[Dict]: 每个样本的得分字典，必须包含 overall 字段（reward 值）
    """
    scores = []
    gpu_ip = os.environ.get("CUDA_SERVER_IP", "10.200.240.10")
    print(f"[Reward Function] CUDA_SERVER_IP from env: {gpu_ip}", file=__import__('sys').stderr, flush=True)
    
    # 输出第 1 个样本的提取代码
    print(f"[Reward Function] Extracted code preview (first 10 samples):", file=__import__('sys').stderr, flush=True)
    for sample_idx, (predict, torch_code) in enumerate(zip(predicts[:1], ground_truths[:1])):
        # 提取代码块（从模型输出中提取 CUDA 代码）
        cuda_code = extract_code_block(predict)
        
        # 输出提取的代码（限制长度，避免日志过长）
        code_preview = cuda_code
        predict_preview = predict
        
        print(f"\n--- Sample {sample_idx} ---", file=__import__('sys').stderr, flush=True)
        print(f"Original predict :\n{predict_preview}", file=__import__('sys').stderr, flush=True)
        print(f"\nExtracted CUDA code :\n{code_preview}", file=__import__('sys').stderr, flush=True)
        print(f"CUDA code full length: {len(cuda_code)} chars", file=__import__('sys').stderr, flush=True)
    
    for predict, torch_code in zip(predicts, ground_truths):
        # 提取代码块（从模型输出中提取 CUDA 代码）
        cuda_code = extract_code_block(predict)
        
        # 调用 API 评估（参考 execution.py 的实现）
        try:
            response = requests.post(
                url=f"http://{gpu_ip}:8000/compute_score",
                data={
                    "cuda_code": cuda_code,
                    "torch_code": torch_code,  # raw_problem 就是 torch_code
                },
                proxies={"http": None, "https": None},
                timeout=120,  # 2分钟超时
            )
            response.raise_for_status()
            res_json = response.json()
            passed = res_json.get("passed", False)
            
            # 根据是否通过设置 reward: passed=1.0, not passed=0.0
            reward = 1.0 if passed else 0.0
            
            scores.append({
                "overall": reward,  # 这是实际使用的 reward 值
            })
            
            # 打印每个样本的评估结果（每10个样本输出一次，避免日志过多）
            sample_idx = len(scores) - 1
            if sample_idx % 10 == 0 or sample_idx == len(predicts) - 1:  # 每10个输出一次，或最后一个样本
                status = "✓ PASSED" if passed else "✗ FAILED"
                print(f"[Reward Sample {sample_idx}] {status} | Reward: {reward:.4f} | "
                      f"CUDA code length: {len(cuda_code)} chars | "
                      f"Extracted from predict length: {len(predict)} chars", 
                      file=__import__('sys').stderr, flush=True)
                
        except requests.exceptions.Timeout:
            sample_idx = len(scores)
            if sample_idx % 10 == 0 or sample_idx == len(predicts) - 1:  # 每10个输出一次，或最后一个样本
                print(f"[Reward Sample {sample_idx}] ✗ TIMEOUT | Reward: 0.0000 | API timeout: {gpu_ip}", 
                      file=__import__('sys').stderr, flush=True)
            scores.append({"overall": 0.0})
        except requests.exceptions.RequestException as e:
            sample_idx = len(scores)
            if sample_idx % 10 == 0 or sample_idx == len(predicts) - 1:  # 每10个输出一次，或最后一个样本
                print(f"[Reward Sample {sample_idx}] ✗ API_ERROR | Reward: 0.0000 | Error: {e}", 
                      file=__import__('sys').stderr, flush=True)
            scores.append({"overall": 0.0})
        except Exception as e:
            sample_idx = len(scores)
            if sample_idx % 10 == 0 or sample_idx == len(predicts) - 1:  # 每10个输出一次，或最后一个样本
                print(f"[Reward Sample {sample_idx}] ✗ UNKNOWN_ERROR | Reward: 0.0000 | Error: {e}", 
                      file=__import__('sys').stderr, flush=True)
            scores.append({"overall": 0.0})
    
    return scores

