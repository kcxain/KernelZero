import os
import re
from typing import Dict, List

import requests


def extract_code_block(text: str) -> str:
    """
    从模型输出中提取代码块
    """
    # 支持 python 和 cuda 代码块
    codeblock_seps = ["python", "cuda"]
    languages_pattern = "|".join(map(re.escape, codeblock_seps))
    codeblock_start = f"```({languages_pattern})"
    # 要求代码块标记后必须有换行符，使用更严格的边界匹配
    pattern = re.compile(codeblock_start + r"\n(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)
    matches = list(pattern.finditer(text))
    
    if matches:
        # 返回最后一个代码块（通常是最终答案）
        last_match = matches[-1]
        code_content = last_match.group(2).rstrip()
        return code_content
    # 提取失败时返回特殊标记，而不是整个文本
    return "CANNOT EXTRACT CODE"


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
    
    for predict, torch_code in zip(predicts, ground_truths):
        # 提取代码块（从模型输出中提取 CUDA 代码）
        cuda_code = extract_code_block(predict)
        
        # 如果提取失败，直接标记为失败，不发送到评估 API
        if cuda_code == "CANNOT EXTRACT CODE":
            sample_idx = len(scores)
            if sample_idx % 1 == 0 or sample_idx == len(predicts) - 1:
                print(f"[Reward Sample {sample_idx}] ✗ EXTRACT_FAILED | Reward: 0.0000 | Cannot extract code from model output", 
                      file=__import__('sys').stderr, flush=True)
                # # 输出ground_truth（源代码）和模型推理结果
                # print(f"\n{'='*100}", file=__import__('sys').stderr, flush=True)
                # print(f"[Reward Sample {sample_idx}] Ground Truth (Source Code):", file=__import__('sys').stderr, flush=True)
                # print(f"{'-'*100}", file=__import__('sys').stderr, flush=True)
                # # 检查torch_code的类型和值
                # if torch_code is None:
                #     print("[DEBUG] torch_code is None", file=__import__('sys').stderr, flush=True)
                # elif isinstance(torch_code, str):
                #     if len(torch_code.strip()) == 0:
                #         print("[DEBUG] torch_code is empty string", file=__import__('sys').stderr, flush=True)
                #     else:
                #         print(torch_code, file=__import__('sys').stderr, flush=True)
                # else:
                #     print(f"[DEBUG] torch_code type: {type(torch_code)}, value: {torch_code}", file=__import__('sys').stderr, flush=True)
                #     print(str(torch_code), file=__import__('sys').stderr, flush=True)
                # print(f"\n[Reward Sample {sample_idx}] Model Output:", file=__import__('sys').stderr, flush=True)
                # print(f"{'-'*100}", file=__import__('sys').stderr, flush=True)
                # if predict is None:
                #     print("[DEBUG] predict is None", file=__import__('sys').stderr, flush=True)
                # elif isinstance(predict, str):
                #     if len(predict.strip()) == 0:
                #         print("[DEBUG] predict is empty string", file=__import__('sys').stderr, flush=True)
                #     else:
                #         print(predict, file=__import__('sys').stderr, flush=True)
                # else:
                #     print(f"[DEBUG] predict type: {type(predict)}, value: {predict}", file=__import__('sys').stderr, flush=True)
                #     print(str(predict), file=__import__('sys').stderr, flush=True)
                # print(f"{'='*100}\n", file=__import__('sys').stderr, flush=True)
            scores.append({"overall": 0.0})
            continue
        
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
            if sample_idx % 1 == 0 or sample_idx == len(predicts) - 1:  # 每10个输出一次，或最后一个样本
                status = "✓ PASSED" if passed else "✗ FAILED"
                print(f"[Reward Sample {sample_idx}] {status} | Reward: {reward:.4f} | "
                      f"CUDA code length: {len(cuda_code)} chars | "
                      f"Extracted from predict length: {len(predict)} chars", 
                      file=__import__('sys').stderr, flush=True)
                
                # # 输出ground_truth（源代码）和模型推理结果
                # print(f"\n{'='*100}", file=__import__('sys').stderr, flush=True)
                # print(f"[Reward Sample {sample_idx}] Ground Truth (Source Code):", file=__import__('sys').stderr, flush=True)
                # print(f"{'-'*100}", file=__import__('sys').stderr, flush=True)
                # # 检查torch_code的类型和值
                # if torch_code is None:
                #     print("[DEBUG] torch_code is None", file=__import__('sys').stderr, flush=True)
                # elif isinstance(torch_code, str):
                #     if len(torch_code.strip()) == 0:
                #         print("[DEBUG] torch_code is empty string", file=__import__('sys').stderr, flush=True)
                #     else:
                #         print(torch_code, file=__import__('sys').stderr, flush=True)
                # else:
                #     print(f"[DEBUG] torch_code type: {type(torch_code)}, value: {torch_code}", file=__import__('sys').stderr, flush=True)
                #     print(str(torch_code), file=__import__('sys').stderr, flush=True)
                # print(f"\n[Reward Sample {sample_idx}] Model Output:", file=__import__('sys').stderr, flush=True)
                # print(f"{'-'*100}", file=__import__('sys').stderr, flush=True)
                # if predict is None:
                #     print("[DEBUG] predict is None", file=__import__('sys').stderr, flush=True)
                # elif isinstance(predict, str):
                #     if len(predict.strip()) == 0:
                #         print("[DEBUG] predict is empty string", file=__import__('sys').stderr, flush=True)
                #     else:
                #         print(predict, file=__import__('sys').stderr, flush=True)
                # else:
                #     print(f"[DEBUG] predict type: {type(predict)}, value: {predict}", file=__import__('sys').stderr, flush=True)
                #     print(str(predict), file=__import__('sys').stderr, flush=True)
                # print(f"{'='*100}\n", file=__import__('sys').stderr, flush=True)
                
        except requests.exceptions.Timeout:
            sample_idx = len(scores)
            if sample_idx % 1 == 0 or sample_idx == len(predicts) - 1:  # 每10个输出一次，或最后一个样本
                print(f"[Reward Sample {sample_idx}] ✗ TIMEOUT | Reward: 0.0000 | API timeout: {gpu_ip}", 
                      file=__import__('sys').stderr, flush=True)
                # # 输出ground_truth（源代码）和模型推理结果
                # print(f"\n{'='*100}", file=__import__('sys').stderr, flush=True)
                # print(f"[Reward Sample {sample_idx}] Ground Truth (Source Code):", file=__import__('sys').stderr, flush=True)
                # print(f"{'-'*100}", file=__import__('sys').stderr, flush=True)
                # # 检查torch_code的类型和值
                # if torch_code is None:
                #     print("[DEBUG] torch_code is None", file=__import__('sys').stderr, flush=True)
                # elif isinstance(torch_code, str):
                #     if len(torch_code.strip()) == 0:
                #         print("[DEBUG] torch_code is empty string", file=__import__('sys').stderr, flush=True)
                #     else:
                #         print(torch_code, file=__import__('sys').stderr, flush=True)
                # else:
                #     print(f"[DEBUG] torch_code type: {type(torch_code)}, value: {torch_code}", file=__import__('sys').stderr, flush=True)
                #     print(str(torch_code), file=__import__('sys').stderr, flush=True)
                # print(f"\n[Reward Sample {sample_idx}] Model Output:", file=__import__('sys').stderr, flush=True)
                # print(f"{'-'*100}", file=__import__('sys').stderr, flush=True)
                # if predict is None:
                #     print("[DEBUG] predict is None", file=__import__('sys').stderr, flush=True)
                # elif isinstance(predict, str):
                #     if len(predict.strip()) == 0:
                #         print("[DEBUG] predict is empty string", file=__import__('sys').stderr, flush=True)
                #     else:
                #         print(predict, file=__import__('sys').stderr, flush=True)
                # else:
                #     print(f"[DEBUG] predict type: {type(predict)}, value: {predict}", file=__import__('sys').stderr, flush=True)
                #     print(str(predict), file=__import__('sys').stderr, flush=True)
                # print(f"{'='*100}\n", file=__import__('sys').stderr, flush=True)
            scores.append({"overall": 0.0})
        except requests.exceptions.RequestException as e:
            sample_idx = len(scores)
            if sample_idx % 1 == 0 or sample_idx == len(predicts) - 1:  # 每10个输出一次，或最后一个样本
                print(f"[Reward Sample {sample_idx}] ✗ API_ERROR | Reward: 0.0000 | Error: {e}", 
                      file=__import__('sys').stderr, flush=True)
                # # 输出ground_truth（源代码）和模型推理结果
                # print(f"\n{'='*100}", file=__import__('sys').stderr, flush=True)
                # print(f"[Reward Sample {sample_idx}] Ground Truth (Source Code):", file=__import__('sys').stderr, flush=True)
                # print(f"{'-'*100}", file=__import__('sys').stderr, flush=True)
                # # 检查torch_code的类型和值
                # if torch_code is None:
                #     print("[DEBUG] torch_code is None", file=__import__('sys').stderr, flush=True)
                # elif isinstance(torch_code, str):
                #     if len(torch_code.strip()) == 0:
                #         print("[DEBUG] torch_code is empty string", file=__import__('sys').stderr, flush=True)
                #     else:
                #         print(torch_code, file=__import__('sys').stderr, flush=True)
                # else:
                #     print(f"[DEBUG] torch_code type: {type(torch_code)}, value: {torch_code}", file=__import__('sys').stderr, flush=True)
                #     print(str(torch_code), file=__import__('sys').stderr, flush=True)
                # print(f"\n[Reward Sample {sample_idx}] Model Output:", file=__import__('sys').stderr, flush=True)
                # print(f"{'-'*100}", file=__import__('sys').stderr, flush=True)
                # if predict is None:
                #     print("[DEBUG] predict is None", file=__import__('sys').stderr, flush=True)
                # elif isinstance(predict, str):
                #     if len(predict.strip()) == 0:
                #         print("[DEBUG] predict is empty string", file=__import__('sys').stderr, flush=True)
                #     else:
                #         print(predict, file=__import__('sys').stderr, flush=True)
                # else:
                #     print(f"[DEBUG] predict type: {type(predict)}, value: {predict}", file=__import__('sys').stderr, flush=True)
                #     print(str(predict), file=__import__('sys').stderr, flush=True)
                # print(f"{'='*100}\n", file=__import__('sys').stderr, flush=True)
            scores.append({"overall": 0.0})
        except Exception as e:
            sample_idx = len(scores)
            if sample_idx % 1 == 0 or sample_idx == len(predicts) - 1:  # 每10个输出一次，或最后一个样本
                print(f"[Reward Sample {sample_idx}] ✗ UNKNOWN_ERROR | Reward: 0.0000 | Error: {e}", 
                      file=__import__('sys').stderr, flush=True)
                # # 输出ground_truth（源代码）和模型推理结果
                # print(f"\n{'='*100}", file=__import__('sys').stderr, flush=True)
                # print(f"[Reward Sample {sample_idx}] Ground Truth (Source Code):", file=__import__('sys').stderr, flush=True)
                # print(f"{'-'*100}", file=__import__('sys').stderr, flush=True)
                # print(torch_code, file=__import__('sys').stderr, flush=True)
                # print(f"\n[Reward Sample {sample_idx}] Model Output:", file=__import__('sys').stderr, flush=True)
                # print(f"{'-'*100}", file=__import__('sys').stderr, flush=True)
                # if predict is None:
                #     print("[DEBUG] predict is None", file=__import__('sys').stderr, flush=True)
                # elif isinstance(predict, str):
                #     if len(predict.strip()) == 0:
                #         print("[DEBUG] predict is empty string", file=__import__('sys').stderr, flush=True)
                #     else:
                #         print(predict, file=__import__('sys').stderr, flush=True)
                # else:
                #     print(f"[DEBUG] predict type: {type(predict)}, value: {predict}", file=__import__('sys').stderr, flush=True)
                #     print(str(predict), file=__import__('sys').stderr, flush=True)
                # print(f"{'='*100}\n", file=__import__('sys').stderr, flush=True)
            scores.append({"overall": 0.0})
    
    return scores

