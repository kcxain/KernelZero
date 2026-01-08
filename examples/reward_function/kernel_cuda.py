import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any

import requests


class ThreadSafeCounter:
    """线程安全的计数器，用于控制日志输出频率"""
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()


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


def _evaluate_single_sample(
    sample_idx: int, predict: str, torch_code: str, gpu_ip: str, 
    log_counter: ThreadSafeCounter, log_lock: threading.Lock
) -> Tuple[int, Dict[str, float]]:
    """
    评估单个样本的 CUDA 代码
    
    Args:
        sample_idx: 样本索引
        predict: 模型生成的代码
        torch_code: PyTorch 参考代码
        gpu_ip: GPU 服务器 IP
        log_counter: 线程安全的日志计数器
        log_lock: 日志输出锁
        
    Returns:
        Tuple[int, Dict]: (样本索引, 得分字典)
    """
    # 提取代码块（从模型输出中提取 CUDA 代码）
    cuda_code = extract_code_block(predict)
    
    # 如果提取失败，直接标记为失败，不发送到评估 API
    if cuda_code == "CANNOT EXTRACT CODE":
        with log_lock:
            log_counter.value += 1
            current_count = log_counter.value
            should_log = (current_count % 10 == 0) or (current_count == 1)
            if should_log:
                print(f"[Reward Sample {sample_idx}] ✗ EXTRACT_FAILED | Reward: 0.0000 | Count: {current_count} | Cannot extract code from model output", 
                      file=__import__('sys').stderr, flush=True)
        return sample_idx, {"overall": 0.0}
    
    # 调用 API 评估
    try:
        response = requests.post(
            url=f"http://{gpu_ip}:8000/compute_score",
            data={
                "cuda_code": cuda_code,
                "torch_code": torch_code,
            },
            proxies={"http": None, "https": None},
        )
        response.raise_for_status()
        res_json = response.json()
        passed = res_json.get("passed", False)
        
        # 根据是否通过设置 reward
        # 如果通过：reward = 1 + speedup * 0.5
        # 如果失败：reward = 0.0
        speedup = 0.0
        if passed:
            # 从 API 返回中提取 speedup
            # 参考 keck/app/main.py 和 keck/app/helper.py
            # 当 passed=True 时，msg 可能是字典（包含 speedup）或字符串（需要解析）
            if "msg" in res_json:
                msg = res_json["msg"]
                # 方式1: 如果 msg 是字典，直接提取 speedup
                if isinstance(msg, dict) and "speedup" in msg:
                    try:
                        speedup = float(msg["speedup"])
                    except (ValueError, TypeError):
                        speedup = 0.0
                # 方式2: 如果 msg 是字符串，使用正则表达式解析
                elif isinstance(msg, str):
                    # 使用正则表达式从 msg 中提取 speedup: "Torch time: Xs, CUDA time: Ys, Speedup: Zx"
                    speedup_match = re.search(
                        r"Torch time:\s*[\d.]+\s*s,\s*CUDA time:\s*[\d.]+\s*s,\s*Speedup:\s*([\d.]+)x",
                        msg,
                    )
                    if speedup_match:
                        try:
                            speedup = float(speedup_match.group(1))
                        except (ValueError, TypeError):
                            speedup = 0.0
            
            # 限制 speedup 最大值为 2
            if speedup > 2.0:
                speedup = 2.0
            
            reward = 1.0 + speedup * 0.5
        else:
            reward = 0.0
        
        # 打印每个样本的评估结果（每10条输出一次）
        with log_lock:
            log_counter.value += 1
            current_count = log_counter.value
            should_log = (current_count % 10 == 0) or (current_count == 1)
            if should_log:
                status = "✓ PASSED" if passed else "✗ FAILED"
                speedup_str = f" | Speedup: {speedup:.2f}x" if passed and speedup > 0 else ""
                print(f"[Reward Sample {sample_idx}] {status} | Reward: {reward:.4f}{speedup_str} | Count: {current_count} | "
                      f"CUDA code length: {len(cuda_code)} chars | "
                      f"Extracted from predict length: {len(predict)} chars", 
                      file=__import__('sys').stderr, flush=True)
        
        return sample_idx, {"overall": reward}
        
    except requests.exceptions.Timeout:
        with log_lock:
            log_counter.value += 1
            current_count = log_counter.value
            should_log = (current_count % 10 == 0) or (current_count == 1)
            if should_log:
                print(f"[Reward Sample {sample_idx}] ✗ TIMEOUT | Reward: 0.0000 | Count: {current_count} | API timeout: {gpu_ip}", 
                      file=__import__('sys').stderr, flush=True)
        return sample_idx, {"overall": 0.0}
    except requests.exceptions.RequestException as e:
        with log_lock:
            log_counter.value += 1
            current_count = log_counter.value
            should_log = (current_count % 10 == 0) or (current_count == 1)
            if should_log:
                print(f"[Reward Sample {sample_idx}] ✗ API_ERROR | Reward: 0.0000 | Count: {current_count} | Error: {e}", 
                      file=__import__('sys').stderr, flush=True)
        return sample_idx, {"overall": 0.0}
    except Exception as e:
        with log_lock:
            log_counter.value += 1
            current_count = log_counter.value
            should_log = (current_count % 10 == 0) or (current_count == 1)
            if should_log:
                print(f"[Reward Sample {sample_idx}] ✗ UNKNOWN_ERROR | Reward: 0.0000 | Count: {current_count} | Error: {e}", 
                      file=__import__('sys').stderr, flush=True)
        return sample_idx, {"overall": 0.0}


def compute_score(predicts: List[str], ground_truths: List[str]) -> List[Dict[str, float]]:
    """
    计算生成的 CUDA 代码的得分（并行版本）
    
    参考 fs-eval/fs_eval/refine_sampler.py 的并行实现方式
    使用 ThreadPoolExecutor 并行调用 API 评估
    
    Reward 计算规则：
    - 如果 passed=True: reward = 1 + speedup * 0.5
    - 如果 passed=False: reward = 0.0
    
    speedup 从 API 返回中提取：
    - 优先从 res_json["result"]["speedup"] 提取（如果 result 是 dict）
    - 否则从 res_json["msg"] 字符串中通过正则表达式解析
    
    Args:
        predicts: 模型生成的代码列表（包含 CUDA 代码）
        ground_truths: PyTorch 参考代码列表（raw_problem 字段，即 torch_code）
        
    Returns:
        List[Dict]: 每个样本的得分字典，必须包含 overall 字段（reward 值）
    """
    gpu_ip = os.environ.get("CUDA_SERVER_IP", "10.200.240.10")
    print(f"[Reward Function] CUDA_SERVER_IP from env: {gpu_ip}", file=__import__('sys').stderr, flush=True)
    
    # 获取并发数，默认使用环境变量或默认值
    max_workers = int(os.environ.get("REWARD_MAX_WORKERS", "128"))
    print(f"[Reward Function] Using {max_workers} workers for parallel evaluation", 
          file=__import__('sys').stderr, flush=True)
    
    # 使用 ThreadPoolExecutor 并行执行
    results = {}  # 使用字典来保持顺序
    
    # 创建线程安全的日志计数器（用于控制每10条输出一次）
    log_counter = ThreadSafeCounter()
    log_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for idx, (predict, torch_code) in enumerate(zip(predicts, ground_truths)):
            future = executor.submit(
                _evaluate_single_sample, idx, predict, torch_code, gpu_ip, 
                log_counter, log_lock
            )
            futures.append((idx, future))
        
        # 收集结果（按完成顺序，但保持索引映射）
        for idx, future in futures:
            sample_idx, score = future.result()
            results[sample_idx] = score
    
    # 按原始顺序返回结果
    scores = [results[i] for i in range(len(predicts))]
    return scores

