import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional

import requests
import numpy as np

# 用于线程安全的计数器和锁
_score_counter = 0
_score_lock = threading.Lock()


def _log_sample_result(result: Dict[str, Any] = None, error: Exception = None, error_msg: str = None):
    """
    线程安全地更新计数器并每10个样本输出一次日志
    
    Args:
        result: 样本结果字典，包含 formated, compiled, passed, msg
        error: 异常对象（如果发生错误）
        error_msg: 错误消息字符串（用于特殊情况，如提取失败）
    """
    global _score_counter, _score_lock
    
    with _score_lock:
        _score_counter += 1
        if _score_counter % 10 == 0:
            if error_msg:
                # 提取失败的情况
                print(
                    f"Sample {_score_counter}: {error_msg} - formated=0.0, compiled=0.0, passed=False, speedup=N/A",
                    file=sys.stderr, flush=True
                )
            elif error:
                # 异常情况
                print(
                    f"Sample {_score_counter}: ERROR - {error} | formated=0.0, compiled=0.0, passed=False, speedup=N/A",
                    file=sys.stderr, flush=True
                )
            elif result:
                # 正常情况
                formated = result.get("formated", 0.0)
                compiled = result.get("compiled", 0.0)
                passed = result.get("passed", False)
                speedup = result.get("msg", {}).get("speedup", -1)
                speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
                print(
                    f"Sample {_score_counter}: formated={formated}, compiled={compiled}, "
                    f"passed={passed}, speedup={speedup_str}",
                    file=sys.stderr, flush=True
                )


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


def _compute_single_score(args: Tuple[str, str]) -> Dict[str, Any]:
    """
    计算单个样本的得分
    
    Args:
        args: (solution_str, ground_truth) 元组
        
    Returns:
        Dict包含: formated, compiled, passed, msg
    """
    solution_str, ground_truth = args
    
    # 提取代码块
    cuda_code = extract_code_block(solution_str)
    
    # 如果提取失败
    if cuda_code == "CANNOT EXTRACT CODE":
        result = {
            "formated": 0.0,
            "compiled": 0.0,
            "passed": False,
            "msg": {"speedup": -1}
        }
        

        _log_sample_result(error_msg="CANNOT EXTRACT CODE")
        
        return result
    
    # 调用 API 评估
    gpu_ip = os.environ.get("CUDA_SERVER_IP", "10.200.240.10")
    try:
        response = requests.post(
            url=f"http://{gpu_ip}:8000/compute_score",
            data={
                "cuda_code": cuda_code,
                "torch_code": ground_truth,
            },
            proxies={"http": None, "https": None},
        )
        response.raise_for_status()
        res_json = response.json()
        
        # 提取结果
        formated = 1.0 if res_json.get("formated", False) else 0.0
        compiled = 1.0 if res_json.get("compiled", False) else 0.0
        passed = res_json.get("passed", False)
        
        # 提取speedup
        msg = res_json.get("msg", {})
        if isinstance(msg, dict):
            speedup = msg.get("speedup", -1)
        elif isinstance(msg, str):
            # 从字符串中解析speedup
            speedup_match = re.search(
                r"Torch time:\s*[\d.]+\s*s,\s*CUDA time:\s*[\d.]+\s*s,\s*Speedup:\s*([\d.]+)x",
                msg,
            )
            if speedup_match:
                try:
                    speedup = float(speedup_match.group(1))
                except (ValueError, TypeError):
                    speedup = -1
            else:
                speedup = -1
        else:
            speedup = -1
        
        result = {
            "formated": formated,
            "compiled": compiled,
            "passed": passed,
            "msg": {"speedup": speedup}
        }
        

        _log_sample_result(result=result)
        
        return result
        
    except Exception as e:
        result = {
            "formated": 0.0,
            "compiled": 0.0,
            "passed": False,
            "msg": {"speedup": -1}
        }
        

        _log_sample_result(error=e)
        
        return result


def compute_score(
    solution_strs: List[str], 
    ground_truths: List[str], 
    index: List[int],
    dataset_index: Optional[List[int]] = None,
    alpha: float = 0.5,
    penalty_weight: float = 1.0
) -> List[Dict[str, Any]]:
    """
    计算生成的 CUDA 代码的得分（带阈值和分组）
    
    Args:
        solution_strs: 模型生成的代码列表（包含 CUDA 代码）
        ground_truths: PyTorch 参考代码列表
        index: 分组索引列表（uid）
        dataset_index: 全局数据集索引列表，用于日志显示
        alpha: 正确样本比例阈值，默认0.5
        penalty_weight: speedup奖励权重，默认1.0
        
    Returns:
        List[Dict]: 每个样本的得分字典，包含score, formated, compiled, passed, speedup
    """
    # 从环境变量获取参数（如果未提供）
    alpha = float(os.environ.get("REWARD_ALPHA", alpha))
    penalty_weight = float(os.environ.get("REWARD_PENALTY_WEIGHT", penalty_weight))
    
    # 获取并发数
    max_workers = int(os.environ.get("REWARD_MAX_WORKERS", "128"))
    
    # 重置全局计数器
    global _score_counter, _score_lock
    with _score_lock:
        _score_counter = 0
    
    # 并行计算单个样本得分
    total_samples = len(solution_strs)
    print(f"Computing scores for {total_samples} samples with {max_workers} workers...", file=sys.stderr, flush=True)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        args_list = list(zip(solution_strs, ground_truths))
        results = list(executor.map(_compute_single_score, args_list))

    passed = [result["passed"] for result in results]
    speedup = [result["msg"].get("speedup", -1) for result in results]
    formated = [result["formated"] for result in results]
    compiled = [result["compiled"] for result in results]

    # 按index分组处理
    group_ids = []
    grouped_data = {}
    group_dataset_index = {}  # 保存每个组的全局索引
    for i, idx in enumerate(index):
        if idx not in group_ids:
            group_ids.append(idx)
        if idx not in grouped_data:
            grouped_data[idx] = {
                "formated": [],
                "compiled": [],
                "passed": [],
                "speedup": [],
            }
            # 保存该组的全局索引（使用第一个样本的索引）
            if dataset_index is not None and i < len(dataset_index):
                group_dataset_index[idx] = dataset_index[i]
            else:
                group_dataset_index[idx] = -1
        grouped_data[idx]["formated"].append(formated[i])
        grouped_data[idx]["compiled"].append(compiled[i])
        grouped_data[idx]["passed"].append(passed[i])
        grouped_data[idx]["speedup"].append(speedup[i])

    id2scores = {}
    for idx in group_ids:
        group_formated = grouped_data[idx]["formated"]
        group_compiled = grouped_data[idx]["compiled"]
        group_passed = grouped_data[idx]["passed"]
        group_speedup = grouped_data[idx]["speedup"]

        group_rollout_n = len(group_passed)

        # 统计正确样本的个数
        num_correct = sum(1 for score in group_passed if score)

        if num_correct < alpha * group_rollout_n:
            # 对于正确样本少于阈值的情况，直接使用正确性作为reward
            scores = []
            for f, c, p, s in zip(
                group_formated,
                group_compiled,
                group_passed,
                group_speedup,
            ):
                # 如果结果错了，还照抄buggy code，要惩罚
                scores.append(f * 0.1 + c * 0.1 + (1.0 if p else 0.0))
            id2scores[idx] = scores
        else:
            # 有足够正确样本，使用speedup进行奖励
            correct_indices = [i for i, score in enumerate(group_passed) if score]
            if correct_indices:
                correct_speedup = [group_speedup[i] for i in correct_indices]
                # 过滤掉无效的speedup值
                correct_speedup = [s for s in correct_speedup if s > 0]
                
                if len(correct_speedup) > 0:
                    speedup_max = np.max(correct_speedup)
                    speedup_min = np.min(correct_speedup)
                else:
                    speedup_max = speedup_min = 0.0
            else:
                speedup_max = speedup_min = 0.0

            scores = []
            for i, (f, c, p, s) in enumerate(
                zip(
                    group_formated,
                    group_compiled,
                    group_passed,
                    group_speedup,
                )
            ):
                if p:  # 正确的样本
                    # 使用最大最小值归一化speedup
                    if speedup_max - speedup_min < 1e-6 or s <= 0:
                        linear_speedup_reward = 0.0
                    else:
                        norm_speedup = (s - speedup_min) / (speedup_max - speedup_min)
                        linear_speedup_reward = norm_speedup  # 线性奖励

                    score = (
                        f * 0.1
                        + c * 0.1
                        + 1.0  # passed = 1.0
                        + linear_speedup_reward * penalty_weight
                    )
                else:
                    score = f * 0.1 + c * 0.1 + 0.0  # passed = 0.0
                scores.append(float(score))
            id2scores[idx] = scores

        # 获取全局索引用于日志显示
        global_idx = group_dataset_index.get(idx, -1)
        if global_idx >= 0:
            print(
                f"Question[{global_idx}]  Scores: {scores}, Formated: {group_formated}, Compiled: {group_compiled}, "
                f"Passed: {group_passed}, Speedup: {group_speedup}",
                file=sys.stderr, flush=True
            )
        else:
            print(
                f"Question[?]  Scores: {scores}, Formated: {group_formated}, Compiled: {group_compiled}, "
                f"Passed: {group_passed}, Speedup: {group_speedup}",
                file=sys.stderr, flush=True
            )
    
    # 构建返回结果
    res = []
    for i, idx in enumerate(index):
        score = id2scores[idx].pop(0)
        res.append({
            "overall": score
        })
    
    # 计算统计信息
    passed_array = np.array([1.0 if p else 0.0 for p in passed])
    # 只计算通过样本的 speedup
    passed_speedup = [s for p, s in zip(passed, speedup) if p and s > 0]
    if passed_speedup:
        avg_speedup = np.mean(passed_speedup)
    else:
        avg_speedup = 0.0
    print(f"Overall stats - passed: {np.mean(passed_array):.4f}, speedup (passed only): {avg_speedup:.4f}", 
          file=sys.stderr, flush=True)

    return res
