import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests
from distill.prompt import distill_optimize_prompt

_score_counter = 0
_score_lock = threading.Lock()


def _log_sample_result(
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Exception] = None,
    error_msg: Optional[str] = None,
) -> None:
    global _score_counter, _score_lock

    with _score_lock:
        _score_counter += 1
        if _score_counter % 10 != 0:
            return

        if error_msg:
            print(
                f"Sample {_score_counter}: {error_msg} - coder_passed=False",
                file=sys.stderr,
                flush=True,
            )
            return

        if error:
            print(
                f"Sample {_score_counter}: ERROR - {error} | coder_passed=False",
                file=sys.stderr,
                flush=True,
            )
            return

        if result:
            coder_passed = result.get("coder_passed", False)
            error_text = result.get("error_msg")
            if error_text:
                error_text = error_text[:200] + "..." if len(error_text) > 200 else error_text
                print(
                    f"Sample {_score_counter}: coder_passed={coder_passed}, error={error_text}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"Sample {_score_counter}: coder_passed={coder_passed}",
                    file=sys.stderr,
                    flush=True,1
                )


def extract_torch_problem(text: str) -> str:
    codeblock_pattern = re.compile(r"```(?:python|py)?\n(.*?)(?:\n```|$)", re.DOTALL)
    code_matches = list(codeblock_pattern.finditer(text))
    if code_matches:
        code = code_matches[-1].group(1).strip()
        if code:
            return code

    question_matches = re.findall(r"<question>(.*?)</question>", text, re.DOTALL)
    if question_matches:
        question = question_matches[-1].strip()
        if question:
            return question

    stripped = text.strip()
    return stripped if stripped else "CANNOT EXTRACT TORCH PROBLEM"


def extract_cuda_code(text: str) -> str:
    pattern = re.compile(r"```(?:cuda|cpp|c\+\+|python)?\n(.*?)(?:\n```|$)", re.DOTALL)
    matches = list(pattern.finditer(text))
    if matches:
        code = matches[-1].group(1).strip()
        if code:
            return code
    stripped = text.strip()
    return stripped if stripped else "CANNOT EXTRACT CUDA CODE"


def _call_coder(torch_problem: str) -> str:
    base_url = os.environ.get("CODER_VLLM_URL", "http://127.0.0.1:8000")
    model_name = os.environ.get("CODER_MODEL_NAME")
    timeout = float(os.environ.get("CODER_TIMEOUT", "300"))
    user_prompt = _build_coder_prompt(torch_problem)

    payload: Dict[str, Any] = {
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(os.environ.get("CODER_TEMPERATURE", "0.0")),
        "top_p": float(os.environ.get("CODER_TOP_P", "1.0")),
        "max_tokens": int(os.environ.get("CODER_MAX_TOKENS", "4096")),
    }
    if model_name:
        payload["model"] = model_name

    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=timeout,
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices", [])
    if not choices:
        raise ValueError("coder response missing choices")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        content = "".join(text_parts)

    if not isinstance(content, str) or not content.strip():
        raise ValueError("coder response content is empty")
    return content


def _build_coder_prompt(torch_problem: str) -> str:
    custom_template = os.environ.get("CODER_PROMPT_TEMPLATE")
    if custom_template:
        return custom_template.format(torch_problem=torch_problem)
    return distill_optimize_prompt.replace("__py_code_to_be_fill__", torch_problem)


def _evaluate_cuda(cuda_code: str, torch_problem: str) -> Dict[str, Any]:
    gpu_ip = os.environ.get("CUDA_SERVER_IP", "10.200.240.10")
    response = requests.post(
        url=f"http://{gpu_ip}:8000/compute_score",
        data={
            "cuda_code": cuda_code,
            "torch_code": torch_problem,
        },
        timeout=float(os.environ.get("CUDA_EVAL_TIMEOUT", "300")),
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    return response.json()


def _compute_single_score(solution_str: str) -> Dict[str, Any]:
    torch_problem = extract_torch_problem(solution_str)
    if torch_problem == "CANNOT EXTRACT TORCH PROBLEM":
        result = {
            "coder_passed": False,
            "error_msg": "CANNOT EXTRACT TORCH PROBLEM",
        }
        _log_sample_result(error_msg=result["error_msg"])
        return result

    try:
        coder_response = _call_coder(torch_problem)
        cuda_code = extract_cuda_code(coder_response)
        if cuda_code == "CANNOT EXTRACT CUDA CODE":
            result = {
                "coder_passed": False,
                "error_msg": "CANNOT EXTRACT CUDA CODE",
            }
            _log_sample_result(result=result)
            return result

        eval_result = _evaluate_cuda(cuda_code, torch_problem)
        passed = bool(eval_result.get("passed", False))
        error_msg = eval_result.get("error")
        if error_msg is None:
            msg = eval_result.get("msg")
            if isinstance(msg, dict):
                error_msg = msg.get("error")
            elif isinstance(msg, str) and "speedup" not in msg.lower():
                error_msg = msg

        result = {
            "coder_passed": passed,
            "error_msg": error_msg,
        }
        _log_sample_result(result=result)
        return result
    except Exception as exc:
        result = {
            "coder_passed": False,
            "error_msg": str(exc),
        }
        _log_sample_result(error=exc)
        return result


def _accuracy_to_reward(accuracy: float) -> float:
    return max(0.0, 1.0 - 2.0 * abs(accuracy - 0.5))


def compute_score(
    solution_strs: List[str],
    ground_truths: List[str],
    index: List[int],
    dataset_index: Optional[List[int]] = None,
    alpha: float = 0.5,
    penalty_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    del ground_truths
    del index
    del dataset_index
    del alpha
    del penalty_weight

    max_workers = int(os.environ.get("REWARD_MAX_WORKERS", "64"))

    global _score_counter, _score_lock
    with _score_lock:
        _score_counter = 0

    total_samples = len(solution_strs)
    print(
        f"Computing proposer rewards for {total_samples} samples with {max_workers} workers...",
        file=sys.stderr,
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_compute_single_score, solution_strs))

    overall_accuracy = (
        sum(1.0 for item in results if item["coder_passed"]) / len(results) if results else 0.0
    )
    overall_reward = _accuracy_to_reward(overall_accuracy)

    print(
        f"Batch stats - coder_accuracy: {overall_accuracy:.4f}, proposer_reward: {overall_reward:.4f}",
        file=sys.stderr,
        flush=True,
    )

    rewards = [{"overall": float(overall_reward)} for _ in solution_strs]
    print(
        f"Overall stats - coder_accuracy: {overall_accuracy:.4f}, proposer_reward: {overall_reward:.4f}",
        file=sys.stderr,
        flush=True,
    )
    return rewards
