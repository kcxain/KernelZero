"""Nested frontier reward for training the KernelZero Proposer.

Each Proposer rollout is a PyTorch module.  The reward service renders the
exact Triton Coder prompt used during Coder RL, samples a group of Triton
implementations from a remote vLLM server, validates every implementation with
Keck, and rewards modules close to the Coder's correctness frontier.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
_print_lock = threading.Lock()
_prompt_template: Optional[str] = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _without_proxy() -> Dict[str, None]:
    return {"http": None, "https": None}


def _extract_python(text: str) -> Optional[str]:
    matches = re.findall(r"```(?:python|py)?\s*\n(.*?)(?:\n```|$)", text, re.DOTALL | re.IGNORECASE)
    if matches:
        code = matches[-1].strip()
        return code or None

    stripped = text.strip()
    if "class Model" in stripped:
        return stripped
    return None


def _extract_triton(text: str) -> Optional[str]:
    matches = re.findall(
        r"```(?:python|py|triton)?\s*\n(.*?)(?:\n```|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if matches:
        code = matches[-1].strip()
        return code or None

    stripped = text.strip()
    if "class ModelNew" in stripped and ("triton" in stripped or "@triton.jit" in stripped):
        return stripped
    return None


def _load_prompt_template() -> str:
    global _prompt_template
    if _prompt_template is not None:
        return _prompt_template

    default_path = Path(__file__).resolve().parents[1] / "format_prompt" / "kernelbench_triton_cot_grpo.jinja"
    prompt_path = Path(os.environ.get("CODER_PROMPT_PATH", str(default_path)))
    text = prompt_path.read_text(encoding="utf-8")
    if "{{ py_code }}" not in text:
        raise ValueError(f"Coder prompt is missing '{{{{ py_code }}}}': {prompt_path}")
    _prompt_template = text
    return _prompt_template


def _request_json(
    method: str,
    url: str,
    *,
    retries: int,
    timeout: float,
    **kwargs: Any,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.request(
                method,
                url,
                timeout=timeout,
                proxies=_without_proxy(),
                **kwargs,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError(f"Expected JSON object from {url}")
            return result
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 5))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}: {last_error}")


def _check_module(torch_code: str) -> bool:
    if not _env_bool("PROPOSER_VALIDATE_MODULE", True):
        return True

    keck_url = os.environ["KECK_BASE_URL"].rstrip("/")
    result = _request_json(
        "POST",
        f"{keck_url}/check_torch",
        retries=int(os.environ.get("SERVICE_RETRIES", "3")),
        timeout=float(os.environ.get("KECK_TIMEOUT", "300")),
        data={"gt_torch": torch_code, "to_check_torch": torch_code},
    )
    return bool(result.get("passed", False))


def _sample_coder(torch_code: str) -> List[str]:
    coder_url = os.environ["CODER_VLLM_URL"].rstrip("/")
    coder_model = os.environ.get("CODER_MODEL_NAME", "kernelzero-triton-step80")
    rollout_n = int(os.environ.get("CODER_ROLLOUT_N", "5"))
    prompt = _load_prompt_template().replace("{{ py_code }}", torch_code)

    payload: Dict[str, Any] = {
        "model": coder_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(os.environ.get("CODER_TEMPERATURE", "0.6")),
        "top_p": float(os.environ.get("CODER_TOP_P", "0.95")),
        "top_k": int(os.environ.get("CODER_TOP_K", "20")),
        "max_tokens": int(os.environ.get("CODER_MAX_TOKENS", "16384")),
        "n": rollout_n,
    }
    result = _request_json(
        "POST",
        f"{coder_url}/v1/chat/completions",
        retries=int(os.environ.get("SERVICE_RETRIES", "3")),
        timeout=float(os.environ.get("CODER_TIMEOUT", "1200")),
        json=payload,
    )

    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != rollout_n:
        raise RuntimeError(f"Coder returned {0 if not isinstance(choices, list) else len(choices)} choices, expected {rollout_n}")

    outputs: List[str] = []
    for choice in choices:
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content", "")
        if not isinstance(content, str):
            content = ""
        outputs.append(content)
    return outputs


def _keck_passed(triton_response: str, torch_code: str) -> bool:
    triton_code = _extract_triton(triton_response)
    if triton_code is None:
        return False

    keck_url = os.environ["KECK_BASE_URL"].rstrip("/")
    result = _request_json(
        "POST",
        f"{keck_url}/compute_score",
        retries=int(os.environ.get("SERVICE_RETRIES", "3")),
        timeout=float(os.environ.get("KECK_TIMEOUT", "300")),
        data={"triton_code": triton_code, "torch_code": torch_code},
    )
    msg = result.get("msg", "")
    if isinstance(msg, str) and "service error" in msg.lower():
        raise RuntimeError(f"Keck service error: {msg}")
    return bool(result.get("passed", False))


def _frontier_reward(accuracy: float) -> float:
    target = float(os.environ.get("PROPOSER_FRONTIER_TARGET", "0.5"))
    return max(0.0, 1.0 - 2.0 * abs(accuracy - target))


def _score_one(solution_str: str) -> Dict[str, float]:
    torch_code = _extract_python(solution_str)
    if torch_code is None:
        return {
            "overall": float(os.environ.get("PROPOSER_INVALID_REWARD", "-1.0")),
            "validity": 0.0,
            "coder_accuracy": 0.0,
        }

    if not _check_module(torch_code):
        return {
            "overall": float(os.environ.get("PROPOSER_INVALID_REWARD", "-1.0")),
            "validity": 0.0,
            "coder_accuracy": 0.0,
        }

    coder_outputs = _sample_coder(torch_code)
    keck_workers = min(len(coder_outputs), int(os.environ.get("KECK_WORKERS_PER_MODULE", "5")))
    with ThreadPoolExecutor(max_workers=max(1, keck_workers)) as executor:
        passed = list(executor.map(lambda output: _keck_passed(output, torch_code), coder_outputs))

    accuracy = sum(bool(item) for item in passed) / len(passed)
    return {
        "overall": float(_frontier_reward(accuracy)),
        "validity": 1.0,
        "coder_accuracy": float(accuracy),
    }


def compute_score(
    solution_strs: List[str],
    ground_truths: List[str],
    index: List[int],
    dataset_index: Optional[List[int]] = None,
) -> List[Dict[str, float]]:
    """Compute one nested frontier reward for every Proposer rollout."""

    del ground_truths
    del index
    del dataset_index

    required = ["CODER_VLLM_URL", "KECK_BASE_URL"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    max_workers = int(os.environ.get("PROPOSER_REWARD_MAX_WORKERS", "8"))
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        scores = list(executor.map(_score_one, solution_strs))

    valid = [score["validity"] for score in scores]
    accuracies = [score["coder_accuracy"] for score in scores if score["validity"] > 0]
    rewards = [score["overall"] for score in scores]
    with _print_lock:
        print(
            "Proposer frontier batch: "
            f"samples={len(scores)}, valid_rate={sum(valid) / max(1, len(valid)):.4f}, "
            f"coder_accuracy={sum(accuracies) / max(1, len(accuracies)):.4f}, "
            f"reward={sum(rewards) / max(1, len(rewards)):.4f}",
            file=sys.stderr,
            flush=True,
        )
    return scores
