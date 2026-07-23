import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

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
                f"Sample {_score_counter}: {error_msg} - formated=0.0, compiled=0.0, passed=False, speedup=N/A",
                file=sys.stderr,
                flush=True,
            )
            return

        if error:
            print(
                f"Sample {_score_counter}: ERROR - {error} | formated=0.0, compiled=0.0, passed=False, speedup=N/A",
                file=sys.stderr,
                flush=True,
            )
            return

        if result:
            formated = result.get("formated", 0.0)
            compiled = result.get("compiled", 0.0)
            passed = result.get("passed", False)
            speedup = result.get("msg", {}).get("speedup", -1)
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            sample_error = result.get("error_msg")
            if not passed and sample_error:
                error_display = sample_error[:200] + "..." if len(sample_error) > 200 else sample_error
                print(
                    f"Sample {_score_counter}: formated={formated}, compiled={compiled}, "
                    f"passed={passed}, speedup={speedup_str}, error={error_display}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"Sample {_score_counter}: formated={formated}, compiled={compiled}, "
                    f"passed={passed}, speedup={speedup_str}",
                    file=sys.stderr,
                    flush=True,
                )


def extract_code_block(text: str) -> str:
    codeblock_seps = ["python", "triton"]
    languages_pattern = "|".join(map(re.escape, codeblock_seps))
    pattern = re.compile(rf"```(?:{languages_pattern})\n(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].group(1).rstrip()
    return "CANNOT EXTRACT CODE"


def _parse_speedup(msg: Any) -> float:
    if isinstance(msg, dict):
        try:
            return float(msg.get("speedup", -1))
        except (TypeError, ValueError):
            return -1
    if isinstance(msg, str):
        speedup_match = re.search(
            r"Torch time:\s*[\d.]+\s*s,\s*(?:CUDA|Triton) time:\s*[\d.]+\s*s,\s*Speedup:\s*([\d.]+)x",
            msg,
        )
        if speedup_match:
            try:
                return float(speedup_match.group(1))
            except (TypeError, ValueError):
                return -1
    return -1


def _extract_error(res_json: Dict[str, Any], msg: Any) -> Optional[str]:
    error_msg = res_json.get("error")
    if error_msg is not None:
        return error_msg

    if isinstance(msg, str):
        lower_msg = msg.lower()
        if any(keyword in lower_msg for keyword in ["error", "failed", "exception", "traceback"]):
            return msg
        if "speedup" not in lower_msg and "torch time" not in lower_msg:
            return msg
    elif isinstance(msg, dict):
        return msg.get("error")
    return None


def _build_eval_payload(triton_code: str, torch_code: str) -> Dict[str, str]:
    code_field = os.environ.get("TRITON_EVAL_CODE_FIELD", "triton_code")
    payload = {
        code_field: triton_code,
        "torch_code": torch_code,
        "backend": "triton",
    }

    if os.environ.get("TRITON_EVAL_INCLUDE_CUDA_ALIAS", "0") == "1":
        payload["cuda_code"] = triton_code

    return payload


def _compute_single_score(args: Tuple[str, str]) -> Dict[str, Any]:
    solution_str, ground_truth = args
    triton_code = extract_code_block(solution_str)

    if triton_code == "CANNOT EXTRACT CODE":
        result = {
            "formated": 0.0,
            "compiled": 0.0,
            "passed": False,
            "msg": {"speedup": -1},
            "error_msg": "CANNOT EXTRACT CODE",
        }
        _log_sample_result(error_msg="CANNOT EXTRACT CODE")
        return result

    gpu_ip = os.environ.get("TRITON_SERVER_IP", os.environ.get("CUDA_SERVER_IP", "10.200.240.10"))
    endpoint = os.environ.get("TRITON_EVAL_ENDPOINT", "/compute_score")
    timeout = float(os.environ.get("TRITON_EVAL_TIMEOUT", os.environ.get("CUDA_EVAL_TIMEOUT", "300")))

    try:
        response = requests.post(
            url=f"http://{gpu_ip}:8000{endpoint}",
            data=_build_eval_payload(triton_code, ground_truth),
            proxies={"http": None, "https": None},
            timeout=timeout,
        )
        response.raise_for_status()
        res_json = response.json()

        msg = res_json.get("msg", {})
        passed = res_json.get("passed", False)
        result = {
            "formated": 1.0 if res_json.get("formated", False) else 0.0,
            "compiled": 1.0 if res_json.get("compiled", False) else 0.0,
            "passed": passed,
            "msg": {"speedup": _parse_speedup(msg)},
            "error_msg": None if passed else _extract_error(res_json, msg),
        }
        _log_sample_result(result=result)
        return result
    except Exception as e:
        result = {
            "formated": 0.0,
            "compiled": 0.0,
            "passed": False,
            "msg": {"speedup": -1},
            "error_msg": str(e),
        }
        _log_sample_result(error=e)
        return result


def compute_score(
    solution_strs: List[str],
    ground_truths: List[str],
    index: List[int],
    dataset_index: Optional[List[int]] = None,
    alpha: float = 1.1,
    penalty_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    alpha = float(os.environ.get("REWARD_ALPHA", alpha))
    penalty_weight = float(os.environ.get("REWARD_PENALTY_WEIGHT", penalty_weight))
    max_workers = int(os.environ.get("REWARD_MAX_WORKERS", "128"))

    global _score_counter, _score_lock
    with _score_lock:
        _score_counter = 0

    total_samples = len(solution_strs)
    print(f"Computing Triton scores for {total_samples} samples with {max_workers} workers...", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_compute_single_score, zip(solution_strs, ground_truths)))

    passed = [result["passed"] for result in results]
    speedup = [result["msg"].get("speedup", -1) for result in results]
    formated = [result["formated"] for result in results]
    compiled = [result["compiled"] for result in results]

    group_ids = []
    grouped_data: Dict[Any, Dict[str, List[Any]]] = {}
    group_dataset_index = {}
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
            group_dataset_index[idx] = dataset_index[i] if dataset_index is not None and i < len(dataset_index) else -1
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
        num_correct = sum(1 for score in group_passed if score)

        if num_correct < alpha * group_rollout_n:
            scores = [
                f * 0.1 + c * 0.1 + (1.0 if p else 0.0)
                for f, c, p in zip(group_formated, group_compiled, group_passed)
            ]
        else:
            correct_speedup = [s for p, s in zip(group_passed, group_speedup) if p and s > 0]
            if correct_speedup:
                speedup_max = np.max(correct_speedup)
                speedup_min = np.min(correct_speedup)
            else:
                speedup_max = speedup_min = 0.0

            scores = []
            for f, c, p, s in zip(group_formated, group_compiled, group_passed, group_speedup):
                if p:
                    if speedup_max - speedup_min < 1e-6 or s <= 0:
                        linear_speedup_reward = 0.0
                    else:
                        linear_speedup_reward = (s - speedup_min) / (speedup_max - speedup_min)
                    score = f * 0.1 + c * 0.1 + 1.0 + linear_speedup_reward * penalty_weight
                else:
                    score = f * 0.1 + c * 0.1
                scores.append(float(score))

        id2scores[idx] = scores
        original_idx = group_dataset_index.get(idx, -1)
        prompt_label = original_idx if original_idx >= 0 else "?"
        print(
            f"[Prompt {prompt_label}] Scores: {scores}, Passed: {group_passed}, Speedup: {group_speedup}, "
            f"Mean: {np.mean(scores):.4f}, Max: {np.max(scores):.4f}, Min: {np.min(scores):.4f}",
            file=sys.stderr,
            flush=True,
        )

    response = []
    for idx in index:
        response.append({"overall": id2scores[idx].pop(0)})

    passed_array = np.array([1.0 if p else 0.0 for p in passed])
    passed_speedup = [s for p, s in zip(passed, speedup) if p and s > 0]
    avg_speedup = np.mean(passed_speedup) if passed_speedup else 0.0
    print(
        f"Overall Triton stats - passed: {np.mean(passed_array):.4f}, speedup (passed only): {avg_speedup:.4f}",
        file=sys.stderr,
        flush=True,
    )

    return response

# ------------------------------------------------------------------
# Fail-fast wrapper: abort the RL step when Keck infrastructure fails.
# ------------------------------------------------------------------
import json as _ff_json
import os as _ff_os
import time as _ff_time
import urllib.request as _ff_request

_original_compute_score = compute_score

_INFRA_MARKERS = (
    "connection refused",
    "connection error",
    "failed to establish",
    "newconnectionerror",
    "max retries exceeded",
    "server disconnected",
    "remote end closed",
    "connect timeout",
    "read timed out",
    "service unavailable",
)

def _require_keck_healthy():
    base_url = _ff_os.environ.get("KECK_BASE_URL", "").rstrip("/")
    if not base_url:
        host = _ff_os.environ.get("TRITON_SERVER_IP", "")
        if not host:
            raise RuntimeError("KECK_BASE_URL and TRITON_SERVER_IP are unset")
        base_url = f"http://{host}:8000"

    health_url = f"{base_url}/healthz"
    last_error = None

    for _ in range(3):
        try:
            request = _ff_request.Request(
                health_url,
                headers={"Connection": "close"},
            )
            with _ff_request.urlopen(request, timeout=3.0) as response:
                payload = _ff_json.loads(response.read().decode("utf-8"))

            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")

            if payload.get("ray_initialized") is not True:
                raise RuntimeError(f"invalid health response: {payload}")

            return
        except Exception as exc:
            last_error = exc
            _ff_time.sleep(1.0)

    raise RuntimeError(
        f"KECK_INFRASTRUCTURE_FAILURE: {health_url}: {last_error}"
    )

def compute_score(
    solution_strs,
    ground_truths,
    index,
    dataset_index=None,
    alpha=1.1,
    penalty_weight=1.0,
):
    _require_keck_healthy()

    result = _original_compute_score(
        solution_strs,
        ground_truths,
        index,
        dataset_index=dataset_index,
        alpha=alpha,
        penalty_weight=penalty_weight,
    )

    result_text = repr(result).lower()
    if any(marker in result_text for marker in _INFRA_MARKERS):
        raise RuntimeError(
            "KECK_INFRASTRUCTURE_FAILURE: connection error in reward result"
        )

    _require_keck_healthy()
    return result
