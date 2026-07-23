"""Nested frontier reward for training the KernelZero Proposer.

Each Proposer rollout is a PyTorch module.  The reward service renders the
exact Triton Coder prompt used during Coder RL, samples a group of Triton
implementations from a remote vLLM server, validates every implementation with
Keck, and rewards modules close to the Coder's correctness frontier.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from kernelzero.proposer.quality_checks import quality_issues
from torch_evaluate.ast_checker import check_operators_used

_print_lock = threading.Lock()
_prompt_template: Optional[str] = None
_operator_maps: Optional[tuple[Dict[str, List[str]], Dict[int, List[str]]]] = None


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
    _raise_if_keck_infrastructure(result, "check_torch")
    return bool(result.get("passed", False))


def _raise_if_keck_infrastructure(result: Dict[str, Any], stage: str) -> None:
    """Never turn a Keck infrastructure failure into a negative sample reward."""
    msg = result.get("msg", "")
    if not isinstance(msg, str):
        return
    lowered = msg.lower()
    if "service error" in lowered or "infrastructure error" in lowered:
        raise RuntimeError(f"Keck {stage} infrastructure failure: {msg}")


def _load_operator_maps() -> tuple[Dict[str, List[str]], Dict[int, List[str]]]:
    """Load the required API list from the unchanged Proposer RL input file."""
    global _operator_maps
    if _operator_maps is not None:
        return _operator_maps

    data_path = os.environ.get("PROPOSER_TRAIN_DATA")
    if not data_path:
        raise RuntimeError("PROPOSER_TRAIN_DATA is required for API coverage checks")
    records = json.loads(Path(data_path).read_text(encoding="utf-8"))
    by_code: Dict[str, List[str]] = {}
    by_index: Dict[int, List[str]] = {}
    for position, record in enumerate(records):
        operators = [str(item) for item in record.get("operators", [])]
        if not operators:
            raise RuntimeError(f"Missing operators in Proposer record {position}")
        code = str(record.get("code", "")).strip()
        if code:
            by_code[code] = operators
        by_index[int(record.get("combination_index", position))] = operators
        by_index.setdefault(position, operators)
    _operator_maps = (by_code, by_index)
    return _operator_maps


def _required_operators(ground_truth: str, dataset_index: Optional[Any]) -> List[str]:
    by_code, by_index = _load_operator_maps()
    operators = by_code.get(str(ground_truth).strip())
    if operators:
        return operators
    if dataset_index is not None:
        try:
            operators = by_index.get(int(dataset_index))
        except (TypeError, ValueError):
            operators = None
    if not operators:
        raise RuntimeError("Unable to resolve required operators for a Proposer rollout")
    return operators


def _semantic_api_issues(torch_code: str, required_operators: List[str]) -> List[str]:
    """Return only high-confidence API-use failures from the dataset filter.

    The full dataset quality filter also enforces length, formatting, output
    dependence, and other style/structure policies.  Those are intentionally
    excluded from the RL reward to avoid rejecting otherwise valid modules.
    """
    issues = quality_issues(
        {
            "operators": required_operators,
            "code": torch_code,
            "raw_response": torch_code,
        }
    )
    graph_issues = [
        issue
        for issue in issues
        if issue.startswith("required_operator_not_in_forward_graph:")
        or issue.startswith("unused_required_result:")
    ]
    if not graph_issues:
        return []

    # The dataset checker only inspects Model.forward directly.  Preserve its
    # high-confidence failures, but do not reject a valid operator call placed
    # in a helper method that is transitively called by forward.
    return [
        issue
        for issue in graph_issues
        if not (
            issue.startswith("required_operator_not_in_forward_graph:")
            and _operator_used_on_reachable_path(
                torch_code, issue.split(":", 1)[1]
            )
        )
    ]


def _log_semantic_rejection(
    torch_code: str,
    required_operators: List[str],
    issues: List[str],
) -> None:
    """Persist rejected modules for false-positive audits without changing reward."""
    output = os.environ.get("PROPOSER_REJECT_LOG")
    if not output:
        return
    record = {
        "timestamp": time.time(),
        "required_operators": required_operators,
        "issues": issues,
        "torch_code": torch_code,
    }
    path = Path(output)
    with _print_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _import_aliases(tree: ast.Module) -> Dict[str, str]:
    aliases = {"torch": "torch", "nn": "torch.nn", "F": "torch.nn.functional"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name == "torch" or name.name.startswith("torch."):
                    aliases[name.asname or name.name.split(".")[0]] = name.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "torch" or node.module.startswith("torch."):
                for name in node.names:
                    if name.name != "*":
                        aliases[name.asname or name.name] = f"{node.module}.{name.name}"
    return aliases


def _canonical_name(name: str, aliases: Dict[str, str]) -> str:
    if not name:
        return ""
    head, separator, tail = name.partition(".")
    canonical_head = aliases.get(head)
    if canonical_head is None:
        return name
    return canonical_head + (separator + tail if separator else "")


def _operator_used_on_reachable_path(torch_code: str, operator: str) -> bool:
    """Conservatively recognize exact API use reachable from Model.forward."""
    try:
        tree = ast.parse(torch_code)
    except SyntaxError:
        return False

    model = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model"),
        None,
    )
    if model is None:
        return False
    methods = {
        node.name: node for node in model.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "forward" not in methods:
        return False

    aliases = _import_aliases(tree)
    reachable = {"forward"}
    pending = ["forward"]
    while pending:
        method = methods[pending.pop()]
        for call in (node for node in ast.walk(method) if isinstance(node, ast.Call)):
            name = _dotted_name(call.func)
            if name.startswith("self."):
                candidate = name.split(".", 1)[1]
                if candidate in methods and candidate not in reachable:
                    reachable.add(candidate)
                    pending.append(candidate)

    reachable_nodes = [methods[name] for name in reachable]
    reachable_calls = {
        _canonical_name(_dotted_name(call.func), aliases)
        for method in reachable_nodes
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
    }
    if operator in reachable_calls:
        return True

    # nn.Module and nn.Parameter objects are normally constructed in __init__
    # and consumed through a self attribute in forward or a reachable helper.
    init = methods.get("__init__")
    if init is None:
        return False
    reachable_attributes = {
        _dotted_name(node)
        for method in reachable_nodes
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute) and _dotted_name(node).startswith("self.")
    }
    for statement in ast.walk(init):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        constructor_calls = {
            _canonical_name(_dotted_name(call.func), aliases)
            for call in ast.walk(value)
            if isinstance(call, ast.Call)
        }
        if operator not in constructor_calls:
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        assigned = {_dotted_name(target) for target in targets}
        if any(attribute in reachable_attributes for attribute in assigned):
            return True
    return False


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
    _raise_if_keck_infrastructure(result, "compute_score")
    return bool(result.get("passed", False))


def _frontier_reward(accuracy: float) -> float:
    target = float(os.environ.get("PROPOSER_FRONTIER_TARGET", "0.5"))
    return max(0.0, 1.0 - 2.0 * abs(accuracy - target))


def _score_one(solution_str: str, required_operators: List[str]) -> Dict[str, float]:
    torch_code = _extract_python(solution_str)
    if torch_code is None:
        return {
            "overall": float(os.environ.get("PROPOSER_INVALID_REWARD", "-1.0")),
            "validity": 0.0,
            "api_coverage": 0.0,
            "semantic_coverage": 0.0,
            "execution": 0.0,
            "coder_accuracy": 0.0,
        }

    coverage = check_operators_used(required_operators, torch_code)
    coverage_ok = coverage["all_used"] or all(
        _operator_used_on_reachable_path(torch_code, operator)
        for operator in required_operators
    )
    if not coverage_ok:
        return {
            "overall": float(os.environ.get("PROPOSER_INVALID_REWARD", "-1.0")),
            "validity": 0.0,
            "api_coverage": 0.0,
            "semantic_coverage": 0.0,
            "execution": 0.0,
            "coder_accuracy": 0.0,
        }

    semantic_issues = _semantic_api_issues(torch_code, required_operators)
    if semantic_issues:
        _log_semantic_rejection(torch_code, required_operators, semantic_issues)
        return {
            "overall": float(os.environ.get("PROPOSER_INVALID_REWARD", "-1.0")),
            "validity": 0.0,
            "api_coverage": 1.0,
            "semantic_coverage": 0.0,
            "execution": 0.0,
            "coder_accuracy": 0.0,
        }

    if not _check_module(torch_code):
        return {
            "overall": float(os.environ.get("PROPOSER_INVALID_REWARD", "-1.0")),
            "validity": 0.0,
            "api_coverage": 1.0,
            "semantic_coverage": 1.0,
            "execution": 0.0,
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
        "api_coverage": 1.0,
        "semantic_coverage": 1.0,
        "execution": 1.0,
        "coder_accuracy": float(accuracy),
    }


def compute_score(
    solution_strs: List[str],
    ground_truths: List[str],
    index: List[int],
    dataset_index: Optional[List[int]] = None,
) -> List[Dict[str, float]]:
    """Compute one nested frontier reward for every Proposer rollout."""

    del index

    required = ["CODER_VLLM_URL", "KECK_BASE_URL"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    max_workers = int(os.environ.get("PROPOSER_REWARD_MAX_WORKERS", "8"))
    dataset_indices = list(dataset_index) if dataset_index is not None else [None] * len(solution_strs)
    required_operators = [
        _required_operators(ground_truth, dataset_indices[position])
        for position, ground_truth in enumerate(ground_truths)
    ]
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        scores = list(executor.map(_score_one, solution_strs, required_operators))

    valid = [score["validity"] for score in scores]
    api_coverage = [score["api_coverage"] for score in scores]
    semantic_coverage = [score["semantic_coverage"] for score in scores]
    execution = [score["execution"] for score in scores]
    accuracies = [score["coder_accuracy"] for score in scores if score["validity"] > 0]
    rewards = [score["overall"] for score in scores]
    with _print_lock:
        print(
            "Proposer frontier batch: "
            f"samples={len(scores)}, api_coverage_rate={sum(api_coverage) / max(1, len(api_coverage)):.4f}, "
            f"semantic_coverage_rate={sum(semantic_coverage) / max(1, len(semantic_coverage)):.4f}, "
            f"execution_rate={sum(execution) / max(1, len(execution)):.4f}, "
            f"valid_rate={sum(valid) / max(1, len(valid)):.4f}, "
            f"coder_accuracy={sum(accuracies) / max(1, len(accuracies)):.4f}, "
            f"reward={sum(rewards) / max(1, len(rewards)):.4f}",
            file=sys.stderr,
            flush=True,
        )
    return scores
