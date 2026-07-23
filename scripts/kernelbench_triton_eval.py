#!/usr/bin/env python3
"""Run KernelBench Triton generation through a vLLM API and score with Keck."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Any

import requests
from jinja2 import Template
from openai import OpenAI


@dataclass
class EvalRecord:
    level: int
    problem_id: int
    problem_file: str
    sample_id: int
    prompt_path: str
    response_path: str
    kernel_path: str
    score: dict[str, Any]
    ok: bool
    elapsed_sec: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_kernelbench_root() -> Path:
    return _repo_root() / "resources" / "KernelBench"


def _find_level_dir(kernelbench_root: Path, level: int) -> Path:
    candidates = [
        kernelbench_root / f"level{level}",
        kernelbench_root / "KernelBench" / f"level{level}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find KernelBench level directory. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _problem_id(path: Path) -> int:
    match = re.match(r"(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse problem id from filename: {path.name}")
    return int(match.group(1))


def _select_problems(level_dir: Path, problem_ids: str | None, offset: int, limit: int | None) -> list[Path]:
    files = sorted(level_dir.glob("*.py"), key=lambda p: (_problem_id(p), p.name))
    if problem_ids:
        wanted = {int(x) for x in re.split(r"[,\s]+", problem_ids.strip()) if x}
        files = [p for p in files if _problem_id(p) in wanted]
    if offset:
        files = files[offset:]
    if limit is not None:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No KernelBench problem files found in {level_dir}")
    return files


def _load_prompt_template(path: Path) -> Template:
    return Template(path.read_text(encoding="utf-8"))


def _render_prompt(template: Template, py_code: str, gpu_name: str) -> str:
    sys.path.insert(0, str(_repo_root()))
    from examples.format_prompt.examples import softmax_example
    from examples.format_prompt.hardware.gpu_specs import GPU_SPEC_INFO

    return template.render(
        GPU_name=gpu_name,
        GPU_list=GPU_SPEC_INFO,
        examples=softmax_example,
        py_code=py_code,
        GPU_info_format="bullets",
    )


def _extract_code_block(text: str) -> str:
    languages = ["python", "triton"]
    pattern = re.compile(
        rf"```(?:{'|'.join(languages)})\s*\n(.*?)(?:\n```|$)",
        re.DOTALL | re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].group(1).rstrip()
    if "class ModelNew" in text:
        return text.strip()
    return ""


def _normalise_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def _query_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


def _score_kernel(
    keck_base_url: str,
    triton_code: str,
    torch_code: str,
    timeout: float,
) -> dict[str, Any]:
    if not triton_code:
        return {
            "formated": False,
            "compiled": False,
            "passed": False,
            "msg": {"speedup": -1},
            "error": "CANNOT EXTRACT CODE",
        }

    response = requests.post(
        f"{keck_base_url.rstrip('/')}/compute_score",
        data={
            "triton_code": triton_code,
            "torch_code": torch_code,
        },
        proxies={"http": None, "https": None},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator used by code-generation benchmarks."""
    if num_samples <= 0:
        return 0.0
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


def _summarise(records: list[EvalRecord], run_name: str, level: int, run_dir: Path, num_samples: int) -> dict[str, Any]:
    by_problem: dict[int, list[EvalRecord]] = {}
    for record in records:
        by_problem.setdefault(record.problem_id, []).append(record)

    problem_rows = []
    for problem_id, problem_records in sorted(by_problem.items()):
        correct = sum(1 for record in problem_records if record.ok)
        n = len(problem_records)
        pass_at = {
            f"pass@{k}": _estimate_pass_at_k(n, correct, k)
            for k in (1, 5, 10)
            if k <= n
        }
        problem_rows.append(
            {
                "problem_id": problem_id,
                "num_samples": n,
                "num_correct": correct,
                "empirical_pass_rate": correct / n if n else 0.0,
                **pass_at,
            }
        )

    aggregate_pass_at = {}
    for k in (1, 5, 10):
        vals = [row[f"pass@{k}"] for row in problem_rows if f"pass@{k}" in row]
        aggregate_pass_at[f"pass@{k}"] = sum(vals) / len(vals) if vals else 0.0

    passed = sum(1 for record in records if record.ok)
    return {
        "run_name": run_name,
        "level": level,
        "run_dir": str(run_dir),
        "num_problems": len(by_problem),
        "num_samples_per_problem": num_samples,
        "total_samples": len(records),
        "passed_samples": passed,
        "sample_pass_rate": passed / len(records) if records else 0.0,
        **aggregate_pass_at,
        "problem_summaries": problem_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("full", "generate", "score"),
        default=os.environ.get("EVAL_MODE", os.environ.get("MODE", "full")),
        help="full: generate and score; generate: only call vLLM; score: only call Keck on saved kernels",
    )
    parser.add_argument("--kernelbench-root", default=os.environ.get("KERNELBENCH_ROOT", str(_default_kernelbench_root())))
    parser.add_argument("--level", type=int, default=int(os.environ.get("LEVEL", "1")))
    parser.add_argument("--problem-ids", default=os.environ.get("PROBLEM_IDS"))
    parser.add_argument("--offset", type=int, default=int(os.environ.get("OFFSET", "0")))
    parser.add_argument("--limit", type=int, default=int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None)
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", f"kernelbench_triton_{int(time.time())}"))
    parser.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR"))
    parser.add_argument("--model-base-url", default=os.environ.get("MODEL_BASE_URL") or os.environ.get("OSS_BASE_URL"))
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME") or os.environ.get("OSS_MODEL_NAME"))
    parser.add_argument("--api-key", default=os.environ.get("OSS_API_KEY", "EMPTY"))
    parser.add_argument("--keck-base-url", default=os.environ.get("KECK_BASE_URL"))
    parser.add_argument("--gpu-name", default=os.environ.get("HARDWARE_GPU_NAME", "A100-80GB"))
    parser.add_argument("--num-samples", type=int, default=int(os.environ.get("GEN_NUM_SAMPLES", "1")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("GEN_TEMPERATURE", "0.7")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("GEN_TOP_P", "0.95")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("GEN_MAX_TOKENS", "10240")))
    parser.add_argument("--model-timeout", type=float, default=float(os.environ.get("MODEL_TIMEOUT", "1800")))
    parser.add_argument("--keck-timeout", type=float, default=float(os.environ.get("KECK_TIMEOUT", "1800")))
    parser.add_argument("--prompt-template", default=os.environ.get("PROMPT_TEMPLATE", str(_repo_root() / "examples" / "format_prompt" / "kernelbench_triton.jinja")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offset < 0:
        raise ValueError("--offset/OFFSET must be >= 0")
    if args.mode in {"full", "generate"} and not args.model_base_url:
        raise ValueError("Missing --model-base-url or MODEL_BASE_URL/OSS_BASE_URL")
    if args.mode in {"full", "generate"} and not args.model_name:
        raise ValueError("Missing --model-name or MODEL_NAME/OSS_MODEL_NAME")
    if args.mode in {"full", "score"} and not args.keck_base_url:
        raise ValueError("Missing --keck-base-url or KECK_BASE_URL")

    kernelbench_root = Path(args.kernelbench_root).resolve()
    level_dir = _find_level_dir(kernelbench_root, args.level)
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else kernelbench_root / "runs"
    run_dir = runs_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts").mkdir(exist_ok=True)
    (run_dir / "responses").mkdir(exist_ok=True)
    (run_dir / "kernels").mkdir(exist_ok=True)

    template = _load_prompt_template(Path(args.prompt_template))
    client = None
    if args.mode in {"full", "generate"}:
        client = OpenAI(base_url=_normalise_base_url(args.model_base_url), api_key=args.api_key)
    problems = _select_problems(level_dir, args.problem_ids, args.offset, args.limit)

    config = vars(args).copy()
    config["api_key"] = "***REDACTED***"
    config["kernelbench_root"] = str(kernelbench_root)
    config["level_dir"] = str(level_dir)
    config["run_dir"] = str(run_dir)
    (run_dir / "generation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    records: list[EvalRecord] = []
    generated_rows: list[dict[str, Any]] = []
    for problem_path in problems:
        problem_id = _problem_id(problem_path)
        torch_code = problem_path.read_text(encoding="utf-8")
        prompt = _render_prompt(template, torch_code, args.gpu_name)

        for sample_id in range(args.num_samples):
            stem = f"level_{args.level}_problem_{problem_id}_sample_{sample_id}"
            prompt_path = run_dir / "prompts" / f"{stem}_prompt.txt"
            response_path = run_dir / "responses" / f"{stem}_response.txt"
            kernel_path = run_dir / "kernels" / f"{stem}_kernel.py"
            prompt_path.write_text(prompt, encoding="utf-8")

            start = time.time()
            triton_code = ""
            try:
                if args.mode in {"full", "generate"}:
                    if client is None:
                        raise RuntimeError("OpenAI client is not initialised")
                    response = _query_model(
                        client,
                        args.model_name,
                        prompt,
                        args.temperature,
                        args.top_p,
                        args.max_tokens,
                        args.model_timeout,
                    )
                    response_path.write_text(response, encoding="utf-8")
                    triton_code = _extract_code_block(response)
                    kernel_path.write_text(triton_code, encoding="utf-8")
                elif kernel_path.exists():
                    triton_code = kernel_path.read_text(encoding="utf-8")
                else:
                    raise FileNotFoundError(f"Missing generated kernel: {kernel_path}")

                if args.mode in {"full", "score"}:
                    score = _score_kernel(args.keck_base_url, triton_code, torch_code, args.keck_timeout)
                    ok = bool(score.get("passed", False))
                else:
                    score = {
                        "formated": bool(triton_code),
                        "compiled": False,
                        "passed": False,
                        "msg": {"speedup": -1},
                        "status": "generated_not_scored",
                    }
                    ok = False
            except Exception as exc:
                score = {
                    "formated": False,
                    "compiled": False,
                    "passed": False,
                    "msg": {"speedup": -1},
                    "error": repr(exc),
                }
                ok = False

            record = EvalRecord(
                level=args.level,
                problem_id=problem_id,
                problem_file=str(problem_path),
                sample_id=sample_id,
                prompt_path=str(prompt_path),
                response_path=str(response_path),
                kernel_path=str(kernel_path),
                score=score,
                ok=ok,
                elapsed_sec=time.time() - start,
            )
            row = asdict(record)
            if args.mode == "generate":
                generated_rows.append(row)
            else:
                records.append(record)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    if args.mode == "generate":
        _write_jsonl(run_dir / "generation_records.jsonl", generated_rows)
        summary = {
            "run_name": args.run_name,
            "mode": args.mode,
            "level": args.level,
            "run_dir": str(run_dir),
            "num_problems": len(problems),
            "num_samples_per_problem": args.num_samples,
            "total_samples": len(generated_rows),
        }
        (run_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    else:
        rows = [asdict(record) for record in records]
        _write_jsonl(run_dir / "keck_score_results.jsonl", rows)
        summary = _summarise(records, args.run_name, args.level, run_dir, args.num_samples)
        summary["mode"] = args.mode
        _write_jsonl(run_dir / "problem_summary.jsonl", summary["problem_summaries"])
        (run_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
