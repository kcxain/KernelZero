#!/usr/bin/env python3
import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _request_with_retry(
    method: str,
    url: str,
    timeout: float,
    retries: int,
    retry_interval: float,
    json_payload: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.request(
                method=method,
                url=url,
                json=json_payload,
                timeout=timeout,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                print(
                    f"[test_coder_vllm] attempt {attempt}/{retries} failed: {exc}; retry in {retry_interval}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(retry_interval)
    raise RuntimeError(f"request failed after {retries} attempts: {last_err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test OpenAI-compatible vLLM endpoint for coder model."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CODER_VLLM_URL", "http://127.0.0.1:8000"),
        help="vLLM base URL, e.g. http://10.0.0.1:8000",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CODER_MODEL_NAME", ""),
        help="Model name served by vLLM. If empty, use first model from /v1/models.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a CUDA kernel for vector add in C++ and explain briefly.",
        help="User prompt sent to /v1/chat/completions",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=5, help="Max retry attempts")
    parser.add_argument(
        "--retry-interval", type=float, default=5.0, help="Seconds between retries"
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="max_tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p")
    parser.add_argument(
        "--require-substring",
        action="append",
        default=[],
        help="Require substring in output (can pass multiple times).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        t0 = time.time()
        model_resp = _request_with_retry(
            method="GET",
            url=f"{base_url}/v1/models",
            timeout=args.timeout,
            retries=args.retries,
            retry_interval=args.retry_interval,
        )
        models_data = model_resp.json()
        model_list = models_data.get("data", [])
        if not isinstance(model_list, list) or not model_list:
            raise RuntimeError(f"/v1/models returned empty data: {models_data}")

        model_name = args.model
        if not model_name:
            first = model_list[0]
            if isinstance(first, dict):
                model_name = str(first.get("id", "")).strip()
        if not model_name:
            raise RuntimeError("cannot determine model name from --model or /v1/models")

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }

        chat_resp = _request_with_retry(
            method="POST",
            url=f"{base_url}/v1/chat/completions",
            timeout=args.timeout,
            retries=args.retries,
            retry_interval=args.retry_interval,
            json_payload=payload,
        )
        chat_data = chat_resp.json()
        choices = chat_data.get("choices", [])
        if not choices:
            raise RuntimeError(f"chat response missing choices: {chat_data}")

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        output_text = _extract_text_content(message.get("content", ""))
        output_text = output_text.strip()
        if not output_text:
            raise RuntimeError("chat response content is empty")

        for s in args.require_substring:
            if s not in output_text:
                raise RuntimeError(f"missing required substring in output: {s!r}")

        dt = time.time() - t0
        print("[test_coder_vllm] PASS")
        print(f"[test_coder_vllm] base_url={base_url}")
        print(f"[test_coder_vllm] model={model_name}")
        print(f"[test_coder_vllm] elapsed={dt:.2f}s")
        print("[test_coder_vllm] response_preview:")
        print(output_text[:1000])
        return 0
    except Exception as exc:
        print(f"[test_coder_vllm] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
