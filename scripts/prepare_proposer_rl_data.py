#!/usr/bin/env python3
"""Convert validated v2 modules into the 400 Proposer RL input prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_user_prompt(prompt: str) -> str:
    marker = "<|im_start|>user\n"
    if marker not in prompt:
        return prompt.strip()
    user_text = prompt.rsplit(marker, 1)[1]
    return user_text.split("<|im_end|>", 1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(source, list) or len(source) != 400:
        raise ValueError(f"Expected exactly 400 records, got {len(source) if isinstance(source, list) else type(source)}")

    prepared = []
    for item in source:
        prepared.append(
            {
                "prompt": extract_user_prompt(item["prompt"]),
                "code": item["code"],
                "operators": item["operators"],
                "combination_index": item["combination_index"],
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(prepared)} records to {output}")


if __name__ == "__main__":
    main()
