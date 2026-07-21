#!/usr/bin/env python3
"""Run one real Proposer-module -> Triton Coder -> Keck integration test."""

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    reward_path = Path(__file__).resolve().parents[1] / "examples" / "reward_function" / "proposer_triton.py"
    spec = importlib.util.spec_from_file_location("proposer_triton", reward_path)
    reward = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reward)

    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    torch_code = records[args.index]["code"]
    print(f"module_index={args.index}")
    print(f"module_valid={reward._check_module(torch_code)}")
    outputs = reward._sample_coder(torch_code)
    print(f"coder_outputs={len(outputs)}")
    results = [reward._keck_passed(output, torch_code) for output in outputs]
    print(f"keck_passed={results}")
    print(f"coder_accuracy={sum(results) / len(results):.4f}")


if __name__ == "__main__":
    main()
