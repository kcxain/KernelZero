"""
Script to check all entries in the distilled_data JSONL file using compute_score_remote.
"""

import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple
from threading import Lock

from distill.check import compute_score_remote
import re

def extract_code(response: str) -> str:
    """
    Extract the code from the response.
    """
    # 用正则
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "None"

def check_single_entry(
    entry: Dict[str, Any]
) -> Tuple[int, Dict[str, Any]]:
    """
    Check a single entry using compute_score_remote.

    Args:
        entry: The JSON entry to check

    Returns:
        Tuple of (index, entry_with_result)
    """
    idx = entry.get("index", -1)

    # Extract required fields
    module = entry.get("module", "")
    response = entry.get("response", "").get("answer", "")
    code = extract_code(response)

    problem = {"ground_truth": {"pytorch_module": module}}

    # Check the entry
    try:
        res_json = compute_score_remote(problem, code)
        # Preserve all original fields; only add `result`.
        entry["result"] = res_json

    except Exception as e:
        logger.error(f"Index {idx}: Error during check - {e}")
        entry["result"] = {"passed": False, "error": str(e)}

    return idx, entry


def check_jsonl_file(jsonl_path: str, output_path: str = None, max_workers: int = 32):
    """
    Check all entries in a JSONL file using compute_score_remote.

    Args:
        jsonl_path: Path to the JSONL file to check
        output_path: Optional path to save results (default: same as input with _checked suffix)
        max_workers: Number of parallel workers (default: 32)
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        logger.error(f"File not found: {jsonl_path}")
        return

    if output_path is None:
        output_path = (
            jsonl_path.parent / f"{jsonl_path.stem}_checked{jsonl_path.suffix}"
        )
    else:
        output_path = Path(output_path)

    logger.info(f"Reading from: {jsonl_path}")
    logger.info(f"Writing results to: {output_path}")
    logger.info(f"Using {max_workers} parallel workers")

    # First, read all entries
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if "index" not in entry:
                    logger.error(f"Line {line_num}: Missing `index` field; skipping")
                    continue
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error - {e}")
                continue

    total_lines = len(entries)
    logger.info(f"Total entries to check: {total_lines}")

    # Statistics (using list for thread-safe updates)
    stats = {"passed": 0, "failed": 0, "error": 0, "total": 0}

    write_lock = Lock()

    # Open output file for writing
    f_out = open(output_path, "w", encoding="utf-8")

    def write_result(checked_entry: Dict[str, Any]):
        """Write a checked entry immediately (real-time flush)."""
        with write_lock:
            f_out.write(json.dumps(checked_entry, ensure_ascii=False) + "\n")
            f_out.flush()  # Ensure immediate write to disk

            # Update statistics
            stats["total"] += 1
            result = checked_entry.get("result", {})
            if result.get("passed", False):
                stats["passed"] += 1
            elif "error" in result:
                stats["error"] += 1
            else:
                stats["failed"] += 1

    # Process entries in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(check_single_entry, entry): entry["index"] for entry in entries}

        # Process completed tasks with progress bar
        with tqdm(total=total_lines, desc="Checking entries") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    _idx, checked_entry = future.result()
                    # Write result immediately (real-time)
                    write_result(checked_entry)

                except Exception as e:
                    logger.error(f"Index {idx}: Unexpected error - {e}")
                    # Preserve original entry data where possible
                    original_entry = None
                    for ent in entries:
                        if ent.get("index") == idx:
                            original_entry = ent.copy()
                            break
                    if original_entry is None:
                        original_entry = {"index": idx}
                    original_entry["result"] = {"passed": False, "error": str(e)}
                    write_result(original_entry)

                pbar.update(1)

    f_out.close()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CHECK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total entries processed: {stats['total']}")
    logger.info(f"Passed: {stats['passed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Errors: {stats['error']}")
    if stats["total"] > 0:
        pass_rate = (stats["passed"] / stats["total"]) * 100
        logger.info(f"Pass rate: {pass_rate:.2f}%")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check JSONL file using compute_score_remote"
    )
    parser.add_argument("jsonl_path", type=str, help="Path to the JSONL file to check")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for checked results (default: input_path with _checked suffix)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32)",
    )

    args = parser.parse_args()
    check_jsonl_file(args.jsonl_path, args.output, max_workers=args.workers)
