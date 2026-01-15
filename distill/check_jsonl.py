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


def check_single_entry(
    entry: Dict[str, Any], line_num: int
) -> Tuple[int, Dict[str, Any]]:
    """
    Check a single entry using compute_score_remote.

    Args:
        entry: The JSON entry to check
        line_num: Line number for logging

    Returns:
        Tuple of (line_num, entry_with_check_result)
    """
    # Extract required fields
    original_code = entry.get("original_code", "")
    extracted_response = entry.get("extracted_response", "")

    if not original_code:
        logger.warning(f"Line {line_num}: Missing original_code")
        entry["check_result"] = {
            "passed": False,
            "error": "Missing original_code",
        }
        return line_num, entry

    if not extracted_response:
        logger.warning(f"Line {line_num}: Missing extracted_response")
        entry["check_result"] = {
            "passed": False,
            "error": "Missing extracted_response",
        }
        return line_num, entry

    # Construct problem dict as expected by compute_score_remote
    problem = {"ground_truth": {"pytorch_module": original_code}}

    # Check the entry
    try:
        passed, msg = compute_score_remote(problem, extracted_response)
        entry["check_result"] = {"passed": passed, "msg": msg}
        logger.info(entry["check_result"])
        if passed:
            logger.debug(f"Line {line_num}: PASSED")
        else:
            logger.debug(f"Line {line_num}: FAILED - {msg}")

    except Exception as e:
        logger.error(f"Line {line_num}: Error during check - {e}")
        entry["check_result"] = {"passed": False, "error": str(e)}

    return line_num, entry


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
                entries.append((line_num, entry))
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error - {e}")
                continue

    total_lines = len(entries)
    logger.info(f"Total entries to check: {total_lines}")

    # Statistics (using list for thread-safe updates)
    stats = {"passed": 0, "failed": 0, "error": 0, "total": 0}

    # Dictionary to store results temporarily (for ordered writing)
    results_dict = {}
    next_write_line = 1  # Track which line should be written next
    write_lock = Lock()

    # Open output file for writing
    f_out = open(output_path, "w", encoding="utf-8")

    def write_result(line_num: int, checked_entry: Dict[str, Any]):
        """Write a result if it's the next one in order, or store it for later."""
        nonlocal next_write_line, stats

        with write_lock:
            results_dict[line_num] = checked_entry

            # Write all consecutive results that are ready
            while next_write_line in results_dict:
                entry = results_dict.pop(next_write_line)
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f_out.flush()  # Ensure immediate write to disk

                # Update statistics
                stats["total"] += 1
                check_result = entry.get("check_result", {})
                if check_result.get("passed", False):
                    stats["passed"] += 1
                elif "error" in check_result:
                    stats["error"] += 1
                else:
                    stats["failed"] += 1

                next_write_line += 1

    # Process entries in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_line = {
            executor.submit(check_single_entry, entry, line_num): line_num
            for line_num, entry in entries
        }

        # Process completed tasks with progress bar
        with tqdm(total=total_lines, desc="Checking entries") as pbar:
            for future in as_completed(future_to_line):
                line_num = future_to_line[future]
                try:
                    result_line_num, checked_entry = future.result()
                    # Write result (will write immediately if in order, or store for later)
                    write_result(result_line_num, checked_entry)

                except Exception as e:
                    logger.error(f"Line {line_num}: Unexpected error - {e}")
                    # Create error entry - find original entry to preserve its data
                    original_entry = None
                    for ln, ent in entries:
                        if ln == line_num:
                            original_entry = ent.copy()
                            break
                    if original_entry:
                        original_entry["check_result"] = {
                            "passed": False,
                            "error": str(e),
                        }
                        write_result(line_num, original_entry)
                    else:
                        # Fallback if entry not found
                        error_entry = {
                            "check_result": {"passed": False, "error": str(e)}
                        }
                        write_result(line_num, error_entry)

                pbar.update(1)

    # Ensure all remaining results are written
    with write_lock:
        while next_write_line <= total_lines:
            if next_write_line in results_dict:
                entry = results_dict.pop(next_write_line)
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f_out.flush()

                stats["total"] += 1
                check_result = entry.get("check_result", {})
                if check_result.get("passed", False):
                    stats["passed"] += 1
                elif "error" in check_result:
                    stats["error"] += 1
                else:
                    stats["failed"] += 1

                next_write_line += 1
            else:
                break

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
