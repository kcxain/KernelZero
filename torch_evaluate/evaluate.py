#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
    This script evaluates generated PyTorch modules from question_generate.py.
    It performs AST validation to check if the code uses the expected operators,
    and dynamic CUDA validation to ensure the code executes correctly without NaN/Inf.
    The script is designed to run as a batch job, often in parallel across multiple GPUs.

Refactoring Notes:
    - Adapted from math problem evaluation to PyTorch module evaluation
    - Uses ast_checker for static analysis and dynamic_checker for runtime validation
    - Filters modules based on operator usage and execution correctness
    - Refactored into a class-based structure for better organization

Setup:
    pip install torch

Example Usage (in a shell script):
    # This would run the script for GPU 0, with a specific save name.
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --suffix 0 --save_name "my_experiment" &
"""

import json
import os
import argparse
import warnings
from typing import List, Dict, Optional
from ast_checker import check_operators_used
from dynamic_checker import execute_model_on_cuda

warnings.filterwarnings("ignore")


class PyTorchModuleEvaluator:
    """Evaluator for generated PyTorch modules."""

    def __init__(
        self,
        suffix: str,
        save_name: str,
        check_cuda: bool = True,
        check_ast: bool = True,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            suffix: A unique suffix for file naming, often the GPU index.
            save_name: A base name for input and output files.
            check_cuda: Whether to perform CUDA execution validation.
            check_ast: Whether to perform AST operator validation.
            storage_path: Path to storage directory. If None, uses STORAGE_PATH env var.
        """
        self.suffix = suffix
        self.save_name = save_name
        self.check_cuda = check_cuda
        self.check_ast = check_ast

        # Set up paths
        if storage_path is None:
            storage_path = os.getenv("STORAGE_PATH")
        if storage_path is None:
            raise ValueError("STORAGE_PATH environment variable not set")

        self.storage_path = storage_path
        self.input_file = f"{storage_path}/generated_modules/{save_name}_{suffix}.json"
        self.output_file = (
            f"{storage_path}/generated_modules/{save_name}_{suffix}_results.json"
        )

        # Results storage
        self.modules_to_process: List[Dict] = []
        self.results: List[Dict] = []

    def load_data(self) -> bool:
        """
        Load input data from file.

        Returns:
            True if data loaded successfully, False otherwise.
        """
        print(f"[{self.suffix}] Loading data from: {self.input_file}")
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Clean up the input file immediately after loading to save space
            # os.remove(self.input_file)

            # Filter data into modules that need processing
            # Only process items with score >= 0 (score=-1 means parsing failed, skip those)
            self.modules_to_process = [
                item for item in data
            ]

            if not self.modules_to_process:
                print(f"[{self.suffix}] No modules to process. Exiting.")
                # Create an empty results file to signal completion
                with open(self.output_file, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=4, ensure_ascii=False)
                return False

            print(
                f"[{self.suffix}] Found {len(self.modules_to_process)} modules to process."
            )
            return True

        except FileNotFoundError:
            print(f"[{self.suffix}] ERROR: Input file not found. Exiting.")
            return False
        except Exception as e:
            print(f"[{self.suffix}] ERROR: Failed to load data: {e}")
            return False

    def validate_module(self, item: Dict) -> Dict:
        """
        Validate a single module.

        Args:
            item: Module data dictionary containing operators, code, etc.

        Returns:
            Validation result dictionary.
        """
        operators = item.get("operators", [])
        code = item.get("code", "")
        combination_index = item.get("combination_index", -1)
        prompt = item.get("prompt", "")
        raw_response = item.get("raw_response", "")

        # Handle empty code
        if not code:
            print(f"[{self.suffix}] WARNING: Empty code for module")
            return {
                "combination_index": combination_index,
                "operators": operators,
                "code": code,
                "score": -1,
                "validation_passed": False,
                "errors": ["Empty code"],
                "ast_check": None,
                "cuda_check": None,
            }

        validation_errors = []
        ast_check_result = None
        cuda_check_result = None

        # AST Validation: Check if code uses expected operators
        if self.check_ast and operators:
            try:
                check_result = check_operators_used(operators, code)
                passed = check_result["all_used"]
                ast_check_result = {
                    "passed": passed,
                    "used_apis": list(check_result["used_apis"]),
                    "missing_apis": list(check_result["missing_apis"]),
                    "details": check_result["details"],
                }
            except Exception as e:
                ast_check_result = {
                    "passed": False,
                    "error": str(e),
                }
                validation_errors.append(f"AST check exception: {str(e)}")

        # CUDA Dynamic Validation: Check if code executes correctly
        if self.check_cuda:
            try:
                success, error = execute_model_on_cuda(code, timeout=10)
                if success:
                    cuda_check_result = {
                        "passed": True,
                        "error": None,
                    }
                else:
                    cuda_check_result = {
                        "passed": False,
                        "error": error,
                    }
                    validation_errors.append(f"CUDA check failed: {error}")
            except Exception as e:
                cuda_check_result = {
                    "passed": False,
                    "error": str(e),
                }
                validation_errors.append(f"CUDA check exception: {str(e)}")
        else:
            cuda_check_result = {"passed": True, "note": "CUDA check disabled"}

        score = 1
        if not ast_check_result["passed"]:
            score = -1
        if not cuda_check_result["passed"]:
            score = -1
        # Build result
        result = {
            "combination_index": combination_index,
            "operators": operators,
            "code": code,
            "score": score,
            "errors": validation_errors if validation_errors else None,
            "ast_check": ast_check_result,
            "cuda_check": cuda_check_result,
        }

        # Optionally include prompt and raw_response if needed
        if prompt:
            result["prompt"] = prompt
        if raw_response:
            result["raw_response"] = raw_response

        return result

    def process_all(self):
        """Process all modules and validate them."""
        print(f"[{self.suffix}] Validating modules...")

        for idx, item in enumerate(self.modules_to_process):
            result = self.validate_module(item)
            self.results.append(result)

            # Progress reporting
            if (idx + 1) % 100 == 0:
                passed_count = sum(1 for r in self.results if r.get("score", -1) == 1)
                print(
                    f"[{self.suffix}] Processed {idx + 1}/{len(self.modules_to_process)} modules, "
                    f"{passed_count} passed validation"
                )

    def print_statistics(self):
        """Print validation statistics."""
        if not self.results:
            return

        passed_count = sum(1 for r in self.results if r.get("score", -1) == 1)
        failed_count = sum(1 for r in self.results if r.get("score", -1) == -1)

        total = len(self.results)
        print(f"[{self.suffix}] Validation Statistics:")
        print(f"  Total modules: {total}")
        print(f"  Overall passed: {passed_count} ({100 * passed_count / total:.1f}%)")
        print(f"  Overall failed: {failed_count} ({100 * failed_count / total:.1f}%)")

    def save_results(self):
        """Save validation results to output file."""
        print(
            f"[{self.suffix}] Processed {len(self.results)} modules. Saving results to: {self.output_file}"
        )

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)

    def run(self):
        """
        Run the complete evaluation pipeline.

        Returns:
            True if evaluation completed successfully, False otherwise.
        """
        # Load data
        if not self.load_data():
            return False

        # Process all modules
        self.process_all()

        # Print statistics
        self.print_statistics()

        # Save results
        self.save_results()

        print(f"[{self.suffix}] Script finished.")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate generated PyTorch modules.")
    parser.add_argument(
        "--suffix",
        type=str,
        default="0",
        help="A unique suffix for file naming, often the GPU index.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
        help="A base name for input and output files.",
    )
    parser.add_argument(
        "--check_cuda",
        action="store_true",
        default=True,
        help="Whether to perform CUDA execution validation (default: True).",
    )
    parser.add_argument(
        "--check_ast",
        action="store_true",
        default=True,
        help="Whether to perform AST operator validation (default: True).",
    )
    args = parser.parse_args()

    # Create evaluator and run
    evaluator = PyTorchModuleEvaluator(
        suffix=args.suffix,
        save_name=args.save_name,
        check_cuda=args.check_cuda,
        check_ast=args.check_ast,
    )

    evaluator.run()


if __name__ == "__main__":
    main()


"""输入格式 (from question_generate.py)
[
    {
        "combination_index": 0,
        "operators": ["torch.nn.Conv3d", "torch.softmax", "torch.nn.MaxPool3d"],
        "code": "生成的PyTorch模块代码",
        "raw_response": "模型的原始响应",
        "prompt": "用于生成该样本的完整prompt"
    },
    // ... 更多模块
]
"""

"""输出格式
[
    {
        "combination_index": 0,
        "operators": ["torch.nn.Conv3d", "torch.softmax", "torch.nn.MaxPool3d"],
        "code": "生成的PyTorch模块代码",
        "code_hash": "sha256哈希值（用于去重）",
        "score": 1,  // 原始分数
        "validation_passed": true,  // 是否通过验证
        "errors": null,  // 错误列表（如果有）
        "ast_check": {
            "passed": true,
            "note": "AST check passed"
        },
        "cuda_check": {
            "passed": true,
            "error": null
        },
        "prompt": "用于生成该样本的完整prompt（可选）",
        "raw_response": "模型的原始响应（可选）"
    },
    // ... 更多评估结果
]
"""
