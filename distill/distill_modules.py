"""
Distillation module for calling vLLM API to generate optimized CUDA kernels.
Reads successful modules from results.json files, deduplicates them,
constructs prompts, and calls vLLM API for distillation.
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI
import re
from distill.prompt import distill_optimize_prompt


# Default vLLM API configuration
DEFAULT_API_URL = os.getenv("OPENAI_API_URL", "http://10.200.108.4:8000/v1")
DEFAULT_API_KEY = "token-abc123"
DEFAULT_MODEL = "oss"


def find_results_files(base_dir: Path) -> List[Path]:
    """Find all files ending with results.json in the given directory."""
    results_files = []
    for f in base_dir.glob("*results.json"):
        results_files.append(f)
    return sorted(results_files)


def load_results_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load and parse a results.json file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def filter_successful_modules(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter entries with score=1."""
    return [entry for entry in results if entry.get("score") == 1]


def compute_hash(code: str, operators: List[str]) -> str:
    """Compute a hash for deduplication based on code and operators."""
    # Normalize: sort operators and strip whitespace from code
    ops_str = "|".join(sorted(operators))
    content = f"{code.strip()}|{ops_str}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def construct_prompt(py_code: str) -> str:
    """Construct the full prompt for distillation using the template."""
    return distill_optimize_prompt.replace("__py_code_to_be_fill__", py_code)


def get_openai_client(api_url: str = None, api_key: str = None) -> OpenAI:
    """Create an OpenAI client for vLLM API."""
    return OpenAI(
        api_key=api_key or DEFAULT_API_KEY,
        base_url=api_url or DEFAULT_API_URL,
    )


def call_vllm_api(
    prompt: str,
    client: OpenAI,
    model: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Tuple[bool, str]:
    """
    Call vLLM API with the given prompt using OpenAI client.

    Args:
        prompt: The input prompt
        client: OpenAI client instance
        model: Model name/path
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Tuple of (success, response_text or error_message)
    """
    try:
        completion = client.completions.create(
            model=model or DEFAULT_MODEL,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["```\n\n", "<|im_end|>", "<|endoftext|>"],
        )

        if completion.choices and len(completion.choices) > 0:
            return True, completion.choices[0].text
        else:
            return False, "No choices in response"

    except Exception as e:
        return False, f"Error: {e}"


def call_vllm_chat_api(
    prompt: str,
    client: OpenAI,
    model: str = None,
    max_tokens: int = 20480,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Tuple[bool, str]:
    """
    Call vLLM Chat API (OpenAI compatible) with the given prompt.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in PyTorch and CUDA programming.",
                },
                {"role": "user", "content": prompt},
            ],
            model=model or DEFAULT_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # print(chat_completion)
        res = {
            "answer": chat_completion.choices[0].message.content,
            "reasoning": chat_completion.choices[0].message.reasoning,
        }
        # print(res)
        return True, res
    
    except Exception as e:
        return False, f"Error: {e}"


def extract_code(response: str) -> str:
    """Extract the last code block (preferably python) from the response using regex."""
    try:
        # Match all code blocks with language hints (e.g., ```python ... ```)
        matches = re.findall(r"```(?:python)?\s*([\s\S]*?)```", response, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        return response
    except Exception:
        return response


def process_single_module(
    module: Dict[str, Any],
    client: OpenAI,
    model: str = None,
    use_chat_api: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Process a single module through the API."""
    code = module.get("module", "")
    operators = module.get("ops", [])

    prompt = construct_prompt(code)
    # print(prompt)

    if use_chat_api:
        success, response = call_vllm_chat_api(prompt, client, model, **kwargs)
    else:
        success, response = call_vllm_api(prompt, client, model, **kwargs)

    # extracted_response = extract_code(response) if success else ""

    return {
        "index": module.get("index"),
        "ops": operators,
        "module": code,
        # "prompt": prompt,
        "call_success": success,
        "response": response,
    }


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def run_distillation(
    input_file: str = None,
    api_url: str = None,
    model: str = None,
    api_key: str = None,
    output_file: str = None,
    max_workers: int = 4,
    use_chat_api: bool = True,
    max_tokens: int = 10240,
    temperature: float = 0.7,
    top_p: float = 0.95,
    limit: int = None,
):
    """
    Main function to run the distillation process.

    Args:
        api_url: vLLM API endpoint URL (default: from env or http://10.200.108.4:8000/v1)
        model: Model name/path (default: oss)
        api_key: API key (default: token-abc123)
        output_file: Path to save results
        max_workers: Number of parallel workers
        use_chat_api: Use chat completions API vs completions API
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        limit: Limit number of modules to process (for testing)
    """
    # Create OpenAI client
    client = get_openai_client(api_url, api_key)
    model = model or DEFAULT_MODEL

    data = load_jsonl(input_file)
    if limit:
        data = data[:limit]


    print(f"\nStarting distillation with {len(data)} modules...")
    print(f"API URL: {api_url or DEFAULT_API_URL}")
    print(f"Model: {model}")
    print(f"Output: {output_file}")
    print(f"Workers: {max_workers}")

    results = []

    # Process modules with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_module,
                d,
                client,
                model,
                use_chat_api=use_chat_api,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ): d['index']
            for d in data
        }

        with open(output_file, "w", encoding="utf-8") as f:
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Distilling"
            ):
                idx = futures[future]
                try:
                    result = future.result()
                    result["index"] = idx
                    results.append(result)
                    # Write incrementally
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"Error processing module {idx}: {e}")

    # Summary
    successful = sum(1 for r in results if r.get("success"))
    print("\nDistillation complete!")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Distill CUDA kernels using vLLM API")
    parser.add_argument(
        "--api-url",
        type=str,
        required=False,
        help="vLLM API endpoint URL (e.g., http://localhost:8000)",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20480, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of modules to process"
    )
    parser.add_argument(
        "--use-completions",
        action="store_true",
        help="Use completions API instead of chat API",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export unique modules without API calls",
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input file path")

    args = parser.parse_args()
    run_distillation(
        input_file=args.input_file,
        api_url=args.api_url,
        model=args.model,
        output_file=args.output_file,
        max_workers=args.workers,
        use_chat_api=not args.use_completions,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
