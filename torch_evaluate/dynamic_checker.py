import ast
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple


class DynamicValidationError(Exception):
    """Raised when dynamic validation fails."""

    pass


def execute_model_on_cuda(code: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Execute generated model code on CUDA device with random tensors.

    Args:
        code: Python code string containing Model class and helper functions.
        timeout: Maximum execution time in seconds.

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
        - (True, None) if execution succeeded without NaN/Inf
        - (False, error_msg) if execution failed or produced NaN/Inf
    """
    try:
        import torch

        # Check CUDA availability
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # Prepare execution namespace
        namespace: Dict[str, Any] = {
            "torch": torch,
            "__builtins__": __builtins__,
        }

        # Decode escaped strings if necessary
        try:
            exec(code, namespace)
        except SyntaxError:
            # Try unescaping
            try:
                unescaped = code.encode("utf-8").decode("unicode_escape")
                exec(unescaped, namespace)
            except Exception as e:
                return False, f"Code execution failed: {str(e)}"

        # Extract Model class and helper functions
        if "Model" not in namespace:
            return False, "Model class not found in code"

        Model = namespace["Model"]
        get_inputs = namespace.get("get_inputs")
        get_init_inputs = namespace.get("get_init_inputs")

        if get_inputs is None:
            return False, "get_inputs function not found"

        # Get initialization inputs
        init_inputs = get_init_inputs() if get_init_inputs else []
        init_inputs = [
            inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in init_inputs
        ]

        # Instantiate model and move to CUDA
        try:
            model = Model(*init_inputs)
            model = model.cuda()
            model.eval()
        except Exception as e:
            return False, f"Model instantiation failed: {str(e)}"

        # Get input tensors and move to CUDA
        try:
            inputs = get_inputs()
            inputs = [
                inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs
            ]
        except Exception as e:
            return False, f"Input generation failed: {str(e)}"

        # Execute forward pass
        try:
            with torch.no_grad():
                output = model(*inputs)
        except Exception as e:
            return False, f"Forward pass failed: {str(e)}"

        # Check for NaN/Inf in output
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                return False, "Output contains NaN values"
            if torch.isinf(output).any():
                return False, "Output contains Inf values"
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    if torch.isnan(out).any():
                        return False, f"Output[{i}] contains NaN values"
                    if torch.isinf(out).any():
                        return False, f"Output[{i}] contains Inf values"

        return True, None

    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def normalize_code(code: str) -> str:
    """
    Normalize code by:
    - Removing comments
    - Normalizing whitespace
    - Sorting imports
    - Standardizing formatting

    This helps with deduplication.
    """
    try:
        # Try to parse and normalize via AST
        tree = ast.parse(code)
        normalized = ast.unparse(tree)
        return normalized
    except:
        # Fallback: basic string normalization
        # Remove comments
        lines = []
        for line in code.split("\n"):
            # Remove inline comments but preserve strings
            if "#" in line:
                # Simple heuristic: if # is not in a string
                parts = line.split("#")
                if len(parts) > 1:
                    line = parts[0].rstrip()
            if line.strip():
                lines.append(line)

        normalized = "\n".join(lines)
        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()


def compute_code_hash(code: str) -> str:
    """
    Compute a hash of normalized code for deduplication.
    """
    normalized = normalize_code(code)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SampleDeduplicator:
    """
    Track and deduplicate code samples based on normalized form.
    """

    def __init__(self):
        self.seen_hashes: set = set()
        self.samples: List[str] = []

    def add_if_unique(self, code: str) -> bool:
        """
        Add code if it's unique.

        Returns:
            True if added (was unique), False if duplicate
        """
        code_hash = compute_code_hash(code)
        if code_hash in self.seen_hashes:
            return False

        self.seen_hashes.add(code_hash)
        self.samples.append(code)
        return True

    def get_unique_samples(self) -> List[str]:
        """Return all unique samples."""
        return self.samples.copy()

    def __len__(self) -> int:
        return len(self.samples)


def validate_and_filter_samples(
    samples: List[str], check_cuda: bool = True, verbose: bool = False
) -> List[str]:
    """
    Validate and filter samples:
    1. Execute on CUDA (if enabled and available)
    2. Filter out samples with NaN/Inf
    3. Deduplicate

    Args:
        samples: List of code samples to validate
        check_cuda: Whether to perform CUDA execution validation
        verbose: Print progress information

    Returns:
        List of valid, unique samples
    """
    deduplicator = SampleDeduplicator()
    valid_samples = []

    for i, sample in enumerate(samples):
        if verbose:
            print(f"Validating sample {i + 1}/{len(samples)}...", end=" ")

        # Dynamic validation
        if check_cuda:
            success, error = execute_model_on_cuda(sample)
            if not success:
                if verbose:
                    print(f"FAILED: {error}")
                continue

        # Deduplication
        if deduplicator.add_if_unique(sample):
            valid_samples.append(sample)
            if verbose:
                print("PASSED (unique)")
        else:
            if verbose:
                print("SKIPPED (duplicate)")

    if verbose:
        print(
            f"\nValidation complete: {len(valid_samples)}/{len(samples)} samples passed"
        )

    return valid_samples
