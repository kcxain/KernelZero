import ast
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple


class DynamicValidationError(Exception):
    """Raised when dynamic validation fails."""

    pass


def validate_init_inputs(init_inputs: List[Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate initialization inputs to prevent CUDA assertion errors.

    Args:
        init_inputs: List of initialization arguments

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    import torch

    for i, inp in enumerate(init_inputs):
        if isinstance(inp, torch.Tensor):
            # Check for invalid tensor values
            if torch.isnan(inp).any() or torch.isinf(inp).any():
                return False, f"Init input[{i}] contains NaN/Inf"
            # Check for negative values in dimensions/sizes (scalar tensors)
            if inp.numel() == 1:
                val = inp.item()
                if val < 0:
                    return False, f"Init input[{i}] is negative: {val}"
                if val == 0:
                    return (
                        False,
                        f"Init input[{i}] is zero (likely invalid for dimension/size)",
                    )
            # Check for negative values in multi-element tensors
            elif inp.numel() > 1:
                if (inp < 0).any():
                    return False, f"Init input[{i}] contains negative values"
        elif isinstance(inp, (int, float)):
            # Check for negative values (some parameters like padding can be negative in special cases,
            # but for most common cases, negative values indicate errors)
            if inp < 0:
                # Be lenient: only flag if it's clearly invalid (e.g., very negative)
                # Most PyTorch operations don't accept negative dimensions
                if inp < -100:  # Very negative values are likely errors
                    return False, f"Init input[{i}] is very negative: {inp}"
        elif isinstance(inp, (tuple, list)):
            # Recursively check nested structures
            for j, sub_inp in enumerate(inp):
                if isinstance(sub_inp, (int, float)) and sub_inp < 0:
                    if sub_inp < -100:  # Very negative values are likely errors
                        return (
                            False,
                            f"Init input[{i}][{j}] is very negative: {sub_inp}",
                        )
                elif isinstance(sub_inp, torch.Tensor):
                    if torch.isnan(sub_inp).any() or torch.isinf(sub_inp).any():
                        return False, f"Init input[{i}][{j}] contains NaN/Inf"
                    if (sub_inp < 0).any():
                        return False, f"Init input[{i}][{j}] contains negative values"

    return True, None


def cleanup_cuda_context(device_id: int = None):
    """
    Thoroughly clean up CUDA context to avoid state pollution.
    This helps prevent device-side asserts caused by residual state from previous executions.

    Args:
        device_id: Specific CUDA device to clean. If None, cleans current device.
    """
    import torch

    if not torch.cuda.is_available():
        return

    try:
        if device_id is not None:
            torch.cuda.set_device(device_id)

        # Synchronize all CUDA operations
        torch.cuda.synchronize()

        # Clear all CUDA caches
        torch.cuda.empty_cache()

        # Reset peak memory stats (helps with memory fragmentation)
        torch.cuda.reset_peak_memory_stats()

        # Force garbage collection to free Python references
        import gc

        gc.collect()

        # Clear cache again after GC
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    except Exception:
        # If cleanup fails, try basic cleanup
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def execute_model_on_cpu(code: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Execute generated model code on CPU device with random tensors.

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
        try:
            init_inputs = get_init_inputs() if get_init_inputs else []
        except Exception as e:
            return False, f"get_init_inputs() failed: {str(e)}"

        # Validate initialization inputs
        is_valid, validation_error = validate_init_inputs(init_inputs)
        if not is_valid:
            return False, f"Invalid init inputs: {validation_error}"

        # Instantiate model on CPU (keep tensors on CPU)
        try:
            model = Model(*init_inputs)
            model.eval()
        except Exception as e:
            return False, f"Model instantiation failed: {str(e)}"

        # Get input tensors (keep on CPU)
        try:
            inputs = get_inputs()
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            # Validate input tensors
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    if inp.numel() == 0:
                        return False, f"Input[{i}] is empty tensor"
                    if torch.isnan(inp).any() or torch.isinf(inp).any():
                        return False, f"Input[{i}] contains NaN/Inf"
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


def execute_model_on_cuda(
    code: str, timeout: int = 10, device_id: int = None, cleanup_before: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Execute generated model code on CUDA device with random tensors.

    This function includes thorough CUDA context cleanup to avoid state pollution
    that can cause device-side asserts when CPU execution succeeds but CUDA fails.

    Args:
        code: Python code string containing Model class and helper functions.
        timeout: Maximum execution time in seconds.
        device_id: Specific CUDA device to use. If None, uses current device.
        cleanup_before: Whether to clean CUDA context before execution (recommended).

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
        - (True, None) if execution succeeded without NaN/Inf
        - (False, error_msg) if execution failed or produced NaN/Inf
    """
    try:
        import torch
        import os

        # Check CUDA availability
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # Set device if specified
        if device_id is not None:
            torch.cuda.set_device(device_id)

        current_device = torch.cuda.current_device()

        # Thorough cleanup before execution to avoid state pollution
        # This is critical when multiple processes/threads share GPU resources
        if cleanup_before:
            cleanup_cuda_context(device_id)

        # Set CUDA_LAUNCH_BLOCKING for better error messages (temporarily)
        # Note: This might slow down execution, but helps with debugging
        original_cuda_launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "0")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        try:
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
            try:
                init_inputs = get_init_inputs() if get_init_inputs else []
            except Exception as e:
                return False, f"get_init_inputs() failed: {str(e)}"

            # Validate initialization inputs before using them
            is_valid, validation_error = validate_init_inputs(init_inputs)
            if not is_valid:
                return False, f"Invalid init inputs: {validation_error}"

            # Move tensors to CUDA with explicit device
            init_inputs = [
                inp.cuda(device=current_device)
                if isinstance(inp, torch.Tensor)
                else inp
                for inp in init_inputs
            ]

            # Synchronize before model instantiation to ensure all transfers complete
            torch.cuda.synchronize()

            # Instantiate model and move to CUDA
            try:
                model = Model(*init_inputs)
                model = model.cuda(device=current_device)
                model.eval()
                # Synchronize after model creation
                torch.cuda.synchronize()
            except RuntimeError as e:
                error_msg = str(e)
                # Thorough cleanup after error to prevent state pollution
                cleanup_cuda_context(device_id)
                # Check if it's a device-side assert
                if (
                    "device-side assert triggered" in error_msg
                    or "CUDA error" in error_msg
                ):
                    # Provide more context about the parameters
                    params_str = ", ".join([
                        str(inp)
                        if not isinstance(inp, torch.Tensor) or inp.numel() == 1
                        else f"tensor(shape={inp.shape})"
                        for inp in init_inputs[:10]
                    ])  # Limit to first 10 params
                    if len(init_inputs) > 10:
                        params_str += f", ... (total {len(init_inputs)} params)"
                    return (
                        False,
                        f"Model instantiation failed: CUDA assertion error (device={current_device}, params: [{params_str}])",
                    )
                return False, f"Model instantiation failed: {error_msg}"
            except Exception as e:
                cleanup_cuda_context(device_id)
                return False, f"Model instantiation failed: {str(e)}"

            # Get input tensors and move to CUDA with explicit device
            try:
                inputs = get_inputs()
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]
                inputs = [
                    inp.cuda(device=current_device)
                    if isinstance(inp, torch.Tensor)
                    else inp
                    for inp in inputs
                ]

                # Synchronize after input transfer
                torch.cuda.synchronize()

                # Validate input tensors
                for i, inp in enumerate(inputs):
                    if isinstance(inp, torch.Tensor):
                        if inp.numel() == 0:
                            return False, f"Input[{i}] is empty tensor"
                        if torch.isnan(inp).any() or torch.isinf(inp).any():
                            return False, f"Input[{i}] contains NaN/Inf"
            except Exception as e:
                cleanup_cuda_context(device_id)
                return False, f"Input generation failed: {str(e)}"

            # Execute forward pass
            try:
                with torch.no_grad():
                    output = model(*inputs)
                    # Synchronize after forward pass to catch any async errors
                    torch.cuda.synchronize()
            except RuntimeError as e:
                error_msg = str(e)
                # Thorough cleanup after error
                cleanup_cuda_context(device_id)
                # Check if it's a device-side assert
                if (
                    "device-side assert triggered" in error_msg
                    or "CUDA error" in error_msg
                ):
                    return (
                        False,
                        f"Forward pass failed: CUDA assertion error (device={current_device}, likely shape mismatch or invalid operation)",
                    )
                return False, f"Forward pass failed: {error_msg}"
            except Exception as e:
                cleanup_cuda_context(device_id)
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

        finally:
            # Restore original CUDA_LAUNCH_BLOCKING setting
            os.environ["CUDA_LAUNCH_BLOCKING"] = original_cuda_launch_blocking
            # Thorough cleanup at the end
            try:
                del model
            except Exception:
                pass
            cleanup_cuda_context(device_id)

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
    except Exception:
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
