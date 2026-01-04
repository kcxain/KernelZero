"""
AST-based checker for PyTorch operators in code.
Based on the API extraction method from torch_pool/operator_query.py
"""

import ast
import re
from typing import List, Set, Optional, Dict


class PyTorchAPIVisitor(ast.NodeVisitor):
    """AST visitor to extract PyTorch API calls from code."""

    def __init__(self):
        self.apis: Set[str] = set()

    def visit_Attribute(self, node: ast.Attribute):
        """Visit attribute nodes to extract torch.xxx, nn.xxx, F.xxx, etc."""
        # Recursively get the full attribute chain
        full_name = self._get_full_attribute_name(node)

        if full_name:
            # Check if it's a PyTorch API
            if full_name.startswith("torch."):
                # Exclude torch.nn and torch.nn.functional (they will be handled separately)
                if not full_name.startswith("torch.nn"):
                    self.apis.add(full_name)
            elif full_name.startswith("nn.") or full_name.startswith("torch.nn."):
                if not full_name.startswith("torch.nn.functional"):
                    # Normalize: nn.xxx -> torch.nn.xxx
                    if full_name.startswith("nn."):
                        normalized = f"torch.{full_name}"
                    else:
                        normalized = full_name
                    self.apis.add(normalized)
            elif full_name.startswith("F.") or full_name.startswith("torch.nn.functional."):
                # Normalize: F.xxx -> torch.nn.functional.xxx
                if full_name.startswith("F."):
                    normalized = f"torch.nn.functional.{full_name[2:]}"
                else:
                    normalized = full_name
                self.apis.add(normalized)

        self.generic_visit(node)

    def _get_full_attribute_name(self, node: ast.Attribute) -> Optional[str]:
        """Get the full attribute name, e.g., torch.add, nn.Linear"""
        parts = []
        current = node

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        # Handle base name (could be Name or Constant)
        if isinstance(current, ast.Name):
            parts.append(current.id)
        elif isinstance(current, ast.Constant) and isinstance(current.value, str):
            parts.append(current.value)
        else:
            return None

        # Reverse the list since it was built from right to left
        parts.reverse()
        return ".".join(parts)

    def visit_Call(self, node: ast.Call):
        """Visit function call nodes"""
        # If calling an attribute (like torch.add()), visit_Attribute will handle it
        # This mainly handles direct calls
        self.generic_visit(node)


def extract_apis_from_code(code: str) -> Set[str]:
    """
    Extract PyTorch APIs from code using AST parsing.
    
    Args:
        code: Python code string
        
    Returns:
        Set of API names in normalized format (e.g., "torch.add", "torch.nn.Linear")
    """
    if not code:
        return set()

    apis = set()

    try:
        # Try AST parsing first
        tree = ast.parse(code)
        visitor = PyTorchAPIVisitor()
        visitor.visit(tree)
        apis.update(visitor.apis)
    except SyntaxError:
        # If AST parsing fails, use regex as fallback
        apis.update(_extract_apis_with_regex(code))

    return apis


def _extract_apis_with_regex(source_code: str) -> Set[str]:
    """Extract APIs using regex (fallback method)"""
    apis = set()

    # Match patterns like torch.xxx, nn.xxx, F.xxx, etc.
    patterns = [
        (r"\btorch\.(\w+)\b", "torch"),
        (r"\bnn\.(\w+)\b", "torch.nn"),
        (r"\btorch\.nn\.(\w+)\b", "torch.nn"),
        (r"\bF\.(\w+)\b", "torch.nn.functional"),
        (r"\btorch\.nn\.functional\.(\w+)\b", "torch.nn.functional"),
    ]

    for pattern, prefix in patterns:
        matches = re.findall(pattern, source_code)
        for match in matches:
            # Skip if it's torch.nn or torch.nn.functional (they're handled separately)
            if prefix == "torch" and match in ["nn", "cuda", "cpu"]:
                continue
            apis.add(f"{prefix}.{match}")

    return apis


def normalize_api_name(api: str) -> str:
    """
    Normalize API name to standard format.
    
    Args:
        api: API name in various formats (e.g., "torch.nn.Conv3d", "nn.Conv3d", "Conv3d")
        
    Returns:
        Normalized API name (e.g., "torch.nn.Conv3d")
    """
    # Already in normalized format
    if api.startswith("torch."):
        return api
    
    # Handle nn.xxx -> torch.nn.xxx
    if api.startswith("nn."):
        return f"torch.{api}"
    
    # Handle F.xxx -> torch.nn.functional.xxx
    if api.startswith("F."):
        return f"torch.nn.functional.{api[2:]}"
    
    # Handle bare names - try to infer from context
    # This is a best-effort approach
    if "." not in api:
        # Could be torch.xxx, torch.nn.xxx, or torch.nn.functional.xxx
        # We'll check all possibilities
        return api  # Return as-is, let the caller handle it
    
    return api


def check_operators_used(required_apis: List[str], code: str) -> Dict[str, bool]:
    """
    Check if all required APIs are used in the code.
    
    Args:
        required_apis: List of required API names (e.g., ["torch.nn.Conv3d", "torch.softmax"])
        code: Python code string to check
        
    Returns:
        Dictionary with:
        - "all_used": bool - whether all required APIs are used
        - "used_apis": Set[str] - APIs found in code
        - "missing_apis": Set[str] - APIs not found in code
        - "details": Dict[str, bool] - per-API check results
    """
    # Extract APIs from code
    code_apis = extract_apis_from_code(code)
    
    # Normalize required APIs
    normalized_required = {normalize_api_name(api) for api in required_apis}
    
    # Check each required API
    details = {}
    missing_apis = set()
    
    for required_api in normalized_required:
        # Try exact match first
        if required_api in code_apis:
            details[required_api] = True
            continue
        
        # Try flexible matching
        # Extract the operator name (last part after dot)
        op_name = required_api.split(".")[-1]
        
        # Check various formats
        found = False
        
        # 1. Check for exact operator name in code_apis
        for code_api in code_apis:
            if code_api.endswith(f".{op_name}") or code_api == op_name:
                found = True
                break
        
        # 2. Check in code string directly (for cases like self.conv3d, nn.Conv3d, etc.)
        if not found:
            # Check for nn.OperatorName
            if f"nn.{op_name}" in code or f"torch.nn.{op_name}" in code:
                found = True
            # Check for F.operator_name (functional)
            elif f"F.{op_name.lower()}" in code or f"torch.nn.functional.{op_name.lower()}" in code:
                found = True
            # Check for torch.operator_name
            elif f"torch.{op_name.lower()}" in code:
                found = True
            # Check for self.operator_name (instance attributes)
            elif f"self.{op_name.lower()}" in code:
                found = True
        
        details[required_api] = found
        if not found:
            missing_apis.add(required_api)
    
    return {
        "all_used": len(missing_apis) == 0,
        "used_apis": code_apis,
        "missing_apis": missing_apis,
        "details": details,
    }


def check_operators_used_simple(required_apis: List[str], code: str) -> bool:
    """
    Simple version that just returns True/False.
    
    Args:
        required_apis: List of required API names
        code: Python code string to check
        
    Returns:
        True if all required APIs are used, False otherwise
    """
    result = check_operators_used(required_apis, code)
    return result["all_used"]


# Backward compatibility
def normalize_code(code: str) -> str:
    """
    Normalize code by removing comments and normalizing whitespace.
    This is kept for backward compatibility.
    """
    try:
        # Try to parse and normalize via AST
        tree = ast.parse(code)
        normalized = ast.unparse(tree)
        return normalized
    except:
        # Fallback: basic string normalization
        lines = []
        for line in code.split("\n"):
            # Remove inline comments but preserve strings
            if "#" in line:
                parts = line.split("#")
                if len(parts) > 1:
                    line = parts[0].rstrip()
            if line.strip():
                lines.append(line)

        normalized = "\n".join(lines)
        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()
