"""
Graph-based similarity metrics for PyTorch modules.
Inspired by TVM's workload signature and graph matching approaches.
"""

import torch
import torch.fx
from typing import List, Tuple, Optional, Dict
from difflib import SequenceMatcher


def get_operator_family(op_name: str) -> str:
    """
    Map operator to its family for semantic similarity.
    Similar operators in the same family should be considered more similar.

    Args:
        op_name: Operator name (e.g., 'Conv2d', 'ReLU', 'MaxPool2d')

    Returns:
        Operator family name
    """
    op_name_lower = op_name.lower()

    # Convolution family
    if "conv" in op_name_lower:
        if "transpose" in op_name_lower:
            return "CONV_TRANSPOSE"
        return "CONV"

    # Normalization family
    if any(
        norm in op_name_lower
        for norm in ["batchnorm", "layernorm", "groupnorm", "instancenorm"]
    ):
        return "NORM"

    # Activation family
    if any(
        act in op_name_lower
        for act in ["relu", "gelu", "sigmoid", "tanh", "leakyrelu", "elu", "swish"]
    ):
        return "ACTIVATION"

    # Pooling family
    if "pool" in op_name_lower:
        if "max" in op_name_lower:
            return "MAX_POOL"
        elif "avg" in op_name_lower or "mean" in op_name_lower:
            return "AVG_POOL"
        return "POOL"

    # Linear/Dense family
    if "linear" in op_name_lower or "dense" in op_name_lower:
        return "LINEAR"

    # Dropout family
    if "dropout" in op_name_lower:
        return "DROPOUT"

    # Loss family
    if "loss" in op_name_lower:
        return "LOSS"

    # Attention family
    if "attention" in op_name_lower or "multihead" in op_name_lower:
        return "ATTENTION"

    # Embedding family
    if "embedding" in op_name_lower:
        return "EMBEDDING"

    # Default: return original name
    return op_name


def get_graph_signature(
    model: torch.nn.Module,
    include_shape: bool = False,
    use_family: bool = True,
    include_dependencies: bool = True,
) -> List[str]:
    """
    Extract graph signature from a PyTorch model using FX tracing.
    Similar to TVM's workload signature but adapted for PyTorch modules.

    Args:
        model: PyTorch model to trace
        include_shape: Whether to include shape information in signature
        use_family: Whether to use operator family instead of exact name
        include_dependencies: Whether to include input dependency offsets

    Returns:
        List of signature strings representing the computation graph
    """
    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception as e:
        # If tracing fails, return empty signature
        return []

    nodes = list(traced.graph.nodes)
    node_to_id = {node: i for i, node in enumerate(nodes)}

    signature = []
    for node in nodes:
        if node.op in ["call_module", "call_function"]:
            # Extract operator name
            op_name = str(node.target).split(".")[-1]

            # Use family if requested
            if use_family:
                op_name = get_operator_family(op_name)

            # Build signature components
            sig_parts = [op_name]

            # Add input dependencies (like TVM's dataflow edges)
            if include_dependencies:
                input_offsets = []
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        offset = node_to_id[node] - node_to_id[arg]
                        input_offsets.append(offset)

                if input_offsets:
                    # Sort for stability (order-independent)
                    input_offsets = sorted(input_offsets)
                    sig_parts.append(f"deps:{input_offsets}")

            # Add shape information if available and requested
            if include_shape and hasattr(node, "meta") and "tensor_meta" in node.meta:
                shape = node.meta["tensor_meta"].shape
                if shape:
                    # Use simplified shape representation (batch, channels, spatial...)
                    sig_parts.append(f"shape:{tuple(shape)}")

            signature.append("|".join(sig_parts))

    return signature


def lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Calculate Longest Common Subsequence (LCS) length between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Length of LCS
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def normalized_lcs_similarity(seq1: List[str], seq2: List[str]) -> float:
    """
    Calculate normalized LCS similarity between two sequences.
    Returns a value between 0 and 1.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Normalized similarity score (0 to 1)
    """
    if not seq1 or not seq2:
        return 0.0

    lcs_len = lcs_length(seq1, seq2)
    max_len = max(len(seq1), len(seq2))

    return lcs_len / max_len if max_len > 0 else 0.0


def graph_similarity(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    use_family: bool = True,
    include_dependencies: bool = True,
) -> float:
    """
    Calculate similarity between two PyTorch models based on their computation graphs.
    Uses LCS on graph signatures, similar to TVM's approach.

    Args:
        model1: First model
        model2: Second model
        use_family: Whether to use operator family for semantic matching
        include_dependencies: Whether to include dependency information

    Returns:
        Similarity score between 0 and 1
    """
    sig1 = get_graph_signature(
        model1,
        include_shape=False,
        use_family=use_family,
        include_dependencies=include_dependencies,
    )
    sig2 = get_graph_signature(
        model2,
        include_shape=False,
        use_family=use_family,
        include_dependencies=include_dependencies,
    )

    return normalized_lcs_similarity(sig1, sig2)


def batch_graph_similarity(
    models: List[torch.nn.Module],
    use_family: bool = True,
    include_dependencies: bool = True,
) -> List[List[float]]:
    """
    Calculate pairwise similarity matrix for a batch of models.

    Args:
        models: List of models
        use_family: Whether to use operator family
        include_dependencies: Whether to include dependencies

    Returns:
        Symmetric similarity matrix (n x n)
    """
    n = len(models)
    similarity_matrix = [[0.0] * n for _ in range(n)]

    # Pre-compute signatures for efficiency
    signatures = [
        get_graph_signature(
            model,
            include_shape=False,
            use_family=use_family,
            include_dependencies=include_dependencies,
        )
        for model in models
    ]

    # Calculate pairwise similarities
    for i in range(n):
        similarity_matrix[i][i] = 1.0  # Self-similarity is 1.0
        for j in range(i + 1, n):
            sim = normalized_lcs_similarity(signatures[i], signatures[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim  # Symmetric

    return similarity_matrix


def cluster_share_per_problem_graph(
    models: List[torch.nn.Module],
    distance_threshold: float = 0.3,
    use_family: bool = True,
    include_dependencies: bool = True,
) -> List[float]:
    """
    Calculate cluster proportions for models based on graph similarity.
    Similar to the BLEU-based clustering but using graph LCS similarity.

    Args:
        models: List of models to cluster
        distance_threshold: Maximum distance for clustering (1 - similarity)
        use_family: Whether to use operator family
        include_dependencies: Whether to include dependencies

    Returns:
        List of cluster proportions (same length as models)
    """
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    if not models:
        return []

    if len(models) < 2:
        return [1.0] * len(models)

    # Calculate similarity matrix
    similarity_matrix = batch_graph_similarity(
        models, use_family=use_family, include_dependencies=include_dependencies
    )

    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1.0 - np.array(similarity_matrix)

    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    # Calculate cluster proportions
    from collections import Counter

    total = len(models)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions


if __name__ == "__main__":
    # Test with the example model
    from challenger.oracle import Model, get_init_inputs

    model1 = Model(*get_init_inputs())
    model2 = Model(*get_init_inputs())

    print("Graph signatures:")
    print("Model 1:", get_graph_signature(model1))
    print("Model 2:", get_graph_signature(model2))

    print("\nSimilarity:", graph_similarity(model1, model2))

    # Test clustering
    models = [Model(*get_init_inputs()) for _ in range(5)]
    proportions = cluster_share_per_problem_graph(models)
    print("\nCluster proportions:", proportions)
