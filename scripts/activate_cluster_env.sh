#!/usr/bin/env bash

# Shared, cluster-neutral environment setup for Slurm entry points.
# Source this file after setting ROOT to the KernelZero repository root.

set -euo pipefail

ROOT="${ROOT:-${KERNELZERO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"
VENV_PATH="${VENV_PATH:-$ROOT/.venv}"

if [[ -n "${MODULE_INIT_SCRIPT:-}" ]]; then
  # shellcheck source=/dev/null
  source "$MODULE_INIT_SCRIPT"
fi

if command -v module >/dev/null 2>&1; then
  if [[ -n "${CLUSTER_MODULES:-}" ]]; then
    read -r -a modules <<< "$CLUSTER_MODULES"
    module load "${modules[@]}"
  fi
  if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE"
  fi
fi

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Python environment not found: $VENV_PATH" >&2
  echo "Set VENV_PATH to an environment created from pyproject.toml or uv.lock." >&2
  return 1
fi

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  export CUDA_HOME
fi
