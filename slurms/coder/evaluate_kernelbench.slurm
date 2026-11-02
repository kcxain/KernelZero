#!/usr/bin/env bash
#SBATCH -J kb_triton_eval
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/kb-triton-eval-%j.out
#SBATCH -e logs/kb-triton-eval-%j.err

set -euo pipefail

trap 'echo "[kernelbench_triton_eval] failed at line ${LINENO}: ${BASH_COMMAND} (exit $?)" >&2' ERR

PROJECT_ROOT="${KERNELZERO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
KERNELBENCH_ROOT="${KERNELBENCH_ROOT:-${PROJECT_ROOT}/resources/KernelBench}"
RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs/kernelbench}"

mkdir -p "${PROJECT_ROOT}/logs" "${RUNS_DIR}"
cd "${PROJECT_ROOT}"

echo "Job start at $(date '+%F %T')"
echo "hostname=$(hostname)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "KERNELBENCH_ROOT=${KERNELBENCH_ROOT}"
echo "RUNS_DIR=${RUNS_DIR}"
echo "MODEL_BASE_URL=${MODEL_BASE_URL:-${OSS_BASE_URL:-}}"
echo "MODEL_NAME=${MODEL_NAME:-${OSS_MODEL_NAME:-}}"
echo "KECK_BASE_URL=${KECK_BASE_URL:-}"
echo "EVAL_MODE=${EVAL_MODE:-${MODE:-full}}"
echo "LEVEL=${LEVEL:-1}"
echo "PROBLEM_IDS=${PROBLEM_IDS:-}"
echo "OFFSET=${OFFSET:-0}"
echo "LIMIT=${LIMIT:-}"

ROOT="$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/activate_cluster_env.sh"

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
export PYTHONUNBUFFERED=1
export KERNELBENCH_ROOT RUNS_DIR

python scripts/kernelbench_triton_eval.py

echo "Job end at $(date '+%F %T')"
