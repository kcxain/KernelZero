#!/usr/bin/env bash
#SBATCH -J kz-validate-modules
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -t 06:00:00
#SBATCH -o logs/kz-validate-modules-%j.out
#SBATCH -e logs/kz-validate-modules-%j.err

set -euo pipefail

ROOT="${KERNELZERO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SAVE_NAME="${SAVE_NAME:?SAVE_NAME is required}"
export STORAGE_PATH="${STORAGE_PATH:-$ROOT/train_model/generated}"

cd "$ROOT"
mkdir -p logs "$STORAGE_PATH/generated_modules"
source "$ROOT/scripts/activate_cluster_env.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/kzv${SLURM_JOB_ID}}"
export RAY_PORT="${RAY_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
mkdir -p "$RAY_TMPDIR"
ray stop --force >/dev/null 2>&1 || true
ray start --head --disable-usage-stats --include-dashboard=false \
  --port="$RAY_PORT" --num-cpus="${SLURM_CPUS_PER_TASK:-64}" --num-gpus=8 \
  --temp-dir="$RAY_TMPDIR"
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

for suffix in {0..7}; do
  python torch_evaluate/evaluate.py \
    --suffix "$suffix" \
    --save_name "$SAVE_NAME" \
    --use_ray \
    --ray_address "$RAY_ADDRESS"
done
