#!/usr/bin/env bash
#SBATCH -J kz-generate-modules
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -t 06:00:00
#SBATCH -o logs/kz-generate-modules-%j.out
#SBATCH -e logs/kz-generate-modules-%j.err

set -euo pipefail

ROOT="${KERNELZERO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROPOSER_MODEL="${PROPOSER_MODEL:?PROPOSER_MODEL is required}"
COMBINATION_FILE="${COMBINATION_FILE:?COMBINATION_FILE is required}"
SAVE_NAME="${SAVE_NAME:-generated_modules}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
export STORAGE_PATH="${STORAGE_PATH:-$ROOT/train_model/generated}"

cd "$ROOT"
mkdir -p logs "$STORAGE_PATH/generated_modules"
source "$ROOT/scripts/activate_cluster_env.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
test -s "$PROPOSER_MODEL/config.json"
test -s "$COMBINATION_FILE"

bash torch_generate/question_generate.bash \
  "$PROPOSER_MODEL" \
  "$NUM_SAMPLES" \
  "$SAVE_NAME" \
  "$COMBINATION_FILE"
