#!/usr/bin/env bash
#SBATCH -J kz-proposer-api-exec-80
#SBATCH -p r8nv-gpu-hw-80g
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -t 1-06:00:00
#SBATCH -o logs/kz-proposer-api-exec-80-%j.out
#SBATCH -e logs/kz-proposer-api-exec-80-%j.err
#SBATCH --exclude=r8a100-d00,r8a100-d05

set -euo pipefail

ROOT="${KERNELZERO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TRAIN_DATA="${TRAIN_DATA:?TRAIN_DATA is required}"
PROPOSER_MODEL="${PROPOSER_MODEL:?PROPOSER_MODEL is required}"
CODER_VLLM_URL="${CODER_VLLM_URL:?CODER_VLLM_URL is required}"
KECK_BASE_URL="${KECK_BASE_URL:?KECK_BASE_URL is required}"
EXP="${EXP:-qwen25_coder7b_proposer_fo_grpo_v2}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/${EXP}}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
MAX_STEPS="${MAX_STEPS:-80}"

cd "$ROOT"
mkdir -p logs "$SAVE_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# /share/personal may be full; keep all framework caches on /nfs_global.
CACHE_ROOT="${CACHE_ROOT:-$ROOT/.cache/proposer_rl}"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME"

source /tools/module_env.sh
module load cluster-tools/v1.0 || true
module load slurm-tools/v1.0 || true
module load cuda-cudnn/12.4-9.1.1
source "$ROOT/.venv/bin/activate"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export CODER_VLLM_URL KECK_BASE_URL
export CODER_MODEL_NAME="${CODER_MODEL_NAME:-kernelzero-triton-step80}"
export CODER_PROMPT_PATH="${CODER_PROMPT_PATH:-$ROOT/examples/format_prompt/kernelbench_triton_cot_grpo.jinja}"
export CODER_ROLLOUT_N="${CODER_ROLLOUT_N:-5}"
export CODER_TEMPERATURE="${CODER_TEMPERATURE:-0.6}"
export CODER_TOP_P="${CODER_TOP_P:-0.95}"
export CODER_TOP_K="${CODER_TOP_K:-20}"
export CODER_MAX_TOKENS="${CODER_MAX_TOKENS:-16384}"
export PROPOSER_FRONTIER_TARGET="${PROPOSER_FRONTIER_TARGET:-0.5}"
export PROPOSER_INVALID_REWARD="${PROPOSER_INVALID_REWARD:--1.0}"
export PROPOSER_REWARD_MAX_WORKERS="${PROPOSER_REWARD_MAX_WORKERS:-8}"
export KECK_WORKERS_PER_MODULE="${KECK_WORKERS_PER_MODULE:-5}"
export PROPOSER_TRAIN_DATA="$TRAIN_DATA"
export PROPOSER_REJECT_LOG="${PROPOSER_REJECT_LOG:-$ROOT/logs/proposer-semantic-rejections-${SLURM_JOB_ID}.jsonl}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"

test -s "$TRAIN_DATA"
test -s "$PROPOSER_MODEL/config.json"
test -s "$CODER_PROMPT_PATH"
curl --noproxy '*' -fsS "$CODER_VLLM_URL/health"
curl --noproxy '*' -fsS "$KECK_BASE_URL/healthz"

export RAY_TMPDIR="/tmp/kzp${SLURM_JOB_ID}"
export RAY_PORT="$((20000 + SLURM_JOB_ID % 20000))"
mkdir -p "$RAY_TMPDIR"
ray stop --force >/dev/null 2>&1 || true
ray start --head --disable-usage-stats --include-dashboard=false \
  --port="$RAY_PORT" --num-cpus="${SLURM_CPUS_PER_TASK:-64}" --num-gpus="$NUM_GPUS" \
  --object-store-memory=268435456 --temp-dir="$RAY_TMPDIR" --plasma-directory="$RAY_TMPDIR"
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

cd "$ROOT"

python3 -m verl.trainer.main \
  config="$ROOT/examples/config.yaml" \
  data.train_files="$TRAIN_DATA" \
  data.prompt_key=prompt \
  data.answer_key=code \
  data.format_prompt="$ROOT/examples/format_prompt/proposer_api.jinja" \
  data.max_prompt_length=8192 \
  data.max_response_length=6144 \
  data.rollout_batch_size=8 \
  worker.rollout.n=4 \
  worker.rollout.gpu_memory_utilization=0.55 \
  worker.rollout.max_num_batched_tokens=49152 \
  worker.actor.global_batch_size=8 \
  worker.actor.model.model_path="$PROPOSER_MODEL" \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.offload.offload_params=false \
  worker.actor.offload.offload_optimizer=false \
  worker.ref.offload.offload_params=false \
  worker.reward.reward_function="$ROOT/examples/reward_function/proposer_triton_api_exec.py:compute_score" \
  trainer.experiment_name="$EXP" \
  trainer.save_checkpoint_path="$SAVE_DIR/" \
  trainer.load_checkpoint_path="$LOAD_CHECKPOINT" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.total_epochs=100 \
  trainer.max_steps="$MAX_STEPS" \
  trainer.save_freq=20 \
  trainer.save_limit=-1 \
  trainer.val_freq=-1 \
  trainer.val_before_train=false
