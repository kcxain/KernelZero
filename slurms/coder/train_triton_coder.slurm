#!/usr/bin/env bash
#SBATCH -J kz-coder-grpo
#SBATCH -p r8nv-gpu-hw-80g
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -t 1-06:00:00
#SBATCH -o logs/kz-coder-grpo-%j.out
#SBATCH -e logs/kz-coder-grpo-%j.err
#SBATCH --exclude=r8a100-d00,r8a100-d05

set -euo pipefail

ROOT="${KERNELZERO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
EXP="${EXP:?EXP is required}"
MODEL="${MODEL:?MODEL is required}"
TRAIN_DATA="${TRAIN_DATA:?TRAIN_DATA is required}"
KECK_BASE_URL="${KECK_BASE_URL:?KECK_BASE_URL is required}"
PROMPT="${PROMPT:-$ROOT/examples/format_prompt/kernelbench_triton_cot_grpo.jinja}"
REWARD="${REWARD:-$ROOT/examples/reward_function/kernel_triton_alpha1p1_failfast.py}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/$EXP}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
MAX_STEPS="${MAX_STEPS:-80}"

cd "$ROOT"
source /tools/module_env.sh
module load cluster-tools/v1.0 || true
module load slurm-tools/v1.0 || true
module load cuda-cudnn/12.4-9.1.1
source "$ROOT/.venv/bin/activate"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset PYTORCH_CUDA_ALLOC_CONF
ulimit -n 8192 || true

CACHE_ROOT="${CACHE_ROOT:-$ROOT/.cache/coder_rl}"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME"

export KECK_BASE_URL
if [[ -z "${TRITON_SERVER_IP:-}" ]]; then
  TRITON_SERVER_IP="${KECK_BASE_URL#*://}"
  TRITON_SERVER_IP="${TRITON_SERVER_IP%%:*}"
fi
export TRITON_SERVER_IP

test -s "$MODEL/config.json"
test -s "$TRAIN_DATA"
test -s "$PROMPT"
test -s "$REWARD"

mkdir -p "$SAVE_DIR"

if [[ -z "$LOAD_CHECKPOINT" ]] && find "$SAVE_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "ERROR: SAVE_DIR is not empty: $SAVE_DIR"
  exit 2
fi
if [[ -n "$LOAD_CHECKPOINT" ]]; then
  test -d "$LOAD_CHECKPOINT"
fi

echo "EXP=$EXP"
echo "MODEL=$MODEL"
echo "REWARD=$REWARD"
echo "SAVE_DIR=$SAVE_DIR"
echo "LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-<fresh>}"
echo "MAX_STEPS=$MAX_STEPS"
echo "KECK_BASE_URL=$KECK_BASE_URL"
grep -n "\balpha\b" "$REWARD"
curl -sf "$KECK_BASE_URL/healthz"

export RAY_TMPDIR=/tmp/kzr${SLURM_JOB_ID}
export RAY_PORT=$((20000 + SLURM_JOB_ID % 20000))
mkdir -p "$RAY_TMPDIR"

ray stop --force >/dev/null 2>&1 || true

ray start \
  --head \
  --disable-usage-stats \
  --include-dashboard=false \
  --port="$RAY_PORT" \
  --num-cpus="${SLURM_CPUS_PER_TASK:-64}" \
  --num-gpus=4 \
  --object-store-memory=268435456 \
  --temp-dir="$RAY_TMPDIR" \
  --plasma-directory="$RAY_TMPDIR"

export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
echo "RAY_ADDRESS=$RAY_ADDRESS"

python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="$TRAIN_DATA" \
  data.format_prompt="$PROMPT" \
  data.answer_key=code \
  data.max_response_length=16384 \
  data.rollout_batch_size=64 \
  worker.rollout.n=5 \
  worker.rollout.gpu_memory_utilization=0.30 \
  worker.rollout.max_num_batched_tokens=32768 \
  worker.actor.global_batch_size=64 \
  worker.actor.model.model_path="$MODEL" \
  worker.actor.offload.offload_params=false \
  worker.actor.offload.offload_optimizer=false \
  worker.ref.offload.offload_params=false \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.reward.reward_function="$REWARD:compute_score" \
  trainer.experiment_name="$EXP" \
  trainer.save_checkpoint_path="$SAVE_DIR/" \
  trainer.load_checkpoint_path="$LOAD_CHECKPOINT" \
  trainer.n_gpus_per_node=4 \
  trainer.total_epochs=100 \
  trainer.max_steps="$MAX_STEPS" \
  trainer.save_freq=20 \
  trainer.save_limit=-1 \
  trainer.milestone_save_freq=-1 \
  trainer.milestone_max_step=-1 \
  trainer.val_freq=-1 \
  trainer.val_before_train=false
