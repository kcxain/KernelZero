#!/bin/bash

#- Job parameters

#SBATCH -J coder_vllm
#SBATCH -o logs/coder_vllm-%j.out
#SBATCH -e logs/coder_vllm-%j.err

#- Resources

#SBATCH -p r8nv-gpu-hw-80g
#SBATCH -t 0-3:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos=gpu-normal
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --exclude=r8a100-d00
#SBATCH --exclude=r8a100-d05
#SBATCH --exclude=r8l40s-a04
#SBATCH --exclude=r8l40s-a02

set -euo pipefail

# ---------------- User Config ----------------
# 1) 推荐直接改这里
MODEL_PATH="${MODEL_PATH:-/path/to/your/coder-model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-coder-model}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# 2) 多卡并行时改 TP_SIZE，并把 #SBATCH --gres=gpu:N 改成一致
TP_SIZE="${TP_SIZE:-4}"
DTYPE="${DTYPE:-auto}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
# 额外参数示例: --trust-remote-code --enforce-eager
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
# ---------------------------------------------

if [[ ! -d "${MODEL_PATH}" && ! -f "${MODEL_PATH}" ]]; then
  echo "[coder_vllm] MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi

mkdir -p logs

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"
echo "$(df -h | grep -v tmpfs)"

ulimit -l unlimited
ulimit -v unlimited
ulimit -n 65535
ulimit -u 4125556

#- Load environments
source /tools/module_env.sh
module list

module load cluster-tools/v1.0
module load slurm-tools/v1.0
module load cuda-cudnn/12.4-9.1.1

export CUDA_HOME=/tools/cluster-software/cuda-cudnn/cuda-12.4.1-9.1.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cluster-quota
nvidia-smi --format=csv --query-gpu=index,name,driver_version,memory.total

echo "Using GPU(s) ${CUDA_VISIBLE_DEVICES}"
echo "This job is assigned the following resources by SLURM:"
scontrol show jobid "$SLURM_JOB_ID" -dd | awk '/IDX/ {print $2, $4}'

unset http_proxy
unset https_proxy
unset no_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset NO_PROXY
echo -e "\033[31m[×] 已关闭代理\033[0m"

NODE_IP="$(hostname -I | awk '{print $1}')"
echo "[coder_vllm] Node IP: ${NODE_IP}"
echo "[coder_vllm] Endpoint for proposer:"
echo "  export CODER_VLLM_URL=http://${NODE_IP}:${PORT}"
echo "  export CODER_MODEL_NAME=${SERVED_MODEL_NAME}"

BASE_ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
)

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  BASE_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

echo "[coder_vllm] Starting vLLM server..."
if command -v vllm >/dev/null 2>&1; then
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2086
    vllm serve "${MODEL_PATH}" "${BASE_ARGS[@]}" ${VLLM_EXTRA_ARGS}
  else
    vllm serve "${MODEL_PATH}" "${BASE_ARGS[@]}"
  fi
else
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2086
    python -m vllm.entrypoints.openai.api_server --model "${MODEL_PATH}" "${BASE_ARGS[@]}" ${VLLM_EXTRA_ARGS}
  else
    python -m vllm.entrypoints.openai.api_server --model "${MODEL_PATH}" "${BASE_ARGS[@]}"
  fi
fi

echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
