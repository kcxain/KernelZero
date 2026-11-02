#!/usr/bin/env bash
#SBATCH -J kz-triton-vllm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/kz-triton-vllm-%j.out
#SBATCH -e logs/kz-triton-vllm-%j.err

set -euo pipefail

ROOT="${KERNELZERO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kernelzero-triton-step80}"
PORT="${PORT:-18080}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"

cd "$ROOT"
mkdir -p logs
source "$ROOT/scripts/activate_cluster_env.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

test -s "$MODEL_PATH/config.json"
NODE_IP="$(hostname -I | awk '{print $1}')"
echo "CODER_NODE=$(hostname)"
echo "CODER_VLLM_URL=http://${NODE_IP}:${PORT}"
echo "CODER_MODEL_NAME=${SERVED_MODEL_NAME}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code
