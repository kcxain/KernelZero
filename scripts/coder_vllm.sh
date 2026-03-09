#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/coder_vllm.sh start /path/to/model [port] [cuda_visible_devices]
#   bash scripts/coder_vllm.sh stop [port]
#   bash scripts/coder_vllm.sh status [port]

ACTION="${1:-}"
MODEL_PATH="${2:-}"
PORT="${3:-${CODER_VLLM_PORT:-8000}}"
CUDA_DEVICES="${4:-${CODER_CUDA_DEVICES:-}}"

HOST="${CODER_VLLM_HOST:-0.0.0.0}"
DTYPE="${CODER_VLLM_DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${CODER_VLLM_GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${CODER_VLLM_MAX_MODEL_LEN:-}"
TENSOR_PARALLEL_SIZE="${CODER_VLLM_TP_SIZE:-}"
SERVED_MODEL_NAME="${CODER_MODEL_NAME:-coder-model}"
LOG_DIR="${CODER_VLLM_LOG_DIR:-./logs/tmp}"
PID_FILE="${LOG_DIR}/coder_vllm_${PORT}.pid"
LOG_FILE="${LOG_DIR}/coder_vllm_${PORT}.log"
EXTRA_ARGS="${CODER_VLLM_EXTRA_ARGS:-}"

mkdir -p "${LOG_DIR}"

die() {
  echo "[coder_vllm] $*" >&2
  exit 1
}

ensure_port() {
  [[ "${PORT}" =~ ^[0-9]+$ ]] || die "Invalid port: ${PORT}"
}

get_tp_from_cuda_devices() {
  if [[ -n "${CUDA_DEVICES}" ]]; then
    IFS=',' read -r -a arr <<< "${CUDA_DEVICES}"
    echo "${#arr[@]}"
    return
  fi
  echo "1"
}

is_running() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

start_server() {
  [[ -n "${MODEL_PATH}" ]] || die "Missing model path. Example: bash scripts/coder_vllm.sh start /path/to/model"
  [[ -d "${MODEL_PATH}" || -f "${MODEL_PATH}" ]] || die "Model path does not exist: ${MODEL_PATH}"
  ensure_port

  if is_running; then
    echo "[coder_vllm] Already running on port ${PORT} (PID: $(cat "${PID_FILE}"))."
    exit 0
  fi

  if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
    TENSOR_PARALLEL_SIZE="$(get_tp_from_cuda_devices)"
  fi

  local launch_cmd
  if command -v vllm >/dev/null 2>&1; then
    launch_cmd=(
      vllm serve "${MODEL_PATH}"
      --host "${HOST}"
      --port "${PORT}"
      --served-model-name "${SERVED_MODEL_NAME}"
      --dtype "${DTYPE}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    )
    if [[ -n "${MAX_MODEL_LEN}" ]]; then
      launch_cmd+=(--max-model-len "${MAX_MODEL_LEN}")
    fi
  else
    launch_cmd=(
      python -m vllm.entrypoints.openai.api_server
      --model "${MODEL_PATH}"
      --host "${HOST}"
      --port "${PORT}"
      --served-model-name "${SERVED_MODEL_NAME}"
      --dtype "${DTYPE}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    )
    if [[ -n "${MAX_MODEL_LEN}" ]]; then
      launch_cmd+=(--max-model-len "${MAX_MODEL_LEN}")
    fi
  fi

  if [[ -n "${CUDA_DEVICES}" ]]; then
    echo "[coder_vllm] Starting with CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" nohup bash -lc "${launch_cmd[*]} ${EXTRA_ARGS}" >"${LOG_FILE}" 2>&1 &
  else
    echo "[coder_vllm] Starting with current CUDA visibility"
    nohup bash -lc "${launch_cmd[*]} ${EXTRA_ARGS}" >"${LOG_FILE}" 2>&1 &
  fi

  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  sleep 2

  if kill -0 "${pid}" 2>/dev/null; then
    echo "[coder_vllm] Started. PID=${pid}, port=${PORT}"
    echo "[coder_vllm] Log: ${LOG_FILE}"
    echo "[coder_vllm] Set env for proposer:"
    echo "  export CODER_VLLM_URL=http://127.0.0.1:${PORT}"
    echo "  export CODER_MODEL_NAME=${SERVED_MODEL_NAME}"
  else
    rm -f "${PID_FILE}"
    die "Failed to start. Check log: ${LOG_FILE}"
  fi
}

stop_server() {
  ensure_port
  if ! is_running; then
    echo "[coder_vllm] No running server found for port ${PORT}."
    rm -f "${PID_FILE}"
    exit 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  echo "[coder_vllm] Stopping PID ${pid} ..."
  kill "${pid}" 2>/dev/null || true
  sleep 1
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  echo "[coder_vllm] Stopped."
}

status_server() {
  ensure_port
  if is_running; then
    echo "[coder_vllm] Running on port ${PORT} (PID: $(cat "${PID_FILE}"))."
    echo "[coder_vllm] Log: ${LOG_FILE}"
  else
    echo "[coder_vllm] Not running on port ${PORT}."
  fi
}

case "${ACTION}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  status)
    status_server
    ;;
  *)
    cat <<EOF
Usage:
  bash scripts/coder_vllm.sh start <model_path> [port] [cuda_visible_devices]
  bash scripts/coder_vllm.sh stop [port]
  bash scripts/coder_vllm.sh status [port]

Environment overrides:
  CODER_MODEL_NAME             default: coder-model
  CODER_VLLM_HOST              default: 0.0.0.0
  CODER_VLLM_PORT              default: 8000
  CODER_CUDA_DEVICES           example: 0 or 0,1
  CODER_VLLM_TP_SIZE           default: inferred from CUDA devices or 1
  CODER_VLLM_DTYPE             default: auto
  CODER_VLLM_GPU_MEM_UTIL      default: 0.90
  CODER_VLLM_MAX_MODEL_LEN     optional
  CODER_VLLM_EXTRA_ARGS        optional extra args, e.g. "--trust-remote-code"
  CODER_VLLM_LOG_DIR           default: ./logs/tmp
EOF
    exit 1
    ;;
esac
