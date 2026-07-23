#!/usr/bin/env bash
#SBATCH -J serve_keck
#SBATCH -p r8nv-gpu-hw-80g
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 30:00:00
#SBATCH --exclude=r8a100-d05
#SBATCH -o logs/serve-keck-%j.out
#SBATCH -e logs/serve-keck-%j.err

set -euo pipefail

trap 'echo "[serve_keck] failed at line ${LINENO}: ${BASH_COMMAND} (exit $?)" >&2' ERR

KERNELZERO_ROOT="${KERNELZERO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
KECK_ROOT="${KECK_ROOT:-$KERNELZERO_ROOT/keck}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
VENV_PATH="${VENV_PATH:-$KERNELZERO_ROOT/.venv}"
CUDA_MODULE="${CUDA_MODULE:-cuda-cudnn/12.4-9.1.1}"
CUDA_HOME="${CUDA_HOME:-/tools/cluster-software/cuda-cudnn/cuda-12.4.1-9.1.1}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_CPUS="${NUM_CPUS:-8}"
COMPILER_CPUS="${COMPILER_CPUS:-2}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
KECK_GPU_MAX_CALLS="${KECK_GPU_MAX_CALLS:-1}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-0}"
RAY_PORT="${RAY_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/kzk${SLURM_JOB_ID}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${KECK_ROOT}/logs"
cd "${KECK_ROOT}"

echo "Job start at $(date '+%F %T')"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "KECK_ROOT=${KECK_ROOT}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"
echo "VENV_PATH=${VENV_PATH}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "NUM_CPUS=${NUM_CPUS}"
echo "COMPILER_CPUS=${COMPILER_CPUS}"
echo "UVICORN_WORKERS=${UVICORN_WORKERS}"
echo "MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS}"
echo "KECK_GPU_MAX_CALLS=${KECK_GPU_MAX_CALLS}"
echo "RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD}"
echo "RAY_PORT=${RAY_PORT}"
echo "RAY_TMPDIR=${RAY_TMPDIR}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

module use /tools/cluster-modulefiles || true
module load cluster-tools/v1.0 || true
module load slurm-tools/v1.0 || true
module unload cuda-cudnn cuda >/dev/null 2>&1 || true
module load "${CUDA_MODULE}"

unset PYTHONHOME || true
export CUDA_HOME="${CUDA_HOME}"
export NUM_GPUS="${NUM_GPUS}"
export NUM_CPUS="${NUM_CPUS}"
export COMPILER_CPUS="${COMPILER_CPUS}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS}"
export KECK_GPU_MAX_CALLS="${KECK_GPU_MAX_CALLS}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}"
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="${RAY_TMPDIR}"
unset RAY_ADDRESS || true
mkdir -p "${RAY_TMPDIR}"

if [[ -x "${VENV_PATH}/bin/python" ]]; then
  export VIRTUAL_ENV="${VENV_PATH}"
  export PATH="${VENV_PATH}/bin:${PATH}"
fi
hash -r

python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("ray", "uvicorn", "fastapi") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python modules: {missing}. Set VENV_PATH or install dependencies.")
print("Python executable:", sys.executable)
PY

nvidia-smi || true

cleanup() {
  echo "Stopping Keck service at $(date '+%F %T')"
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

ray stop --force >/dev/null 2>&1 || true
ray_start_cmd=(
  ray start
  --head
  --port="${RAY_PORT}"
  --temp-dir="${RAY_TMPDIR}"
  --num-cpus="${NUM_CPUS}"
  --num-gpus="${NUM_GPUS}"
)

if [[ "${RAY_INCLUDE_DASHBOARD}" == "1" ]]; then
  ray_start_cmd+=(--include-dashboard=true)
else
  ray_start_cmd+=(--include-dashboard=false)
fi

printf -v ray_start_cmd_str "%q " "${ray_start_cmd[@]}"
echo "CMD: ${ray_start_cmd_str}"
"${ray_start_cmd[@]}"

export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

echo "Launching Keck API..."
echo "Endpoint: http://$(hostname):${PORT}"
exec python -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" --workers "${UVICORN_WORKERS}"
