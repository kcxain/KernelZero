model_path=$1
run_id=$2
export VLLM_DISABLE_COMPILE_CACHE=1
# Disable torch compile to avoid cache directory issues
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
# Force Python to use spawn method for multiprocessing (required for CUDA)
export PYTHON_MULTIPROCESSING_START_METHOD=spawn

echo "[start.sh] Starting vLLM servers..."
CUDA_VISIBLE_DEVICES=4 python vllm_service_init/start_vllm_server.py --port 5000 --model_path $model_path > ./logs/tmp/vllm_server_5000.log 2>&1 &
echo "[start.sh] Started server on port 5000 (PID: $!)"

CUDA_VISIBLE_DEVICES=5 python vllm_service_init/start_vllm_server.py --port 5001 --model_path $model_path > ./logs/tmp/vllm_server_5001.log 2>&1 &
echo "[start.sh] Started server on port 5001 (PID: $!)"

CUDA_VISIBLE_DEVICES=6 python vllm_service_init/start_vllm_server.py --port 5002 --model_path $model_path > ./logs/tmp/vllm_server_5002.log 2>&1 &
echo "[start.sh] Started server on port 5002 (PID: $!)"

CUDA_VISIBLE_DEVICES=7 python vllm_service_init/start_vllm_server.py --port 5003 --model_path $model_path > ./logs/tmp/vllm_server_5003.log 2>&1 &
echo "[start.sh] Started server on port 5003 (PID: $!)"

echo "[start.sh] All vLLM servers started, waiting 5 seconds for initialization..."
sleep 5
echo "[start.sh] vLLM servers should be ready now"