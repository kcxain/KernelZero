# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
combination_file=$4
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHON_MULTIPROCESSING_START_METHOD=spawn
export VLLM_USE_V1=0

python3 -c "
import subprocess
import sys
import os

model_name = sys.argv[1]
num_samples = sys.argv[2]
save_name = sys.argv[3]
combination_file = sys.argv[4]
# Ensure environment variables are set
env = os.environ.copy()
env['VLLM_DISABLE_COMPILE_CACHE'] = '1'
env['PYTHON_MULTIPROCESSING_START_METHOD'] = 'spawn'
env['VLLM_USE_V1'] = '0'  # Disable V1 engine to avoid fork issues

processes = []
for gpu_id in range(8):
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cmd = [
        sys.executable,
        'torch_generate/question_generate.py',
        '--model', model_name,
        '--suffix', str(gpu_id),
        '--num_samples', num_samples,
        '--combination_file', combination_file,
        '--save_name', save_name
    ]
    p = subprocess.Popen(cmd, env=env)
    processes.append(p)

# Wait for all processes to complete
for p in processes:
    p.wait()
" "$model_name" "$num_samples" "$save_name" "$combination_file"
