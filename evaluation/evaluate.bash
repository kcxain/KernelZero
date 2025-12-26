#!/bin/bash
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHON_MULTIPROCESSING_START_METHOD=spawn
export VLLM_USE_V1=0
model_name=$1

MODEL_NAMES=(
  $model_name
)

TASKS=(
  "math"
  "gsm8k" 
  "amc"
  "minerva"
  "olympiad"
  "aime2024"
  "aime2025"
)

# 使用 Python 来管理并行任务，避免 bash fork 问题
python3 -c "
import subprocess
import sys
import os
import time
from datetime import datetime

model_name = sys.argv[1]
tasks = sys.argv[2:]

# Ensure environment variables are set
env = os.environ.copy()
env['VLLM_DISABLE_COMPILE_CACHE'] = '1'
env['PYTHON_MULTIPROCESSING_START_METHOD'] = 'spawn'
env['VLLM_USE_V1'] = '0'

# Get available GPUs
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    gpu_queue = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
except:
    gpu_queue = ['0']  # Fallback to single GPU

print(f'Available GPUs: {gpu_queue}')

processes = {}  # {gpu_id: (process, task)}

task_index = 0
while task_index < len(tasks) or processes:
    # Start new tasks if GPUs are available
    while gpu_queue and task_index < len(tasks):
        gpu_id = gpu_queue.pop(0)
        task = tasks[task_index]
        task_index += 1
        
        print(f'==> [{datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}] Start task [{task}] with model [{model_name}] on GPU [{gpu_id}] ...')
        
        cmd = [sys.executable, 'evaluation/generate.py', '--model', model_name, '--dataset', task]
        env['CUDA_VISIBLE_DEVICES'] = gpu_id
        p = subprocess.Popen(cmd, env=env)
        processes[gpu_id] = (p, task)
    
    # Check for finished processes
    for gpu_id in list(processes.keys()):
        p, task = processes[gpu_id]
        if p.poll() is not None:  # Process finished
            print(f'==> [{datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}] GPU [{gpu_id}] job finished (task: {task}).')
            del processes[gpu_id]
            gpu_queue.append(gpu_id)
    
    if task_index < len(tasks) or processes:
        time.sleep(1)
" "$model_name" "${TASKS[@]}"

# 使用 Python subprocess 而不是 bash & 来避免 fork 问题
python3 -c "
import subprocess
import sys
import os

model_name = sys.argv[1]

# Ensure environment variables are set
env = os.environ.copy()
env['VLLM_DISABLE_COMPILE_CACHE'] = '1'
env['PYTHON_MULTIPROCESSING_START_METHOD'] = 'spawn'
env['VLLM_USE_V1'] = '0'  # Disable V1 engine to avoid fork issues

# Run results_recheck in background
p1 = subprocess.Popen(
    [sys.executable, 'evaluation/results_recheck.py', '--model_name', model_name],
    env=env
)

# Run evaluation scripts sequentially (they use tensor_parallel_size=4, so they need multiple GPUs)
scripts = [
    ['evaluation/eval_supergpqa.py', '--model_path', model_name],
    ['evaluation/eval_bbeh.py', '--model_path', model_name],
    ['evaluation/eval_mmlupro.py', '--model_path', model_name],
]

for script in scripts:
    p = subprocess.Popen([sys.executable] + script, env=env)
    p.wait()

# Wait for results_recheck to finish
p1.wait()
" "$model_name"


echo "==> All tasks have finished!"
