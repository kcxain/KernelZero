#!/bin/bash

#- Job parameters

#SBATCH -J R-Zero                  # The job name
#SBATCH -o logs/R-Zero-%j.out         # Write the standard output to file named 'train-<job_number>.out'
#SBATCH -e logs/R-Zero-%j.err         # Write the standard error to file named 'train-<job_number>.err'


#- Resources

#SBATCH -p r8nv-gpu-hw-80g           # Submit to 'r8nv-gpu-hw-80g' Partition (A100 80GB)
#SBATCH -t 0-30:00:00                # Run for a maximum time of 0 days, 24 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request 1 node
#SBATCH --gres=gpu:8                 # Request 8 GPUs per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --qos=gpu-normal             # Request QOS Type
#SBATCH --mem=512G                   # CPU 内存 (降低内存请求，避免资源不足)
#SBATCH --cpus-per-task=64           # CPU 核数 (8 GPUs × 8 cores = 64 cores)         
#SBATCH --exclude=r8a100-d00
#SBATCH --exclude=r8a100-d05         # Exclude broken node d05
#SBATCH --exclude=r8l40s-a04
#SBATCH --exclude=r8l40s-a02
###SBATCH --nodelist=r8a100-c05   
###SBATCH --constraint="L40S"      # Request a specific list of hosts 
###
### The system will alloc 8 or 16 cores per gpu by default.
### If you need more or less, use following:
### #SBATCH --cpus-per-task=K            # Request K cores
###
### 
### Without specifying the constraint, any available nodes that meet the requirement will be allocated
### You can specify the characteristics of the compute nodes, and even the names of the compute nodes
###
### #SBATCH --nodelist=r8a30-a0          # Request a specific list of hosts 
### #SBATCH --constraint="A30|A100"      # Request GPU Type: A30 or A100_40GB
###

#- Log information

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

##- Tools
module load cluster-tools/v1.0
module load slurm-tools/v1.0

##- CUDA
module load cuda-cudnn/12.4-9.1.1
export CUDA_HOME=/tools/cluster-software/cuda-cudnn/cuda-12.4.1-9.1.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

#- storage path
export STORAGE_PATH=/nfs_global/S/fangzixiang/train_model/R-zero/test
#  建立存储文件
mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"

##- virtualenv
# 初始化 conda（如果尚未初始化）
if [ -f "/workspace/S/fangzixiang/miniconda/etc/profile.d/conda.sh" ]; then
    source /workspace/S/fangzixiang/miniconda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda/etc/profile.d/conda.sh
fi

# 激活 conda 环境
conda activate r-zero || {
    echo "✗ Failed to activate conda environment 'r-zero'. Exiting."
    exit 1
}

# 验证 conda 环境是否成功激活
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "✗ Warning: Conda environment may not be activated properly."
    echo "Current Python: $(which python)"
else
    echo "✓ Conda environment activated: $CONDA_DEFAULT_ENV"
    echo "✓ Python path: $(which python)"
fi

#- Other

cluster-quota                 

nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit 


echo "Using GPU(s) ${CUDA_VISIBLE_DEVICES}"                    
echo "This job is assigned the following resources by SLURM:"
scontrol show jobid $SLURM_JOB_ID -dd | awk '/IDX/ {print $2, $4}'

unset http_proxy
unset https_proxy
unset no_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset NO_PROXY
echo -e "\033[31m[×] 已关闭代理\033[0m"

# 设置环境变量（在 Ray 启动之前设置，以便传递给 worker 进程）
export CUDA_SERVER_IP="10.200.198.13"
export REWARD_MAX_WORKERS=128  

ray stop --force
ray start --head

# 启动脚本
bash scripts/main.sh /nfs_global/S/fangzixiang/train_model/Qwen2.5-Coder-7B-Instruct-CUDA-speedup/full/speed_up_batch_234/checkpoint-606/ qwen2.5-coder-7b-instruct_speedup_update
# 测试用
#bash scripts/main.sh /share/collab/codemodel/models/Qwen/Qwen2.5-Coder-1.5B-Instruct qwen2.5-coder-1.5b-instruct
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
