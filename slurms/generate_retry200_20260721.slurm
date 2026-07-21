#!/bin/bash

#- Job parameters

#SBATCH -J kz_proposer_retry200
#SBATCH -o logs/kz_proposer_retry200-%j.out
#SBATCH -e logs/kz_proposer_retry200-%j.err


#- Resources

#SBATCH -p r8nv-gpu-hw-80g               # Submit to 'r8nv-gpu-hw-80g' Partition (A100 80GB)
#SBATCH -t 0-6:00:00                # Run for a maximum time of 0 days, 24 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request 1 node
#SBATCH --gres=gpu:8                 # Request 8 GPUs per node
#SBATCH --qos=gpu-short              # Same A100-80G partition; 6-hour queue
### #SBATCH --mem=512G                   # CPU 内存 (降低内存请求，避免资源不足)
#SBATCH --cpus-per-task=32           # Fit the current A100-80G node CPU/GPU binding
### #SBATCH --constraint="L40S"
### #SBATCH --nodelist=r8l40s-a01        # Request a specific list of hosts 
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

#- Project virtualenv
source .venv/bin/activate

##- CUDA
module load cuda-cudnn/12.4-9.1.1
export CUDA_HOME=/tools/cluster-software/cuda-cudnn/cuda-12.4.1-9.1.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

#- storage path
export STORAGE_PATH=./train_model/kernelzero/v0
#  建立存储文件
mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"

##- virtualenv
# 初始化 conda（如果尚未初始化）
# if [ -f "/workspace/S/fangzixiang/miniconda/etc/profile.d/conda.sh" ]; then
#     source /workspace/S/fangzixiang/miniconda/etc/profile.d/conda.sh
# elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
#     source $HOME/miniconda/etc/profile.d/conda.sh
# fi

# # 激活 conda 环境
# conda activate r-zero || {
#     echo "✗ Failed to activate conda environment 'r-zero'. Exiting."
#     exit 1
# }

# # 验证 conda 环境是否成功激活
# if [ -z "$CONDA_DEFAULT_ENV" ]; then
#     echo "✗ Warning: Conda environment may not be activated properly."
#     echo "Current Python: $(which python)"
# else
#     echo "✓ Conda environment activated: $CONDA_DEFAULT_ENV"
#     echo "✓ Python path: $(which python)"
# fi

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

ray stop --force
ray start --head

export STORAGE_PATH=./train_model/kernelzero/proposer_retry200_20260721
# 启动脚本
bash torch_generate/question_generate.bash /nfs_global/S/kechangxin/proj/llmpiler/checkpoints/qwen2.5-coder-cudallm_5k 1 retry200 resources/sampled_torch_ops/retry200_20260721.json

echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
