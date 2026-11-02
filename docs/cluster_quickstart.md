# Cluster quick start

Run all commands from the repository root and activate the checked environment:

```bash
source .venv/bin/activate
mkdir -p logs checkpoints
```

Configure site-specific modules once:

```bash
export MODULE_INIT_SCRIPT=/path/to/modules-init.sh
export CUDA_MODULE=cuda/12.4
export VENV_PATH="$PWD/.venv"
```

Pass `--partition`, `--account`, and `--qos` to each `sbatch` command according
to the local Slurm configuration.

## Start Keck

```bash
sbatch --partition=<gpu-partition> \
  --export=ALL,PORT=8000 \
  keck/slurms/serve_keck.slurm
```

Wait for the job to start, then verify:

```bash
curl http://<keck-node>:8000/healthz
```

## Start the Coder service

```bash
sbatch --partition=<gpu-partition> \
  --export=ALL,MODEL_PATH=<merged-coder-checkpoint>,PORT=18080 \
  slurms/services/serve_coder_vllm.slurm
```

## Train the Coder

```bash
sbatch --partition=<gpu-partition> --export=ALL,\
EXP=<experiment>,\
MODEL=<base-model>,\
TRAIN_DATA=<modules.json>,\
KECK_BASE_URL=http://<keck-node>:8000 \
slurms/coder/train_triton_coder.slurm
```

Override `REWARD_ALPHA` for reward ablations. The default reward is the
fail-fast Triton reward used by the current experiments.

## Train the Proposer

```bash
sbatch --partition=<gpu-partition> --export=ALL,\
EXP=<experiment>,\
PROPOSER_MODEL=<sft-model>,\
TRAIN_DATA=<proposer-inputs.json>,\
CODER_VLLM_URL=http://<coder-node>:18080,\
KECK_BASE_URL=http://<keck-node>:8000 \
slurms/proposer/train_frontier_grpo.slurm
```

To resume, add `LOAD_CHECKPOINT=<complete-global-step-directory>`. Checkpoints
and framework caches default to repository-local ignored directories. Set
`SAVE_DIR` and `CACHE_ROOT` to mounted high-capacity storage for production runs.

A resumable directory contains actor model shards, optimizer shards,
extra state, a Hugging Face export, and dataloader state.
