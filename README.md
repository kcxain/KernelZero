# KernelZero

KernelZero trains a pair of code-generation models that improve each other:
a **Proposer** constructs realistic PyTorch modules and a **Coder** generates
optimized Triton or CUDA implementations. The integrated **Keck** service
executes generated programs, checks correctness, and measures speedup.

## Training workflow

1. Sample a combination of Torch APIs from the co-occurrence distribution.
2. Generate a PyTorch module with the Proposer.
3. Generate multiple optimized kernels with the Coder.
4. Validate modules and kernels with Keck.
5. Train the Proposer with frontier reward and the Coder with
   correctness-aware performance reward.

See [docs/architecture.md](docs/architecture.md) for the data flow.

## Repository layout

| Path | Purpose |
| --- | --- |
| `kernelzero/` | Shared Proposer and dataset-quality utilities |
| `verl/` | GRPO training runtime |
| `examples/reward_function/` | Coder and Proposer rewards |
| `examples/format_prompt/` | Coder and Proposer prompt templates |
| `slurms/coder/` | Triton Coder training and KernelBench evaluation |
| `slurms/proposer/` | Proposer frontier-GRPO training |
| `slurms/data/` | PyTorch module generation and validation |
| `slurms/services/` | Coder vLLM service |
| `keck/` | Integrated correctness and performance validation service |
| `torch_evaluate/` | Static and execution-side PyTorch checks |
| `torch_generate/`, `torch_pool/` | Module generation and API sampling |
| `tests/` | Reward and integration tests |

Generated logs, checkpoints, analysis outputs, datasets, and local
environments are deliberately excluded from Git.

## Environment

The current cluster environment uses Python 3.12, PyTorch 2.6, CUDA 12.4,
Ray, vLLM, and the dependencies pinned in `uv.lock`.

```bash
uv sync
source .venv/bin/activate
```

Cluster jobs should be submitted from the repository root. A complete sequence
for starting Keck, serving the Coder, and training both models is in
[docs/cluster_quickstart.md](docs/cluster_quickstart.md).

## Main entry points

### Triton Coder GRPO

```bash
sbatch --export=ALL,\
EXP=<name>,\
MODEL=<base-model>,\
TRAIN_DATA=<modules.json>,\
KECK_BASE_URL=http://<keck-node>:8000 \
slurms/coder/train_triton_coder.slurm
```

This entry point retains the reward, sampling parameters, and 20-step
checkpoint cadence used by the current Triton experiments. Set
`REWARD_ALPHA` for the alpha ablation.

### Proposer frontier GRPO

```bash
sbatch --export=ALL,\
EXP=<name>,\
PROPOSER_MODEL=<sft-model>,\
TRAIN_DATA=<proposer-inputs.json>,\
CODER_VLLM_URL=http://<coder-node>:18080,\
KECK_BASE_URL=http://<keck-node>:8000 \
slurms/proposer/train_frontier_grpo.slurm
```

The Proposer reward checks requested API use and PyTorch executability before
sampling five Coder outputs and computing frontier reward.

### Keck

Keck is included in `keck/`; it is no longer maintained as a separate project
copy. Start it with:

```bash
sbatch --export=ALL,PORT=8000 keck/slurms/serve_keck.slurm
```

Use a separate Keck allocation for each concurrent RL experiment.

## Storage policy

Model weights and framework caches must live on `/nfs_global`. The maintained
training scripts default to:

```text
<repository>/checkpoints/<experiment>
<repository>/.cache/<training-component>
```

Do not store checkpoints under `/share/personal`. Resume only from a complete
checkpoint containing actor model shards, optimizer shards, extra state,
Hugging Face export, and dataloader state.

## Validation

Run lightweight checks before submitting expensive jobs:

```bash
uv run --with pytest pytest \
  tests/test_proposer_triton_reward.py \
  tests/test_proposer_quality_checks.py
python scripts/test_proposer_triton_integration.py --help
bash -n slurms/coder/train_triton_coder.slurm
bash -n slurms/proposer/train_frontier_grpo.slurm
bash -n keck/slurms/serve_keck.slurm
```

## Acknowledgments

KernelZero builds on [R-Zero](https://github.com/Chengsong-Huang/R-Zero) and
uses KernelBench-style PyTorch modules for kernel-generation training and
evaluation.
