# Data layout

Large datasets and generated artifacts are not committed to Git. The following
directories are expected to be mounted or created locally:

| Path | Contents |
| --- | --- |
| `resources/torch_operators/` | Canonical Torch API inventory |
| `resources/sampled_torch_ops/` | Sampled API combinations |
| `resources/KernelBench/` | KernelBench reference modules |
| `train_model/` | Generated Proposer/Coder training datasets |
| `analysis/` | Experiment-specific metrics and figures |
| `runs/` | Evaluation outputs |
| `checkpoints/` | Model and optimizer checkpoints |

Keep reusable API inventories and final datasets. Treat raw model responses,
retry shards, temporary validation outputs, logs, and partial checkpoints as
experiment artifacts. Archive them outside the repository before removal.

The maintained Slurm scripts accept data paths through environment variables;
they do not require user-specific absolute paths.
