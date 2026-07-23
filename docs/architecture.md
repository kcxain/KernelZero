# Architecture

KernelZero alternates between two learned components:

1. The **Proposer** generates PyTorch modules from sampled Torch API
   combinations.
2. The **Coder** generates Triton implementations for those modules.
3. **Keck** executes the reference module and generated kernel, checks
   correctness, and measures performance.

Coder GRPO uses correctness and speedup from Keck. Proposer frontier GRPO first
checks syntax, requested API use, and PyTorch executability, then queries the
fixed Coder for multiple Triton implementations. Modules near the Coder's
correctness frontier receive the largest reward.

Runtime data flows through HTTP:

```text
API combinations -> Proposer -> PyTorch module
                                  |
                                  v
                            Coder vLLM service
                                  |
                                  v
                         Triton candidates -> Keck
                                  |
                                  v
                    correctness, speedup, frontier reward
```

The Proposer, Coder vLLM server, and Keck use separate Slurm jobs. Independent
training runs should use independent Keck services.
