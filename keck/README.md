# Keck validation service

Keck is the execution backend used by KernelZero rewards. It exposes:

- `POST /check_torch` to validate generated PyTorch modules.
- `POST /compute_score` to compile, execute, and compare CUDA or Triton kernels.
- `GET /healthz` for service readiness checks.

The service runs generated code in Ray workers and should be deployed on a
dedicated GPU allocation. Do not share one Keck instance between independent
RL jobs because request queues and worker failures would couple the runs.

From the KernelZero repository root:

```bash
mkdir -p logs
sbatch --export=ALL,PORT=8000 keck/slurms/serve_keck.slurm
```

Useful environment variables include `PORT`, `NUM_GPUS`, `NUM_CPUS`,
`COMPILER_CPUS`, `MAX_CONCURRENT_REQUESTS`, `KECK_GPU_MAX_CALLS`, and
`RAY_TMPDIR`. The default virtual environment is the repository `.venv`.

The standalone historical service directory is kept outside this repository
until all active jobs finish. New jobs should use this integrated copy.
