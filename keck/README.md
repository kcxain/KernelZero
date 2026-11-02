# Keck validation service

Keck is the execution backend used by KernelZero rewards. It exposes:

- `POST /check_torch` to validate generated PyTorch modules.
- `POST /compute_score` to compile, execute, and compare CUDA or Triton kernels.
- `GET /healthz` for service readiness checks.

The service runs generated code in Ray workers on a dedicated GPU allocation.
Each independent RL job uses its own Keck instance, which keeps request queues
and worker lifecycles isolated.

From the KernelZero repository root:

```bash
mkdir -p logs
sbatch --partition=<gpu-partition> \
  --export=ALL,PORT=8000 \
  keck/slurms/serve_keck.slurm
```

Useful environment variables include `PORT`, `NUM_GPUS`, `NUM_CPUS`,
`COMPILER_CPUS`, `MAX_CONCURRENT_REQUESTS`, `KECK_GPU_MAX_CALLS`, and
`RAY_TMPDIR`. The default virtual environment is the repository `.venv`.
Site-specific module initialization is configured through
`MODULE_INIT_SCRIPT`, `CLUSTER_MODULES`, `CUDA_MODULE`, `CUDA_HOME`, and
`VENV_PATH`.

After Slurm assigns a node, verify readiness with:

```bash
export KECK_BASE_URL=http://<keck-node>:8000
curl -fsS "$KECK_BASE_URL/healthz" | python -m json.tool
```

`POST /compute_score` accepts `torch_code` plus `cuda_code` or `triton_code` as
multipart form fields. `POST /check_torch` accepts `gt_torch` and
`to_check_torch`.
