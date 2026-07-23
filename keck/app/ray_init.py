import ray
from app.config import settings


def init_ray() -> None:
    """初始化 Ray，支持连接已有集群"""
    if ray.is_initialized():
        return

    cfg = settings.ray
    runtime_env = {"env_vars": cfg.env_vars}

    try:
        # 首选带资源限制的初始化（适用于新启动的本地集群）
        ray.init(
            runtime_env=runtime_env,
            num_cpus=cfg.num_cpus,
            num_gpus=cfg.num_gpus,
        )
    except ValueError as e:
        # 如果正在连接到已有集群，不能传 num_cpus/num_gpus
        if "When connecting to an existing cluster" in str(e):
            ray.init(runtime_env=runtime_env)
        else:
            raise
