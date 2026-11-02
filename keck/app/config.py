import os
from dataclasses import dataclass, field
from pathlib import Path

_APP_DIR = Path(__file__).parent


@dataclass
class RayConfig:
    """Ray 集群配置"""

    num_cpus: int = field(default_factory=lambda: int(os.getenv("NUM_CPUS", "64")))
    num_gpus: int = field(default_factory=lambda: int(os.getenv("NUM_GPUS", "8")))
    env_vars: dict = field(
        default_factory=lambda: {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "0",
            "BPEX_NO_WARN_ON_UNTUNED_CASE": "1",
        }
    )


@dataclass
class CompileConfig:
    """CUDA 编译配置"""

    # 编译任务占用的 CPU 核数
    compiler_cpus: int = field(
        default_factory=lambda: int(os.getenv("COMPILER_CPUS", "4"))
    )
    # 编译超时时间（秒）
    compile_timeout: int = field(
        default_factory=lambda: int(os.getenv("COMPILE_TIMEOUT", "180"))
    )
    # CUDA 安装路径
    cuda_home: str = field(default_factory=lambda: os.getenv("CUDA_HOME", "/usr/local/cuda"))
    # 目标 CUDA 架构（A100 = 8.0）
    cuda_arch_list: str = field(
        default_factory=lambda: os.getenv("TORCH_CUDA_ARCH_LIST", "8.0")
    )
    # nvcc 并行编译任务数：默认与 compiler_cpus 对齐，充分利用已分配的 CPU 核数
    max_jobs: str = field(
        default_factory=lambda: os.getenv(
            "MAX_JOBS", os.getenv("COMPILER_CPUS", "4")
        )
    )


@dataclass
class ExecuteConfig:
    """GPU 执行配置"""

    # 执行超时时间（秒）
    exec_timeout: int = field(
        default_factory=lambda: int(os.getenv("EXEC_TIMEOUT", "120"))
    )
    # GPU 资源占用（1 = 独占一张卡，0.5 = 两个任务共享一张卡）
    execute_gpus: float = field(
        default_factory=lambda: float(os.getenv("EXECUTE_GPUS_PER_TASK", "1"))
    )


@dataclass
class Settings:
    """全局配置"""

    ray: RayConfig = field(default_factory=RayConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    execute: ExecuteConfig = field(default_factory=ExecuteConfig)
    max_concurrent_requests: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))
    )

    # 模板文件路径（绝对路径，避免依赖启动目录）
    templates_dir: str = field(default_factory=lambda: str(_APP_DIR / "templates"))


# 全局单例
settings = Settings()
