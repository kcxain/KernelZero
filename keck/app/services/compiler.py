import glob
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

from app.config import settings
from app.schemas import CompileResult
from app.utils import rewrite_load_inline


def compile_cuda_extension(cuda_code: str) -> Tuple[bool, CompileResult]:
    """
    编译用户提供的 CUDA 扩展代码

    流程:
    1. 把 cuda_code 写入临时目录的 model_new.py
    2. 执行 python model_new.py 触发 load_inline 编译
    3. 收集生成的 .so 文件
    4. 用 AST 重写生成 model_new_patch.py（把 load_inline 改为 import）

    Returns:
        (success, CompileResult)
    """
    cfg = settings.compile

    with tempfile.TemporaryDirectory() as tmpdir:
        model_new_path = os.path.join(tmpdir, "model_new.py")
        Path(model_new_path).write_text(cuda_code)

        compile_log = ""
        try:
            # 构建子进程环境：继承当前环境，覆盖编译相关变量
            # 注意：不使用 patch.dict(os.environ) 以避免多任务并发时的 race condition
            env = os.environ.copy()
            env.update({
                "CUDA_HOME": cfg.cuda_home,
                "TORCH_CUDA_ARCH_LIST": cfg.cuda_arch_list,
                "TORCH_EXTENSIONS_DIR": os.path.join(tmpdir, "build"),
                "MAX_JOBS": cfg.max_jobs,
                # 编译阶段不需要 GPU，禁止访问以实现严格的 CPU/GPU 隔离
                "CUDA_VISIBLE_DEVICES": "",
            })

            result = subprocess.run(
                    "python model_new.py",
                    timeout=cfg.compile_timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    cwd=tmpdir,
                    env=env,
                )

            compile_log = result.stdout.decode()

            # 查找生成的 .so 文件
            build_dir = os.path.join(tmpdir, "build")
            so_files = glob.glob(f"{build_dir}/**/*.so", recursive=True)
            if len(so_files) != 1:
                raise RuntimeError(
                    f"Expected 1 .so file, got {len(so_files)}: {so_files}"
                )

            # 读取编译产物
            ext_filename = os.path.basename(so_files[0])
            ext_content = Path(so_files[0]).read_bytes()

            # 生成 model_new_patch.py
            patch_path = os.path.join(tmpdir, "model_new_patch.py")
            rewrite_load_inline(src_path=model_new_path, dst_path=patch_path)
            model_new_patch = Path(patch_path).read_text()
            # print(model_new_patch)
            return True, CompileResult(
                success=True,
                ext_filename=ext_filename,
                ext_content=ext_content,
                model_new_patch=model_new_patch,
                message="compile success",
            )

        except subprocess.TimeoutExpired:
            return False, CompileResult(
                success=False,
                message="failed: compilation timed out",
            )
        except Exception as e:
            return False, CompileResult(
                success=False,
                message=f"failed: compilation error: [{e}] log: [{compile_log}]",
            )
