"""Runtime compatibility shims for cluster training jobs.

Python imports ``sitecustomize`` automatically when this repository is on
``PYTHONPATH``.  Keep patches tiny and gated so normal local tooling is not
surprised.
"""

from __future__ import annotations

import importlib
import copy
import functools
import os
import subprocess
import warnings
from functools import lru_cache
from types import SimpleNamespace

warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\. Please install nvidia-ml-py instead\.",
    category=FutureWarning,
)


def _enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off", ""}


def _split_visible_devices(value: str | None) -> list[str]:
    if not value:
        return []
    value = value.strip()
    if not value or value == "NoDevFiles":
        return []
    return [device.strip() for device in value.split(",") if device.strip()]


@lru_cache(maxsize=1)
def _gpu_uuid_by_index() -> dict[str, str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}

    mapping: dict[str, str] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        index, uuid = parts
        if index and uuid:
            mapping[index] = uuid
    return mapping


@lru_cache(maxsize=1)
def _gpu_index_by_uuid() -> dict[str, str]:
    return {uuid: index for index, uuid in _gpu_uuid_by_index().items()}


def _canonical_cuda_visible_device(device: str) -> str:
    device = device.strip()
    if not device:
        return device
    if not _enabled("KECK_RAY_USE_GPU_UUIDS", "1"):
        return device
    if not device.isdigit():
        return device

    uuid = _gpu_uuid_by_index().get(device)
    return uuid or device


def _physical_cuda_device_index(device: str) -> str:
    device = device.strip()
    if not device:
        return device
    if device.isdigit():
        return device
    if device.startswith("GPU-"):
        return _gpu_index_by_uuid().get(device, device)
    return device


def _patch_nccl_env_defaults() -> None:
    if not _enabled("KECK_PATCH_NCCL_ENV_DEFAULTS", "1"):
        return

    os.environ.setdefault("NCCL_DEBUG", "WARN")
    # NCCL 2.30+ can allow multiple ranks on one physical GPU.  Older NCCL
    # builds ignore this, so the Slurm script still preflights duplicate buses.
    os.environ.setdefault("NCCL_MULTI_RANK_GPU_ENABLE", "1")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def _assign_ray_worker_cuda_visible_devices() -> bool:
    if os.environ.get("WG_BACKEND") != "ray":
        return False

    rank_value = os.environ.get("RANK")
    local_world_size_value = os.environ.get("RAY_LOCAL_WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE")
    if rank_value is None or local_world_size_value is None:
        return False

    try:
        rank = int(rank_value)
        local_world_size = max(1, int(local_world_size_value))
    except ValueError:
        return False

    visible_devices = _split_visible_devices(os.environ.get("KECK_PARENT_CUDA_VISIBLE_DEVICES"))
    if not visible_devices:
        visible_devices = _split_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if len(visible_devices) < local_world_size and _enabled("KECK_ALLOW_SYNTHETIC_CUDA_VISIBLE_DEVICES", "1"):
        visible_devices = [str(index) for index in range(local_world_size)]
    if not visible_devices:
        return False

    local_rank = rank % local_world_size
    assigned_device = visible_devices[local_rank % len(visible_devices)]
    assigned_device = _canonical_cuda_visible_device(assigned_device)
    previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned_device
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["KECK_ASSIGNED_CUDA_VISIBLE_DEVICES"] = assigned_device

    if _enabled("KECK_LOG_RAY_WORKER_CUDA_VISIBLE_DEVICES", "1"):
        import sys

        print(
            "[keck] ray worker cuda mapping: "
            f"rank={rank} local_rank={local_rank} "
            f"parent={','.join(visible_devices)} "
            f"previous={previous_visible_devices or 'unset'} "
            f"assigned={assigned_device}",
            file=sys.stderr,
            flush=True,
        )
    return True


def _patch_ray_worker_cuda_visible_devices() -> None:
    if not _enabled("KECK_PATCH_RAY_WORKER_CUDA_VISIBLE_DEVICES"):
        return

    if _enabled("KECK_PATCH_RAY_WORKER_CUDA_VISIBLE_DEVICES_EARLY"):
        _assign_ray_worker_cuda_visible_devices()
        return

    target_module_name = "verl.single_controller.base.worker"

    def _patch_module(module):  # type: ignore[no-untyped-def]
        cls = getattr(module, "Worker", None)
        if cls is None or getattr(cls, "_keck_cuda_visible_patch", False):
            return

        original_setup = cls._setup_env_cuda_visible_devices

        def patched_setup(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            result = original_setup(self, *args, **kwargs)
            _assign_ray_worker_cuda_visible_devices()
            return result

        cls._setup_env_cuda_visible_devices = patched_setup
        cls._keck_cuda_visible_patch = True

    try:
        import builtins
        import sys
    except Exception:
        return

    existing_module = sys.modules.get(target_module_name)
    if existing_module is not None:
        _patch_module(existing_module)
        return

    if getattr(builtins, "_keck_verl_worker_cuda_visible_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        target_module = sys.modules.get(target_module_name)
        if target_module is not None:
            _patch_module(target_module)
        return module

    builtins.__import__ = patched_import
    builtins._keck_verl_worker_cuda_visible_import_patch = True


def _patch_ray_accelerator_ids_uuid_compat() -> None:
    if not _enabled("KECK_RAY_USE_GPU_UUIDS", "1"):
        return

    target_module_name = "ray._private.worker"

    def _patch_module(module):  # type: ignore[no-untyped-def]
        worker_cls = getattr(module, "Worker", None)
        if worker_cls is None or getattr(worker_cls, "_keck_uuid_accelerator_patch", False):
            return

        original_method = getattr(worker_cls, "get_accelerator_ids_for_accelerator_resource", None)
        if original_method is None:
            return

        def patched_method(self, resource_name=None, resource_regex=None):  # type: ignore[no-untyped-def]
            try:
                return original_method(self, resource_name, resource_regex)
            except IndexError:
                visible = _split_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
                assigned = os.environ.get("KECK_ASSIGNED_CUDA_VISIBLE_DEVICES", "").strip()
                fallback = []
                if assigned:
                    fallback = [assigned]
                elif len(visible) == 1:
                    fallback = visible
                if fallback:
                    return {str(item) for item in fallback}
                raise

        worker_cls.get_accelerator_ids_for_accelerator_resource = patched_method
        worker_cls._keck_uuid_accelerator_patch = True

    try:
        import builtins
        import sys
    except Exception:
        return

    existing_module = sys.modules.get(target_module_name)
    if existing_module is not None:
        _patch_module(existing_module)
        return

    if getattr(builtins, "_keck_ray_uuid_accelerator_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        target_module = sys.modules.get(target_module_name)
        if target_module is not None:
            _patch_module(target_module)
        return module

    builtins.__import__ = patched_import
    builtins._keck_ray_uuid_accelerator_import_patch = True


def _patch_vllm_uuid_cuda_visible_devices() -> None:
    if not _enabled("KECK_PATCH_VLLM_UUID_VISIBLE_DEVICES", "1"):
        return

    target_module_names = {
        "vllm.platforms.interface",
        "vllm.platforms.cuda",
    }

    def _fallback_physical_device_id(device_id):  # type: ignore[no-untyped-def]
        try:
            logical_index = int(device_id)
        except Exception:
            logical_index = 0
        visible_devices = _split_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
        if 0 <= logical_index < len(visible_devices):
            physical = _physical_cuda_device_index(visible_devices[logical_index])
            if physical.isdigit():
                return int(physical)
        assigned = _physical_cuda_device_index(os.environ.get("KECK_ASSIGNED_CUDA_VISIBLE_DEVICES", ""))
        if assigned.isdigit():
            return int(assigned)
        raise ValueError(f"Unable to resolve CUDA device id {device_id!r} from UUID mapping")

    def _patch_callable(owner, attr_name: str) -> bool:  # type: ignore[no-untyped-def]
        descriptor = getattr(owner, "__dict__", {}).get(attr_name)
        if descriptor is None:
            return False

        if isinstance(descriptor, classmethod):
            original = descriptor.__func__

            def patched(cls, device_id, *args, **kwargs):  # type: ignore[no-untyped-def]
                try:
                    return original(cls, device_id, *args, **kwargs)
                except ValueError as exc:
                    if "invalid literal for int()" not in str(exc):
                        raise
                    return _fallback_physical_device_id(device_id)

            setattr(owner, attr_name, classmethod(patched))
            return True

        if isinstance(descriptor, staticmethod):
            original = descriptor.__func__

            def patched(device_id, *args, **kwargs):  # type: ignore[no-untyped-def]
                try:
                    return original(device_id, *args, **kwargs)
                except ValueError as exc:
                    if "invalid literal for int()" not in str(exc):
                        raise
                    return _fallback_physical_device_id(device_id)

            setattr(owner, attr_name, staticmethod(patched))
            return True

        if callable(descriptor):
            original = descriptor

            def patched(device_id, *args, **kwargs):  # type: ignore[no-untyped-def]
                try:
                    return original(device_id, *args, **kwargs)
                except ValueError as exc:
                    if "invalid literal for int()" not in str(exc):
                        raise
                    return _fallback_physical_device_id(device_id)

            setattr(owner, attr_name, patched)
            return True

        return False

    def _patch_module(module):  # type: ignore[no-untyped-def]
        if getattr(module, "_keck_uuid_visible_devices_patch", False):
            return

        patched_any = False
        if hasattr(module, "device_id_to_physical_device_id"):
            patched_any = _patch_callable(module, "device_id_to_physical_device_id") or patched_any

        for value in vars(module).values():
            if isinstance(value, type) and hasattr(value, "device_id_to_physical_device_id"):
                patched_any = _patch_callable(value, "device_id_to_physical_device_id") or patched_any

        if patched_any:
            module._keck_uuid_visible_devices_patch = True

    try:
        import builtins
        import sys
    except Exception:
        return

    for module_name in target_module_names:
        existing_module = sys.modules.get(module_name)
        if existing_module is not None:
            _patch_module(existing_module)

    if getattr(builtins, "_keck_vllm_uuid_visible_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        for module_name in target_module_names:
            target_module = sys.modules.get(module_name)
            if target_module is not None:
                _patch_module(target_module)
        return module

    builtins.__import__ = patched_import
    builtins._keck_vllm_uuid_visible_import_patch = True


def _patch_vllm_lora_module_alias() -> None:
    if not _enabled("KECK_PATCH_VLLM_LORA_ALIAS", "1"):
        return

    try:
        import builtins
        import sys
    except Exception:
        return

    if getattr(builtins, "_keck_vllm_lora_alias_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "vllm.lora.lora_model":
            try:
                module = importlib.import_module("vllm.lora.models")
            except ModuleNotFoundError:
                pass
            else:
                sys.modules.setdefault("vllm.lora.lora_model", module)
                return module
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = patched_import
    builtins._keck_vllm_lora_alias_import_patch = True


def _patch_transformers_tokenizer_special_tokens() -> None:
    if not _enabled("KECK_PATCH_TOKENIZER_COMPAT"):
        return

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    @property
    def all_special_tokens_extended(self):  # type: ignore[no-untyped-def]
        special_map = getattr(self, "special_tokens_map_extended", None)
        if not special_map:
            special_map = getattr(self, "special_tokens_map", {})

        tokens = []
        seen = set()
        for value in special_map.values():
            values = value if isinstance(value, (list, tuple)) else [value]
            for token in values:
                key = str(token)
                if key in seen:
                    continue
                seen.add(key)
                tokens.append(token)
        return tokens

    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


def _patch_qwen35_text_config_for_vllm_mrope() -> None:
    if not _enabled("KECK_PATCH_QWEN35_VLLM_COMPAT"):
        return

    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    except Exception:
        return

    if getattr(Qwen3_5TextConfig, "_keck_vllm_mrope_patch", False):
        return

    original_getattribute = Qwen3_5TextConfig.__getattribute__

    def _default_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            return default

    def _namespace_from_mapping(value):  # type: ignore[no-untyped-def]
        if isinstance(value, dict):
            merged = dict(value)
            merged.setdefault(
                "spatial_merge_size",
                _default_int("VLLM_VISION_SPATIAL_MERGE_SIZE", 2),
            )
            return SimpleNamespace(**merged)
        if value is None:
            return SimpleNamespace(
                spatial_merge_size=_default_int("VLLM_VISION_SPATIAL_MERGE_SIZE", 2)
            )
        if not hasattr(value, "spatial_merge_size"):
            try:
                setattr(
                    value,
                    "spatial_merge_size",
                    _default_int("VLLM_VISION_SPATIAL_MERGE_SIZE", 2),
                )
            except Exception:
                return SimpleNamespace(
                    spatial_merge_size=_default_int("VLLM_VISION_SPATIAL_MERGE_SIZE", 2)
                )
        return value

    def patched_getattribute(self, name):  # type: ignore[no-untyped-def]
        if name == "vision_config":
            try:
                value = original_getattribute(self, name)
            except AttributeError:
                value = None
            return _namespace_from_mapping(value)

        defaults = {
            "image_token_id": ("VLLM_IMAGE_TOKEN_ID", -1),
            "video_token_id": ("VLLM_VIDEO_TOKEN_ID", -1),
            "vision_start_token_id": ("VLLM_VISION_START_TOKEN_ID", -1),
        }
        if name in defaults:
            try:
                return original_getattribute(self, name)
            except AttributeError:
                env_name, default = defaults[name]
                return _default_int(env_name, default)

        return original_getattribute(self, name)

    Qwen3_5TextConfig.__getattribute__ = patched_getattribute
    Qwen3_5TextConfig._keck_vllm_mrope_patch = True


def _patch_verl_flash_attn_bert_padding_fallback() -> None:
    if not _enabled("KECK_PATCH_FLASH_ATTN_FALLBACK"):
        return

    try:
        from verl.utils import attention_utils
    except Exception:
        return

    if getattr(attention_utils, "_keck_flash_attn_fallback_patch", False):
        return

    def index_first_axis(input_tensor, indices):  # type: ignore[no-untyped-def]
        return input_tensor.index_select(0, indices.to(device=input_tensor.device).long())

    def pad_input(hidden_states, indices, batch, seqlen, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs

        trailing_shape = tuple(hidden_states.shape[1:])
        output = hidden_states.new_zeros((batch * seqlen, *trailing_shape))
        output[indices.to(device=hidden_states.device).long()] = hidden_states
        return output.reshape(batch, seqlen, *trailing_shape)

    def rearrange(*args, **kwargs):  # type: ignore[no-untyped-def]
        from einops import rearrange as einops_rearrange

        return einops_rearrange(*args, **kwargs)

    def unpad_input(hidden_states, attention_mask, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs

        import torch

        attention_mask = attention_mask.to(dtype=torch.bool)
        batch, seqlen = attention_mask.shape
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).reshape(-1)
        cu_seqlens = torch.zeros(
            seqlens.numel() + 1,
            dtype=torch.int32,
            device=seqlens.device,
        )
        cu_seqlens[1:] = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
        max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0

        trailing_shape = tuple(hidden_states.shape[2:])
        flat = hidden_states.reshape(batch * seqlen, *trailing_shape)
        unpadded = flat.index_select(0, indices.to(device=flat.device).long())
        return unpadded, indices, cu_seqlens, max_seqlen

    @lru_cache(maxsize=1)
    def _get_attention_functions():  # type: ignore[no-untyped-def]
        return index_first_axis, pad_input, rearrange, unpad_input

    attention_utils._get_attention_functions = _get_attention_functions
    attention_utils._keck_flash_attn_fallback_patch = True


def _patch_torch_cuda_capability_fallback() -> None:
    if not _enabled("KECK_PATCH_CUDA_CAPABILITY_FALLBACK"):
        return

    try:
        import torch
    except Exception:
        return

    if getattr(torch.cuda, "_keck_capability_fallback_patch", False):
        return

    original_get_device_capability = torch.cuda.get_device_capability

    def _fallback_capability() -> tuple[int, int]:
        raw_value = os.environ.get("KECK_CUDA_CAPABILITY", "8,0")
        try:
            major, minor = raw_value.replace(".", ",").split(",", 1)
            return int(major.strip()), int(minor.strip())
        except Exception:
            return 8, 0

    def patched_get_device_capability(device=None):  # type: ignore[no-untyped-def]
        try:
            return original_get_device_capability(device)
        except AssertionError as exc:
            if "Invalid device id" not in str(exc):
                raise
            return _fallback_capability()

    torch.cuda.get_device_capability = patched_get_device_capability
    torch.cuda._keck_capability_fallback_patch = True


def _patch_verl_resource_pool_max_colocate() -> None:
    if not _enabled("KECK_PATCH_VERL_MAX_COLOCATE"):
        return

    target_module_names = {
        "verl.single_controller.ray.base",
        "verl.trainer.ppo.ray_trainer",
    }

    def _desired_count() -> int:
        try:
            return max(1, int(os.environ.get("KECK_VERL_MAX_COLOCATE_COUNT", "1")))
        except ValueError:
            return 1

    def _patch_module(module):  # type: ignore[no-untyped-def]
        cls = getattr(module, "ResourcePoolManager", None)
        if cls is None or getattr(cls, "_keck_max_colocate_patch", False):
            return

        original_init = cls.__init__
        try:
            import inspect

            parameters = [
                name
                for name in inspect.signature(original_init).parameters
                if name != "self"
            ]
            position = parameters.index("max_colocate_count")
        except ValueError:
            return
        except Exception:
            position = None

        def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            desired = _desired_count()
            if position is not None and len(args) > position:
                args = (*args[:position], desired, *args[position + 1 :])
            else:
                kwargs["max_colocate_count"] = desired

            return original_init(self, *args, **kwargs)

        cls.__init__ = patched_init
        cls._keck_max_colocate_patch = True

        try:
            cls.__dataclass_fields__["max_colocate_count"].default = _desired_count()
        except Exception:
            pass

    try:
        import builtins
        import sys
    except Exception:
        return

    patched_existing_module = False
    for target_module_name in target_module_names:
        existing_module = sys.modules.get(target_module_name)
        if existing_module is not None:
            _patch_module(existing_module)
            patched_existing_module = True
    if patched_existing_module:
        return

    if getattr(builtins, "_keck_verl_max_colocate_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        for target_module_name in target_module_names:
            target_module = sys.modules.get(target_module_name)
            if target_module is not None:
                _patch_module(target_module)
        return module

    builtins.__import__ = patched_import
    builtins._keck_verl_max_colocate_import_patch = True


def _keck_ray_rank_context() -> tuple[int | None, int, int]:
    rank_value = os.environ.get("RANK")
    local_world_size_value = os.environ.get("RAY_LOCAL_WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE")
    try:
        rank = int(rank_value) if rank_value is not None else None
    except ValueError:
        rank = None
    try:
        local_world_size = max(1, int(local_world_size_value or "1"))
    except ValueError:
        local_world_size = 1

    if rank is None:
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except ValueError:
            local_rank = 0
    else:
        local_rank = rank % local_world_size
    return rank, local_world_size, max(0, local_rank)


def _keck_select_cuda_index(device_count: int) -> tuple[int | None, int | None, int, int]:
    if device_count <= 0:
        return None, None, 1, 0

    rank, local_world_size, local_rank = _keck_ray_rank_context()
    if rank is not None and device_count >= local_world_size:
        device_index = local_rank
    else:
        # Ray may narrow each actor to one CUDA device.  In that per-actor view,
        # torch cuda:0 is the assigned physical GPU.
        device_index = 0
    return min(device_count - 1, max(0, device_index)), rank, local_world_size, local_rank


def _keck_set_torch_cuda_device(context: str) -> int | None:
    if os.environ.get("WG_BACKEND") == "ray":
        _assign_ray_worker_cuda_visible_devices()

    try:
        import torch
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return None
        device_count = torch.cuda.device_count()
    except Exception:
        return None

    selected, rank, local_world_size, local_rank = _keck_select_cuda_index(device_count)
    if selected is None:
        return None

    try:
        before = torch.cuda.current_device()
    except Exception:
        before = "unknown"

    try:
        torch.cuda.set_device(selected)
    except Exception as exc:
        if _enabled("KECK_LOG_VERL_FSDP_CUDA_DEVICE", "1"):
            import sys

            print(
                "[keck] cuda device set failed: "
                f"context={context} rank={rank if rank is not None else 'unset'} "
                f"local_rank={local_rank} local_world_size={local_world_size} "
                f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
                f"torch_device_count={device_count} selected={selected} error={exc}",
                file=sys.stderr,
                flush=True,
            )
        return None

    os.environ["LOCAL_RANK"] = str(selected)
    os.environ["KECK_TORCH_CUDA_DEVICE_INDEX"] = str(selected)

    if _enabled("KECK_LOG_VERL_FSDP_CUDA_DEVICE", "1"):
        import sys

        try:
            current = torch.cuda.current_device()
        except Exception:
            current = "unknown"
        print(
            "[keck] cuda device selected: "
            f"context={context} rank={rank if rank is not None else 'unset'} "
            f"local_rank={local_rank} local_world_size={local_world_size} "
            f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
            f"assigned={os.environ.get('KECK_ASSIGNED_CUDA_VISIBLE_DEVICES', 'unset')} "
            f"torch_device_count={device_count} before={before} "
            f"selected={selected} current={current}",
            file=sys.stderr,
            flush=True,
        )
    return selected


def _patch_torch_fsdp_device_id() -> None:
    if not _enabled("KECK_PATCH_VERL_FSDP_CUDA_DEVICE"):
        return

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception:
        return

    if getattr(FSDP, "_keck_device_id_patch", False):
        return

    original_init = FSDP.__init__

    def patched_init(self, module, *args, **kwargs):  # type: ignore[no-untyped-def]
        selected = _keck_set_torch_cuda_device("torch.distributed.fsdp.FSDP.__init__")
        if selected is not None and os.environ.get("WG_BACKEND") == "ray":
            previous_device_id = kwargs.get("device_id", None)
            kwargs["device_id"] = selected
            if _enabled("KECK_LOG_VERL_FSDP_CUDA_DEVICE", "1"):
                import sys

                print(
                    "[keck] fsdp device_id override: "
                    f"previous={previous_device_id if previous_device_id is not None else 'unset'} "
                    f"device_id={selected}",
                    file=sys.stderr,
                    flush=True,
                )
        return original_init(self, module, *args, **kwargs)

    FSDP.__init__ = patched_init
    FSDP._keck_device_id_patch = True


def _patch_verl_get_device_id() -> None:
    if not _enabled("KECK_PATCH_VERL_FSDP_CUDA_DEVICE"):
        return

    target_module_names = {
        "verl.utils.device",
        "verl.workers.engine.fsdp.transformer_impl",
    }

    def _patch_module(module):  # type: ignore[no-untyped-def]
        original_get_device_id = getattr(module, "get_device_id", None)
        if original_get_device_id is None or getattr(original_get_device_id, "_keck_device_id_patch", False):
            return

        def patched_get_device_id(*args, **kwargs):  # type: ignore[no-untyped-def]
            selected = _keck_set_torch_cuda_device(f"{getattr(module, '__name__', 'module')}.get_device_id")
            if selected is not None:
                return selected
            return original_get_device_id(*args, **kwargs)

        patched_get_device_id._keck_device_id_patch = True
        module.get_device_id = patched_get_device_id

    try:
        import builtins
        import sys
    except Exception:
        return

    for target_module_name in target_module_names:
        existing_module = sys.modules.get(target_module_name)
        if existing_module is not None:
            _patch_module(existing_module)

    if getattr(builtins, "_keck_verl_get_device_id_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        for target_module_name in target_module_names:
            target_module = sys.modules.get(target_module_name)
            if target_module is not None:
                _patch_module(target_module)
        return module

    builtins.__import__ = patched_import
    builtins._keck_verl_get_device_id_import_patch = True


def _patch_verl_fsdp_cuda_device_selection() -> None:
    if not _enabled("KECK_PATCH_VERL_FSDP_CUDA_DEVICE"):
        return

    target_module_name = "verl.workers.engine.fsdp.transformer_impl"

    def _rank_context() -> tuple[int | None, int, int]:
        rank_value = os.environ.get("RANK")
        local_world_size_value = os.environ.get("RAY_LOCAL_WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE")
        try:
            rank = int(rank_value) if rank_value is not None else None
        except ValueError:
            rank = None
        try:
            local_world_size = max(1, int(local_world_size_value or "1"))
        except ValueError:
            local_world_size = 1

        if rank is None:
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            except ValueError:
                local_rank = 0
        else:
            local_rank = rank % local_world_size
        return rank, local_world_size, max(0, local_rank)

    def _select_cuda_index(device_count: int) -> tuple[int | None, int | None, int, int]:
        if device_count <= 0:
            return None, None, 1, 0

        rank, local_world_size, local_rank = _rank_context()
        if rank is not None and device_count >= local_world_size:
            device_index = local_rank
        else:
            # Ray often narrows CUDA_VISIBLE_DEVICES to one physical device per
            # actor.  In that view, torch device 0 is the assigned GPU.
            device_index = 0

        return min(device_count - 1, max(0, device_index)), rank, local_world_size, local_rank

    def _set_torch_cuda_device(context: str) -> None:
        try:
            import torch
        except Exception:
            return

        try:
            if not torch.cuda.is_available():
                return
            device_count = torch.cuda.device_count()
        except Exception:
            return

        selected, rank, local_world_size, local_rank = _select_cuda_index(device_count)
        if selected is None:
            return

        try:
            before = torch.cuda.current_device()
        except Exception:
            before = "unknown"

        try:
            torch.cuda.set_device(selected)
        except Exception as exc:
            if _enabled("KECK_LOG_VERL_FSDP_CUDA_DEVICE", "1"):
                import sys

                print(
                    "[keck] fsdp cuda device set failed: "
                    f"context={context} rank={rank if rank is not None else 'unset'} "
                    f"local_rank={local_rank} local_world_size={local_world_size} "
                    f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
                    f"torch_device_count={device_count} selected={selected} error={exc}",
                    file=sys.stderr,
                    flush=True,
                )
            return

        os.environ["LOCAL_RANK"] = str(selected)
        os.environ["KECK_TORCH_CUDA_DEVICE_INDEX"] = str(selected)

        if _enabled("KECK_LOG_VERL_FSDP_CUDA_DEVICE", "1"):
            import sys

            try:
                current = torch.cuda.current_device()
            except Exception:
                current = "unknown"
            print(
                "[keck] fsdp cuda device: "
                f"context={context} rank={rank if rank is not None else 'unset'} "
                f"local_rank={local_rank} local_world_size={local_world_size} "
                f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
                f"assigned={os.environ.get('KECK_ASSIGNED_CUDA_VISIBLE_DEVICES', 'unset')} "
                f"torch_device_count={device_count} before={before} "
                f"selected={selected} current={current}",
                file=sys.stderr,
                flush=True,
            )

    def _wrap_method(cls, method_name: str) -> bool:  # type: ignore[no-untyped-def]
        original_method = getattr(cls, method_name, None)
        if original_method is None or getattr(original_method, "_keck_fsdp_cuda_device_patch", False):
            return False

        def patched_method(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            _set_torch_cuda_device(f"{cls.__name__}.{method_name}")
            return original_method(self, *args, **kwargs)

        patched_method._keck_fsdp_cuda_device_patch = True
        setattr(cls, method_name, patched_method)
        return True

    def _patch_module(module):  # type: ignore[no-untyped-def]
        if getattr(module, "_keck_fsdp_cuda_device_module_patch", False):
            return

        patched_any = False
        for value in vars(module).values():
            if not isinstance(value, type):
                continue
            for method_name in ("initialize", "_build_model_optimizer", "_build_fsdp_module"):
                patched_any = _wrap_method(value, method_name) or patched_any

        if patched_any:
            module._keck_fsdp_cuda_device_module_patch = True

    try:
        import builtins
        import sys
    except Exception:
        return

    existing_module = sys.modules.get(target_module_name)
    if existing_module is not None:
        _patch_module(existing_module)
        return

    if getattr(builtins, "_keck_verl_fsdp_cuda_device_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        target_module = sys.modules.get(target_module_name)
        if target_module is not None:
            _patch_module(target_module)
        return module

    builtins.__import__ = patched_import
    builtins._keck_verl_fsdp_cuda_device_import_patch = True


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return _enabled(name)


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _set_existing_attr(obj, attr: str, value) -> bool:  # type: ignore[no-untyped-def]
    if obj is None or value is None or not hasattr(obj, attr):
        return False
    try:
        current = getattr(obj, attr)
        if current == value:
            return False
        setattr(obj, attr, value)
        return True
    except Exception:
        return False


def _patch_vllm_engine_config_from_env() -> None:
    if not _enabled("KECK_PATCH_VLLM_ENGINE_CONFIG", "1"):
        return

    target_module_names = {
        "vllm.v1.engine.core": "EngineCore",
        "vllm.engine.llm_engine": "LLMEngine",
    }

    def _apply_config(vllm_config):  # type: ignore[no-untyped-def]
        changed = []
        cache_config = getattr(vllm_config, "cache_config", None)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)

        gpu_memory_utilization = (
            _env_float("ROLLOUT_GPU_MEMORY_UTILIZATION")
            or _env_float("VLLM_GPU_MEMORY_UTILIZATION")
        )
        if gpu_memory_utilization is not None:
            for obj_name, obj in (
                ("vllm_config", vllm_config),
                ("cache_config", cache_config),
            ):
                if _set_existing_attr(obj, "gpu_memory_utilization", gpu_memory_utilization):
                    changed.append(f"{obj_name}.gpu_memory_utilization={gpu_memory_utilization}")

        enable_chunked_prefill = (
            _env_bool("ROLLOUT_ENABLE_CHUNKED_PREFILL")
            if os.environ.get("ROLLOUT_ENABLE_CHUNKED_PREFILL") is not None
            else _env_bool("VLLM_ENABLE_CHUNKED_PREFILL")
        )
        if enable_chunked_prefill is not None:
            for attr in ("enable_chunked_prefill", "chunked_prefill_enabled"):
                for obj_name, obj in (
                    ("vllm_config", vllm_config),
                    ("scheduler_config", scheduler_config),
                ):
                    if _set_existing_attr(obj, attr, enable_chunked_prefill):
                        changed.append(f"{obj_name}.{attr}={enable_chunked_prefill}")

        enable_prefix_caching = (
            _env_bool("ROLLOUT_ENABLE_PREFIX_CACHING")
            if os.environ.get("ROLLOUT_ENABLE_PREFIX_CACHING") is not None
            else _env_bool("VLLM_ENABLE_PREFIX_CACHING")
        )
        if enable_prefix_caching is not None:
            for obj_name, obj in (
                ("vllm_config", vllm_config),
                ("cache_config", cache_config),
            ):
                if _set_existing_attr(obj, "enable_prefix_caching", enable_prefix_caching):
                    changed.append(f"{obj_name}.enable_prefix_caching={enable_prefix_caching}")

        max_num_batched_tokens = _env_float("ROLLOUT_MAX_NUM_BATCHED_TOKENS")
        if max_num_batched_tokens is not None:
            value = int(max_num_batched_tokens)
            if _set_existing_attr(scheduler_config, "max_num_batched_tokens", value):
                changed.append(f"scheduler_config.max_num_batched_tokens={value}")

        max_num_seqs = _env_float("ROLLOUT_MAX_NUM_SEQS")
        if max_num_seqs is not None:
            value = int(max_num_seqs)
            if _set_existing_attr(scheduler_config, "max_num_seqs", value):
                changed.append(f"scheduler_config.max_num_seqs={value}")

        if changed and _enabled("KECK_LOG_VLLM_ENGINE_CONFIG", "1"):
            import sys

            print(
                "[keck] vllm engine config override: " + ", ".join(changed),
                file=sys.stderr,
                flush=True,
            )

    def _patch_module(module, class_name: str):  # type: ignore[no-untyped-def]
        cls = getattr(module, class_name, None)
        if cls is None or getattr(cls, "_keck_engine_config_patch", False):
            return

        original_init = cls.__init__

        def patched_init(self, vllm_config, *args, **kwargs):  # type: ignore[no-untyped-def]
            _apply_config(vllm_config)
            return original_init(self, vllm_config, *args, **kwargs)

        cls.__init__ = patched_init
        cls._keck_engine_config_patch = True

    try:
        import builtins
        import sys
    except Exception:
        return

    patched_existing_module = False
    for target_module_name, class_name in target_module_names.items():
        existing_module = sys.modules.get(target_module_name)
        if existing_module is not None:
            _patch_module(existing_module, class_name)
            patched_existing_module = True
    if patched_existing_module:
        return

    if getattr(builtins, "_keck_vllm_engine_config_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        for target_module_name, class_name in target_module_names.items():
            target_module = sys.modules.get(target_module_name)
            if target_module is not None:
                _patch_module(target_module, class_name)
        return module

    builtins.__import__ = patched_import
    builtins._keck_vllm_engine_config_import_patch = True


def _patch_verl_async_rollout_fsdp() -> None:
    if not _enabled("KECK_PATCH_VERL_ASYNC_ROLLOUT_FSDP", "1"):
        return

    target_module_name = "verl.workers.fsdp_workers"

    def _strip_decorators(source: str) -> str:
        lines = source.splitlines()
        while lines and lines[0].lstrip().startswith("@"):
            lines.pop(0)
        return "\n".join(lines)

    def _compile_function(source: str, func_name: str, globals_dict):  # type: ignore[no-untyped-def]
        namespace = {}
        exec(compile(source, f"<keck patch {func_name}>", "exec"), globals_dict, namespace)
        return namespace[func_name]

    def _patch_module(module):  # type: ignore[no-untyped-def]
        worker_cls = getattr(module, "ActorRolloutRefWorker", None)
        if worker_cls is None or getattr(worker_cls, "_keck_async_rollout_fsdp_patch", False):
            return

        import inspect
        import re
        import textwrap

        init_source = textwrap.dedent(inspect.getsource(worker_cls.init_model))
        init_source = _strip_decorators(init_source)

        init_pattern = re.compile(r"(?s)(    if self\._is_actor or self\._is_rollout:\n)(.*?)(\n    if self\._is_ref:.*)")

        def _rewrite_init(match):  # type: ignore[no-untyped-def]
            body = match.group(2).rstrip("\n")
            return (
                "    if self._is_actor or self._is_rollout:\n"
                "        if self._is_rollout and getattr(self.config.rollout, \"mode\", None) == \"async\":\n"
                "            self.actor_module_fsdp = None\n"
                "            self.actor_module = None\n"
                "            self.actor = None\n"
                "            self.actor_optimizer = None\n"
                "            self.actor_lr_scheduler = None\n"
                "            self.actor_model_config = None\n"
                "            self.checkpoint_manager = None\n"
                "            self._build_rollout(trust_remote_code=self.config.model.get(\"trust_remote_code\", False))\n"
                "        else:\n"
                + textwrap.indent(body, "    ")
                + match.group(3)
            )

        init_source, init_count = init_pattern.subn(_rewrite_init, init_source, count=1)
        if init_count != 1:
            return

        build_rollout_source = textwrap.dedent(inspect.getsource(worker_cls._build_rollout))
        build_rollout_source = _strip_decorators(build_rollout_source)
        build_rollout_source = build_rollout_source.replace(
            "    if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:",
            "    if getattr(self, \"actor_module_fsdp\", None) is not None and torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:",
            1,
        )

        try:
            patched_init_model = _compile_function(init_source, "init_model", worker_cls.init_model.__globals__)
            patched_build_rollout = _compile_function(build_rollout_source, "_build_rollout", worker_cls._build_rollout.__globals__)
        except Exception:
            return

        worker_cls.init_model = patched_init_model
        worker_cls._build_rollout = patched_build_rollout
        worker_cls._keck_async_rollout_fsdp_patch = True

    try:
        import builtins
        import sys
    except Exception:
        return

    existing_module = sys.modules.get(target_module_name)
    if existing_module is not None:
        _patch_module(existing_module)
        if getattr(existing_module, "_keck_async_rollout_fsdp_module_patch", False):
            return

    if getattr(builtins, "_keck_verl_async_rollout_fsdp_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        target_module = sys.modules.get(target_module_name)
        if target_module is not None:
            _patch_module(target_module)
            if getattr(target_module, "ActorRolloutRefWorker", None) is not None:
                target_module._keck_async_rollout_fsdp_module_patch = True
        return module

    builtins.__import__ = patched_import
    builtins._keck_verl_async_rollout_fsdp_import_patch = True


def _patch_verl_async_vllm_prompt_truncation() -> None:
    if not _enabled("KECK_PATCH_VERL_ASYNC_VLLM_PROMPT_TRUNCATION", "1"):
        return

    target_module_name = "verl.workers.rollout.vllm_rollout.vllm_async_server"

    def _positive_int(value) -> int | None:  # type: ignore[no-untyped-def]
        try:
            value = int(value)
        except Exception:
            return None
        return value if value > 0 else None

    def _log_truncation(instance, message: str) -> None:  # type: ignore[no-untyped-def]
        if not _enabled("KECK_LOG_ASYNC_VLLM_PROMPT_TRUNCATION", "1"):
            return
        max_logs = _env_int("KECK_ASYNC_VLLM_PROMPT_TRUNCATION_MAX_LOGS")
        if max_logs is None:
            max_logs = 20
        count_attr = "_keck_async_vllm_prompt_truncation_log_count"
        count = getattr(instance, count_attr, 0)
        if count >= max_logs:
            return
        try:
            setattr(instance, count_attr, count + 1)
        except Exception:
            pass
        import sys

        print(f"[keck] async vllm prompt truncation: {message}", file=sys.stderr, flush=True)

    def _patch_module(module):  # type: ignore[no-untyped-def]
        server_cls = getattr(module, "vLLMHttpServer", None)
        if server_cls is None or getattr(server_cls, "_keck_prompt_truncation_patch", False):
            return

        original_generate = server_cls.generate

        async def patched_generate(self, prompt_ids, sampling_params, request_id, *args, **kwargs):  # type: ignore[no-untyped-def]
            next_sampling_params = dict(sampling_params or {})
            min_generate_tokens = _env_int("KECK_ASYNC_VLLM_MIN_GENERATE_TOKENS")
            if min_generate_tokens is None:
                min_generate_tokens = 1
            min_generate_tokens = max(1, min_generate_tokens)

            for key in ("max_tokens", "max_new_tokens"):
                if key not in next_sampling_params:
                    continue
                try:
                    if int(next_sampling_params[key]) < min_generate_tokens:
                        next_sampling_params[key] = min_generate_tokens
                except Exception:
                    pass

            next_prompt_ids = prompt_ids
            try:
                next_prompt_ids = module.normalize_token_ids(prompt_ids)
            except Exception:
                pass

            try:
                prompt_len = len(next_prompt_ids)
            except Exception:
                return await original_generate(self, prompt_ids, next_sampling_params, request_id, *args, **kwargs)

            config = getattr(self, "config", None)
            prompt_length = _positive_int(getattr(config, "prompt_length", None))
            max_model_len = _positive_int(getattr(config, "max_model_len", None))

            limits = []
            if prompt_length is not None:
                limits.append(prompt_length)
            if max_model_len is not None:
                limits.append(max(1, max_model_len - min_generate_tokens))

            if limits:
                keep_tokens = max(1, min(limits))
                if prompt_len > keep_tokens:
                    next_prompt_ids = list(next_prompt_ids)[-keep_tokens:]
                    _log_truncation(
                        self,
                        "left-trimmed prompt_ids "
                        f"request_id={request_id} original={prompt_len} kept={keep_tokens} "
                        f"prompt_length={prompt_length or 'unset'} max_model_len={max_model_len or 'unset'} "
                        f"min_generate_tokens={min_generate_tokens}",
                    )

            return await original_generate(self, next_prompt_ids, next_sampling_params, request_id, *args, **kwargs)

        server_cls.generate = patched_generate
        server_cls._keck_prompt_truncation_patch = True

    try:
        import builtins
        import sys
    except Exception:
        return

    existing_module = sys.modules.get(target_module_name)
    if existing_module is not None:
        _patch_module(existing_module)
        return

    if getattr(builtins, "_keck_verl_async_vllm_prompt_truncation_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        target_module = sys.modules.get(target_module_name)
        if target_module is not None:
            _patch_module(target_module)
        return module

    builtins.__import__ = patched_import
    builtins._keck_verl_async_vllm_prompt_truncation_import_patch = True


def _patch_verl_dataproto_concat_padding() -> None:
    if not _enabled("KECK_PATCH_VERL_DATAPROTO_CONCAT_PADDING", "1"):
        return

    target_module_name = "verl.protocol"

    def _leaf_items(tensordict):  # type: ignore[no-untyped-def]
        for kwargs in (
            {"include_nested": True, "leaves_only": True},
            {"include_nested": True},
            {},
        ):
            try:
                return list(tensordict.items(**kwargs))
            except TypeError:
                continue
            except Exception:
                break
        try:
            return [(key, tensordict.get(key)) for key in list(tensordict.keys())]
        except Exception:
            return []

    def _get_value(tensordict, key):  # type: ignore[no-untyped-def]
        try:
            return tensordict.get(key)
        except Exception:
            try:
                return tensordict[key]
            except Exception:
                return None

    def _set_value(tensordict, key, value) -> None:  # type: ignore[no-untyped-def]
        try:
            tensordict.set(key, value)
            return
        except Exception:
            pass
        try:
            tensordict[key] = value
        except Exception:
            pass

    def _pad_tensor_to_shape(tensor, target_shape):  # type: ignore[no-untyped-def]
        try:
            import torch
            import torch.nn.functional as functional
        except Exception:
            return tensor

        if not isinstance(tensor, torch.Tensor) or tensor.dim() < 2:
            return tensor
        if len(target_shape) != tensor.dim():
            return tensor
        slices = [slice(None)] * tensor.dim()
        sliced = False
        for dim in range(1, tensor.dim()):
            current = int(tensor.shape[dim])
            target = int(target_shape[dim])
            if target <= 0:
                return tensor
            if target < current:
                slices[dim] = slice(current - target, current)
                sliced = True
        if sliced:
            tensor = tensor[tuple(slices)]
        pads = []
        changed = False
        for dim in range(tensor.dim() - 1, 0, -1):
            current = int(tensor.shape[dim])
            target = int(target_shape[dim])
            if target < current:
                return tensor
            pads.extend([0, target - current])
            changed = changed or target != current
        if not changed:
            return tensor
        return functional.pad(tensor, pads, value=0)

    def _pad_dataproto_batches(data):  # type: ignore[no-untyped-def]
        try:
            import torch
        except Exception:
            return data

        batches = [getattr(item, "batch", None) for item in data]
        if not batches or batches[0] is None:
            return data

        sequence_cap = _env_int("KECK_DATAPROTO_MAX_SEQUENCE_LENGTH")
        target_shapes = {}
        for batch in batches:
            if batch is None:
                continue
            for key, value in _leaf_items(batch):
                if not isinstance(value, torch.Tensor) or value.dim() < 2:
                    continue
                shape = tuple(int(dim) for dim in value.shape)
                previous = target_shapes.get(key)
                if previous is None:
                    target_shapes[key] = tuple(
                        dim_size
                        if dim == 0 or sequence_cap is None or dim_size <= sequence_cap
                        else sequence_cap
                        for dim, dim_size in enumerate(shape)
                    )
                    continue
                if len(previous) != len(shape):
                    continue
                next_shape = []
                for dim in range(len(shape)):
                    if dim == 0:
                        next_shape.append(shape[0])
                        continue
                    target_dim = max(previous[dim], shape[dim])
                    if sequence_cap is not None and target_dim > sequence_cap:
                        target_dim = sequence_cap
                    next_shape.append(target_dim)
                target_shapes[key] = tuple(next_shape)

        if not target_shapes:
            return data

        padded = []
        for item in data:
            batch = getattr(item, "batch", None)
            if batch is None:
                padded.append(item)
                continue
            try:
                next_batch = batch.clone()
            except Exception:
                try:
                    next_batch = copy.deepcopy(batch)
                except Exception:
                    padded.append(item)
                    continue
            for key, target_shape in target_shapes.items():
                value = _get_value(next_batch, key)
                next_value = _pad_tensor_to_shape(value, target_shape)
                if next_value is not value:
                    _set_value(next_batch, key, next_value)
            try:
                next_item = copy.copy(item)
                next_item.batch = next_batch
            except Exception:
                try:
                    next_item = type(item)(
                        batch=next_batch,
                        non_tensor_batch=getattr(item, "non_tensor_batch", None),
                        meta_info=getattr(item, "meta_info", None),
                    )
                except Exception:
                    next_item = item
            padded.append(next_item)
        return padded

    def _log_padding(message: str) -> None:
        if not _enabled("KECK_LOG_DATAPROTO_CONCAT_PADDING", "1"):
            return
        import sys

        print(f"[keck] dataproto concat padding: {message}", file=sys.stderr, flush=True)

    def _patch_module(module):  # type: ignore[no-untyped-def]
        cls = getattr(module, "DataProto", None)
        if cls is None or getattr(cls, "_keck_concat_padding_patch", False):
            return

        original_concat = cls.concat

        def patched_concat(data):  # type: ignore[no-untyped-def]
            prepared_data = data
            if _env_int("KECK_DATAPROTO_MAX_SEQUENCE_LENGTH") is not None:
                prepared_data = _pad_dataproto_batches(data)
            try:
                return original_concat(prepared_data)
            except RuntimeError as exc:
                if "Sizes of tensors must match" not in str(exc):
                    raise
                padded_data = _pad_dataproto_batches(prepared_data)
                _log_padding(str(exc))
                return original_concat(padded_data)

        cls.concat = staticmethod(patched_concat)
        cls._keck_concat_padding_patch = True

    try:
        import builtins
        import sys
    except Exception:
        return

    existing_module = sys.modules.get(target_module_name)
    if existing_module is not None:
        _patch_module(existing_module)
        return

    if getattr(builtins, "_keck_verl_dataproto_concat_padding_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        target_module = sys.modules.get(target_module_name)
        if target_module is not None:
            _patch_module(target_module)
        return module

    builtins.__import__ = patched_import
    builtins._keck_verl_dataproto_concat_padding_import_patch = True


def _patch_verl_vllm_weight_transfer_bucket() -> None:
    if not _enabled("KECK_PATCH_VERL_VLLM_WEIGHT_TRANSFER_BUCKET", "1"):
        return

    target_module_name = "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"

    def _max_bucket_bytes() -> int:
        mb = _env_int("KECK_VLLM_WEIGHT_TRANSFER_BUCKET_MB")
        if mb is None:
            mb = 1
        return max(1, mb) * 1024 * 1024

    def _as_scalar_size(value):  # type: ignore[no-untyped-def]
        try:
            return int(value.__index__())
        except Exception:
            pass
        if isinstance(value, (list, tuple)) and len(value) == 1:
            try:
                return int(value[0].__index__())
            except Exception:
                try:
                    return int(value[0])
                except Exception:
                    return None
        try:
            return int(value)
        except Exception:
            return None

    def _cap_instance_bucket(instance, context: str) -> None:  # type: ignore[no-untyped-def]
        current = getattr(instance, "bucket_size", None)
        if current is None:
            return
        try:
            current_int = int(current)
        except Exception:
            return

        max_bytes = _max_bucket_bytes()
        if current_int <= max_bytes:
            return

        try:
            setattr(instance, "bucket_size", max_bytes)
        except Exception:
            return

        if _enabled("KECK_LOG_VERL_VLLM_WEIGHT_TRANSFER_BUCKET", "1"):
            logged_attr = "_keck_weight_transfer_bucket_logged"
            if getattr(instance, logged_attr, False):
                return
            try:
                setattr(instance, logged_attr, True)
            except Exception:
                pass
            import sys

            print(
                "[keck] vllm weight transfer bucket override: "
                f"context={context} previous={current_int} bucket_size={max_bytes}",
                file=sys.stderr,
                flush=True,
            )

    def _patch_class(cls) -> bool:  # type: ignore[no-untyped-def]
        patched_any = False

        original_init = getattr(cls, "__init__", None)
        if original_init is not None and not getattr(original_init, "_keck_bucket_patch", False):

            def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                result = original_init(self, *args, **kwargs)
                _cap_instance_bucket(self, f"{cls.__name__}.__init__")
                return result

            patched_init._keck_bucket_patch = True
            cls.__init__ = patched_init
            patched_any = True

        for method_name in ("_init_buffer", "async_send_weights", "send_weights"):
            original_method = getattr(cls, method_name, None)
            if original_method is None or getattr(original_method, "_keck_bucket_patch", False):
                continue

            if method_name == "async_send_weights":

                async def patched_async_method(self, *args, _original=original_method, **kwargs):  # type: ignore[no-untyped-def]
                    _cap_instance_bucket(self, f"{cls.__name__}.async_send_weights")
                    return await _original(self, *args, **kwargs)

                patched_async_method._keck_bucket_patch = True
                setattr(cls, method_name, patched_async_method)
            elif method_name == "_init_buffer":

                def patched_init_buffer(self, *args, _original=original_method, **kwargs):  # type: ignore[no-untyped-def]
                    _cap_instance_bucket(self, f"{cls.__name__}._init_buffer")
                    try:
                        import torch
                    except Exception:
                        return _original(self, *args, **kwargs)

                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                    original_empty = torch.empty

                    def capped_empty(*empty_args, **empty_kwargs):  # type: ignore[no-untyped-def]
                        max_bytes = _max_bucket_bytes()
                        next_args = empty_args
                        size = empty_args[0] if empty_args else empty_kwargs.get("size")
                        dtype = empty_kwargs.get("dtype")
                        device = empty_kwargs.get("device")
                        scalar_size = _as_scalar_size(size)
                        current_bucket_size = _as_scalar_size(getattr(self, "bucket_size", None))
                        is_bucket_allocation = (
                            scalar_size is not None
                            and current_bucket_size is not None
                            and scalar_size == current_bucket_size
                        )
                        dtype_matches = dtype is None or dtype == torch.uint8 or str(dtype) == "torch.uint8"
                        device_matches = device is None or "cuda" in str(device).lower()
                        if (
                            is_bucket_allocation
                            and dtype_matches
                            and device_matches
                            and scalar_size is not None
                            and scalar_size > max_bytes
                        ):
                            if empty_args:
                                next_args = (max_bytes, *empty_args[1:])
                            else:
                                empty_kwargs = dict(empty_kwargs)
                                empty_kwargs["size"] = max_bytes
                            try:
                                setattr(self, "bucket_size", max_bytes)
                            except Exception:
                                pass
                            if _enabled("KECK_LOG_VERL_VLLM_WEIGHT_TRANSFER_BUCKET", "1"):
                                import sys

                                print(
                                    "[keck] vllm weight transfer buffer allocation capped: "
                                    f"requested={scalar_size} bucket_size={max_bytes}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                        return original_empty(*next_args, **empty_kwargs)

                    torch.empty = capped_empty
                    try:
                        result = _original(self, *args, **kwargs)
                    finally:
                        torch.empty = original_empty
                    _cap_instance_bucket(self, f"{cls.__name__}._init_buffer.after")
                    return result

                patched_init_buffer._keck_bucket_patch = True
                setattr(cls, method_name, patched_init_buffer)
            else:

                def patched_method(self, *args, _original=original_method, _method_name=method_name, **kwargs):  # type: ignore[no-untyped-def]
                    _cap_instance_bucket(self, f"{cls.__name__}.{_method_name}")
                    return _original(self, *args, **kwargs)

                patched_method._keck_bucket_patch = True
                setattr(cls, method_name, patched_method)
            patched_any = True

        return patched_any

    def _patch_module(module):  # type: ignore[no-untyped-def]
        if getattr(module, "_keck_weight_transfer_bucket_patch", False):
            return

        patched_any = False
        max_bytes = _max_bucket_bytes()
        for name, value in list(vars(module).items()):
            if isinstance(value, int) and "BUCKET" in name.upper() and value > max_bytes:
                try:
                    setattr(module, name, max_bytes)
                    patched_any = True
                except Exception:
                    pass
            if isinstance(value, type):
                patched_any = _patch_class(value) or patched_any

        if patched_any:
            module._keck_weight_transfer_bucket_patch = True

    try:
        import builtins
        import sys
    except Exception:
        return

    existing_module = sys.modules.get(target_module_name)
    if existing_module is not None:
        _patch_module(existing_module)
        return

    if getattr(builtins, "_keck_verl_vllm_weight_transfer_bucket_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        target_module = sys.modules.get(target_module_name)
        if target_module is not None:
            _patch_module(target_module)
        return module

    builtins.__import__ = patched_import
    builtins._keck_verl_vllm_weight_transfer_bucket_import_patch = True


def _patch_verl_log_prob_release_rollout_memory() -> None:
    if not _enabled("KECK_PATCH_VERL_LOG_PROB_RELEASE_ROLLOUT_MEMORY", "0"):
        return

    target_module_name = "verl.workers.engine_workers"
    method_names = (
        "sleep",
        "free_cache_engine",
        "free_cache",
        "release_memory",
        "reset_prefix_cache",
    )

    def _log(message: str) -> None:
        if not _enabled("KECK_LOG_VERL_LOG_PROB_RELEASE_ROLLOUT_MEMORY", "1"):
            return
        import sys

        print(f"[keck] log_prob release rollout memory: {message}", file=sys.stderr, flush=True)

    def _try_call(obj, method_name: str) -> bool:  # type: ignore[no-untyped-def]
        method = getattr(obj, method_name, None)
        if method is None or not callable(method):
            return False
        for args, kwargs in (
            ((), {"level": 2}),
            ((2,), {}),
            ((), {"level": 1}),
            ((1,), {}),
            ((), {}),
        ):
            try:
                method(*args, **kwargs)
                return True
            except TypeError:
                continue
            except Exception as exc:
                _log(f"{type(obj).__name__}.{method_name} failed: {exc!r}")
                return False
        return False

    def _candidate_objects(root):  # type: ignore[no-untyped-def]
        seen = set()
        names = (
            "rollout",
            "rollout_engine",
            "rollout_sharding_manager",
            "vllm_engine",
            "llm",
            "async_llm",
            "engine",
            "server",
        )
        queue = []
        for name in names:
            value = getattr(root, name, None)
            if value is not None:
                queue.append((name, value))
        try:
            for name, value in vars(root).items():
                if value is not None and ("rollout" in name.lower() or "vllm" in name.lower()):
                    queue.append((name, value))
        except Exception:
            pass

        while queue:
            name, value = queue.pop(0)
            ident = id(value)
            if ident in seen:
                continue
            seen.add(ident)
            yield name, value
            if len(seen) > 64:
                break
            try:
                nested_engine = getattr(value, "engine", None)
                if nested_engine is not None:
                    queue.append((f"{name}.engine", nested_engine))
            except Exception:
                pass
            try:
                nested_llm = getattr(value, "llm", None)
                if nested_llm is not None:
                    queue.append((f"{name}.llm", nested_llm))
            except Exception:
                pass

    def _release_memory(worker) -> None:  # type: ignore[no-untyped-def]
        called = []
        for name, obj in _candidate_objects(worker):
            for method_name in method_names:
                if _try_call(obj, method_name):
                    called.append(f"{name}.{method_name}")
                    break
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        if called:
            _log("called " + ",".join(called))
        else:
            _log("no rollout/vllm release methods found")

    def _patch_module(module):  # type: ignore[no-untyped-def]
        for _, cls in list(vars(module).items()):
            if not isinstance(cls, type) or getattr(cls, "_keck_log_prob_release_rollout_memory_patch", False):
                continue
            original = getattr(cls, "compute_log_prob", None)
            if original is None or not callable(original):
                continue

            @functools.wraps(original)
            def patched_compute_log_prob(self, *args, __original=original, **kwargs):  # type: ignore[no-untyped-def]
                _release_memory(self)
                return __original(self, *args, **kwargs)

            try:
                patched_compute_log_prob.__dict__.update(getattr(original, "__dict__", {}))
            except Exception:
                pass
            cls.compute_log_prob = patched_compute_log_prob
            cls._keck_log_prob_release_rollout_memory_patch = True

    try:
        import builtins
        import sys
    except Exception:
        return

    existing_module = sys.modules.get(target_module_name)
    if existing_module is not None:
        _patch_module(existing_module)
        return

    if getattr(builtins, "_keck_verl_log_prob_release_rollout_memory_import_patch", False):
        return

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        module = original_import(name, globals, locals, fromlist, level)
        if getattr(builtins, "_keck_verl_log_prob_release_rollout_memory_in_patch", False):
            return module
        target_module = sys.modules.get(target_module_name)
        if target_module is not None and not getattr(target_module, "_keck_log_prob_release_rollout_memory_module_patch", False):
            builtins._keck_verl_log_prob_release_rollout_memory_in_patch = True
            try:
                _patch_module(target_module)
                target_module._keck_log_prob_release_rollout_memory_module_patch = True
            finally:
                builtins._keck_verl_log_prob_release_rollout_memory_in_patch = False
        return module

    builtins.__import__ = patched_import
    builtins._keck_verl_log_prob_release_rollout_memory_import_patch = True


_patch_nccl_env_defaults()
_patch_ray_worker_cuda_visible_devices()
_patch_ray_accelerator_ids_uuid_compat()
_patch_vllm_uuid_cuda_visible_devices()
_patch_vllm_lora_module_alias()
_patch_transformers_tokenizer_special_tokens()
_patch_torch_cuda_capability_fallback()
_patch_verl_resource_pool_max_colocate()
_patch_torch_fsdp_device_id()
_patch_verl_get_device_id()
_patch_verl_fsdp_cuda_device_selection()
_patch_vllm_engine_config_from_env()
_patch_verl_async_rollout_fsdp()
_patch_verl_async_vllm_prompt_truncation()
_patch_verl_dataproto_concat_padding()
_patch_verl_vllm_weight_transfer_bucket()
_patch_verl_log_prob_release_rollout_memory()
_patch_qwen35_text_config_for_vllm_mrope()
_patch_verl_flash_attn_bert_padding_fallback()
