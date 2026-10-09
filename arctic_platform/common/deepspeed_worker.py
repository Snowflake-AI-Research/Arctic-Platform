# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSpeed Ray worker actor used by the on-prem RL server."""

from __future__ import annotations

import asyncio
import logging
import numbers
import os
import socket
import time
from typing import Any

import deepspeed
import ray
import torch
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator

from arctic_platform.common.ray_cluster import primary_ip
from arctic_platform.common.utils import combine_metric_microbatches
from arctic_platform.common.utils import dp_sp_world_size
from arctic_platform.common.utils import log_dp_shard_tokens
from arctic_platform.common.utils import merge_dict_shards
from arctic_platform.common.utils import sp_size_from_job_config
from arctic_platform.common.utils import split_dict
from arctic_platform.common.utils import unpack_batch
from arctic_platform.common.utils.bf16_zero_norm import is_bf16_zero_norm_assert
from arctic_platform.common.utils.debug import enable_full_determinism
from arctic_platform.common.utils.debug import pr0
from arctic_platform.common.utils.debug import see_memory_usage
from arctic_platform.model import ModelSpec
from arctic_platform.model import build_model

logger = logging.getLogger(__name__)

_COMPOSITE_NAME_PREFIXES = (("model.", "model.language_model."), ("", "language_model."))
_HF_CONFIG_FILES = ("config.json", "generation_config.json")
_HF_INDEX_FILE = "model.safetensors.index.json"
_DTYPE_BYTES = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _safetensors_names(model_dir: str) -> dict[str, str]:
    """Map tensor names to shard names for a model directory."""
    from safetensors import safe_open

    names: dict[str, str] = {}
    for shard in sorted(os.listdir(model_dir)):
        if shard.endswith(".safetensors"):
            with safe_open(os.path.join(model_dir, shard), framework="pt") as handle:
                names.update({name: shard for name in handle.keys()})
    return names


def _canonical_hf_export_name(name: str) -> str:
    """Remove activation-checkpoint wrapper segments from a Hugging Face parameter name."""
    return name.replace("._checkpoint_wrapped_module", "")


def _composite_renames(names, checkpoint_names: set[str]) -> dict[str, str]:
    """Map text-only trained names onto a composite source checkpoint."""
    checkpoint_names = {_canonical_hf_export_name(name) for name in checkpoint_names}
    renames: dict[str, str] = {}
    unmapped: list[str] = []
    for name in names:
        if name in checkpoint_names:
            continue
        for old, new in _COMPOSITE_NAME_PREFIXES:
            if name.startswith(old) and new + name[len(old) :] in checkpoint_names:
                renames[name] = new + name[len(old) :]
                break
        else:
            unmapped.append(name)
    if not renames:
        return {}
    unmapped = [name for name in unmapped if name != "lm_head.weight"]
    if unmapped:
        examples = ", ".join(unmapped[:10])
        raise RuntimeError(
            f"checkpoint name translation mapped {len(renames)} name(s) but not {len(unmapped)} (e.g. {examples})"
        )
    return renames


def _hf_config_class_name(model_type: object) -> str | None:
    """Return the Hugging Face config class name registered for ``model_type``."""
    if not isinstance(model_type, str) or not model_type:
        return None
    from transformers import AutoConfig

    try:
        return type(AutoConfig.for_model(model_type)).__name__
    except (AttributeError, KeyError, OSError, ValueError):
        return None


def _read_hf_config(model_dir: str) -> dict | None:
    import json

    path = os.path.join(model_dir, _HF_CONFIG_FILES[0])
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def restore_source_weight_layout(source_model_dir: str, out_dir: str) -> bool:
    """Complete a text-only weights export into its composite source checkpoint layout."""
    import json

    from safetensors import safe_open
    from safetensors.torch import save_file

    source_config = _read_hf_config(source_model_dir)
    saved_config = _read_hf_config(out_dir)
    if source_config is None or saved_config is None:
        return False
    if "text_config" not in source_config:
        return False
    source = _safetensors_names(source_model_dir)
    saved = _safetensors_names(out_dir)
    if not source or not saved:
        return False
    saved_architectures = saved_config.get("architectures") or []
    saved_is_text_model = "text_config" not in saved_config or any(
        str(architecture).endswith("ForCausalLM") for architecture in saved_architectures
    )
    source_class = _hf_config_class_name(source_config.get("model_type"))
    saved_class = _hf_config_class_name(saved_config.get("model_type"))
    nested_text = saved_config.get("text_config")
    nested_class = _hf_config_class_name(nested_text.get("model_type")) if isinstance(nested_text, dict) else None
    # vLLM Qwen3_5ProcessingInfo.get_hf_config accepts Qwen3_5Config and rejects Qwen3_5TextConfig.
    engine_requires_qwen3_5_config = source_class == "Qwen3_5Config" and (
        saved_class == "Qwen3_5TextConfig" or (saved_class == "Qwen3_5Config" and nested_class == "Qwen3_5TextConfig")
    )
    if engine_requires_qwen3_5_config:
        saved_is_text_model = False
    if saved_is_text_model:
        source_layout: set[str] = set()
        saved_layout = {name for name in saved if not name.startswith(("model.visual.", "visual."))}
        renames = {}
        for name in saved_layout:
            if name.startswith("model.language_model."):
                renames[name] = "model." + name.removeprefix("model.language_model.")
            elif name.startswith("language_model."):
                renames[name] = name.removeprefix("language_model.")
    else:
        source_layout = set(source)
        saved_layout = {
            name
            for name in saved
            if name in source_layout
            or any(
                name.startswith(old) and new + name[len(old) :] in source_layout
                for old, new in _COMPOSITE_NAME_PREFIXES
            )
            or name == "lm_head.weight"
        }
        renames = _composite_renames(saved_layout, source_layout)
    excluded = set(saved) - saved_layout
    weight_map: dict[str, str] = {}
    total_size = 0

    def account(handle, keys, shard_name: str) -> None:
        nonlocal total_size
        for key in keys:
            weight_map[renames.get(key, key)] = shard_name
            sliced = handle.get_slice(key)
            numel = 1
            for dimension in sliced.get_shape():
                numel *= dimension
            total_size += numel * _DTYPE_BYTES[sliced.get_dtype()]

    for shard in sorted(set(saved.values())):
        path = os.path.join(out_dir, shard)
        target = "model-00001-of-00001.safetensors" if shard == "model.safetensors" else shard
        with safe_open(path, framework="pt") as handle:
            keys = [key for key in handle.keys() if key not in excluded]
            account(handle, keys, target)
            tensors = (
                {renames.get(key, key): handle.get_tensor(key) for key in keys}
                if any(key in renames or key in excluded for key in handle.keys())
                else None
            )
        if tensors is not None:
            os.remove(path)
            if tensors:
                save_file(tensors, os.path.join(out_dir, target), metadata={"format": "pt"})
        elif target != shard:
            os.replace(path, os.path.join(out_dir, target))

    source_only = sorted(
        name for name in source_layout if name not in weight_map and _canonical_hf_export_name(name) not in weight_map
    )
    if source_only:
        extra = "model-source-only.safetensors"
        tensors = {}
        by_shard: dict[str, list[str]] = {}
        for name in source_only:
            by_shard.setdefault(source[name], []).append(name)
        for shard, names in sorted(by_shard.items()):
            with safe_open(os.path.join(source_model_dir, shard), framework="pt") as handle:
                tensors.update({name: handle.get_tensor(name) for name in names})
                account(handle, names, extra)
        save_file(tensors, os.path.join(out_dir, extra), metadata={"format": "pt"})
    with open(os.path.join(out_dir, _HF_INDEX_FILE), "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}, handle)
    config_path = os.path.join(out_dir, _HF_CONFIG_FILES[0])
    if saved_class == "Qwen3_5TextConfig" and source_class == "Qwen3_5Config":
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(source_config, handle)
    elif saved_is_text_model and isinstance(nested_text, dict):
        text_config = dict(nested_text)
        text_config["architectures"] = saved_architectures
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(text_config, handle)
    logger.info(
        "weights-only save: completed source layout (%d renamed, %d excluded, %d copied from %s)",
        len(renames),
        len(excluded),
        len(source_only),
        source_model_dir,
    )
    return True


def replace_exported_qwen3_5_text_config(source_model_dir: str | None, out_dir: str) -> bool:
    """Rewrite an exported ``Qwen3_5TextConfig`` to ``Qwen3_5Config``.

    ``AutoModelForCausalLM`` maps model type ``qwen3_5`` to ``Qwen3_5ForCausalLM``, and that class
    serializes ``Qwen3_5TextConfig``. The serving engine resolves ``Qwen3_5ForCausalLM`` to
    ``Qwen3_5ForConditionalGeneration`` and accepts only ``Qwen3_5Config``. The saved text fields stay the
    exported text config; vision fields come from the source checkpoint when that checkpoint is ``qwen3_5``.
    """
    import json

    from transformers import Qwen3_5Config

    saved = _read_hf_config(out_dir)
    if not isinstance(saved, dict) or saved.get("model_type") != "qwen3_5_text":
        return False
    source = _read_hf_config(source_model_dir) if source_model_dir else None
    text_config = dict(saved)
    architectures = text_config.pop("architectures", None)
    kwargs: dict = {"text_config": text_config}
    if isinstance(source, dict) and source.get("model_type") == "qwen3_5":
        for key in (
            "vision_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "tie_word_embeddings",
        ):
            if key in source:
                kwargs[key] = source[key]
    exported = Qwen3_5Config(**kwargs).to_dict()
    if architectures:
        exported["architectures"] = architectures
    with open(os.path.join(out_dir, _HF_CONFIG_FILES[0]), "w", encoding="utf-8") as handle:
        json.dump(exported, handle)
    return True


def _copy_source_sidecars(source_model_dir: str, out_dir: str) -> None:
    """Copy non-weight source assets without replacing the exported model config."""
    import shutil

    for name in sorted(os.listdir(source_model_dir)):
        source = os.path.join(source_model_dir, name)
        target = os.path.join(out_dir, name)
        if not os.path.isfile(source):
            continue
        lower = name.lower()
        if lower.endswith((".safetensors", ".bin", ".pt", ".pth", ".index.json")):
            continue
        if name == "config.json" and os.path.isfile(target):
            continue
        shutil.copyfile(source, target)


def _canonical_hf_export_state_dict(model, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert internal weights and remove activation-checkpoint wrappers before Hugging Face serialization."""
    state_dict = {_canonical_hf_export_name(name): tensor for name, tensor in state_dict.items()}
    is_prime_state_dict = getattr(model, "is_prime_state_dict", None)
    if is_prime_state_dict is not None and is_prime_state_dict(state_dict):
        state_dict = model.convert_to_hf(state_dict)
    return state_dict


def _gather_live_hf_export_state_dict(model, rank: int) -> dict[str, torch.Tensor] | None:
    """Gather expert-parallel shards and return the complete live state on rank zero."""
    import deepspeed.utils.groups as ds_groups

    state_dict = {} if rank == 0 else None
    for name, parameter in model.named_parameters():
        group_name = getattr(parameter, "group_name", None)
        if group_name is not None and getattr(parameter, "allreduce", True) is False:
            group = ds_groups._get_expert_parallel_group(group_name)
            local = parameter.detach().contiguous()
            shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group=group))]
            dist.all_gather(shards, local, group=group)
            if rank == 0:
                state_dict[name] = torch.cat(shards, dim=0).cpu()
        elif rank == 0:
            state_dict[name] = parameter.detach().cpu()
    return state_dict


def _model_full_hf_export_state_dict(model, rank: int) -> dict[str, torch.Tensor] | None:
    """Collect a model-provided full Hugging Face state dict on rank zero."""
    iterator = getattr(model, "_iter_full_hf_weights", None)
    if iterator is None:
        return None
    state_dict = dict(iterator())
    return state_dict if rank == 0 else None


# ---------------------------------------------------------------------------
# Request / response models (mirrors dss-platform sftp_server)
# ---------------------------------------------------------------------------

ENABLE_TIMERS = False
if ENABLE_TIMERS:
    from arctic_platform.common.utils.debug import SynchronizedWallClockTimerSimple

    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.common.utils.debug import SynchronizedWallClockTimerSimpleDummy

    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)


def make_model_gradient_checkpointing_compatible(model):
    # Taken from arctic_platform/model/hf_factory.py
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model


# ---------------------------------------------------------------------------
# DeepSpeed training actor
# ---------------------------------------------------------------------------


def _worker_debug_config(job_config: dict) -> dict:
    """Resolve debug settings from native and legacy on-prem job payloads."""
    training_debug = (job_config.get("training_config") or {}).get("debug") or {}
    worker_debug = (job_config.get("ds_worker_config") or {}).get("debug") or {}
    debug = {**training_debug, **worker_debug}
    if "full_determinism" in job_config:
        debug["full_determinism"] = job_config["full_determinism"]
    return debug


def _setup_model_parallel_groups(spec: ModelSpec) -> dict[str, Any]:
    """Create the runtime process groups consumed by the native model loaders.

    The native worker supports either no sequence parallelism or full-world sequence parallelism. Expert groups
    remain node-local contiguous blocks; ranks at the same position in each block form the expert-data-parallel
    replica groups that DeepSpeed uses for gradient reduction and optimizer sharding.
    """
    import deepspeed.utils.groups as ds_groups
    from deepspeed.runtime.sequence_parallel import parallel_state_sp as sp_mpu

    ep_size = spec.parallelism.expert_parallel
    sp_size = spec.parallelism.sequence_parallel
    world_size = dist.get_world_size()
    if world_size % ep_size:
        raise ValueError(f"world_size={world_size} must be divisible by expert_parallel={ep_size}")

    if sp_size == 1:
        if ep_size > 1:
            ds_groups._create_expert_and_data_parallel(ep_size)
        return {
            "ep_group": ds_groups._get_expert_parallel_group(f"ep_size_{ep_size}") if ep_size > 1 else None,
            "sp_group": None,
            "mpu": None,
        }

    if sp_size != world_size:
        raise ValueError(
            f"the native worker supports sequence_parallel=1 or the full world ({world_size}), got {sp_size}"
        )
    sp_mpu.initialize_sequence_parallel(sp_size)

    ep_name = f"ep_size_{ep_size}"
    rank = dist.get_rank()
    if ep_size > 1:
        for start in range(0, world_size, ep_size):
            ranks = list(range(start, start + ep_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                ds_groups._EXPERT_PARALLEL_GROUP[ep_name] = group
                ds_groups._EXPERT_PARALLEL_GROUP_RANKS[ep_name] = ranks
        for offset in range(ep_size):
            ranks = list(range(offset, world_size, ep_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                ds_groups._EXPERT_DATA_PARALLEL_GROUP[ep_name] = group
                ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[ep_name] = ranks

    ds_groups.mpu = sp_mpu
    return {
        "ep_group": ds_groups._get_expert_parallel_group(ep_name) if ep_size > 1 else None,
        "sp_group": sp_mpu.get_sequence_parallel_group(),
        "mpu": sp_mpu,
    }


def _deepspeed_init_kwargs(model, ds_config: dict, parallel_groups: dict[str, Any], has_optimizer: bool) -> dict:
    """Build DeepSpeed initialization arguments with the runtime's model-parallel topology."""
    kwargs = {"model": model, "config": ds_config}
    mpu = parallel_groups.get("mpu")
    if mpu is not None:
        kwargs["mpu"] = mpu
    if has_optimizer:
        kwargs["model_parameters"] = model.parameters()
    return kwargs


async def spawn_and_initialize_workers(gpus, master_port, config_dict, actor_options):
    """Create DeepSpeed ranks and initialize them. Destroy every spawned actor if any step fails.

    ``actor_options(rank)`` is the kwargs for ``DeepSpeedWorker.options``. Rank 0's host is the
    distributed rendezvous master so off-node ranks do not hang on their own localhost.
    """
    workers = []
    try:
        for rank in range(gpus):
            workers.append(DeepSpeedWorker.options(**actor_options(rank)).remote(rank, gpus, master_port))
        master_addr = await workers[0].get_ip.remote()
        await asyncio.gather(*[w.initialize.remote(master_addr, config_dict) for w in workers])
    except Exception:
        await asyncio.gather(*[w.destroy.remote() for w in workers], return_exceptions=True)
        raise
    return workers


@ray.remote
class DeepSpeedWorker:
    """Single-GPU worker for DeepSpeed training."""

    def __init__(self, rank: int, world_size: int, master_port: int):
        self.rank = rank
        self.world_size = world_size
        self.my_addr = socket.gethostname()
        self.master_addr = primary_ip()
        self.master_port = master_port
        self.engine = None
        self.sp_size = 1
        self._weight_sender = None
        self._on_gpu = True
        self._gradient_norms_per_param = False
        self._source_model_dir: str | None = None

    def get_ip(self) -> str:
        return self.my_addr

    def initialize(self, master_addr: str, job_config: dict) -> bool:
        self.master_addr = master_addr
        os.environ.update(
            {
                "RANK": str(self.rank),
                "LOCAL_RANK": "0",
                "WORLD_SIZE": str(self.world_size),
                "MASTER_ADDR": self.master_addr,
                "MASTER_PORT": str(self.master_port),
            }
        )

        debug_config = _worker_debug_config(job_config)
        determinism_config = {"debug": debug_config}
        from arctic_platform.model.implementations.debug.determinism import configure_full_determinism
        from arctic_platform.model.implementations.debug.determinism import determinism_worker_env
        from arctic_platform.model.implementations.debug.determinism import full_determinism_enabled
        from arctic_platform.model.implementations.debug.determinism import resolve_seed

        seed = resolve_seed(job_config.get("seed"), determinism_config)
        os.environ.update(determinism_worker_env(determinism_config, seed))
        if full_determinism_enabled(determinism_config):
            enable_full_determinism(seed=seed)
            configure_full_determinism(determinism_config)

        # aws-ofi-nccl generates a per-process topology and hands it to NCCL by setting NCCL_TOPO_FILE to a
        # /proc/self/fd/<N> path (an in-memory fd). That handle is only valid in the process that created it: once
        # inherited by a child, fd <N> resolves to an unrelated file, and because the plugin skips regenerating when
        # NCCL_TOPO_FILE is already set, NCCL loads a bogus topology and the multi-rank rendezvous deadlocks. This
        # worker has not run OFI yet, so any value present here is necessarily inherited and stale -- drop it.
        #
        # Dropping it does NOT cost performance: the plugin sets NCCL_TOPO_FILE only as an in-process mechanism to
        # pass its generated topology to NCCL, and it regenerates that topology during OFI init whenever the var is
        # unset. So popping the stale handle simply makes the plugin produce a fresh, correct, platform-optimal
        # topology for this process -- exactly what happens on a clean first init -- instead of reusing a poisoned
        # one. The narrow /proc/self/fd/ check also leaves an admin-provided static topology path untouched.
        topo_file = os.environ.get("NCCL_TOPO_FILE", "")
        if topo_file.startswith("/proc/self/fd/"):
            logger.warning(f"dropping inherited stale NCCL_TOPO_FILE={topo_file}; OFI will regenerate per-process")
            os.environ.pop("NCCL_TOPO_FILE", None)

        deepspeed.init_distributed()

        model_name = job_config["model_name"]
        ds_config = job_config.get("ds_config") or {}
        self.job_type = job_config.get("job_type")
        pr0(f"{self.job_type=} {job_config=}")
        pr0(f"ds_worker[before_modify]: {self.job_type=} {ds_config=}")

        ds_worker_config = job_config.get("ds_worker_config") or {}
        ds_worker_config["world_size"] = self.world_size
        self.ds_worker_config = ds_worker_config
        self._source_model_dir = model_name
        self.sp_size = sp_size_from_job_config(job_config)
        gradient_norms_per_param = debug_config.get("gradient_norms_per_param", False)
        if not isinstance(gradient_norms_per_param, bool):
            raise TypeError(
                "gradient_norms_per_param in the training debug config must be a bool, "
                f"got {type(gradient_norms_per_param).__name__}"
            )
        self._gradient_norms_per_param = gradient_norms_per_param

        # Build the DeepSpeed config per job type. Training engines get an
        # optimizer; the reference/log-prob engine is forward-only and is
        # configured from log_prob_config with no optimizer state.
        if self.job_type == "log_prob":
            log_prob_config = job_config.get("log_prob_config") or {}
            ds_config = self.ds_inference_config(log_prob_config, ds_worker_config)
            self._has_optimizer = False
        else:
            ds_config = self.ds_training_config(job_config, ds_config, ds_worker_config)
            self._has_optimizer = True

        pr0(f"ds_worker[after_modify]: {self.job_type=} {ds_config=} {ds_worker_config=}")

        # HF load + patches via ModelSpec (world_size already injected above).
        spec = ModelSpec.from_ds_worker_config(model_name, ds_worker_config)
        from arctic_platform.model.implementations.debug.determinism import maybe_partial_determinism_support

        maybe_partial_determinism_support(model_name, spec.attn_implementation, ds_worker_config)
        parallel_groups = _setup_model_parallel_groups(spec)
        loaded = build_model(spec, parallel_groups=parallel_groups)
        model = loaded.model

        zorro_train_enable = ds_worker_config.get("zorro_train_enable", False)
        self.dedup_actor_model_once_patcher = getattr(model, "_arctic_zorro_once_patcher", None)

        # Forward-only (log-prob) engines omit model parameters so DeepSpeed allocates no optimizer state.
        init_kwargs = _deepspeed_init_kwargs(model, ds_config, parallel_groups, self._has_optimizer)
        self.engine, _, _, _ = deepspeed.initialize(**init_kwargs)
        self._device = get_accelerator().device_name(self.engine.local_rank)

        gpu_id = torch.cuda.current_device()
        gpu_uuid = torch.cuda.get_device_properties(gpu_id).uuid
        logger.info("Rank %d initialized on GPU %d (uuid=%s, device=%s)", self.rank, gpu_id, gpu_uuid, self._device)
        self.cpu_device = torch.device("cpu")

        pr0(
            "ds_worker[after_initialize]:"
            f" {self.job_type=} {self.engine.global_steps=} {zorro_train_enable=} {model_name=}"
        )

        return True

    def ds_training_config(self, job_config: dict, ds_config: dict, ds_worker_config: dict) -> dict:
        """Build the DeepSpeed config for a trainable engine (with optimizer).

        Prefer the high-level training_config when provided (the framework sends
        training_config and the server owns the DeepSpeed details); otherwise
        fall back to a default AdamW optimizer.
        """
        training_config = job_config.get("training_config")
        if training_config is not None:
            opt_cfg = training_config.get("optimizer", {})
            betas = opt_cfg.get("betas")
            if betas is None:
                betas = [opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)]
            # Forward DeepSpeed-recognized AdamW knobs (torch_adam selects
            # torch.optim.AdamW instead of FusedAdam — required for Axolotl
            # adamw_torch parity).
            opt_params: dict[str, Any] = {
                "lr": opt_cfg.get("lr", 1e-5),
                "betas": list(betas),
                "eps": opt_cfg.get("eps", 1e-8),
                "weight_decay": opt_cfg.get("weight_decay", 0.0),
            }
            if opt_cfg.get("torch_adam"):
                opt_params["torch_adam"] = True
            if "adam_w_mode" in opt_cfg:
                opt_params["adam_w_mode"] = opt_cfg["adam_w_mode"]
            ds_config["optimizer"] = {
                "type": "AdamW",
                "params": opt_params,
            }
            if "gradient_accumulation_steps" in training_config:
                ds_config["gradient_accumulation_steps"] = training_config["gradient_accumulation_steps"]
            # Prefer explicit optimizer.gradient_clipping; accept top-level alias.
            if "gradient_clipping" in opt_cfg:
                ds_config["gradient_clipping"] = opt_cfg["gradient_clipping"]
            elif "gradient_clipping" in training_config:
                ds_config["gradient_clipping"] = training_config["gradient_clipping"]
            elif "max_grad_norm" in opt_cfg:
                ds_config["gradient_clipping"] = opt_cfg["max_grad_norm"]

            sched_cfg = training_config.get("lr_scheduler", None)
            horizon = training_config.get("training_horizon", 0)
            if sched_cfg is not None and horizon > 0:
                lr = opt_cfg.get("lr", 1e-5)
                # DeepSpeed's WarmupLR/WarmupCosineLR require warmup_num_steps to
                # be an integer; warmup_ratio * horizon (or an explicit fractional
                # warmup_steps override) is generally fractional.
                if "warmup_steps" in sched_cfg:
                    warmup_steps = round(float(sched_cfg["warmup_steps"]))
                else:
                    warmup_steps = round(sched_cfg.get("warmup_ratio", 0.0) * horizon)
                if sched_cfg.get("type", "constant") == "cosine":
                    ds_config["scheduler"] = {
                        "type": "WarmupCosineLR",
                        "params": {
                            "total_num_steps": horizon,
                            "warmup_num_steps": warmup_steps,
                            "warmup_min_ratio": 0.0,
                            "cos_min_ratio": sched_cfg.get("min_lr_ratio") or 0.0,
                            "warmup_type": "linear",
                        },
                    }
                elif warmup_steps > 0:
                    ds_config["scheduler"] = {
                        "type": "WarmupLR",
                        "params": {
                            "warmup_min_lr": 0.0,
                            "warmup_max_lr": lr,
                            "warmup_num_steps": warmup_steps,
                            "warmup_type": "linear",
                        },
                    }
                else:
                    # No LR scheduler, use constant LR
                    pass

        # Set reasonable defaults as fallbacks
        ds_config.setdefault("train_micro_batch_size_per_gpu", 1)
        fp16_on = bool((ds_config.get("fp16") or {}).get("enabled"))
        if not fp16_on:
            ds_config.setdefault("bf16", {"enabled": True})
        ds_config.setdefault(
            "optimizer",
            {
                "type": "AdamW",
                "params": {"lr": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8},
            },
        )
        ds_config.setdefault("torch_autocast", {"enabled": True, "dtype": "bfloat16"})
        ds_config.setdefault("communication_data_type", "fp32")
        ds_config.setdefault("data_types", {"grad_accum_dtype": "fp32"})

        return ds_config

    def ds_inference_config(self, log_prob_config: dict, ds_worker_config: dict) -> dict:
        """Build a forward-only DeepSpeed config (no optimizer state).

        Used for the reference / log-prob engine. Keeps ZeRO param sharding
        (stage + offload_param) and bf16, but omits the optimizer, gradient
        accumulation, gradient clipping, train_batch_size, and any
        offload_optimizer so DeepSpeed allocates no optimizer state.
        """
        src = dict(log_prob_config or {})
        zero = dict(src.get("zero_optimization", {}) or {})
        zero.pop("offload_optimizer", None)

        cfg: dict = {
            "train_micro_batch_size_per_gpu": src.get("train_micro_batch_size_per_gpu", 1),
        }
        if zero:
            cfg["zero_optimization"] = zero
        if "sequence_parallel_size" in src:
            cfg["sequence_parallel_size"] = src["sequence_parallel_size"]
        if ds_worker_config.get("use_autocast", False):
            cfg["torch_autocast"] = {"enabled": True, "dtype": "bfloat16"}
        cfg["bf16"] = {"enabled": True}
        return cfg

    # move batch to device
    def _move_batch_to_device(self, batch: Any, device: torch.device):
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(v, device) for v in batch]
        elif isinstance(batch, torch.Tensor):
            # Pin + non_blocking H2D: copy overlaps with subsequent CPU work on the
            # default stream; the next CUDA kernel that consumes the tensor waits.
            # ``device`` may be a str ("cuda:0") or a torch.device.
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            if dev.type == "cuda" and batch.device.type == "cpu":
                if not batch.is_pinned():
                    try:
                        batch = batch.pin_memory()
                    except RuntimeError:
                        pass  # non-pageable / shared storage — fall through
                return batch.to(dev, non_blocking=True)
            return batch.to(dev)
        return batch

    def _inject_sft_global_token_meta(self, loss_fn: str, batch_data, meta_data: dict) -> None:
        """All-reduce valid-target count into ``meta["global_num_tokens"]`` + ``dp_size``.

        The loss callback scales by the logical data-parallel degree. Under sequence parallelism, the SP ranks are
        one logical replica and their label shards are summed into the global token denominator.
        Opt-in via ``SFT_GLOBAL_TOKEN_LOSS_FNS``. No-op when labels are absent.
        """
        from arctic_platform.sft.processor import SFT_GLOBAL_TOKEN_LOSS_FNS
        from arctic_platform.sft.processor import count_valid_target_tokens

        if loss_fn not in SFT_GLOBAL_TOKEN_LOSS_FNS:
            return

        local_tokens = count_valid_target_tokens(batch_data, meta_data)
        if local_tokens is None:
            return

        global_tokens = local_tokens
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.world_size > 1:
            tok = torch.tensor([local_tokens], device=self._device, dtype=torch.long)
            torch.distributed.all_reduce(tok, op=torch.distributed.ReduceOp.SUM)
            global_tokens = int(tok.item())
        meta_data["global_num_tokens"] = global_tokens
        meta_data["dp_size"] = dp_sp_world_size(self.world_size, self.sp_size)

    def _forward_maybe_backward(self, batch: dict, backward: bool) -> dict:
        # torch.autograd.set_detect_anomaly(True)

        pr0(f"_forward_maybe_backward mode: {backward=}")
        PROFILE = False
        # if backward:
        #     PROFILE = True
        if PROFILE:
            torch.cuda.memory._record_memory_history(max_entries=int(1e12))
        see_memory_usage("_forward_maybe_backward start", force=True)

        from arctic_platform import sft_profile
        from arctic_platform.rl.processors.base_loss import _pop_loss_object

        loss_object = _pop_loss_object(batch)
        args, batch_data, meta_data, processing = unpack_batch(batch)
        with sft_profile.timed("h2d"):
            if isinstance(batch_data, list):
                batch_data = [self._move_batch_to_device(mb, self._device) for mb in batch_data]
            else:
                batch_data = self._move_batch_to_device(batch_data, self._device)
            if sft_profile.enabled() and torch.cuda.is_available():
                torch.cuda.synchronize()

        tag = "forward_only" if not backward else "forward_backward"

        if isinstance(batch_data, list):
            for i, mb in enumerate(batch_data):
                log_dp_shard_tokens(self.rank, f"{tag} shard mb{i}", mb, meta_data)
            pr0(f"[DeepSpeedWorker] {tag}: gas_list={len(batch_data)} {meta_data.keys()=} {processing.keys()=}")
        else:
            log_dp_shard_tokens(self.rank, f"{tag} shard", batch_data, meta_data)
            pr0(f"[DeepSpeedWorker] {tag}: {batch_data.keys()=} {meta_data.keys()=} {processing.keys()=}")
            for k, v in batch_data.items():
                pr0(f"[DeepSpeedWorker] {tag}: {k=}: shape={getattr(v, 'shape', type(v).__name__)}")

        grad_accum_steps = self.engine.gradient_accumulation_steps()
        # H3: list-of-microbatches from the client skips concat→split_dict.
        if isinstance(batch_data, list):
            micro_batch_data = batch_data
            if len(micro_batch_data) != grad_accum_steps:
                raise ValueError(
                    f"Received {len(micro_batch_data)} GAS microbatches but "
                    f"engine.gradient_accumulation_steps()={grad_accum_steps}"
                )
        else:
            micro_batch_data = split_dict(batch_data, grad_accum_steps)
        num_micro_batches = len(micro_batch_data)
        pipeline_micro_batch_outputs = []
        return_tensors = meta_data.get("worker_return_tensors", False)

        # Resolve before selecting a specialized pipeline so class registry
        # precedence applies equally to SFT and RL names.
        loss_fn = processing.get("loss_fn", "ap_grpo")
        from arctic_platform.common.registry import LOSS_FNS
        from arctic_platform.rl.processors import resolve_loss
        from arctic_platform.sft.processor import SFT_LOSS_FNS

        if loss_object is None and loss_fn is not None:
            loss_object = resolve_loss(loss_fn)
        if loss_object is not None:
            loss_object.model_call_count_callback(
                [num_micro_batches],
                processing.get("config") or {},
            )
        legacy_sft_loss = LOSS_FNS.get(loss_fn) if loss_fn in SFT_LOSS_FNS else None
        use_sft_pipeline = (
            legacy_sft_loss is not None
            and loss_object is not None
            and loss_object.is_legacy_adapter_for(legacy_sft_loss)
        )

        if use_sft_pipeline:
            self._inject_sft_global_token_meta(loss_fn, batch_data, meta_data)

        loss_reduction = None
        if not use_sft_pipeline:
            from arctic_platform.rl.processors import resolve_packed_loss_reduction

            loss_reduction = resolve_packed_loss_reduction(
                processing,
                [{**meta_data, **micro_batch} for micro_batch in micro_batch_data],
                require_declared=False,
                loss_object=loss_object,
            )

        pr0(f"mbs {len(micro_batch_data)=} {grad_accum_steps=}")

        for i, micro_batch in enumerate(micro_batch_data):

            # time.sleep(1)
            # pr0(f"{i=}")
            # pr0(f"{micro_batch.keys()=}")

            log_dp_shard_tokens(
                self.rank,
                f"{tag} micro_batch {i}/{num_micro_batches}",
                micro_batch,
                meta_data,
            )

            DEBUG = False
            if DEBUG:
                from arctic_platform.rl.zorro_train import analyze_normal_batch_via_attention_mask

                analyze_normal_batch_via_attention_mask(
                    micro_batch["input_ids"], micro_batch["attention_mask"], response_len=meta_data["max_response_len"]
                )

            # die
            see_memory_usage(f"_forward_maybe_backward mb {i=}", force=True)
            if i == 0:
                pr0(f"[DeepSpeedWorker] {tag}: {i=}/{num_micro_batches=} {meta_data.keys()=} {processing.keys()=}")

            if use_sft_pipeline:
                from arctic_platform.sft.processor import run_sft_pipeline

                # Match the GRPO packed path: only the last microbatch is an
                # accumulation boundary so DeepSpeed does not all-reduce grads
                # (or run optimizer bookkeeping) on every microbatch when gas>1.
                if backward and hasattr(self.engine, "set_gradient_accumulation_boundary"):
                    self.engine.set_gradient_accumulation_boundary(i == num_micro_batches - 1)

                micro_batch_output = run_sft_pipeline(
                    self.engine,
                    micro_batch,
                    meta_data,
                    processing,
                    device=self._device,
                    backward=backward,
                )
            else:
                from arctic_platform.rl.processors import apply_packed_loss_reduction
                from arctic_platform.rl.processors import run_pipeline

                pipeline_backward = "loss_only" if backward and loss_reduction is not None else backward
                micro_batch_output = run_pipeline(
                    self.engine,
                    args,
                    micro_batch,
                    meta_data,
                    processing,
                    device=self._device,
                    backward=pipeline_backward,
                    pack=False,
                    return_tensors=return_tensors,
                    loss_object=loss_object,
                )
                if backward and loss_reduction is not None:
                    apply_packed_loss_reduction(
                        self.engine,
                        micro_batch_output.pop("loss_tensor"),
                        loss_reduction.loss_scales[i],
                        backward=True,
                    )

            if i == 0:
                pr0(f"[DeepSpeedWorker] {tag}: {i=}/{num_micro_batches=} {micro_batch_output.keys()=}")
            pipeline_micro_batch_outputs.append(micro_batch_output)

            # DS requires matching steps for backward pass
            if backward and i < num_micro_batches - 1:
                self._engine_step()

        pipeline_outputs = dict()
        for k, v in pipeline_micro_batch_outputs[0].items():
            if k == "metrics" and isinstance(v, dict):
                if loss_reduction is None:
                    # Legacy losses without packed metadata retain the
                    # historical GAS metric combiner.
                    pipeline_outputs[k] = combine_metric_microbatches([r[k] for r in pipeline_micro_batch_outputs])
                else:
                    from arctic_platform.rl.processors import combine_packed_metrics

                    pipeline_outputs[k] = combine_packed_metrics(
                        [r[k] for r in pipeline_micro_batch_outputs],
                        loss_reduction.reporting_weights,
                    )
            elif isinstance(v, dict):
                pipeline_outputs[k] = merge_dict_shards([r[k] for r in pipeline_micro_batch_outputs])
            elif isinstance(v, numbers.Number):
                values = [r[k] for r in pipeline_micro_batch_outputs]
                if k == "avg_loss" and loss_reduction is not None:
                    from arctic_platform.rl.processors import combine_packed_losses

                    pipeline_outputs[k] = combine_packed_losses(values, loss_reduction)
                else:
                    pipeline_outputs[k] = sum(values) / len(values)

        pipeline_outputs = self._move_batch_to_device(pipeline_outputs, self.cpu_device)

        see_memory_usage("_forward_maybe_backward end", force=True)
        if PROFILE:
            dir = "/tmp/mem-prof"
            rank = 0  # torch.distributed.get_rank()
            from pathlib import Path

            Path(dir).mkdir(exist_ok=True)
            torch.cuda.memory._dump_snapshot(f"{dir}/rank-{rank}.pickle")
            exit()

        pr0(f"[DeepSpeedWorker] {tag}: {pipeline_outputs.keys()=}")
        from arctic_platform import sft_profile

        if sft_profile.enabled():
            metrics = pipeline_outputs.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            metrics = sft_profile.merge_into_metrics(metrics)
            pipeline_outputs["metrics"] = metrics
            sft_profile.maybe_print(f"worker rank{self.rank} {tag}", metrics.get("_profile_ms"))
        return pipeline_outputs

    def forward_backward(self, batch: dict) -> dict:
        tname = timers.start("forward_backward")
        results = self._forward_maybe_backward(batch, backward=True)
        timers.stop_and_print_elapsed(tname)
        return results

    def forward_no_grad(self, batch: dict) -> dict:
        tname = timers.start("forward_no_grad")
        results = self._forward_maybe_backward(batch, backward=False)
        timers.stop_and_print_elapsed(tname)
        return results

    def _per_parameter_gradient_norm_metrics(self) -> dict[str, Any]:
        """Return one full-gradient norm per trainable parameter from rank zero."""
        if not self._gradient_norms_per_param:
            return {}

        from deepspeed.utils import safe_get_full_grad

        model = self.engine.module if hasattr(self.engine, "module") else self.engine
        norms: dict[str, float] = {}
        for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
            if not param.requires_grad:
                continue
            full_grad = safe_get_full_grad(param)
            if full_grad is None:
                continue
            norms[name] = float(torch.linalg.vector_norm(full_grad.detach().float()))
        if self.rank != 0:
            return {}
        return {"gradient_norms_per_param": norms}

    def _per_global_expert_gradient_norm_metrics(self) -> dict[str, Any]:
        """Return norms indexed by global expert for expert-parallel parameters from rank zero."""
        if not self._gradient_norms_per_param:
            return {}

        import deepspeed.utils.groups as ds_groups
        from deepspeed.utils import safe_get_full_grad

        model = self.engine.module if hasattr(self.engine, "module") else self.engine
        norms: dict[str, list[float]] = {}
        for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
            if not param.requires_grad:
                continue
            group_name = getattr(param, "group_name", None)
            if group_name is None or getattr(param, "allreduce", True) is not False:
                continue
            full_grad = safe_get_full_grad(param)
            if full_grad is None or full_grad.dim() < 2:
                continue
            local = torch.linalg.vector_norm(full_grad.detach().float().flatten(1), dim=1).contiguous()
            ep_group = ds_groups._get_expert_parallel_group(group_name)
            shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group=ep_group))]
            dist.all_gather(shards, local, group=ep_group)
            if self.rank == 0:
                norms[name] = [float(value) for value in torch.cat(shards)]
        if self.rank != 0 or not norms:
            return {}
        return {"gradient_norms_per_expert": norms}

    def _is_bf16_zero_norm_assert(self, exc: BaseException) -> bool:
        """True only for BF16_Optimizer's bare ``assert all_groups_norm > 0.``."""
        return is_bf16_zero_norm_assert(exc, getattr(self.engine, "optimizer", None))

    def _engine_step(self) -> None:
        """``engine.step()`` with a skip for DeepSpeed BF16_Optimizer's zero-norm assert.

        ZeRO-1/2 use ``BF16_Optimizer``, which asserts ``all_groups_norm > 0``.
        GRPO can produce an all-zero grad batch (identical group rewards). ZeRO-3
        does not assert; skip the optimizer update instead of crashing.
        Only that BF16 assert is skipped; every other ``AssertionError`` re-raises.
        """
        try:
            self.engine.step()
        except AssertionError as err:
            if self._is_bf16_zero_norm_assert(err):
                pr0("[DeepSpeedWorker] skip optimizer.step: global grad norm is 0")
                return
            raise

    def _set_optimizer_learning_rate(self, learning_rate: float | None) -> None:
        if learning_rate is None:
            return
        optimizer = getattr(self.engine, "optimizer", None)
        if optimizer is None:
            return
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def step(self, learning_rate: float | None = None) -> dict:
        from arctic_platform import sft_profile

        self._set_optimizer_learning_rate(learning_rate)
        gradient_norm_metrics = self._per_parameter_gradient_norm_metrics()
        expert_gradient_norm_metrics = self._per_global_expert_gradient_norm_metrics()
        with sft_profile.timed("step"):
            self._engine_step()
            if sft_profile.enabled() and torch.cuda.is_available():
                torch.cuda.synchronize()
        # Pull grad_norm out of DeepSpeed so it can be logged by the trainer.
        # rename_dict in ray_trainer turns "grad_norm" -> "actor/grad_norm",
        # matching the FSDP baseline path in verl/workers/actor/dp_actor.py.
        grad_norm = self.engine.get_global_grad_norm()
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        metrics = dict(
            global_steps=self.engine.global_steps,
            last_lr=self.engine.get_lr()[0],
            **gradient_norm_metrics,
            **expert_gradient_norm_metrics,
        )
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        metrics = sft_profile.merge_into_metrics(metrics)
        if sft_profile.enabled():
            sft_profile.maybe_print(f"worker rank{self.rank} step", metrics.get("_profile_ms"))
        return dict(metrics=metrics, batch=dict())

    def save_checkpoint(self, path: str, export_hf: bool = False) -> dict:
        """Save DeepSpeed checkpoint; optionally export HF weights to ``{path}/hf/`` (rank 0)."""
        self.engine.save_checkpoint(path)
        hf_path = None
        if export_hf:
            hf_path = self.export_hf_checkpoint(path)
        return {
            "path": path,
            "hf_path": hf_path,
            "global_step": int(self.engine.global_steps),
        }

    def routable_ip(self) -> str:
        """Return this worker's address on the allocation network."""
        from arctic_platform.common.ray_cluster import primary_ip

        return primary_ip()

    def export_node_checkpoint(self, path: str, peers: list[str]) -> list[str]:
        """Copy checkpoint files written on this node to ``peers`` and open them here."""
        from arctic_platform.common.utils.checkpoint import publish_node_checkpoint

        return publish_node_checkpoint(path, peers)

    def require_checkpoint_files(self, path: str, relative_paths: list[str]) -> None:
        """Open every checkpoint file in ``relative_paths`` under ``path``."""
        from arctic_platform.common.utils.checkpoint import require_checkpoint_files

        require_checkpoint_files(path, relative_paths)

    def load_checkpoint(self, path: str) -> int:
        """Restore DeepSpeed state. Returns restored ``global_steps``, or 0 if none found."""
        load_path, _ = self.engine.load_checkpoint(path)
        if load_path is None:
            return 0
        return int(self.engine.global_steps)

    def export_hf_checkpoint(self, checkpoint_dir: str) -> str | None:
        """Convert a DeepSpeed checkpoint dir to HF ``save_pretrained`` under ``hf/``.

        Rank 0 only; other ranks return None after a barrier. Prefer DeepSpeed's
        ``get_fp32_state_dict_from_zero_checkpoint``; fall back to gathering live params.
        """
        import os

        hf_dir = os.path.join(checkpoint_dir, "hf")
        model = self.engine.module
        full_state_dict = _model_full_hf_export_state_dict(model, self.rank)
        if not hasattr(model, "_iter_full_hf_weights"):
            full_state_dict = _gather_live_hf_export_state_dict(model, self.rank)
        if self.rank == 0:
            os.makedirs(hf_dir, exist_ok=True)
            state_dict = full_state_dict
            if not hasattr(model, "_iter_full_hf_weights"):
                state_dict = _canonical_hf_export_state_dict(model, state_dict)
            try:
                model.save_pretrained(hf_dir, state_dict=state_dict, safe_serialization=True)
            except TypeError:
                # Older transformers: no state_dict kwarg — load then save.
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logger.warning("HF export load_state_dict missing=%s unexpected=%s", missing, unexpected)
                model.save_pretrained(hf_dir, safe_serialization=True)
            if self._source_model_dir and restore_source_weight_layout(self._source_model_dir, hf_dir):
                _copy_source_sidecars(self._source_model_dir, hf_dir)
            replace_exported_qwen3_5_text_config(self._source_model_dir, hf_dir)
            logger.info("Exported HF checkpoint to %s", hf_dir)
        # The Ray server waits for every worker result, including rank zero's complete export. A distributed
        # barrier here only leaves nonzero ranks inside NCCL while rank zero reconstructs the full fp32 state.
        return hf_dir if self.rank == 0 else None

    @staticmethod
    def prune_checkpoint_dirs(parent_dir: str, keep: int) -> int:
        """Keep the newest ``keep`` ``checkpoint-*`` dirs under ``parent_dir``; remove older."""
        from arctic_platform.common.utils.checkpoint import prune_checkpoint_dirs

        return prune_checkpoint_dirs(parent_dir, keep)

    def compute_log_probs(self, batch: dict) -> torch.Tensor:
        """Full-sequence per-token log-probs for a ``{"batch","meta","processing"}`` DP shard.

        Takes a dict shard (like ``forward_no_grad``) -- ``unpack_batch`` pulls the encoded ``{input_ids,
        attention_mask}`` out of ``batch["batch"]`` -- runs a forward-only pass, and returns the shifted-label
        per-token log-probs ``[shard_B, S-1]`` on CPU. The DeepSpeed engine ignores ``top_k`` (unlike the vLLM path).
        """
        _, kwargs, _, _ = unpack_batch(batch)
        kwargs = {k: v.to(self._device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        model = self.engine.module
        with torch.no_grad():
            if getattr(model, "_dss_chunked_lm_head_logprobs", False) or getattr(
                model, "_dss_native_lm_head_logprobs", False
            ):
                outputs = self.engine(**kwargs, dss_compute_logprobs=True)
                token_log_probs = outputs["logprobs"] if isinstance(outputs, dict) else outputs.logprobs
            else:
                outputs = self.engine(**kwargs)
                if isinstance(outputs, dict) and outputs.get("logprobs") is not None:
                    token_log_probs = outputs["logprobs"]
                else:
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                    shifted_ids = kwargs["input_ids"][:, 1:]
                    token_log_probs = (
                        torch.log_softmax(logits, dim=-1)[:, :-1].gather(-1, shifted_ids.unsqueeze(-1)).squeeze(-1)
                    )
        return token_log_probs.cpu()

    def max_param_bytes(self) -> int:
        max_bytes = 0
        for p in self.engine.module.parameters():
            elem_size = p.data.element_size()
            numel = p.ds_numel if hasattr(p, "ds_id") else p.data.numel()
            max_bytes = max(max_bytes, numel * elem_size)
        return max_bytes

    def init_weight_sender(self, group, schedule, master_addr, base_port, bucket_size) -> bool:
        from arctic_inference.server.weight_sync.sender import WeightSender

        self._weight_sender = WeightSender(
            group=group,
            schedule=schedule,
            master_addr=master_addr,
            base_port=base_port,
            device=torch.device(self._device),
            bucket_size=bucket_size,
        )
        return True

    def get_weights(self) -> list[tuple[str, torch.Tensor]]:
        weights = []
        for n, p in self.engine.module.named_parameters():
            if hasattr(p, "ds_id"):
                with deepspeed.zero.GatheredParameters([p], enabled=True):
                    weights.append((n, p.data))
            else:
                weights.append((n, p.data))
        return weights

    def weight_norm(self) -> dict:
        """Global L2 norm of the model's parameters (sum of squares + count).

        ZeRO-3 partitions each param across ranks, so ``GatheredParameters``
        materializes the full param on every rank; the resulting sum of
        squares is therefore the whole-model value (the server reads rank 0).
        Summing squares is layout-invariant, so it compares directly against
        the vLLM engine's norm despite vLLM fusing params differently. Used by
        tests to confirm a weight sync landed.
        """
        sq_sum = 0.0
        num_params = 0
        for _, p in self.engine.module.named_parameters():
            if hasattr(p, "ds_id"):
                with deepspeed.zero.GatheredParameters([p], enabled=True):
                    sq_sum += p.data.double().pow(2).sum().item()
            else:
                sq_sum += p.data.double().pow(2).sum().item()
            num_params += 1
        return {"norm": sq_sum**0.5, "sq_sum": sq_sum, "num_params": num_params}

    def send_weights(self) -> dict:
        weights = self.get_weights()
        if self._weight_sender is not None:
            return self._weight_sender.send(weights)
        return {"status": "not_initialized"}

    def send_weights_ipc(self, group_id: int) -> dict:
        """Save weights to shared memory for colocated (same-GPU) transfer."""
        from arctic_inference.server.weight_sync.ipc_engine import save_weights_to_shm

        weights = [(n, p.data) for n, p in self.engine.module.named_parameters()]
        return save_weights_to_shm(weights, group_id)

    def get_cuda_ipc_handles(self) -> dict:
        """Create CUDA IPC handles for all model parameters.

        Returns a dict with names, dtypes, shapes and pickled IPC handles
        that can be opened by another process on the same GPU.

        For ZeRO-2 this reads p.data directly (full params on every rank).
        For ZeRO-3 this is incorrect — use gather_cuda_ipc_handles instead.
        """
        import base64
        import pickle

        from torch.multiprocessing.reductions import reduce_tensor

        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

        names, dtype_names, shapes = [], [], []
        handles = []
        self._ipc_tensor_refs = []

        for name, p in self.engine.module.named_parameters():
            weight = p.data.detach().contiguous()
            self._ipc_tensor_refs.append(weight)
            handle = reduce_tensor(weight)
            handles.append({gpu_uuid: handle})
            names.append(name)
            dtype_names.append(str(weight.dtype).split(".")[-1])
            shapes.append(list(weight.shape))

        torch.cuda.synchronize()
        pickled = base64.b64encode(pickle.dumps(handles)).decode("utf-8")
        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles_pickled": pickled,
            "num_params": len(names),
        }

    def gather_cuda_ipc_handles(self) -> dict:
        """Gather ZeRO-3 partitioned params and create CUDA IPC handles.

        All ranks must call this collectively (GatheredParameters is a
        collective op).  Inside the context manager the full param lives
        on every rank's GPU — each rank clones it and creates an IPC handle
        keyed by its own GPU UUID before the context manager frees the
        gathered tensor.  Every rank returns its payload; the server merges
        them so each colocated inference replica finds a handle for its GPU.
        """
        import base64
        import pickle

        import deepspeed
        from torch.multiprocessing.reductions import reduce_tensor

        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

        t0 = time.monotonic()
        names, dtype_names, shapes = [], [], []
        handles = []
        self._ipc_tensor_refs = []
        model = self.engine.module

        # Every rank builds IPC handles for the full (gathered) weight on its
        # OWN physical GPU, keyed by that GPU's UUID. With colocated multi-GPU
        # inference, each vLLM replica lives on a distinct physical GPU (bundle
        # r == training rank r), so it needs a handle for *its* GPU. The server
        # merges these per-rank dicts so each replica finds its GPU's handle.
        # (Previously only rank 0 produced handles, which only worked when a
        # single sampling GPU was colocated with rank 0.)
        for name, p in model.named_parameters():
            if hasattr(p, "ds_id"):
                with deepspeed.zero.GatheredParameters([p], enabled=True):
                    weight = p.data.detach().clone().contiguous()
                    self._ipc_tensor_refs.append(weight)
                    handle = reduce_tensor(weight)
                    handles.append({gpu_uuid: handle})
                    names.append(name)
                    dtype_names.append(str(weight.dtype).split(".")[-1])
                    shapes.append(list(weight.shape))
            else:
                weight = p.data.detach().contiguous()
                self._ipc_tensor_refs.append(weight)
                handle = reduce_tensor(weight)
                handles.append({gpu_uuid: handle})
                names.append(name)
                dtype_names.append(str(weight.dtype).split(".")[-1])
                shapes.append(list(weight.shape))

        if hasattr(self.engine, "empty_partition_cache"):
            self.engine.empty_partition_cache()
        torch.cuda.synchronize()

        elapsed = time.monotonic() - t0
        pickled = base64.b64encode(pickle.dumps(handles)).decode("utf-8")
        logger.info(
            "Rank %d gathered IPC handles in %.2fs (%d params, gpu=%s)", self.rank, elapsed, len(names), gpu_uuid
        )
        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles_pickled": pickled,
            "num_params": len(names),
            "gpu_uuid": gpu_uuid,
        }

    def release_ipc_handles(self) -> bool:
        """Release tensor references held for IPC handles."""
        self._ipc_tensor_refs = []
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        return True

    def get_parameter_names(self) -> list:
        """Return the model's parameter names in deterministic order.

        Used to drive the low-memory streaming weight sync one parameter at a
        time. Only names cross the Ray boundary; the live parameter is resolved
        on each rank inside ``get_cuda_ipc_handle`` so ZeRO-3 ``ds_id`` / live
        storage is preserved.
        """
        return [name for name, _ in self.engine.module.named_parameters()]

    def _param_by_name(self, name: str):
        """Resolve this rank's live module parameter for ``name`` (cached)."""
        cache = getattr(self, "_param_name_cache", None)
        if cache is None:
            cache = dict(self.engine.module.named_parameters())
            self._param_name_cache = cache
        return cache[name]

    def get_cuda_ipc_handle(self, name: str) -> dict:
        """Create a CUDA IPC handle payload for a single parameter on this
        rank's GPU.

        Memory-efficient counterpart to ``gather_cuda_ipc_handles``: only ONE
        full parameter is materialized at a time instead of the whole model.
        For ZeRO-3 params all ranks must call this collectively with the same
        ``name`` (``GatheredParameters`` is a collective op). The caller must
        invoke ``release_ipc_handles`` between params so peak extra GPU memory
        stays at one full parameter per GPU.
        """
        import base64
        import pickle

        import deepspeed
        from torch.multiprocessing.reductions import reduce_tensor

        p = self._param_by_name(name)

        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

        if hasattr(p, "ds_id"):
            with deepspeed.zero.GatheredParameters([p], enabled=True):
                weight = p.data.detach().clone().contiguous()
        else:
            weight = p.data.detach().contiguous()

        # Hold exactly one source tensor alive until release_ipc_handles().
        self._ipc_tensor_refs = [weight]
        handle = reduce_tensor(weight)
        torch.cuda.synchronize()

        return {
            "names": [name],
            "dtype_names": [str(weight.dtype).split(".")[-1]],
            "shapes": [list(weight.shape)],
            "ipc_handles_pickled": base64.b64encode(pickle.dumps([{gpu_uuid: handle}])).decode("utf-8"),
            "num_params": 1,
            "gpu_uuid": gpu_uuid,
        }

    def save_state_dict_to_path(self, path: str) -> dict:
        """Save model state dict to a file.

        Works for both ZeRO-2 (params are full) and ZeRO-3 (params are
        partitions — we just save whatever is local).  For ZeRO-3, the
        caller must ensure all ranks call this collectively and only
        rank 0's output is used.
        """
        t0 = time.monotonic()
        weights = [(n, p.data.cpu()) for n, p in self.engine.module.named_parameters()]
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(weights, path)
        num_params = len(weights)
        del weights
        import gc

        gc.collect()
        elapsed = time.monotonic() - t0
        logger.info("Rank %d saved state dict to %s in %.2fs (%d params)", self.rank, path, elapsed, num_params)
        return {"num_params": num_params, "elapsed": elapsed}

    def gather_and_save_state_dict(self, path: str) -> dict:
        """Gather ZeRO-3 partitioned params and save full state dict.

        All ranks must call this collectively.  Every parameter is wrapped
        in GatheredParameters so the all-gather runs on all ranks.  Only
        rank 0 copies the full tensor and writes to disk.
        """
        import deepspeed

        t0 = time.monotonic()
        model = self.engine.module
        weights = []
        for n, p in model.named_parameters():
            if hasattr(p, "ds_id"):
                with deepspeed.zero.GatheredParameters([p], enabled=True):
                    if self.rank == 0:
                        weights.append((n, p.data.cpu()))
            else:
                if self.rank == 0:
                    weights.append((n, p.data.cpu()))
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(weights, path)
        num_params = len(weights)
        del weights
        import gc

        gc.collect()
        if hasattr(self.engine, "empty_partition_cache"):
            self.engine.empty_partition_cache()
        torch.cuda.empty_cache()
        elapsed = time.monotonic() - t0
        logger.info("Rank %d gathered+saved state dict in %.2fs (%d params)", self.rank, elapsed, num_params)
        return {"num_params": num_params, "elapsed": elapsed}

    def _log_mem(self, label):
        from deepspeed.runtime.utils import see_memory_usage

        see_memory_usage(f"[Rank {self.rank}] {label}", force=True)

    def _ds_offload(self, include):
        """Offload states using DeepSpeed native API.

        engine.offload_states works for ZeRO-3/ZeRO-2 without
        offload_optimizer.  When offload_optimizer is configured,
        DeepSpeed raises AssertionError ("Moving states across devices
        is not supported"); fall back to optimizer.offload_states which
        bypasses the engine assertion (see DeepSpeed issue #6596).
        """
        from deepspeed.runtime.zero.config import OffloadDeviceEnum

        try:
            self.engine.offload_states(include=include)
        except AssertionError as e:
            if "Moving states across devices" not in str(e):
                raise
            opt = getattr(self.engine, "optimizer", None)
            if opt is None:
                # Forward-only (inference) engine has no optimizer to fall back
                # to; the engine-level assertion only fires with offload_optimizer.
                raise
            opt.offload_states(
                include=include,
                device=OffloadDeviceEnum.cpu,
                pin_memory=True,
            )

    def _ds_reload(self):
        """Reload state to GPU.

        engine.reload_states works for most configs.  When
        offload_optimizer is configured, falls back to optimizer
        directly.  ZeRO-2 reload_states requires .grad on fp32 param
        partitions; create zero grads if missing.  Forward-only engines
        have no optimizer, so the optimizer-specific handling is skipped.
        """
        opt = getattr(self.engine, "optimizer", None)
        if opt is not None and hasattr(opt, "single_partition_of_fp32_groups"):
            for fp32_group in opt.single_partition_of_fp32_groups:
                if fp32_group.grad is None:
                    fp32_group.grad = torch.zeros_like(fp32_group)
        try:
            self.engine.reload_states()
        except AssertionError as e:
            if "Moving states across devices" not in str(e):
                raise
            if opt is None:
                raise
            opt.reload_states()

    def _move_params(self, device) -> None:
        """Move model parameters to ``device`` for a forward-only engine.

        DeepSpeed's offload_states/reload_states require a real optimizer, so a
        no-optimizer (inference) engine can't use them. Instead move each
        parameter's storage directly. Under ZeRO-3 the local shard lives in
        ``param.ds_tensor``; otherwise it is ``param.data``. This is only ever
        called while the engine is idle (the client wakes it before any forward
        and sleeps it after), so it never races a parameter all-gather.
        """
        for p in self.engine.module.parameters():
            shard = getattr(p, "ds_tensor", None)
            if shard is not None:
                shard.data = shard.data.to(device, non_blocking=True)
            else:
                p.data = p.data.to(device, non_blocking=True)

    def offload_to_cpu(self) -> dict:
        """Offload engine state to CPU using DeepSpeed native API.

        Trainable engines offload optimizer/grad states plus model params via
        the DeepSpeed offload_states API. A forward-only (no-optimizer) engine
        has no optimizer for that API, so it moves only its model params.
        """
        if not self._on_gpu:
            return {"status": "already_offloaded"}
        t0 = time.monotonic()
        self._log_mem("before offload_to_cpu")

        if getattr(self, "_has_optimizer", True):
            from deepspeed.runtime.zero.offload_states import OffloadStateTypeEnum

            self._ds_offload(
                include=[
                    OffloadStateTypeEnum.hp_params,
                    OffloadStateTypeEnum.lp_params,
                    OffloadStateTypeEnum.lp_grads,
                    OffloadStateTypeEnum.contiguous_grad_buffer,
                ]
            )
        else:
            self._move_params(self.cpu_device)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self._on_gpu = False
        elapsed = time.monotonic() - t0
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Rank %d offloaded to CPU in %.2fs (%.0f MB GPU remaining)", self.rank, elapsed, mem_mb)
        return {"status": "offloaded", "elapsed": elapsed, "gpu_mb": mem_mb}

    def backload_to_gpu(self) -> dict:
        """Reload engine state to GPU using DeepSpeed native API.

        Forward-only engines move only their model params back (no optimizer
        state to reload)."""
        if self._on_gpu:
            return {"status": "already_on_gpu"}
        t0 = time.monotonic()
        self._log_mem("before reload_states")

        if getattr(self, "_has_optimizer", True):
            self._ds_reload()
        else:
            self._move_params(torch.device(self._device))
        torch.cuda.empty_cache()
        self._log_mem("after reload_states + empty_cache")

        torch.cuda.synchronize()
        self._on_gpu = True
        elapsed = time.monotonic() - t0
        logger.info("Rank %d backloaded to GPU in %.2fs", self.rank, elapsed)
        return {"status": "on_gpu", "elapsed": elapsed}

    def offload_non_lp_states(self) -> dict:
        """Offload everything except bf16 params (for CUDA IPC sync)."""
        t0 = time.monotonic()
        self._log_mem("before offload_non_lp")

        from deepspeed.runtime.zero.offload_states import OffloadStateTypeEnum

        self._ds_offload(
            include=[
                OffloadStateTypeEnum.hp_params,
                OffloadStateTypeEnum.lp_grads,
                OffloadStateTypeEnum.contiguous_grad_buffer,
            ]
        )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        elapsed = time.monotonic() - t0
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Rank %d offloaded non-lp states in %.2fs (%.0f MB GPU)", self.rank, elapsed, mem_mb)
        return {"status": "offloaded_non_lp", "elapsed": elapsed, "gpu_mb": mem_mb}

    def offload_lp_params(self) -> dict:
        """Offload bf16 model params to CPU (after CUDA IPC sync)."""
        t0 = time.monotonic()

        from deepspeed.runtime.zero.offload_states import OffloadStateTypeEnum

        self._ds_offload(include=[OffloadStateTypeEnum.lp_params])

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self._on_gpu = False
        elapsed = time.monotonic() - t0
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Rank %d offloaded lp_params in %.2fs (%.0f MB GPU remaining)", self.rank, elapsed, mem_mb)
        return {"status": "offloaded_lp", "elapsed": elapsed, "gpu_mb": mem_mb}

    def empty_cache(self) -> dict:
        """Release ZeRO-3 partition cache and PyTorch cached memory."""
        if hasattr(self.engine, "empty_partition_cache"):
            self.engine.empty_partition_cache()
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Rank %d empty_cache: %.0f MB GPU remaining", self.rank, mem_mb)
        return {"gpu_mb": mem_mb}

    def destroy(self) -> bool:
        if self._weight_sender is not None:
            self._weight_sender.destroy()
            self._weight_sender = None
        if dist.is_initialized():
            dist.destroy_process_group()
        self.engine = None
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
