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
"""DeepSpeed engine bring-up for a training job.

Builds the DeepSpeed config, constructs the optimizer, and calls
``deepspeed.initialize``. This is separate from model construction. The
on-prem RL worker in ``arctic_platform.common.deepspeed_worker`` translates
its config through ``TrainingEngineSpec`` and uses the same initialization
path.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from arctic_platform.common.optimizer import OptimizerSettings
from arctic_platform.debug.deepspeed_zero_initialization import maybe_trace_zero_initialization
from arctic_platform.debug.training import maybe_configure_router_replay_trace
from arctic_platform.debug.weight_sync import maybe_configure_broadcast_training
from arctic_platform.model.implementations.fp8 import keeps_fp32
from arctic_platform.model.implementations.fp8 import preserve_fp8_params_through_ds_cast
from arctic_platform.model.moe_spec import is_glm_moe_model

logger = logging.getLogger(__name__)

# Qwen3.8-Flash-Next's PLE names its RMSNorms norm_key/norm_query/norm_conv, which "norm.weight" does not match.
_NO_DECAY = (
    "bias",
    "layer_norm.weight",
    "layernorm.weight",
    "norm.weight",
    "ln_f.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
)
_OPTIMIZER_ALIASES = {
    "fusedadam": "fused_adam",
    "fused_adam": "fused_adam",
    "torch_adamw": "torch_adamw",
    "deepspeed_cpu_adam": "deepspeed_cpu_adam",
    "deepspeedcpuadam": "deepspeed_cpu_adam",
    "cpu_adam": "deepspeed_cpu_adam",
    "cpuadam": "deepspeed_cpu_adam",
    "deep_speed_cpu_adam": "deepspeed_cpu_adam",
}


class TrainingEngineSpec(BaseModel):
    """Engine inputs that do not depend on a training service.

    ``fp32_masters`` and ``zero_bucket_size`` come from the caller's job config.
    ``ds_overrides`` is shallow-merged last, matching a client DeepSpeed
    override block.
    """

    model_config = ConfigDict(extra="forbid")

    dtype: str = "bfloat16"
    world_size: int
    sequence_parallel: int = 1
    train_micro_batch_size_per_gpu: int = Field(1, ge=1)
    gradient_accumulation_steps: int = Field(1, ge=1)
    train_batch_size: int | None = Field(None, ge=1)
    zero_stage: int | None = Field(2, ge=0, le=3)
    gradient_clipping: float | None = 1.0
    seed: int | None = None
    ds_overrides: dict[str, Any] | None = None
    fp32_masters: bool = False
    zero_bucket_size: int | None = None
    optimizer: OptimizerSettings | None = None
    adam_w_mode: bool = True

    @classmethod
    def from_deepspeed_config(
        cls,
        config: dict[str, Any],
        *,
        dtype: str,
        world_size: int,
        sequence_parallel: int = 1,
        seed: int | None = None,
    ) -> "TrainingEngineSpec":
        """Represent an existing DeepSpeed config without dropping native blocks.

        Supported AdamW optimizer dictionaries become ``OptimizerSettings``;
        scheduler dictionaries remain in ``ds_overrides`` so DeepSpeed binds
        them to the explicit optimizer.
        """
        source = dict(config)
        zero = source.get("zero_optimization")
        zero_stage = int(zero.get("stage", 0)) if isinstance(zero, dict) else None
        optimizer_config = source.pop("optimizer", None)
        optimizer = _optimizer_settings_from_deepspeed(optimizer_config)
        return cls(
            dtype=dtype,
            world_size=world_size,
            sequence_parallel=sequence_parallel,
            train_micro_batch_size_per_gpu=int(source.get("train_micro_batch_size_per_gpu", 1)),
            gradient_accumulation_steps=int(source.get("gradient_accumulation_steps", 1)),
            train_batch_size=source.get("train_batch_size"),
            zero_stage=zero_stage,
            gradient_clipping=(
                float(source["gradient_clipping"]) if source.get("gradient_clipping") is not None else None
            ),
            seed=seed,
            ds_overrides=source,
            optimizer=optimizer,
            adam_w_mode=_adam_w_mode_from_deepspeed(optimizer_config),
        )


def build_deepspeed_config(spec: TrainingEngineSpec) -> dict[str, Any]:
    """Build the config dict consumed by ``deepspeed.initialize``.

    Sequence parallelism reduces the data-parallel degree used to derive
    ``train_batch_size`` when the caller does not provide it explicitly.
    Gradients accumulate in FP32 unless the merged config says otherwise;
    see ``apply_default_grad_accum_dtype``.
    """
    dtype = (spec.dtype or "bfloat16").lower()
    is_bf16 = dtype in ("bfloat16", "bf16")
    is_fp16 = dtype in ("float16", "fp16")
    # bf16 master weights match FusedAdam's training dynamics. fp32 masters
    # are the opt-in that disables them.
    bf16_masters = is_bf16 and not spec.fp32_masters
    train_batch_size = spec.train_batch_size or (
        _data_parallel_size(spec.world_size, spec.sequence_parallel)
        * spec.train_micro_batch_size_per_gpu
        * spec.gradient_accumulation_steps
    )
    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": spec.train_micro_batch_size_per_gpu,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "bf16": {
            "enabled": is_bf16,
            "bf16_master_weights_and_grads": bf16_masters,
            "bf16_optimizer_states": bf16_masters,
        },
        "fp16": {"enabled": is_fp16},
        # NCCL bf16 all-reduce is not accumulated in fp32 internally.
        "communication_data_type": "fp32" if is_bf16 else None,
    }
    if spec.gradient_clipping is not None:
        ds_config["gradient_clipping"] = spec.gradient_clipping
    if spec.zero_stage is not None:
        ds_config["zero_optimization"] = {"stage": spec.zero_stage}
    if spec.seed is not None:
        ds_config["seed"] = int(spec.seed)
    if spec.sequence_parallel > 1:
        # A DeepSpeedConfig built from this dict without the SP mpu reads this key for the dp degree;
        # without it DeepSpeed checks ``train_batch_size`` against the full world size and asserts.
        ds_config["sequence_parallel_size"] = spec.sequence_parallel
    if spec.zero_bucket_size is not None:
        zero_config = ds_config.setdefault("zero_optimization", {"stage": spec.zero_stage or 0})
        zero_config["reduce_bucket_size"] = spec.zero_bucket_size
        zero_config["allgather_bucket_size"] = spec.zero_bucket_size
    if spec.ds_overrides:
        ds_config.update(spec.ds_overrides)
    apply_default_grad_accum_dtype(ds_config)
    _require_fp32_gradient_reduction(ds_config)
    logger.info(
        "Resolved DeepSpeed precision: bf16=%s fp16=%s communication_data_type=%s",
        ds_config.get("bf16"),
        ds_config.get("fp16"),
        ds_config.get("communication_data_type"),
    )
    return ds_config


def apply_default_grad_accum_dtype(ds_config: dict[str, Any]) -> None:
    """Set ``data_types.grad_accum_dtype`` to ``"fp32"`` in place when the config leaves it unset.

    DeepSpeed accumulates gradients in the model dtype when ``grad_accum_dtype`` is absent or ``None``, so a bf16
    job would sum its microbatch gradients at bfloat16 granularity. The default belongs to the key, not the block: a
    supplied ``data_types`` without it gets FP32 too, and its other fields are kept. An explicit value is never
    changed. The block is replaced by a copy, so a caller's shared ``data_types`` dict is never mutated.

    DeepSpeed refuses FP32 accumulation in two configurations, which keep its model-dtype default instead: ZeRO
    disabled with bf16 or fp16 parameters (``unsupported mix of model_dtype and grad_accum_dtype``), and ZeRO stage 1
    with bf16 parameters, ``compute_grad_norm`` false and no CPU optimizer offload, which selects the BF16 optimizer
    that requires the norm.
    """
    data_types = ds_config.get("data_types")
    if data_types is not None and not isinstance(data_types, dict):
        return
    if (data_types or {}).get("grad_accum_dtype") is not None:
        return
    if not _deepspeed_accepts_fp32_grad_accumulation(ds_config):
        return
    ds_config["data_types"] = {**(data_types or {}), "grad_accum_dtype": "fp32"}


def _deepspeed_accepts_fp32_grad_accumulation(ds_config: dict[str, Any]) -> bool:
    """Whether ``deepspeed.initialize`` accepts this config once ``grad_accum_dtype`` is ``"fp32"``."""
    bf16_config = ds_config.get("bf16")
    if bf16_config is None:
        bf16_config = ds_config.get("bfloat16")
    fp16_config = ds_config.get("fp16")
    bf16 = isinstance(bf16_config, dict) and bool(bf16_config.get("enabled"))
    fp16 = isinstance(fp16_config, dict) and bool(fp16_config.get("enabled"))
    zero_config = ds_config.get("zero_optimization")
    if not isinstance(zero_config, dict):
        zero_config = {}
    stage = int(zero_config.get("stage") or 0)
    if stage == 0:
        return not (bf16 or fp16)
    offload = zero_config.get("offload_optimizer")
    cpu_offload = isinstance(offload, dict) and offload.get("device") == "cpu"
    if stage == 1 and bf16 and not cpu_offload and zero_config.get("compute_grad_norm", True) is False:
        return False
    return True


def split_expert_param_groups(param_groups: list) -> list:
    """Name optimizer groups and split MoE parameters onto the expert-data-parallel group.

    DeepSpeed only does this split when it builds the optimizer itself. A caller-supplied
    optimizer has to hand it groups that already separate expert parameters.
    """
    from deepspeed.moe.utils import configure_moe_param_groups

    named = []
    for index, group in enumerate(param_groups):
        group.setdefault("name", f"group_{index}")
        named.append(group)
    return configure_moe_param_groups(named)


_OFFLOADED_SNAPSHOT_CHUNK_BYTES = 16 * 1024 * 1024


class _OffloadedGradientStorage:
    """Reuse bounded staging when FP32 offload snapshots outgrow a small CPU tensor."""

    def __init__(self):
        self.staging = None

    def snapshot(self, source):
        size = source.numel() * source.element_size()
        if size <= _OFFLOADED_SNAPSHOT_CHUNK_BYTES:
            return source.detach().to(device="cpu", non_blocking=False)
        snapshot = _OffloadedGradientFile(source, self)
        try:
            snapshot.copy_(source)
        except BaseException:
            snapshot.file.close()
            raise
        return snapshot

    def buffer(self, source):
        import torch

        if source.dtype != torch.float32 or not source.is_contiguous():
            raise ValueError("offloaded snapshots require contiguous FP32 partitions")
        if self.staging is None:
            self.staging = torch.empty(
                _OFFLOADED_SNAPSHOT_CHUNK_BYTES // source.element_size(),
                dtype=source.dtype,
                pin_memory=source.device.type == "cuda",
            )
        return self.staging


class _OffloadedGradientFile:
    """An owned, unlinked scratch file; native CPU master gradients remain untouched."""

    def __init__(self, source, storage):
        self.storage = storage
        self.shape = source.shape
        self.size = source.numel() * source.element_size()
        fast = Path("/data-fast/tmp")
        if fast.parent.is_dir():
            fast.mkdir(exist_ok=True)
            directory = str(fast)
        else:
            directory = tempfile.gettempdir()
        self.file = tempfile.TemporaryFile(prefix="ap-fp32-grad-", dir=directory)
        os.ftruncate(self.file.fileno(), self.size)
        os.posix_fadvise(self.file.fileno(), 0, 0, os.POSIX_FADV_RANDOM)

    def copy_(self, source):
        self._transfer(source, writing=True)

    def restore(self, destination):
        self._transfer(destination, writing=False)

    def _transfer(self, tensor, *, writing):
        staging = self.storage.buffer(tensor)
        if tensor.shape != self.shape:
            raise ValueError("offloaded snapshot shape differs from its partition")
        flat = tensor.view(-1)
        descriptor = self.file.fileno()
        operation = os.pwritev if writing else os.preadv
        for offset in range(0, flat.numel(), staging.numel()):
            length = min(staging.numel(), flat.numel() - offset)
            chunk = staging.narrow(0, 0, length)
            view = memoryview(chunk.numpy()).cast("B")
            byte_offset = offset * tensor.element_size()
            if writing:
                chunk.copy_(flat.narrow(0, offset, length), non_blocking=False)
            transferred = 0
            while transferred < len(view):
                count = operation(descriptor, [view[transferred:]], byte_offset + transferred)
                if count <= 0:
                    raise OSError("offloaded snapshot I/O made no progress")
                transferred += count
            if writing:
                # Flush each chunk before advisory eviction so dirty pages cannot grow with model size.
                os.fdatasync(descriptor)
            else:
                flat.narrow(0, offset, length).copy_(chunk, non_blocking=False)
            os.posix_fadvise(descriptor, byte_offset, len(view), os.POSIX_FADV_DONTNEED)


def release_offloaded_grad_partitions(optimizer) -> bool:
    """Evict completed ZeRO-1/2 offload partitions without losing FP32 request-boundary accumulation.

    DeepSpeed copies each reduced parameter to the CPU master gradient at an accumulation boundary but keeps the
    compact FP32 GPU partition until ``step()``. Once every local parameter has been copied, retain a separate
    FP32 snapshot and release the GPU partition. Small snapshots use CPU tensors; larger ones use unlinked local
    scratch files with a shared 16 MiB staging buffer and chunkwise page-cache eviction. Reuse that storage across
    backwards in the same update. A later
    backward restores the snapshot only when recreating its GPU partition, preserving the cumulative sum even when
    master gradients are BF16. Release completed groups during every reduction, including non-boundary microsteps.
    Partial groups are released after the native reduction epilogue, waiting for CPU copies before snapshotting.

    The wrappers are instance-local. Each group drops its snapshot before the native update allocates optimizer
    state: its CPU master gradient has already been unscaled and clipped, so the snapshot is no longer needed.
    Successful steps, including skipped overflow steps, clear all snapshots. Failed steps retain only groups not
    submitted for update; native updates are not transactional. Optimizers without CPU offload keep their methods.
    After native reset has waited for offload copies, discard compact groups' CPU accumulators as well: their sums and
    master-gradient views belong to the completed update, not to the next update or a restored checkpoint. Noncompact
    groups retain native accumulator ownership and cleanup.
    """
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    if type(optimizer) is not DeepSpeedZeroOptimizer or not optimizer.cpu_offload:
        return False
    compact_groups = [
        group_idx
        for group_idx in range(len(optimizer.bit16_groups))
        if optimizer._use_compact_grad_partition_buffer(group_idx)
    ]
    if not compact_groups:
        return False
    compact_param_ids = {
        optimizer.get_param_id(param) for group_idx in compact_groups for param in optimizer.bit16_groups[group_idx]
    }
    retained: dict[int, Any] = {}
    storage = _OffloadedGradientStorage()
    copy_grads_in_partition = optimizer.copy_grads_in_partition
    compact_grad_partition_buffer = optimizer._get_compact_grad_partition_buffer
    step = optimizer.step
    reset_cpu_buffers = optimizer.reset_cpu_buffers
    optimizer_step = optimizer._optimizer_step
    gradient_partition_epilogue = optimizer.independent_gradient_partition_epilogue

    def release_partition(group_idx):
        partition = optimizer.averaged_gradients.get(group_idx)
        if partition is None:
            return
        snapshot = retained.get(group_idx)
        if snapshot is None:
            retained[group_idx] = storage.snapshot(partition.flat_partition.detach())
        else:
            snapshot.copy_(partition.flat_partition.detach())
        optimizer.averaged_gradients[group_idx] = None

    def copy_then_release(param, group_idx=None):
        if group_idx is None:
            group_idx = optimizer._get_param_group_idx(param)
        copy_grads_in_partition(param, group_idx)
        if optimizer._use_compact_grad_partition_buffer(group_idx) and all(
            optimizer.params_already_reduced[optimizer.get_param_id(partition_param)]
            for partition_param in optimizer.params_in_partition[group_idx]
        ):
            release_partition(group_idx)

    def epilogue_then_release(*args, **kwargs):
        result = gradient_partition_epilogue(*args, **kwargs)
        if any(optimizer.averaged_gradients.get(group_idx) is not None for group_idx in compact_groups):
            optimizer._wait_for_offload_copies()
            for group_idx in compact_groups:
                release_partition(group_idx)
        return result

    def partition_buffer(group_idx):
        previous_partition = optimizer.averaged_gradients.get(group_idx)
        partition = compact_grad_partition_buffer(group_idx)
        snapshot = retained.get(group_idx)
        if snapshot is not None and partition is not previous_partition:
            if isinstance(snapshot, _OffloadedGradientFile):
                snapshot.restore(partition.flat_partition)
            else:
                partition.flat_partition.copy_(snapshot, non_blocking=False)
        return partition

    def reset_then_release(*args, **kwargs):
        result = reset_cpu_buffers(*args, **kwargs)
        for param_id in compact_param_ids:
            optimizer.accumulated_grads_in_cpu.pop(param_id, None)
        retained.clear()
        # The overflow path resets CPU buffers without detaching the compact GPU partition.
        for group_idx in optimizer.averaged_gradients:
            if optimizer._use_compact_grad_partition_buffer(group_idx):
                optimizer.averaged_gradients[group_idx] = None
        return result

    def step_then_release(*args, **kwargs):
        result = step(*args, **kwargs)
        retained.clear()
        return result

    def release_then_update(group_idx):
        retained.pop(group_idx, None)
        return optimizer_step(group_idx)

    optimizer.copy_grads_in_partition = copy_then_release
    optimizer.independent_gradient_partition_epilogue = epilogue_then_release
    optimizer._get_compact_grad_partition_buffer = partition_buffer
    optimizer.reset_cpu_buffers = reset_then_release
    optimizer.step = step_then_release
    optimizer._optimizer_step = release_then_update
    return True


def initialize_training_engine(
    model,
    ds_config: dict[str, Any],
    *,
    optimizer: OptimizerSettings | None,
    ep_size: int,
    mpu,
    model_path: str,
    rank: int,
    adam_w_mode: bool = True,
):
    """Build the optimizer, keep FP8 storage, and call ``deepspeed.initialize``.

    Returns ``(engine, optimizer, optimizer_choice)``. ``optimizer`` and
    ``optimizer_choice`` are ``None`` when the job has no optimizer.
    ``mpu`` is forwarded only when it is set, so ZeRO reduces sequence-parallel
    gradients over the sequence-data-parallel group.
    """
    ds_config = maybe_configure_broadcast_training(ds_config)
    ds_config = maybe_configure_router_replay_trace(ds_config, model, rank)
    built = None
    choice = None
    # Marks first: the optimizer groups split on them.
    preserved, marked_fp32 = preserve_fp8_params_through_ds_cast(model)
    if optimizer is not None:
        fused = optimizer.fused
        if fused is None:
            fused = _optimizer_cpu_offload_enabled(ds_config) or not is_glm_moe_model(model_path)
        built = build_training_optimizer(model, optimizer, ds_config, fused=fused, adam_w_mode=adam_w_mode)
        choice = optimizer_choice(optimizer, ds_config, fused=fused, adam_w_mode=adam_w_mode)
        if ep_size > 1:
            built.param_groups[:] = split_expert_param_groups(list(built.param_groups))
    if preserved or marked_fp32:
        from deepspeed.runtime.engine import DeepSpeedEngine

        logger.info(
            "rank=%d preserving %d FP8 parameters and %d FP32 parameters through the "
            "DeepSpeed dtype cast (deepspeed_param_cast_guard=%d)",
            rank,
            preserved,
            marked_fp32,
            int(bool(getattr(DeepSpeedEngine, "_ap_fp8_param_cast_guard", False))),
        )
    import deepspeed

    init_kwargs = {"mpu": mpu} if mpu is not None else {}
    with maybe_trace_zero_initialization(ds_config, rank) as init_config:
        if built is not None:
            engine, built, _, _ = deepspeed.initialize(model=model, optimizer=built, config=init_config, **init_kwargs)
        else:
            if init_config.get("optimizer") is not None:
                init_kwargs["model_parameters"] = model.parameters()
            engine, built, _, _ = deepspeed.initialize(model=model, config=init_config, **init_kwargs)
    if built is not None:
        release_offloaded_grad_partitions(built)
    return engine, built, choice


def resolve_optimizer_name(name: str, ds_config: dict, *, fused: bool = True) -> str:
    """Resolve defaults by offload policy, while preserving explicit optimizer choices."""
    normalized_name = name.strip().lower().replace("-", "_")
    cpu_offload = _optimizer_cpu_offload_enabled(ds_config)

    if normalized_name in {"", "adam", "adamw"}:
        if not cpu_offload:
            return "fused_adam"
        elif fused:
            return "deepspeed_cpu_adam"
        else:
            return "torch_cpu_adamw"

    optimizer_name = _OPTIMIZER_ALIASES.get(normalized_name)
    if optimizer_name is None:
        supported = ", ".join(sorted(set(_OPTIMIZER_ALIASES.values())))
        raise ValueError(f"Unsupported optimizer {name!r}; supported values: {supported}")
    if optimizer_name == "torch_adamw" and cpu_offload:
        return "torch_cpu_adamw"
    return optimizer_name


def optimizer_implementation(optimizer_name: str, *, fused: bool, adam_w_mode: bool = True) -> str:
    """The class ``build_training_optimizer`` constructs for a resolved name."""
    if optimizer_name == "fused_adam" and fused:
        return "fused_adam"
    if optimizer_name == "deepspeed_cpu_adam":
        return "deepspeed_cpu_adam"
    if optimizer_name == "torch_cpu_adamw":
        return "deepspeed_cpu_adam_nonfused"
    if optimizer_name in ("torch_adamw", "fused_adam"):
        return "torch_adamw" if adam_w_mode else "torch_adam"
    raise AssertionError(f"Unhandled optimizer {optimizer_name!r}")


def optimizer_choice(
    optimizer: OptimizerSettings, ds_config: dict, *, fused: bool, adam_w_mode: bool = True
) -> dict[str, str]:
    """The implementation a job runs and the customer-facing reason."""
    optimizer_name = resolve_optimizer_name(optimizer.name, ds_config, fused=fused)
    implementation = optimizer_implementation(optimizer_name, fused=fused, adam_w_mode=adam_w_mode)
    if implementation == "deepspeed_cpu_adam_nonfused":
        reason = "optimizer state is offloaded to CPU, with standard non-fused AdamW arithmetic"
    elif implementation == "deepspeed_cpu_adam":
        reason = (
            "optimizer state is offloaded to CPU, so the CPU implementation (CPUAdam) ran"
            if _optimizer_cpu_offload_enabled(ds_config)
            else f"you selected '{optimizer.name}'"
        )
    elif implementation == "fused_adam":
        reason = (
            "FusedAdam ran in Adam mode because you disabled AdamW mode"
            if not adam_w_mode
            else "AdamW ran as the faster fused GPU implementation (FusedAdam)"
        )
    elif implementation == "torch_adam":
        reason = "you disabled AdamW mode, so standard Adam ran"
    elif optimizer_name == "torch_adamw":
        reason = f"you selected '{optimizer.name}'"
    elif optimizer.fused is False:
        reason = "you set 'optimizer.fused' to false, so standard AdamW ran"
    else:
        reason = "FusedAdam isn't supported for this model, so standard AdamW ran"
    return {"implementation": implementation, "reason": reason}


def cpu_adam_fp32_optimizer_states(ds_config: dict) -> bool:
    bf16_config = ds_config.get("bf16")
    if isinstance(bf16_config, dict) and bf16_config.get("bf16_optimizer_states"):
        return False
    return True


def build_training_optimizer(
    model, optimizer: OptimizerSettings, ds_config: dict, *, fused: bool, adam_w_mode: bool = True
):
    """Build AdamW with decay disabled on bias and norm parameters.

    Generic AdamW uses fused implementations unless the model or caller disables
    them. Explicit CPUAdam remains selected regardless of that policy.
    """
    optimizer_name = resolve_optimizer_name(optimizer.name, ds_config, fused=fused)
    if _optimizer_cpu_offload_enabled(ds_config) and optimizer_name in ("deepspeed_cpu_adam", "torch_cpu_adamw"):
        logger.info(
            "training_config.ds_config.zero_optimization.offload_optimizer is configured; using optimizer=%r",
            optimizer_name,
        )
    decay_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not any(token in name.lower() for token in _NO_DECAY)
    ]
    no_decay_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and any(token in name.lower() for token in _NO_DECAY)
    ]
    # ZeRO flattens each group into one buffer of its first parameter's dtype, so
    # parameters kept in FP32 through the bf16 cast need groups of their own.
    param_groups = [
        {"params": _aligned_first([param for param in params if keeps_fp32(param) == fp32]), "weight_decay": decay}
        for params, decay in ((decay_params, optimizer.weight_decay), (no_decay_params, 0.0))
        for fp32 in (False, True)
    ]
    param_groups = [group for group in param_groups if group["params"]]
    betas = (optimizer.betas[0], optimizer.betas[1])
    implementation = optimizer_implementation(optimizer_name, fused=fused, adam_w_mode=adam_w_mode)
    if implementation == "fused_adam":
        from deepspeed.ops.adam import FusedAdam

        return FusedAdam(
            param_groups,
            lr=optimizer.lr,
            betas=betas,
            eps=optimizer.eps,
            adam_w_mode=adam_w_mode,
        )
    if implementation in ("deepspeed_cpu_adam", "deepspeed_cpu_adam_nonfused"):
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        built = DeepSpeedCPUAdam(
            param_groups,
            lr=optimizer.lr,
            betas=betas,
            eps=optimizer.eps,
            adamw_mode=adam_w_mode,
            fp32_optimizer_states=cpu_adam_fp32_optimizer_states(ds_config),
        )
        if implementation == "deepspeed_cpu_adam_nonfused":
            from arctic_platform.common.cpu_adam import configure_nonfused_cpu_adam

            return configure_nonfused_cpu_adam(built)
        return built
    import torch

    optimizer_class = torch.optim.AdamW if adam_w_mode else torch.optim.Adam
    return optimizer_class(
        param_groups,
        lr=optimizer.lr,
        betas=betas,
        eps=optimizer.eps,
    )


def _optimizer_settings_from_deepspeed(config: Any) -> OptimizerSettings | None:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("DeepSpeed optimizer config must be a dictionary")
    optimizer_type = str(config.get("type") or "AdamW")
    if optimizer_type.lower() not in {"adam", "adamw"}:
        raise ValueError(f"Unsupported DeepSpeed optimizer type {optimizer_type!r}; only Adam/AdamW are supported")
    params = config.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("DeepSpeed optimizer params must be a dictionary")
    if params.get("lr") is None:
        raise ValueError("DeepSpeed optimizer params must set lr")
    betas = params.get("betas", [params.get("beta1", 0.9), params.get("beta2", 0.999)])
    torch_adam = bool(params.get("torch_adam", False))
    return OptimizerSettings(
        name="torch_adamw" if torch_adam else "AdamW",
        lr=float(params["lr"]),
        weight_decay=float(params.get("weight_decay", 0.0)),
        betas=list(betas),
        eps=float(params.get("eps", 1e-8)),
        fused=False if torch_adam else None,
    )


def _adam_w_mode_from_deepspeed(config: Any) -> bool:
    if not isinstance(config, dict):
        return True
    params = config.get("params") or {}
    # DeepSpeed runs AdamW for type AdamW whatever adam_w_mode says.
    return str(config.get("type") or "AdamW").lower() == "adamw" or bool(params.get("adam_w_mode", True))


def _data_parallel_size(world_size: int, sequence_parallel: int) -> int:
    if sequence_parallel > 1 and world_size % sequence_parallel != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by sp_size ({sequence_parallel})")
    return world_size // sequence_parallel


def _optimizer_cpu_offload_enabled(ds_config: dict) -> bool:
    zero_config = ds_config.get("zero_optimization")
    if not isinstance(zero_config, dict):
        return False
    offload = zero_config.get("offload_optimizer")
    if isinstance(offload, dict):
        from deepspeed.runtime.zero.offload_config import DeepSpeedZeroOffloadOptimizerConfig
        from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum

        # Same default as DeepSpeed: missing device is none. Extra keys raise.
        return DeepSpeedZeroOffloadOptimizerConfig(**offload).device != OffloadDeviceEnum.none
    return bool(offload)


def _aligned_first(params: list) -> list:
    """``params`` with every parameter whose size is not a multiple of 8 elements moved to the end.

    ZeRO flattens a group's parameters back to back and makes each one a view at its offset in that buffer, so
    one parameter of, say, 3 elements leaves every parameter after it 2-byte aligned. cuBLAS then cannot use
    Hopper's GEMMs on those weights (TMA needs 16-byte-aligned operands) and falls back to its SM80 kernels:
    GLM-5.3-Flash's 3-element mHC scales, in its BF16 groups, sent 75% of the cuBLAS GEMM time there.
    The sort is stable, so the order within each half is the model's.
    """
    return sorted(params, key=lambda param: param.numel() % 8 != 0)


def _require_fp32_gradient_reduction(ds_config: dict[str, Any]) -> None:
    """Refuse a bfloat16 gradient reduction while the parameters are bfloat16.

    ``communication_data_type`` is the dtype gradient partials are summed in
    when they cross a rank. A bfloat16 job sets it to fp32 here, and a
    shallow client override can lower it afterwards. DeepSpeed's own default
    is the parameter dtype, so dropping the key is the same hazard as setting
    bfloat16. Anything other than fp32 is refused.
    """
    bf16_config = ds_config.get("bf16")
    if not (isinstance(bf16_config, dict) and bf16_config.get("enabled")):
        return
    reduction_dtype = str(ds_config.get("communication_data_type") or "").strip().lower()
    if reduction_dtype in ("fp32", "float32"):
        return
    raise ValueError(
        "communication_data_type must be fp32 while bf16 is enabled, because the gradient partials that cross "
        "a rank boundary are otherwise summed at bfloat16 granularity -- one step of about 2**-8 relative "
        "against 2**-24, which makes the result depend on how many ranks the token axis is split over; got "
        f"{ds_config.get('communication_data_type')!r}. Remove the training_config.ds_config override, or set "
        "dtype to float32 if the whole step is meant to run in float32."
    )
