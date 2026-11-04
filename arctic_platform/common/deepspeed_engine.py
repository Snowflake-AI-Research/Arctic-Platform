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
on-prem RL worker in ``arctic_platform.common.deepspeed_worker`` still builds
its own config in ``ds_training_config`` and calls ``deepspeed.initialize``
itself.

TODO: combine this module with ``deepspeed_worker`` once both callers share
one engine-construction path.
"""

from __future__ import annotations

import errno
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict

from arctic_platform.common.optimizer import OptimizerSettings
from arctic_platform.model.implementations.fp8 import keeps_fp32
from arctic_platform.model.implementations.fp8 import preserve_fp8_params_through_ds_cast

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
    gradient_clipping: float = 1.0
    seed: int | None = None
    ds_overrides: dict[str, Any] | None = None
    fp32_masters: bool = False
    zero_bucket_size: int | None = None
    optimizer: OptimizerSettings | None = None


def build_deepspeed_config(spec: TrainingEngineSpec) -> dict[str, Any]:
    """Build the config dict consumed by ``deepspeed.initialize``.

    The caller steps after each microbatch, so the microbatch size and the
    accumulation count are both 1. Sequence parallelism reduces the
    data-parallel degree, and therefore ``train_batch_size``, to
    ``world_size // sequence_parallel``.
    """
    dtype = (spec.dtype or "bfloat16").lower()
    is_bf16 = dtype in ("bfloat16", "bf16")
    is_fp16 = dtype in ("float16", "fp16")
    # bf16 master weights match FusedAdam's training dynamics. fp32 masters
    # are the opt-in that disables them.
    bf16_masters = is_bf16 and not spec.fp32_masters
    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": _data_parallel_size(spec.world_size, spec.sequence_parallel),
        "gradient_accumulation_steps": 1,
        "gradient_clipping": spec.gradient_clipping,
        "bf16": {
            "enabled": is_bf16,
            "bf16_master_weights_and_grads": bf16_masters,
            "bf16_optimizer_states": bf16_masters,
        },
        "fp16": {"enabled": is_fp16},
        "zero_optimization": {"stage": 2},
        # NCCL bf16 all-reduce is not accumulated in fp32 internally.
        "communication_data_type": "fp32" if is_bf16 else None,
    }
    if spec.seed is not None:
        ds_config["seed"] = int(spec.seed)
    if spec.sequence_parallel > 1:
        # A DeepSpeedConfig built from this dict without the SP mpu reads this key for the dp degree;
        # without it DeepSpeed checks ``train_batch_size`` against the full world size and asserts.
        ds_config["sequence_parallel_size"] = spec.sequence_parallel
    if spec.zero_bucket_size is not None:
        zero_config = ds_config["zero_optimization"]
        zero_config["reduce_bucket_size"] = spec.zero_bucket_size
        zero_config["allgather_bucket_size"] = spec.zero_bucket_size
    if spec.ds_overrides:
        ds_config.update(spec.ds_overrides)
    _require_fp32_gradient_reduction(ds_config)
    logger.info(
        "Resolved DeepSpeed precision: bf16=%s fp16=%s communication_data_type=%s",
        ds_config.get("bf16"),
        ds_config.get("fp16"),
        ds_config.get("communication_data_type"),
    )
    return ds_config


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
        self.sync_bytes = int(
            os.environ.get("ARCTIC_OFFLOADED_GRADIENT_SYNC_BYTES", str(_OFFLOADED_SNAPSHOT_CHUNK_BYTES))
        )
        if self.sync_bytes < 0:
            raise ValueError("ARCTIC_OFFLOADED_GRADIENT_SYNC_BYTES must be nonnegative")
        directory = Path(os.environ.get("ARCTIC_OFFLOADED_GRADIENT_TMPDIR", "/data-fast/tmp"))

        def allocate(path):
            path.mkdir(exist_ok=True)
            file = tempfile.TemporaryFile(prefix="ap-fp32-grad-", dir=path)
            try:
                # Reserve blocks now: ftruncate alone would defer ENOSPC until a backward write.
                os.posix_fallocate(file.fileno(), 0, self.size)
                os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            except BaseException:
                file.close()
                raise
            return file

        try:
            self.file = allocate(directory)
        except OSError as error:
            fallback = Path(tempfile.gettempdir())
            if fallback == directory or (
                not isinstance(error, (PermissionError, FileNotFoundError))
                and error.errno not in {errno.ENOSPC, errno.EOPNOTSUPP, errno.ENOTDIR, errno.EROFS}
            ):
                raise
            logger.info(
                "Offloaded gradient scratch %s is unavailable (%s); falling back to %s", directory, error, fallback
            )
            directory = fallback
            self.file = allocate(directory)
        logger.info("Offloaded gradient snapshot scratch directory: %s", directory)

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
        synced_offset = 0
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
                end = byte_offset + len(view)
                if self.sync_bytes > 0 and end - synced_offset >= self.sync_bytes:
                    # Bounded dirty pages by default; zero selects one sync per complete snapshot.
                    os.fdatasync(descriptor)
                    os.posix_fadvise(descriptor, synced_offset, end - synced_offset, os.POSIX_FADV_DONTNEED)
                    synced_offset = end
            else:
                flat.narrow(0, offset, length).copy_(chunk, non_blocking=False)
                os.posix_fadvise(descriptor, byte_offset, len(view), os.POSIX_FADV_DONTNEED)
        if writing and synced_offset < self.size:
            os.fdatasync(descriptor)
            os.posix_fadvise(descriptor, synced_offset, self.size - synced_offset, os.POSIX_FADV_DONTNEED)


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

    if type(optimizer) is not DeepSpeedZeroOptimizer:
        logger.info("Offloaded partition release does not support optimizer type %s", type(optimizer).__qualname__)
        return False
    if not optimizer.cpu_offload:
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
    finalize_gradient_accumulation = optimizer.finalize_gradient_accumulation_boundary
    finalize_remaining = None

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
        if finalize_remaining is not None and group_idx in finalize_remaining:
            finalize_remaining[group_idx].discard(optimizer.get_param_id(param))
            if len(finalize_remaining[group_idx]) == 0:
                optimizer._wait_for_offload_copies()
                optimizer.averaged_gradients[group_idx] = None
            return
        if optimizer._use_compact_grad_partition_buffer(group_idx) and all(
            optimizer.params_already_reduced[optimizer.get_param_id(partition_param)]
            for partition_param in optimizer.params_in_partition[group_idx]
            if partition_param.requires_grad and partition_param.numel() > 0
        ):
            optimizer._wait_for_offload_copies()
            release_partition(group_idx)

    def epilogue_then_release(*args, **kwargs):
        result = gradient_partition_epilogue(*args, **kwargs)
        if any(optimizer.averaged_gradients.get(group_idx) is not None for group_idx in compact_groups):
            optimizer._wait_for_offload_copies()
            for group_idx in compact_groups:
                release_partition(group_idx)
        return result

    def finalize_then_release(*args, **kwargs):
        nonlocal finalize_remaining
        finalize_remaining = {
            group_idx: {
                optimizer.get_param_id(param)
                for param in optimizer.params_in_partition[group_idx]
                if param.requires_grad and optimizer.get_param_id(param) in optimizer._offload_accumulated_param_ids
            }
            for group_idx in compact_groups
        }
        try:
            result = finalize_gradient_accumulation(*args, **kwargs)
            if any(optimizer.averaged_gradients.get(group_idx) is not None for group_idx in compact_groups):
                optimizer._wait_for_offload_copies()
                for group_idx in compact_groups:
                    optimizer.averaged_gradients[group_idx] = None
            return result
        finally:
            finalize_remaining = None

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
    optimizer.finalize_gradient_accumulation_boundary = finalize_then_release
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
):
    """Build the optimizer, keep FP8 storage, and call ``deepspeed.initialize``.

    Returns ``(engine, optimizer, optimizer_choice)``. ``optimizer`` and
    ``optimizer_choice`` are ``None`` when the job has no optimizer.
    ``mpu`` is forwarded only when it is set, so ZeRO reduces sequence-parallel
    gradients over the sequence-data-parallel group.
    """
    built = None
    choice = None
    # Marks first: the optimizer groups split on them.
    preserved, marked_fp32 = preserve_fp8_params_through_ds_cast(model)
    if optimizer is not None:
        fused = optimizer.fused is not False
        built = build_training_optimizer(model, optimizer, ds_config, fused=fused)
        choice = optimizer_choice(optimizer, ds_config, fused=fused)
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
    if built is not None:
        engine, built, _, _ = deepspeed.initialize(model=model, optimizer=built, config=ds_config, **init_kwargs)
    else:
        engine, *_ = deepspeed.initialize(model=model, config=ds_config, **init_kwargs)
        built = None
    if built is not None:
        release_offloaded_grad_partitions(built)
    return engine, built, choice


def resolve_optimizer_name(name: str, ds_config: dict) -> str:
    """Map a caller optimizer name onto the implementation DeepSpeed will run."""
    resolved = name.strip().lower().replace("-", "_")
    if resolved in {"", "adam", "adamw"}:
        return "deepspeed_cpu_adam" if _optimizer_cpu_offload_enabled(ds_config) else "fused_adam"
    if resolved not in _OPTIMIZER_ALIASES:
        supported = ", ".join(sorted(set(_OPTIMIZER_ALIASES.values())))
        raise ValueError(f"Unsupported optimizer {name!r}; supported values: {supported}")
    return _OPTIMIZER_ALIASES[resolved]


def optimizer_implementation(optimizer_name: str, *, fused: bool) -> str:
    """The class ``build_training_optimizer`` constructs for a resolved name."""
    if optimizer_name == "fused_adam" and fused:
        return "fused_adam"
    if optimizer_name == "deepspeed_cpu_adam":
        return "deepspeed_cpu_adam"
    if optimizer_name in ("torch_adamw", "fused_adam"):
        return "torch_adamw"
    raise AssertionError(f"Unhandled optimizer {optimizer_name!r}")


def optimizer_choice(optimizer: OptimizerSettings, ds_config: dict, *, fused: bool) -> dict[str, str]:
    """The implementation a job runs and the customer-facing reason."""
    optimizer_name = resolve_optimizer_name(optimizer.name, ds_config)
    implementation = optimizer_implementation(optimizer_name, fused=fused)
    if implementation == "deepspeed_cpu_adam":
        reason = (
            "optimizer state is offloaded to CPU, so the CPU implementation (CPUAdam) ran"
            if _optimizer_cpu_offload_enabled(ds_config)
            else f"you selected '{optimizer.name}'"
        )
    elif implementation == "fused_adam":
        reason = "AdamW ran as the faster fused GPU implementation (FusedAdam)"
    elif optimizer_name == "torch_adamw":
        reason = f"you selected '{optimizer.name}'"
    else:
        reason = "you set 'optimizer.fused' to false, so standard AdamW ran"
    return {"implementation": implementation, "reason": reason}


def cpu_adam_fp32_optimizer_states(ds_config: dict) -> bool:
    bf16_config = ds_config.get("bf16")
    if isinstance(bf16_config, dict) and bf16_config.get("bf16_optimizer_states"):
        return False
    return True


def build_training_optimizer(model, optimizer: OptimizerSettings, ds_config: dict, *, fused: bool):
    """Build AdamW with decay disabled on bias and norm parameters.

    GPU ``FusedAdam`` by default, or ``DeepSpeedCPUAdam`` when ZeRO optimizer
    offload is enabled.
    """
    optimizer_name = resolve_optimizer_name(optimizer.name, ds_config)
    if _optimizer_cpu_offload_enabled(ds_config) and optimizer_name == "deepspeed_cpu_adam":
        logger.info(
            "training_config.ds_config.zero_optimization.offload_optimizer is configured; "
            "using optimizer='deepspeed_cpu_adam'"
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
    implementation = optimizer_implementation(optimizer_name, fused=fused)
    if implementation == "fused_adam":
        from deepspeed.ops.adam import FusedAdam

        return FusedAdam(
            param_groups,
            lr=optimizer.lr,
            betas=betas,
            eps=optimizer.eps,
            adam_w_mode=True,
        )
    if implementation == "deepspeed_cpu_adam":
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        return DeepSpeedCPUAdam(
            param_groups,
            lr=optimizer.lr,
            betas=betas,
            eps=optimizer.eps,
            adamw_mode=True,
            fp32_optimizer_states=cpu_adam_fp32_optimizer_states(ds_config),
        )
    import torch

    return torch.optim.AdamW(
        param_groups,
        lr=optimizer.lr,
        betas=betas,
        eps=optimizer.eps,
    )


def _data_parallel_size(world_size: int, sequence_parallel: int) -> int:
    if sequence_parallel > 1 and world_size % sequence_parallel != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by sp_size ({sequence_parallel})")
    return world_size // sequence_parallel


def _optimizer_cpu_offload_enabled(ds_config: dict) -> bool:
    zero_config = ds_config.get("zero_optimization")
    if not isinstance(zero_config, dict):
        return False
    return bool(zero_config.get("offload_optimizer"))


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
