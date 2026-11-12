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
from pydantic import Field

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
    Unspecified BF16 gradient accumulation uses FP32 where DeepSpeed supports
    it, based on the effective precision after overrides. FP16 and FP32 retain
    DeepSpeed's native defaults; explicit accumulation dtypes are preserved.
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
    apply_default_grad_accum_dtype(ds_config, bf16_only=True)
    _require_fp32_gradient_reduction(ds_config)
    logger.info(
        "Resolved DeepSpeed precision: bf16=%s fp16=%s communication_data_type=%s",
        ds_config.get("bf16"),
        ds_config.get("fp16"),
        ds_config.get("communication_data_type"),
    )
    return ds_config


def apply_default_grad_accum_dtype(ds_config: dict[str, Any], *, bf16_only: bool = False) -> None:
    """Default unspecified accumulation to FP32 without mutating a caller-owned block.

    The builder opts into BF16-only defaulting; the worker retains its existing FP32 default for all supported
    precisions. Preserve other ``data_types`` fields and explicit dtypes, and leave invalid blocks to
    DeepSpeed's config validation.

    Mixed-precision ZeRO0 rejects FP32 accumulation. BF16 ZeRO1 without CPU offload selects DeepSpeed's BF16
    optimizer when gradient norms are disabled, and that optimizer refuses this configuration. Leave the native
    default in those cases instead of making a previously accepted job fail initialization.
    """
    data_types = ds_config.get("data_types")
    if data_types is not None and not isinstance(data_types, dict):
        return
    if (data_types or {}).get("grad_accum_dtype") is not None:
        return
    if not _deepspeed_accepts_fp32_grad_accumulation(ds_config, bf16_only=bf16_only):
        return
    ds_config["data_types"] = {**(data_types or {}), "grad_accum_dtype": "fp32"}


def _deepspeed_accepts_fp32_grad_accumulation(ds_config: dict[str, Any], *, bf16_only: bool) -> bool:
    """Whether the effective config supports FP32 accumulation within the caller's precision policy."""
    from deepspeed.runtime.precision_config import get_bfloat16_config
    from deepspeed.runtime.precision_config import get_float16_config
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.runtime.zero.config import get_zero_config
    from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum

    # Use the same typed parsers as DeepSpeedConfig, including "auto" and deprecated fields.
    zero = get_zero_config(ds_config)
    bf16 = get_bfloat16_config(ds_config).enabled
    fp16 = get_float16_config(ds_config).enabled
    if bf16_only and not bf16:
        return False
    if zero.stage == ZeroStageEnum.disabled:
        return not (bf16 or fp16)
    cpu_offload = zero.offload_optimizer is not None and zero.offload_optimizer.device == OffloadDeviceEnum.cpu
    if zero.stage == ZeroStageEnum.optimizer_states and bf16 and not cpu_offload and not zero.compute_grad_norm:
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


def _scratch_filesystem_type(directory: Path) -> str:
    resolved = directory.resolve()
    selected = (0, "unknown")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields, separator, filesystem = line.partition(" - ")
        if not separator:
            continue
        mountpoint = fields.split()[4]
        for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
            mountpoint = mountpoint.replace(escaped, literal)
        mount = Path(mountpoint)
        if (resolved == mount or mount in resolved.parents) and len(mount.parts) > selected[0]:
            selected = (len(mount.parts), filesystem.split()[0])
    return selected[1]


class _OffloadedGradientScratchUnavailable(ValueError):
    """Disk-backed snapshots are unavailable; native DeepSpeed retention remains valid."""


class _OffloadedGradientStorage:
    """Reuse bounded staging when FP32 offload snapshots outgrow a small CPU tensor."""

    def __init__(self):
        self.staging = None
        self.snapshots = {}
        try:
            self.sync_bytes = int(os.environ.get("ARCTIC_OFFLOADED_GRADIENT_SYNC_BYTES", "0"))
        except ValueError as error:
            raise ValueError("ARCTIC_OFFLOADED_GRADIENT_SYNC_BYTES must be a nonnegative integer") from error
        if self.sync_bytes < 0:
            raise ValueError("ARCTIC_OFFLOADED_GRADIENT_SYNC_BYTES must be nonnegative")
        allow_memory = os.environ.get("ARCTIC_OFFLOADED_GRADIENT_ALLOW_MEMORY_BACKED", "0")
        if allow_memory not in {"0", "1"}:
            raise ValueError("ARCTIC_OFFLOADED_GRADIENT_ALLOW_MEMORY_BACKED must be 0 or 1")
        self.allow_memory = allow_memory == "1"
        self.directory = Path(os.environ.get("ARCTIC_OFFLOADED_GRADIENT_TMPDIR", "/data-fast/tmp"))
        try:
            self.prepare_directory(self.directory)
        except OSError as error:
            self.use_fallback(error)
        logger.info("Offloaded gradient snapshot scratch directory: %s", self.directory)

    def prepare_directory(self, directory):
        directory.mkdir(exist_ok=True)
        filesystem = _scratch_filesystem_type(directory)
        if filesystem in {"tmpfs", "ramfs"} and not self.allow_memory:
            raise _OffloadedGradientScratchUnavailable(
                f"Offloaded gradient scratch {directory} is on {filesystem}; choose disk-backed "
                "ARCTIC_OFFLOADED_GRADIENT_TMPDIR or explicitly set ARCTIC_OFFLOADED_GRADIENT_ALLOW_MEMORY_BACKED=1"
            )
        if not os.access(directory, os.W_OK | os.X_OK):
            raise PermissionError(f"Offloaded gradient scratch is not writable: {directory}")

    def use_fallback(self, error):
        fallback = Path(tempfile.gettempdir())
        if fallback == self.directory or (
            not isinstance(error, (PermissionError, FileNotFoundError))
            and error.errno not in {errno.ENOSPC, errno.EOPNOTSUPP, errno.ENOTDIR, errno.EROFS}
        ):
            raise error
        self.prepare_directory(fallback)
        logger.warning(
            "Offloaded gradient scratch %s is unavailable (%s); falling back to %s (filesystem=%s)",
            self.directory,
            error,
            fallback,
            _scratch_filesystem_type(fallback),
        )
        self.directory = fallback

    def allocate(self, size):
        def reserve():
            file = tempfile.TemporaryFile(prefix="ap-fp32-grad-", dir=self.directory)
            try:
                os.posix_fallocate(file.fileno(), 0, size)
                os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            except BaseException:
                file.close()
                raise
            return file

        try:
            return reserve()
        except OSError as error:
            self.use_fallback(error)
            return reserve()

    def snapshot(self, source, *, group_idx=None):
        size = source.numel() * source.element_size()
        if size <= _OFFLOADED_SNAPSHOT_CHUNK_BYTES:
            return source.detach().to(device="cpu", non_blocking=False)
        snapshot = self.snapshots.get(group_idx) if group_idx is not None else None
        if snapshot is None:
            snapshot = _OffloadedGradientFile(source, self)
        try:
            snapshot.copy_(source)
        except BaseException:
            snapshot.file.close()
            self.snapshots.pop(group_idx, None)
            raise
        if group_idx is not None:
            self.snapshots[group_idx] = snapshot
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
        self.sync_bytes = storage.sync_bytes
        self.file = storage.allocate(self.size)

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
                    # Zero (default) syncs once per snapshot; a positive value bounds dirty pages.
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
    backwards and optimizer updates without recreating or reserving its file. A later
    backward restores the snapshot only when recreating its GPU partition, preserving the cumulative sum even when
    master gradients are BF16. Release completed groups during every reduction, including non-boundary microsteps.
    Partial groups are released after the native reduction epilogue, waiting for CPU copies before snapshotting.

    The wrappers are instance-local. Each group drops its snapshot before the native update allocates optimizer
    state: its CPU master gradient has already been unscaled and clipped, so the snapshot is no longer needed.
    Successful steps, including skipped overflow steps, clear all snapshots. Failed steps retain only groups not
    submitted for update; native updates are not transactional. Optimizers without CPU offload keep their methods.
    Native CPU accumulator ownership and cleanup are unchanged.
    """
    if not getattr(optimizer, "cpu_offload", False):
        return False
    enabled = os.environ.get("ARCTIC_OFFLOADED_GRADIENT_RELEASE", "1")
    if enabled not in {"0", "1"}:
        raise ValueError("ARCTIC_OFFLOADED_GRADIENT_RELEASE must be 0 or 1")
    if enabled == "0":
        return False
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    if type(optimizer) is not DeepSpeedZeroOptimizer:
        logger.warning("Offloaded partition release does not support optimizer type %s", type(optimizer).__qualname__)
        return False
    required = (
        "bit16_groups",
        "partition_size",
        "averaged_gradients",
        "params_in_partition",
        "params_already_reduced",
        "_offload_accumulated_param_ids",
        "_use_compact_grad_partition_buffer",
        "_get_compact_grad_partition_buffer",
        "_get_param_group_idx",
        "_wait_for_offload_copies",
        "get_param_id",
        "copy_grads_in_partition",
        "step",
        "reset_cpu_buffers",
        "_optimizer_step",
        "independent_gradient_partition_epilogue",
        "finalize_gradient_accumulation_boundary",
    )
    missing = [name for name in required if not hasattr(optimizer, name)]
    if missing:
        logger.warning("Offloaded partition release requires supported DeepSpeed capabilities; missing: %s", missing)
        return False
    compact_groups = [
        group_idx
        for group_idx in range(len(optimizer.bit16_groups))
        if optimizer._use_compact_grad_partition_buffer(group_idx)
    ]
    if not compact_groups:
        return False
    retained: dict[int, Any] = {}
    # Small partitions never touch scratch storage, including its filesystem requirements.
    try:
        storage = (
            _OffloadedGradientStorage()
            if max(optimizer.partition_size[index] * 4 for index in compact_groups) > _OFFLOADED_SNAPSHOT_CHUNK_BYTES
            else None
        )
    except (OSError, _OffloadedGradientScratchUnavailable) as error:
        logger.info("Offloaded partition release disabled: %s; retaining native DeepSpeed gradients", error)
        return False
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
            source = partition.flat_partition.detach()
            retained[group_idx] = (
                storage.snapshot(source, group_idx=group_idx)
                if storage is not None
                else source.to(device="cpu", non_blocking=False)
            )
        else:
            try:
                snapshot.copy_(partition.flat_partition.detach())
            except BaseException:
                # A partial overwrite is no longer a valid accumulated sum; keep the live GPU partition.
                retained.pop(group_idx, None)
                if isinstance(snapshot, _OffloadedGradientFile):
                    snapshot.file.close()
                    snapshot.storage.snapshots.pop(group_idx, None)
                raise
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
    adam_w_mode: bool = True,
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
    if built is not None:
        engine, built, _, _ = deepspeed.initialize(model=model, optimizer=built, config=ds_config, **init_kwargs)
    else:
        if ds_config.get("optimizer") is not None:
            init_kwargs["model_parameters"] = model.parameters()
        engine, built, _, _ = deepspeed.initialize(model=model, config=ds_config, **init_kwargs)
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
    else:
        reason = "you set 'optimizer.fused' to false, so standard AdamW ran"
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
