"""Shared DeepSpeed lifecycle for Prime-style MoE models."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import struct
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from math import prod
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping
from typing import cast
from urllib.parse import quote

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module

from arctic_platform.model.implementations.debug.token_combine import maybe_use_fixed_order_row_sum
from arctic_platform.model.implementations.fp8 import carry_keep_fp32
from arctic_platform.model.implementations.gpu.tiled_mlp import enable_tiled_mlp
from arctic_platform.model.loader import canonical_parameter_name

from .distributed.ep_backend import uses_dispatch_ep
from .distributed.expert_parallel import DeepEPExpertParallel
from .distributed.expert_parallel import DeepEPShardParallel
from .layers.moe import LatentMoE
from .layers.moe import MoE
from .logging_utils import get_logger
from .parallel_dims import ParallelDims
from .vlm import get_language_model

_EXPORT_GATHER_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class MoEDeepSpeedAdapter:
    dtype_map: Mapping[str, torch.dtype]
    get_model: Callable[..., nn.Module]
    configure_moe_ep_backend: Callable[[nn.Module, Any], None]
    configure_family_backend: Callable[[Any], None]
    inject_lm_head: Callable[..., None]
    apply_sequence_parallelism: Callable[[nn.Module, int, Any], None]
    apply_ac: Callable[[nn.Module, Any], None]
    load_dcp_from_hf: Callable[[nn.Module, Any, ParallelDims], None]
    reset_runtime_moe_buffers: Callable[[nn.Module], None]
    shared_expert_type: type[nn.Module]
    shared_expert_forward: Callable[[nn.Module, torch.Tensor], torch.Tensor]
    build_model_config: Callable[[str, int, int, str, str, Any], Any]
    vllm_weight_export: Callable[[nn.Module], Callable] | None = None


# Largest EP-group gather assembled on the master GPU in one piece during weight export.


def _ep_checkpoint_parameters(model: nn.Module) -> dict[str, dict[str, nn.Parameter]]:
    if not any(isinstance(module, (MoE, LatentMoE)) for module in model.modules()):
        return {}
    groups: dict[str, dict[str, nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if getattr(parameter, "allreduce", True) is False:
            group_name = parameter.group_name
            name = canonical_parameter_name(name)
            parameters = groups.setdefault(group_name, {})
            if name in parameters:
                raise ValueError(f"AP expert checkpoint has duplicate canonical parameter name: {name}")
            parameters[name] = parameter
    return groups


def _ep_checkpoint_identity(group_name: str) -> dict:
    import deepspeed.utils.groups as ds_groups

    return dict(
        group_name=group_name,
        ep_size=ds_groups._get_expert_parallel_world_size(group_name),
        ep_rank=ds_groups._get_expert_parallel_rank(group_name),
    )


def _ep_checkpoint_path(directory, tag, identity: dict) -> Path:
    group_name = quote(identity["group_name"], safe="")
    return Path(directory) / tag / f"ap_ep_{group_name}_rank_{identity['ep_rank']:02d}_model_states.pt"


def _ep_checkpoint_tensor_chunks(tensor: torch.Tensor, max_bytes: int):
    """Yield logical row-major bytes without materializing an entire strided tensor."""
    max_elements = max_bytes // tensor.element_size()
    source = tensor.detach()
    if source.is_contiguous():
        flat = source.view(-1).view(torch.uint8)
        for start in range(0, flat.numel(), max_bytes):
            yield flat.narrow(0, start, min(max_bytes, flat.numel() - start))
    elif source.numel() <= max_elements:
        yield source.contiguous().view(-1).view(torch.uint8)
    else:
        dimension = next(index for index, size in enumerate(source.shape) if size > 1)
        width = max(1, max_elements // prod(source.shape[dimension + 1 :]))
        for part in source.split(width, dim=dimension):
            yield from _ep_checkpoint_tensor_chunks(part, max_bytes)


def _checkpoint_sync(operation, *args) -> None:
    try:
        operation(*args)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP}:
            raise
        get_logger().info("Checkpoint filesystem does not support %s", operation.__name__)


def _validate_legacy_ep_zero_layout(engine, directory, tag, identity: dict) -> None:
    """Require this rank's native ZeRO archive and matching expert group/partition metadata."""
    path = Path(engine._get_zero_ckpt_name(directory, tag))
    alternate = path.with_name(
        path.name.removeprefix("bf16_") if path.name.startswith("bf16_") else "bf16_" + path.name
    )
    candidates = [path, alternate]
    candidates += [
        candidate.with_name(candidate.name.removesuffix("_optim_states.pt") + "optim_states.pt")
        for candidate in candidates
        if candidate.name.endswith("_optim_states.pt")
    ]
    found = next((candidate for candidate in candidates if candidate.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"Legacy EP checkpoint is missing this rank's ZeRO optimizer file: {path}")
    path = found
    # Native ZIP archives expose metadata without materializing masters and Adam moments a second time.
    if zipfile.is_zipfile(path):
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    else:
        saved = engine.checkpoint_engine.load(str(path), map_location=torch.device("cpu"))
    optimizer = saved.get("optimizer_state_dict", {}) if isinstance(saved, dict) else None
    base = optimizer.get("base_optimizer_state") if isinstance(optimizer, dict) else None
    if not isinstance(optimizer, dict) or not isinstance(base, dict):
        raise ValueError("Legacy EP resume requires non-elastic ZeRO optimizer metadata")
    groups = base.get("param_groups")
    counts = optimizer.get("partition_count")
    current_counts = engine.optimizer.partition_count
    if (
        not isinstance(groups, list)
        or not isinstance(counts, list)
        or any(not isinstance(group, dict) for group in groups)
        or counts != current_counts
        or len(groups) != len(counts)
    ):
        raise ValueError("Legacy EP checkpoint DP/EP partition layout does not match the current optimizer")
    matching = [group for group in groups if group.get("moe") and group.get("name") == identity["group_name"]]
    # AP registers native expert groups as ep_size_N; a different saved N must never recover local experts.
    if identity["group_name"] != f"ep_size_{identity['ep_size']}" or len(matching) != 1:
        raise ValueError("Legacy EP checkpoint expert-parallel size/group does not match the current model")


def _validate_ep_checkpoint_storage_offsets(path: Path, tensors: dict, offsets: dict) -> None:
    """Check private Torch offsets against the ZIP records before writing any payload."""
    with zipfile.ZipFile(path) as archive, path.open("rb") as stream:
        records = {
            info.filename.rsplit("/", 1)[-1]: info
            for info in archive.infolist()
            if info.filename.rsplit("/", 2)[-2] == "data"
        }
        if set(records) != {str(index) for index in range(len(tensors))}:
            raise ValueError("AP expert checkpoint storage record inventory is invalid")
        for index, (name, tensor) in enumerate(tensors.items()):
            record = records[str(index)]
            stream.seek(record.header_offset)
            header = stream.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise ValueError("AP expert checkpoint ZIP local header is invalid")
            filename_bytes, extra_bytes = struct.unpack_from("<HH", header, 26)
            actual_offset = record.header_offset + 30 + filename_bytes + extra_bytes
            if (
                record.compress_type != zipfile.ZIP_STORED
                or record.file_size != tensor.numel() * tensor.element_size()
                or offsets[name] != actual_offset
                or actual_offset + record.file_size > archive.start_dir
            ):
                raise ValueError(f"AP expert checkpoint storage offset does not match ZIP record for {name}")


@contextmanager
def _ep_checkpoint_workspace(path: Path):
    prefix = f".ap-ep-checkpoint-{path.stem}-"
    # Sweep this rank's leftovers in every tag, without touching other ranks' active saves.
    for tag in path.parent.parent.iterdir():
        if not tag.is_dir():
            continue
        for workspace in tag.glob(f"{prefix}*"):
            try:
                owner = workspace / ".owner"
                stamp = owner.stat().st_mtime if owner.exists() else workspace.stat().st_mtime
                if workspace.is_dir() and time.time() - stamp > 60:
                    shutil.rmtree(workspace)
            except FileNotFoundError:
                pass
    with tempfile.TemporaryDirectory(prefix=prefix, dir=path.parent) as directory:
        owner = Path(directory) / ".owner"
        owner.touch()
        stopped = threading.Event()

        def heartbeat():
            while not stopped.wait(1):
                try:
                    owner.touch()
                except OSError:
                    get_logger().info("Could not refresh expert checkpoint workspace heartbeat", exc_info=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield Path(directory)
        finally:
            stopped.set()
            thread.join(timeout=2)


def _save_ep_checkpoint_shard(state: dict, path: Path) -> None:
    """Write local expert tensors without copying their ZeRO backing buffers to the host.

    AP-owned sidefiles use a regular torch ZIP archive, independently of the dense checkpoint's serialization
    setting. Reserve compact storage records first, then fill them with a single 16 MiB CPU staging buffer.
    ARCTIC_EXPERT_CHECKPOINT_SYNC_BYTES sets the sync/eviction interval (16 MiB by default, 0 for end-only).
    Full cold-read checksum verification is enabled by default; ARCTIC_EXPERT_CHECKPOINT_VERIFY=0 keeps only
    ZIP storage boundaries, metadata and first/last-byte checks on every tensor.
    Only the AP shard is synced here; DeepSpeed owns durability of dense/optimizer files and the latest pointer.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    sync_env = "ARCTIC_EXPERT_CHECKPOINT_SYNC_BYTES"
    try:
        sync_bytes = int(os.environ.get(sync_env, str(16 * 1024 * 1024)))
    except ValueError as error:
        raise ValueError(f"{sync_env} must be a non-negative integer") from error
    if sync_bytes < 0:
        raise ValueError(f"{sync_env} must be a non-negative integer")
    verify_env = "ARCTIC_EXPERT_CHECKPOINT_VERIFY"
    verify = os.environ.get(verify_env, "1")
    if verify not in {"0", "1"}:
        raise ValueError(f"{verify_env} must be 0 or 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = state["module"]
    with _ep_checkpoint_workspace(path) as directory:
        temporary = directory / path.name
        # Real empty tensors can fill the entire expert layout under deterministic initialization.
        with FakeTensorMode():
            layout = {
                name: torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu") for name, tensor in tensors.items()
            }
        with torch.serialization.skip_data(materialize_fake_tensors=True):
            torch.save({**state, "module": layout}, temporary)
        del layout
        with FakeTensorMode():
            records = torch.load(temporary, weights_only=True)["module"]
        raw_offsets = {
            name: getattr(tensor.untyped_storage(), "_checkpoint_offset", None) for name, tensor in records.items()
        }
        del records
        if any(type(offset) is not int for offset in raw_offsets.values()):
            raise RuntimeError("AP expert checkpoint requires torch storage-offset serialization support (torch>=2.7)")
        offsets = cast(dict[str, int], raw_offsets)
        _validate_ep_checkpoint_storage_offsets(temporary, tensors, offsets)
        records_end = temporary.stat().st_size
        ranges = sorted(
            (offsets[name], offsets[name] + tensor.numel() * tensor.element_size())
            for name, tensor in tensors.items()
            if tensor.numel() > 0
        )
        if any(start < 0 or end > records_end for start, end in ranges) or any(
            left[1] > right[0] for left, right in zip(ranges, ranges[1:])
        ):
            raise ValueError("AP expert checkpoint storage offsets are invalid or overlapping")
        staging = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cpu")
        descriptor = os.open(temporary, os.O_WRONLY)
        checksums = {}
        samples = {}
        try:
            pending_bytes = 0
            for name, tensor in tensors.items():
                start = 0
                checksum = hashlib.sha256()
                first, last = b"", b""
                for source in _ep_checkpoint_tensor_chunks(tensor, staging.numel()):
                    count = source.numel()
                    chunk = staging.narrow(0, 0, count)
                    chunk.copy_(source, non_blocking=False)
                    data = memoryview(chunk.numpy())
                    if not first:
                        first = bytes(data[:8])
                    last = (last + bytes(data[-8:]))[-8:]
                    if verify == "1":
                        checksum.update(data)
                    written = 0
                    while written < count:
                        size = os.pwritev(descriptor, [data[written:]], offsets[name] + start + written)
                        if size <= 0:
                            raise OSError("AP expert checkpoint write made no progress")
                        written += size
                    pending_bytes += count
                    if sync_bytes > 0 and pending_bytes >= sync_bytes:
                        _checkpoint_sync(os.fdatasync, descriptor)
                        _checkpoint_sync(os.posix_fadvise, descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                        pending_bytes = 0
                    start += count
                checksums[name] = checksum.digest()
                samples[name] = (first, last)
            # Also sync archive metadata when the shard contains only empty tensors.
            _checkpoint_sync(os.fdatasync, descriptor)
            _checkpoint_sync(os.posix_fadvise, descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
        # Validate actual ZIP tensor records, independently of the private offsets used for writing.
        written_state = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)["module"]
        if set(written_state) != set(tensors):
            raise ValueError("AP expert checkpoint tensor inventory verification failed")
        for name, tensor in written_state.items():
            if tensor.shape != tensors[name].shape or tensor.dtype != tensors[name].dtype:
                raise ValueError(f"AP expert checkpoint tensor metadata verification failed for {name}")
            payload_bytes = tensor.view(-1).view(torch.uint8)
            if (payload_bytes[:8].numpy().tobytes(), payload_bytes[-8:].numpy().tobytes()) != samples[name]:
                raise ValueError(f"AP expert checkpoint payload verification failed for {name}")
            if verify == "1":
                checksum = hashlib.sha256()
                for chunk in _ep_checkpoint_tensor_chunks(tensor, staging.numel()):
                    checksum.update(memoryview(chunk.numpy()))
                if checksum.digest() != checksums[name]:
                    raise ValueError(f"AP expert checkpoint payload verification failed for {name}")
        del written_state
        os.replace(temporary, path)
        # DeepSpeed creates the checkpoint tree; only our shard's rename is owned here.
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            _checkpoint_sync(os.fsync, directory_descriptor)
        finally:
            os.close(directory_descriptor)


def _probe_ep_checkpoint_serialization() -> None:
    """Fail during model setup if Torch cannot reserve and locate compact expert storage."""
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with tempfile.TemporaryDirectory(prefix="ap-ep-checkpoint-probe-") as directory:
            path = Path(directory) / "probe.pt"
            with FakeTensorMode():
                tensor = torch.empty(3, dtype=torch.uint8)
            with torch.serialization.skip_data(materialize_fake_tensors=True):
                torch.save(dict(module=dict(probe=tensor)), path)
            with FakeTensorMode():
                tensor = torch.load(path, weights_only=True)["module"]["probe"]
            offset = getattr(tensor.untyped_storage(), "_checkpoint_offset", None)
            if type(offset) is not int:
                raise RuntimeError("Torch did not expose its checkpoint storage offset")
            _validate_ep_checkpoint_storage_offsets(path, dict(probe=tensor), dict(probe=offset))
    except (ImportError, AttributeError, TypeError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "AP expert-parallel checkpoints require Torch compact-storage serialization support "
            "(torch>=2.7); the startup capability probe failed"
        ) from error


def _patch_deepspeed_ep_checkpoints(engine_type) -> None:
    if getattr(engine_type, "_ap_ep_ckpt_patched", False):
        return
    _probe_ep_checkpoint_serialization()
    import deepspeed.utils.groups as ds_groups
    from deepspeed.checkpoint.constants import FROZEN_PARAM_FRAGMENTS

    original_load = engine_type._load_checkpoint
    original_public_load = engine_type.load_checkpoint
    original_module_load = engine_type.load_module_state_dict
    original_module_state_dict = engine_type.module_state_dict
    original_non_moe_state_dict = engine_type._get_non_moe_state_dict
    original_frozen_attributes = engine_type._get_zero_frozen_param_attributes
    original_zero_load = getattr(engine_type, "_load_zero_checkpoint", None)
    load_signature = signature(original_load)
    module_load_signature = signature(original_module_load)
    module_state_signature = signature(original_module_state_dict)

    @wraps(original_frozen_attributes)
    def frozen_attributes(self, attr_func):
        parameters = getattr(self, "_ap_ep_checkpoint_save_parameters", None)
        if parameters is None:
            return original_frozen_attributes(self, attr_func)
        excluded = {parameter for named_parameters in parameters.values() for parameter in named_parameters.values()}

        def dense_attribute(parameter):
            # Filter before the native fragment callback copies frozen experts to CPU.
            return None if parameter in excluded else attr_func(parameter)

        attributes = original_frozen_attributes(self, dense_attribute)
        excluded_names = {self.param_names[parameter] for parameter in excluded}
        return {name: value for name, value in attributes.items() if name not in excluded_names}

    @wraps(original_module_state_dict)
    def module_state_dict(self, *args, **kwargs):
        state = original_module_state_dict(self, *args, **kwargs)
        parameters = getattr(self, "_ap_ep_checkpoint_save_parameters", None)
        if parameters is not None and state is not None:
            bound = module_state_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            prefix = bound.arguments["prefix"]
            for named_parameters in parameters.values():
                for name in named_parameters:
                    state.pop(prefix + canonical_parameter_name(name), None)
        return state

    @wraps(original_non_moe_state_dict)
    def non_moe_state_dict(self, full_state_dict):
        state = original_non_moe_state_dict(self, full_state_dict)
        # Exclude AP's rank-local experts before DeepSpeed clones the dense checkpoint to CPU.
        for named_parameters in _ep_checkpoint_parameters(self.module).values():
            for name in named_parameters:
                state.pop(canonical_parameter_name(name), None)
        return state

    def wrap_save(original_save):
        save_signature = signature(original_save)

        @wraps(original_save)
        def save(self, *args, **kwargs):
            bound = save_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            parameters = _ep_checkpoint_parameters(self.module)
            if len(parameters) > 0 and self.zero_optimization_partition_weights():
                raise ValueError("AP expert-parallel checkpoint shards do not support ZeRO stage 3")
            # Ordinary saves bypass _get_non_moe_state_dict; filter before their CPU clone too.
            previous_parameters = getattr(self, "_ap_ep_checkpoint_save_parameters", None)
            self._ap_ep_checkpoint_save_parameters = parameters
            try:
                result = original_save(self, *args, **kwargs)
            finally:
                if previous_parameters is None:
                    del self._ap_ep_checkpoint_save_parameters
                else:
                    self._ap_ep_checkpoint_save_parameters = previous_parameters
            for group_name, named_parameters in parameters.items():
                data_rank = ds_groups._get_expert_data_parallel_rank(group_name)
                if not self.checkpoint_engine.is_data_parallel_writer(data_rank):
                    continue
                identity = _ep_checkpoint_identity(group_name)
                state = {
                    name: parameter.detach()
                    for name, parameter in named_parameters.items()
                    if parameter.requires_grad or not arguments["exclude_frozen_parameters"]
                }
                _save_ep_checkpoint_shard(
                    dict(
                        version=1,
                        **identity,
                        exclude_frozen_parameters=arguments["exclude_frozen_parameters"],
                        module=state,
                    ),
                    _ep_checkpoint_path(arguments["save_dir"], arguments["tag"], identity),
                )
            return result

        return save

    @wraps(original_load)
    def load(self, *args, **kwargs):
        bound = load_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        parameters = _ep_checkpoint_parameters(self.module)
        if len(parameters) == 0:
            return original_load(self, *args, **kwargs)
        if self.zero_optimization_partition_weights():
            raise ValueError("AP expert-parallel checkpoint shards do not support ZeRO stage 3")
        local_state = {}
        legacy_parameters = {}
        restore_legacy_masters = False
        for group_name, named_parameters in parameters.items():
            identity = _ep_checkpoint_identity(group_name)
            path = _ep_checkpoint_path(arguments["load_dir"], arguments["tag"], identity)
            if not path.is_file():
                if identity["ep_size"] == 1:
                    legacy_parameters.update(named_parameters)
                    continue
                # Dense experts are rank-zero values in legacy EP saves. Only saved ZeRO masters can replace them.
                from deepspeed.runtime.utils import is_optimized_parameter

                if (
                    original_zero_load is not None
                    and arguments["load_optimizer_states"]
                    and not arguments["load_module_only"]
                    and callable(getattr(self.optimizer, "update_lp_params", None))
                    and self.zero_load_from_fp32_weights()
                    and all(is_optimized_parameter(parameter) for parameter in named_parameters.values())
                    and callable(getattr(self, "_get_zero_ckpt_name", None))
                ):
                    _validate_legacy_ep_zero_layout(self, arguments["load_dir"], arguments["tag"], identity)
                    legacy_parameters.update(named_parameters)
                    restore_legacy_masters = True
                    get_logger().warning(
                        "Loading legacy EP checkpoint without AP shards; rank-local experts will be restored "
                        "from ZeRO master weights. Model-only and frozen-expert legacy EP loads are unsupported."
                    )
                    continue
                raise FileNotFoundError(f"AP expert-parallel checkpoint is missing its rank-local shard: {path}")
            # AP sidefiles are always Torch ZIP archives, even with a different dense checkpoint engine.
            shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if (
                not isinstance(shard, dict)
                or type(shard.get("version")) is not int
                or shard["version"] != 1
                or any(shard.get(key) != value for key, value in identity.items())
                or type(shard.get("exclude_frozen_parameters")) is not bool
                or not isinstance(shard.get("module"), dict)
            ):
                raise ValueError(f"AP expert-parallel checkpoint shard metadata does not match: {path}")
            canonical_state = {}
            for name, value in shard["module"].items():
                canonical_name = canonical_parameter_name(name)
                if canonical_name in canonical_state:
                    raise ValueError(f"AP expert-parallel checkpoint has duplicate canonical names: {path}")
                canonical_state[canonical_name] = value
            expected_names = {
                name
                for name, parameter in named_parameters.items()
                if parameter.requires_grad or not shard["exclude_frozen_parameters"]
            }
            if set(canonical_state) != expected_names:
                raise ValueError(f"AP expert-parallel checkpoint parameter coverage does not match: {path}")
            for name, value in canonical_state.items():
                if not isinstance(value, torch.Tensor) or value.shape != named_parameters[name].shape:
                    raise ValueError(f"AP expert-parallel checkpoint tensor shape does not match for {name}: {path}")
                if value.dtype != named_parameters[name].dtype:
                    raise ValueError(f"AP expert-parallel checkpoint tensor dtype does not match for {name}: {path}")
            local_state.update(canonical_state)
        self._ap_ep_checkpoint_state = local_state
        self._ap_ep_legacy_parameters = legacy_parameters
        self._ap_ep_restore_legacy_masters = restore_legacy_masters
        try:
            return original_load(*bound.args, **bound.kwargs)
        except BaseException:
            self.__dict__.pop("_ap_ep_restore_legacy_masters", None)
            raise
        finally:
            del self._ap_ep_checkpoint_state
            del self._ap_ep_legacy_parameters

    @wraps(original_public_load)
    def load_checkpoint(self, *args, **kwargs):
        try:
            # DeepSpeed loads ZeRO masters after the private module-load call has returned.
            return original_public_load(self, *args, **kwargs)
        finally:
            self.__dict__.pop("_ap_ep_restore_legacy_masters", None)

    if original_zero_load is not None:

        @wraps(original_zero_load)
        def load_zero(self, *args, **kwargs):
            restoring = getattr(self, "_ap_ep_restore_legacy_masters", False)
            try:
                result = original_zero_load(self, *args, **kwargs)
                if restoring:
                    if not result:
                        raise RuntimeError("Legacy EP resume requires successfully loaded ZeRO master weights")
                    # Native ordinary resume leaves low-precision model values from the dense checkpoint intact.
                    self.optimizer.update_lp_params()
                return result
            finally:
                self.__dict__.pop("_ap_ep_restore_legacy_masters", None)

        engine_type._load_zero_checkpoint = load_zero

    @wraps(original_module_load)
    def load_module(self, *args, **kwargs):
        if not hasattr(self, "_ap_ep_checkpoint_state"):
            return original_module_load(self, *args, **kwargs)
        bound = module_load_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        checkpoint = dict(bound.arguments["checkpoint"])
        checkpoint["module"] = dict(checkpoint["module"])
        for name, parameter in self._ap_ep_legacy_parameters.items():
            value = checkpoint["module"].get(name)
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise FileNotFoundError(
                    f"AP expert-parallel checkpoint has neither a rank-local shard nor compatible dense expert {name}"
                )
        checkpoint["module"].update(
            (canonical_parameter_name(name), value) for name, value in self._ap_ep_checkpoint_state.items()
        )
        # DeepSpeed restores frozen fragments after the module, so both must use the local shard.
        if checkpoint.get(FROZEN_PARAM_FRAGMENTS) is not None:
            from deepspeed.runtime.utils import is_optimized_parameter

            live_parameters = {
                canonical_parameter_name(name): parameter for name, parameter in self.module.named_parameters()
            }
            fragments = dict(checkpoint[FROZEN_PARAM_FRAGMENTS])
            fragments.update(
                (self.param_names[live_parameters[name]], value)
                for name, value in self._ap_ep_checkpoint_state.items()
                if not is_optimized_parameter(live_parameters[name])
            )
            checkpoint[FROZEN_PARAM_FRAGMENTS] = fragments
        bound.arguments["checkpoint"] = checkpoint
        return original_module_load(*bound.args, **bound.kwargs)

    engine_type._get_non_moe_state_dict = non_moe_state_dict
    engine_type._get_zero_frozen_param_attributes = frozen_attributes
    engine_type.module_state_dict = module_state_dict
    engine_type._save_moe_checkpoint = wrap_save(engine_type._save_moe_checkpoint)
    # Frozen expert-only models follow DeepSpeed's ordinary checkpoint path.
    engine_type._save_checkpoint = wrap_save(engine_type._save_checkpoint)
    engine_type._load_checkpoint = load
    engine_type.load_checkpoint = load_checkpoint
    engine_type.load_module_state_dict = load_module
    engine_type._ap_ep_ckpt_patched = True


def patch_deepspeed_moe_detection() -> None:
    import deepspeed.runtime.engine as ds_engine

    if getattr(ds_engine.DeepSpeedEngine, "_ap_moe_patched", False):
        return
    original = ds_engine.DeepSpeedEngine._configure_distributed_model

    def patched(self, model):
        if self.zero_optimization_partition_weights() and _ep_checkpoint_parameters(model):
            raise ValueError("AP expert-parallel models do not support ZeRO stage 3; use ZeRO stage 1 or 2")
        original(self, model)
        if self.has_moe_layers:
            return
        for module in self.module.modules():
            if not isinstance(module, (MoE, LatentMoE)):
                continue
            experts = getattr(module, "experts", None)
            if experts is not None and not any(parameter.requires_grad for parameter in experts.parameters()):
                continue
            self.has_moe_layers = True
            num_experts = getattr(module, "num_experts", None)
            if num_experts is None and experts is not None:
                num_experts = getattr(experts, "num_experts", None)
            if num_experts is not None:
                self.num_experts.append(num_experts)

    _patch_deepspeed_ep_checkpoints(ds_engine.DeepSpeedEngine)
    ds_engine.DeepSpeedEngine._configure_distributed_model = patched
    ds_engine.DeepSpeedEngine._ap_moe_patched = True


def apply_ep_with_mesh(model: nn.Module, config: Any, ep_mesh: DeviceMesh) -> None:
    if not uses_dispatch_ep(config.ep_comm_backend):
        raise NotImplementedError(
            f"EP comm backend must be one of ('deepep', 'uccl'), got {config.ep_comm_backend!r}."
        )
    for transformer_block in get_language_model(model).layers:
        block_mlp = getattr(transformer_block, "mlp", None)
        if isinstance(block_mlp, (MoE, LatentMoE)):
            parallelize_module(
                block_mlp.experts,
                device_mesh=ep_mesh,
                parallelize_plan=DeepEPExpertParallel(),
            )
    for module in model.modules():
        if getattr(module, "_dss_shard_on_ep", False):
            parallelize_module(
                module,
                device_mesh=ep_mesh,
                parallelize_plan=DeepEPShardParallel(),
            )


def convert_dtensors_to_local(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            if not isinstance(parameter.data, DTensor) and not hasattr(parameter, "to_local"):
                continue
            local = parameter.to_local().detach()
            replacement = carry_keep_fp32(
                parameter,
                nn.Parameter(local, requires_grad=parameter.requires_grad),
            )
            if getattr(parameter, "_dss_skip_weight_sync", False):
                replacement._dss_skip_weight_sync = True
            module.register_parameter(name, replacement)
            count += 1
    return count


def tag_expert_params_for_deepspeed(model: nn.Module, ep_group_name: str) -> int:
    count = 0
    for module in model.modules():
        if not isinstance(module, (MoE, LatentMoE)):
            continue
        for parameter in module.experts.parameters(recurse=False):
            parameter.allreduce = False
            parameter.group_name = ep_group_name
            count += 1
    for module in model.modules():
        if not getattr(module, "_dss_ep_sharded", False):
            continue
        for parameter in module.parameters(recurse=False):
            if parameter.numel() == 0:
                continue
            parameter.allreduce = False
            parameter.group_name = ep_group_name
            count += 1
    return count


def tag_expert_lora_adapters_for_deepspeed(model: nn.Module, ep_group_name: str) -> int:
    count = 0
    for module in model.modules():
        if not isinstance(module, (MoE, LatentMoE)):
            continue
        experts = getattr(module, "experts", None)
        if experts is None:
            continue
        for parameter in experts.parameters(recurse=True):
            if parameter.requires_grad and getattr(parameter, "group_name", None) is None:
                parameter.allreduce = False
                parameter.group_name = ep_group_name
                count += 1
    return count


def build_iter_full_hf_weights(model: nn.Module):
    model_type = type(model)
    if getattr(model_type, "convert_layer_to_hf", None) is None:
        raise RuntimeError(f"build_iter_full_hf_weights requires {model_type.__name__}.convert_layer_to_hf")
    # Bound from the instance so a family whose layer mapping depends on its config can read it.
    convert_layer_to_hf = model.convert_layer_to_hf

    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist

    from arctic_platform.model.implementations.moe.weights import hf_export_param_name
    from arctic_platform.model.weight_export import wait_for_export_collective

    def layer_index(name: str) -> int:
        parts = name.split(".")
        if len(parts) >= 4 and parts[:3] == ["model", "language_model", "layers"]:
            index = parts[3]
        elif len(parts) >= 3 and parts[:2] == ["model", "layers"]:
            index = parts[2]
        else:
            return -1
        return int(index) if index.isdigit() else -1

    def _gather_to_master(local: torch.Tensor, world: int, ep_group, is_master: bool):
        """Gather checkpoint-only shards into host memory without an all-gather copy on every rank."""
        if local.dtype != torch.uint8:
            full = _gather_to_master(local.view(torch.uint8), world, ep_group, is_master)
            return None if full is None else full.view(local.dtype)
        rows = local.shape[0]
        row_bytes = (local.numel() // max(rows, 1)) * local.element_size()
        step = max(1, _EXPORT_GATHER_BYTES // max(row_bytes * world, 1))
        step = min(step, max(rows, 1))
        full = torch.empty((world * rows, *local.shape[1:]), dtype=local.dtype, device="cpu") if is_master else None
        buffer = (
            torch.empty((world, step, *local.shape[1:]), dtype=local.dtype, device=local.device)
            if is_master and local.device.type != "cpu"
            else None
        )
        for start in range(0, rows, step):
            count = min(step, rows - start)
            if buffer is not None:
                outputs = [buffer[rank, :count] for rank in range(world)]
            elif full is not None:
                outputs = [full[rank * rows + start : rank * rows + start + count] for rank in range(world)]
            else:
                outputs = None
            chunk = local[start : start + count].contiguous()
            dist.gather(chunk, outputs, dst=0, group=ep_group)
            if not is_master:
                wait_for_export_collective(chunk)
            if full is not None and buffer is not None:
                for rank in range(world):
                    full[rank * rows + start : rank * rows + start + count].copy_(buffer[rank, :count])
        return full

    def iterator(*, include_skipped: bool = False):
        is_master = dist.get_rank() == 0
        from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors

        skipped_parameter_ids = {
            id(parameter)
            for module in model.modules()
            if getattr(module, "_dss_skip_weight_sync", False)
            for parameter in module.parameters(recurse=False)
        }
        by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
        for name, tensor in named_weight_sync_tensors(model, include_export_buffers=include_skipped):
            if not include_skipped and (
                getattr(tensor, "_dss_skip_weight_sync", False) or id(tensor) in skipped_parameter_ids
            ):
                continue
            hf_name = hf_export_param_name(name)
            if hf_name is not None:
                by_layer.setdefault(layer_index(hf_name), []).append((hf_name, tensor))

        ordered_layers = ([-1] if -1 in by_layer else []) + sorted(index for index in by_layer if index >= 0)
        for index in ordered_layers:
            layer_state: dict[str, torch.Tensor] = {}
            for name, tensor in by_layer[index]:
                if hasattr(tensor, "group_name") and getattr(tensor, "allreduce", True) is False:
                    ep_group = ds_groups._get_expert_parallel_group(tensor.group_name)
                    # Only the master's EP replica participates; gather its shards into bounded CPU output.
                    if 0 not in dist.get_process_group_ranks(ep_group):
                        continue
                    if getattr(tensor, "_dss_skip_weight_sync", False) or id(tensor) in skipped_parameter_ids:
                        local = tensor.data.contiguous()
                        full = _gather_to_master(local, dist.get_world_size(group=ep_group), ep_group, is_master)
                        if is_master:
                            layer_state[name] = full
                        del full
                        continue
                    full = _gather_ep_weight_to_cpu(
                        tensor, ep_group, is_master=is_master, max_gather_bytes=_EXPORT_GATHER_BYTES
                    )
                    if is_master:
                        layer_state[name] = full
                elif is_master:
                    layer_state[name] = tensor.data

            if not is_master:
                continue
            layer_state = {
                name.replace("._checkpoint_wrapped_module", ""): tensor for name, tensor in layer_state.items()
            }
            convert_layer_to_hf(layer_state, index)
            yield from layer_state.items()

    return iterator


def setup_model_local_no_train(
    adapter: MoEDeepSpeedAdapter,
    config: Any,
    parallel_dims: ParallelDims,
    ep_mesh: DeviceMesh,
    *,
    fused_cross_entropy: bool | str = False,
    tiled_mlp_token_chunk_size: int | None = None,
    sp_size: int = 1,
    sp_group=None,
) -> nn.Module:
    logger = get_logger()
    model = adapter.get_model(
        config,
        device=torch.device("meta"),
        dtype=adapter.dtype_map[config.optimization_dtype],
    )
    adapter.configure_moe_ep_backend(model, config)
    adapter.configure_family_backend(config)

    chunk_size = (
        config.fused_lm_head_token_chunk_size if isinstance(config.fused_lm_head_token_chunk_size, int) else None
    )
    adapter.inject_lm_head(
        model,
        chunk_size=chunk_size,
        fused_cross_entropy=fused_cross_entropy,
        fp32_lm_head=config.fp32_lm_head,
    )
    if parallel_dims.ep_enabled:
        apply_ep_with_mesh(model, config, ep_mesh)
    adapter.apply_sequence_parallelism(model, sp_size, sp_group)
    if config.ac is not None:
        adapter.apply_ac(model, config.ac)

    adapter.load_dcp_from_hf(model, config, parallel_dims)
    adapter.reset_runtime_moe_buffers(model)
    count = convert_dtensors_to_local(model)
    logger.info(f"Unwrapped {count} DTensor parameters into local tensors")
    enable_tiled_mlp(
        model,
        is_target=lambda module: isinstance(module, adapter.shared_expert_type),
        mlp_forward=adapter.shared_expert_forward,
        token_chunk_size=tiled_mlp_token_chunk_size,
        logger=logger,
    )
    return model


def load_moe_model_for_deepspeed(
    adapter: MoEDeepSpeedAdapter,
    model_config: Any,
    parallel_dims: ParallelDims,
    ep_mesh: DeviceMesh,
    ep_group_name: str,
    *,
    fused_cross_entropy: bool | str = False,
    tiled_mlp_token_chunk_size: int | None = None,
    sp_size: int = 1,
    sp_group=None,
) -> nn.Module:
    model = setup_model_local_no_train(
        adapter,
        model_config,
        parallel_dims,
        ep_mesh,
        fused_cross_entropy=fused_cross_entropy,
        tiled_mlp_token_chunk_size=tiled_mlp_token_chunk_size,
        sp_size=sp_size,
        sp_group=sp_group,
    )
    count = tag_expert_params_for_deepspeed(model, ep_group_name)
    get_logger().info(f"Tagged {count} expert parameters with group_name='{ep_group_name}', allreduce=False")
    from arctic_platform.model.weight_export import WeightExportContract
    from arctic_platform.model.weight_export import register_weight_export

    register_weight_export(
        model,
        WeightExportContract(
            hf=build_iter_full_hf_weights,
            vllm=adapter.vllm_weight_export,
        ),
    )
    return model


def load_moe_model(
    adapter: MoEDeepSpeedAdapter,
    family_loader: Callable,
    *,
    model_name: str,
    optimization_dtype: str,
    attn_implementation: str,
    ep_size: int,
    sp_size: int = 1,
    sp_group=None,
    ep_group=None,
    options: Any,
    patch_moe_detection: Callable[[], None] = patch_deepspeed_moe_detection,
    device_mesh_type=DeviceMesh,
) -> nn.Module:
    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist

    world_size = dist.get_world_size()
    if world_size % ep_size:
        raise ValueError(f"world_size={world_size} must be divisible by ep_size={ep_size}")
    patch_moe_detection()

    ep_group_name = f"ep_size_{ep_size}"
    if ep_group is None:
        ep_group = ds_groups._get_expert_parallel_group(ep_group_name)
    if dist.get_world_size(ep_group) != ep_size:
        raise ValueError("ep_group size must match ep_size")
    if sp_size > 1 and (sp_group is None or dist.get_world_size(sp_group) != sp_size):
        raise ValueError("sp_group size must match sp_size")
    ep_mesh = device_mesh_type.from_group(
        group=ep_group,
        device_type="cuda",
        mesh_dim_names=("ep",),
    )
    ep_backend = options.ep_comm_backend
    if not uses_dispatch_ep(ep_backend):
        raise NotImplementedError(f"EP comm backend must be one of ('deepep', 'uccl'), got {ep_backend!r}.")

    dp_replicate = world_size // ep_size
    model_config = adapter.build_model_config(
        model_name,
        ep_size,
        dp_replicate,
        optimization_dtype,
        attn_implementation,
        options,
    )
    parallel_dims = ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=-1,
        cp=1,
        pp=1,
        ep=ep_size,
        world_size=world_size,
    )
    model = family_loader(
        model_config,
        parallel_dims,
        ep_mesh,
        ep_group_name,
        fused_cross_entropy=options.fused_cross_entropy,
        tiled_mlp_token_chunk_size=options.tiled_mlp_token_chunk_size,
        sp_size=sp_size,
        sp_group=sp_group,
    )
    maybe_use_fixed_order_row_sum(model, options.model_dump())
    return model


def _gather_ep_weight_to_cpu(
    tensor: torch.Tensor,
    ep_group,
    *,
    is_master: bool,
    max_gather_bytes: int = 64 * 1024 * 1024,
) -> torch.Tensor | None:
    import torch.distributed as dist

    from arctic_platform.model.weight_export import wait_for_export_collective

    local = tensor.detach().contiguous()
    ep_size = dist.get_world_size(group=ep_group)
    if local.ndim == 0:
        raise ValueError("expert-parallel export requires a tensor sharded along dimension zero")
    if max_gather_bytes < ep_size * local.element_size():
        raise ValueError("max_gather_bytes must hold at least one expert-parallel element per rank")

    full = None
    if is_master:
        full = torch.empty(
            (ep_size * local.shape[0], *local.shape[1:]),
            dtype=local.dtype,
            device="cpu",
        )
    if local.numel() == 0:
        return full
    if ep_size == 1:
        if full is not None:
            full.view(torch.uint8).copy_(local.view(torch.uint8))
        return full

    # Stage bounded, group-ordered chunks on CPU instead of retaining full GPU shards.
    flat = local.view(torch.uint8).view(-1)
    chunk_elements = min(flat.numel(), max_gather_bytes // ep_size)
    gathered = torch.empty(ep_size * chunk_elements, dtype=torch.uint8, device=local.device)
    full_shards = full.view(torch.uint8).view(ep_size, flat.numel()) if full is not None else None
    use_direct_buffer = local.is_cuda and dist.get_backend(group=ep_group) == "nccl"
    for offset in range(0, flat.numel(), chunk_elements):
        count = min(chunk_elements, flat.numel() - offset)
        outputs = gathered[: ep_size * count]
        inputs = flat[offset : offset + count]
        if use_direct_buffer:
            # NCCL's list gather allocates another packed output; write into the bounded buffer directly.
            dist.all_gather_into_tensor(outputs, inputs, group=ep_group)
        else:
            dist.all_gather(list(outputs.view(ep_size, count).unbind(0)), inputs, group=ep_group)
        if full_shards is not None:
            for rank, shard in enumerate(outputs.view(ep_size, count)):
                full_shards[rank, offset : offset + count].copy_(shard)
        else:
            wait_for_export_collective(inputs)
    return full
