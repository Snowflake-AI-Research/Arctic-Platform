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
"""DeepSpeed checkpoints of expert-parallel models that convert to universal checkpoints.

DeepSpeed's MoE checkpoint writes one file per expert that ``ds_to_universal`` cannot convert. This format saves the
dense state through DeepSpeed's non-MoE path, where the rank-0 model states hold only EP rank 0's experts, and each EP
rank's experts as a plain ``{name: tensor}`` file, ``experts_ep_rank_NNN.pt``. ``convert_to_universal(expert_shards=
expert_shard_paths(...))`` lays those out as DeepSpeed's AutoEP saves them, so the checkpoint resizes to another DP size
at the same EP size.

Expert parameters are the ones DeepSpeed shards over an expert-parallel group: ``allreduce=False`` and a
``group_name``. Torch and DeepSpeed are imported lazily.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import re
from collections.abc import Iterator
from typing import Any

from .universal_checkpoint import expert_parallel_parameter_names
from .universal_checkpoint import latest_tag
from .universal_checkpoint import universal_load

logger = logging.getLogger(__name__)

EXPERT_SHARD_FMT = "experts_ep_rank_{ep_rank:03d}.pt"
_EXPERT_SHARD_RE = re.compile(r"experts_ep_rank_\d{3,}\.pt")


def expert_shard_name(ep_rank: int) -> str:
    return EXPERT_SHARD_FMT.format(ep_rank=ep_rank)


def expert_params_and_ranks(engine) -> tuple[list[tuple[str, Any]], int, int]:
    """Expert params with this rank's ``(ep_rank, expert_data_parallel_rank)``. Raises when the model has none."""
    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist

    items = [
        (name, p)
        for name, p in engine.module.named_parameters()
        if getattr(p, "allreduce", True) is False and getattr(p, "group_name", None)
    ]
    if not items:
        raise RuntimeError("no EP-tagged expert params (allreduce=False + group_name)")
    group_name = items[0][1].group_name
    ep_rank = dist.get_rank(group=ds_groups._get_expert_parallel_group(group_name))
    edp_rank = dist.get_rank(group=ds_groups._get_expert_data_parallel_group(group_name))
    return items, ep_rank, edp_rank


def has_expert_shards(tag_dir: str) -> bool:
    """Whether ``tag_dir`` was saved in this format rather than DeepSpeed's own MoE format."""
    if not os.path.isdir(tag_dir):
        return False
    with os.scandir(tag_dir) as entries:
        return any(_EXPERT_SHARD_RE.fullmatch(entry.name) for entry in entries)


def expert_shard_paths(engine, tag_dir: str) -> list[str]:
    """Each EP rank's expert file under ``tag_dir``, in EP rank order; raises unless the job runs at the saved EP size."""
    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist

    items, _, _ = expert_params_and_ranks(engine)
    ep_size = dist.get_world_size(group=ds_groups._get_expert_parallel_group(items[0][1].group_name))
    saved = sorted(name for name in os.listdir(tag_dir) if _EXPERT_SHARD_RE.fullmatch(name))
    expected = [expert_shard_name(r) for r in range(len(saved))]
    if saved != expected:
        raise RuntimeError(f"expert shards under {tag_dir!r} are {saved}, expected {expected}")
    if len(saved) != ep_size:
        raise RuntimeError(
            f"checkpoint was saved at ep_size {len(saved)} but this job runs at ep_size {ep_size}; "
            "a universal-checkpoint resize keeps the expert-parallel size"
        )
    return [os.path.join(tag_dir, name) for name in saved]


@contextlib.contextmanager
def non_moe_checkpoint(engine) -> Iterator[None]:
    """Save or load through DeepSpeed's dense path, which leaves the experts to the per-EP-rank files."""
    saved = engine.has_moe_layers
    engine.has_moe_layers = False
    try:
        yield
    finally:
        engine.has_moe_layers = saved


@contextlib.contextmanager
def keep_local_expert_params(engine) -> Iterator[None]:
    """Leave EP-sharded params to the expert files during the dense model-states load.

    The rank-0 model states hold rank 0's shard of each, which another rank's shard need not match in shape (DeepSeek
    V4.1's engram embedding splits its rows unevenly), so each entry is loaded as the rank's own tensor.
    """
    from deepspeed.checkpoint.constants import FROZEN_PARAM_FRAGMENTS

    items, _, _ = expert_params_and_ranks(engine)
    local = {name: p.data for name, p in items}
    # Model-states keys drop wrapper segments (``_checkpoint_wrapped_module.``) that parameter names keep.
    expert_ids = {id(p) for _, p in items}
    local.update((key, t.data) for key, t in engine.module.state_dict(keep_vars=True).items() if id(t) in expert_ids)
    old_load = engine.load_module_state_dict

    def _load_dense_only(checkpoint, *args, **kwargs):
        checkpoint = dict(checkpoint)
        for key in ("module", FROZEN_PARAM_FRAGMENTS):
            saved = checkpoint.get(key)
            if saved is not None:
                checkpoint[key] = {name: local.get(name, t) for name, t in saved.items()}
        return old_load(checkpoint, *args, **kwargs)

    engine.load_module_state_dict = _load_dense_only
    try:
        yield
    finally:
        engine.load_module_state_dict = old_load


def save_expert_shards(engine, tag_dir: str) -> None:
    """Write this EP rank's experts to ``tag_dir``; one writer per EP rank, since expert-data-parallel replicas match."""
    import torch

    items, ep_rank, edp_rank = expert_params_and_ranks(engine)
    if edp_rank != 0:
        return
    os.makedirs(tag_dir, exist_ok=True)
    path = os.path.join(tag_dir, expert_shard_name(ep_rank))
    partial = path + ".partial"
    # The shard is a plain ``{name: tensor}`` dict with one standalone CPU storage per tensor. Writing it must
    # neither stage every local expert on the host (``.cpu()`` on every parameter) nor write whole ZeRO flat
    # buffers (``torch.save`` of the parameters: each bf16 ``.data`` is a view of a per-group flat buffer, and
    # torch.save copies and writes every backing storage whole). The archive is laid out first from never-written
    # host placeholders without their bytes. Each parameter is then copied to the host alone, written at its slot,
    # flushed to disk and dropped from the page cache before the next one, so each rank holds at most one tensor
    # in anonymous memory and one tensor of dirty pages.
    # An exception removes ``.partial``; a kill (an OOM killer's SIGKILL) can leave it behind, but the final name only
    # ever appears, by rename, on a complete shard.
    layout = {name: torch.empty(p.shape, dtype=p.dtype) for name, p in items}
    try:
        with torch.serialization.skip_data():
            torch.save(layout, partial)
        del layout
        offsets = _shard_slot_offsets(partial)
        fd = os.open(partial, os.O_WRONLY)
        try:
            for name, p in items:
                host = p.data.detach().contiguous().view(-1).view(torch.uint8).cpu()
                data = memoryview(host.numpy())
                offset, size = offsets[name]
                if len(data) != size:
                    raise RuntimeError(f"expert shard slot for {name!r} holds {size} bytes, not {len(data)}")
                written = 0
                while written < len(data):
                    written += os.pwrite(fd, data[written:], offset + written)
                os.fdatasync(fd)
                os.posix_fadvise(fd, offset, len(data), os.POSIX_FADV_DONTNEED)
                del data, host
        finally:
            os.close(fd)
        os.replace(partial, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(partial)
        raise
    logger.info("saved %d expert params (ep_rank=%d) to %s", len(items), ep_rank, path)


def load_expert_shards(engine, tag_dir: str) -> None:
    """Copy this EP rank's experts from its file under ``tag_dir`` into the module."""
    import struct
    import zipfile

    import torch

    items, ep_rank, _ = expert_params_and_ranks(engine)
    path = os.path.join(tag_dir, expert_shard_name(ep_rank))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing expert shard {path!r} for EP rank {ep_rank}")
    # The shard is a plain ``{name: tensor}`` dict. It may come from storage someone else controls: a full unpickle
    # would run whatever a tampered shard names in every GPU worker on resume, so the archive is opened only by the
    # restricted weights-only unpickler. Never fall back. Any plain dict ``torch.save`` writes loads: standalone
    # storages as save_expert_shards writes them, and tensors that share a storage, stride through it or occupy part of
    # it, as a direct ``torch.save`` of tensors writes them. Every rank of a node resumes at once, so the shard must not
    # be read into host memory whole: the archive is opened mapped for each tensor's dtype, layout and storage record
    # only. Each record that backs a needed expert is then read alone, copied into every parameter it backs and dropped
    # from the page cache before the next one; records that back no needed expert are not read. Each rank holds at most
    # one storage record in anonymous memory (one tensor when storages are standalone) and leaves no shard pages
    # cached. Ranks with the same ep_rank on a node (expert-data-parallel replicas) read one shard file, so after
    # DONTNEED a later reader re-reads its records from disk, a cost only when expert data parallelism is above one. A
    # mapped ``torch.load`` reads only zip archives, which is what ``torch.save`` writes; anything else is refused.
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"expert shard {path!r} is corrupt or not a plain tensor dict (not a zip archive)")
    try:
        shard = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            f"expert shard {path!r} is corrupt or not a plain tensor dict (refused by the weights-only loader): {e}"
        ) from e
    if not isinstance(shard, dict):
        raise TypeError(f"expert shard {path!r} holds {type(shard).__name__}, expected a dict")
    layout = {}
    for name, _ in items:
        if name not in shard:
            raise KeyError(f"expert param {name!r} absent from {path!r}")
        t = shard[name]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"expert param {name!r} in {path!r} holds {type(t).__name__}, expected a tensor")
        layout[name] = (t.dtype, t.shape, t.stride(), t.storage_offset())
    shard = t = None  # the mapping is released before any record is read
    try:
        records = _shard_slot_offsets(path)
    except (zipfile.BadZipFile, struct.error, KeyError, RuntimeError) as e:
        raise RuntimeError(
            f"expert shard {path!r} is corrupt or not a plain tensor dict "
            f"(its zip directory does not locate its tensor storages): {e}"
        ) from e
    backed: dict[tuple[int, int], list] = {}
    for name, p in items:
        backed.setdefault(records[name], []).append((name, p))
    fd = os.open(path, os.O_RDONLY)
    try:
        with torch.no_grad():
            for (offset, size), params in sorted(backed.items()):
                host = torch.empty(size, dtype=torch.uint8)
                data = memoryview(host.numpy())
                read = 0
                while read < size:
                    n = os.preadv(fd, [data[read:]], offset + read)
                    if n == 0:
                        raise RuntimeError(f"expert shard {path!r} ends inside the storage record of {params[0][0]!r}")
                    read += n
                for name, p in params:
                    dtype, shape, stride, storage_offset = layout[name]
                    try:
                        tensor = host.view(dtype).as_strided(shape, stride, storage_offset)
                    except RuntimeError as e:
                        raise RuntimeError(
                            f"expert shard {path!r} does not hold {name!r} ({dtype}, shape {tuple(shape)}, "
                            f"stride {stride}, storage offset {storage_offset}) inside its {size}-byte storage "
                            f"record: {e}"
                        ) from e
                    p.data.copy_(tensor.to(device=p.data.device, dtype=p.data.dtype))
                    del tensor
                os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
                del data, host
    finally:
        os.close(fd)
    logger.info("loaded %d expert params (ep_rank=%d) from %s", len(items), ep_rank, path)


def _shard_slot_offsets(path: str) -> dict:
    """File offset and size of the storage record that holds each tensor's bytes in a ``torch.save`` shard.

    The saver locates slots in the layout it writes under ``torch.serialization.skip_data``, where every placeholder
    has its own storage, so each name's record is that tensor's own slot. The loader locates the records of a finished
    shard, whose tensors may share a storage or occupy part of one. A read-only mapped load places each distinct
    storage at the mapping's base plus its record offset; the zip directory gives each record's offset and size.
    Pairing the two by address assumes nothing about how ``torch.save`` names its records. A record that the directory
    declares past the end of the file is refused, so no caller sizes a buffer or a write from it.
    """
    import struct
    import zipfile

    import torch

    slots = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    records = []
    with zipfile.ZipFile(path) as archive, open(path, "rb") as raw:
        file_size = os.fstat(raw.fileno()).st_size
        for info in archive.infolist():
            if re.search(r"/data/\d+$", info.filename):
                raw.seek(info.header_offset + 26)
                name_len, extra_len = struct.unpack("<HH", raw.read(4))
                offset = info.header_offset + 30 + name_len + extra_len
                if offset + info.file_size > file_size:
                    raise RuntimeError(
                        f"expert shard record {info.filename!r} at byte {offset} declares {info.file_size} bytes, "
                        f"past the end of the {file_size}-byte file"
                    )
                records.append((offset, info.file_size))
    records.sort()
    tensors = [(name, t) for name, t in slots.items() if isinstance(t, torch.Tensor)]
    storages = {}
    for name, t in tensors:
        storages.setdefault(t.untyped_storage().data_ptr(), (name, t.untyped_storage()))
    storages = sorted(storages.items(), key=lambda item: item[0])
    if len(storages) != len(records):
        raise RuntimeError(f"expert shard layout has {len(records)} data records for {len(storages)} storages")
    base = storages[0][0] - records[0][0] if storages else 0
    placed = {}
    for (address, (name, storage)), (offset, size) in zip(storages, records):
        if address - offset != base or storage.nbytes() != size:
            raise RuntimeError(f"expert shard layout does not place the storage of {name!r} at a record of its size")
        placed[address] = (offset, size)
    return {name: placed[t.untyped_storage().data_ptr()] for name, t in tensors}


def resolve_tag(load_dir: str, tag: str | None) -> str:
    """``tag``, or the one DeepSpeed's ``latest`` marker under ``load_dir`` names; raises when there is neither."""
    if tag is not None:
        return tag
    latest = os.path.join(load_dir, "latest")
    if not os.path.isfile(latest):
        raise RuntimeError(f"no tag given and no 'latest' marker at {latest!r}")
    with open(latest, encoding="utf-8") as f:
        return f.read().strip()


def save_expert_parallel_checkpoint(engine, save_dir: str, tag: str | None = None, **save_kwargs) -> str:
    """``engine.save_checkpoint`` in this format; returns the tag. Collective.

    ``tag=None`` saves as ``global_step<N>``, as DeepSpeed does.
    """
    tag = tag if tag is not None else f"global_step{engine.global_steps}"
    with non_moe_checkpoint(engine):
        engine.save_checkpoint(save_dir, tag=tag, **save_kwargs)
    save_expert_shards(engine, os.path.join(save_dir, tag))
    return tag


def load_expert_parallel_checkpoint(
    engine,
    load_dir: str,
    tag: str | None = None,
    *,
    universal: bool = False,
    load_module_only: bool = False,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool | None = None,
):
    """``engine.load_checkpoint`` for a checkpoint in this format; returns what it returns. Collective.

    ``universal``: ``load_dir/tag`` is a universal copy converted with ``expert_shard_paths``. It holds every expert's
    fp32 master; DeepSpeed hands each EP rank its contiguous block and rewrites the bf16 experts from it, so no expert
    file is read. Otherwise this rank's experts come from its file, which holds weights, not optimizer state, so it is
    read for a module-only load too. ``load_lr_scheduler_states`` defaults to a full (not module-only) optimizer load.
    """
    if load_lr_scheduler_states is None:
        load_lr_scheduler_states = load_optimizer_states and not load_module_only
    tag = resolve_tag(load_dir, tag)
    if universal:
        with universal_load(engine), non_moe_checkpoint(engine), keep_local_expert_params(engine):
            return engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_optimizer_states=load_optimizer_states,
                load_lr_scheduler_states=load_lr_scheduler_states,
            )
    with non_moe_checkpoint(engine), keep_local_expert_params(engine):
        result = engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_only=load_module_only,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
    if result[0] is None:
        return result
    load_expert_shards(engine, os.path.join(load_dir, tag))
    # A module-only load reinitializes the optimizer, and DeepSpeed refreshes its fp32 masters from the module inside
    # load_checkpoint, before the experts above were loaded. Refresh again, or the first step's master-to-bf16 copy
    # would put the experts back to their initial values. A full load restores the masters from the optimizer shards.
    if load_module_only:
        engine.optimizer.refresh_fp32_params()
    return result


def load_deepspeed_checkpoint(
    engine, load_dir: str, tag: str | None = None, *, universal: bool = False, **load_kwargs
):
    """``engine.load_checkpoint`` for a checkpoint in DeepSpeed's own format or in this one. Collective.

    ``universal``: ``load_dir/tag`` is a universal copy, which an expert-parallel engine loads as this format's. Otherwise
    the format is that of the files under the tag. Returns what ``engine.load_checkpoint`` returns.
    """
    if universal:
        if expert_parallel_parameter_names(engine.module):
            return load_expert_parallel_checkpoint(engine, load_dir, tag, universal=True, **load_kwargs)
        with universal_load(engine):
            return engine.load_checkpoint(load_dir, tag, **load_kwargs)
    resolved = tag if tag is not None else latest_tag(load_dir)
    if resolved is not None and has_expert_shards(os.path.join(load_dir, resolved)):
        return load_expert_parallel_checkpoint(engine, load_dir, resolved, **load_kwargs)
    return engine.load_checkpoint(load_dir, tag, **load_kwargs)
