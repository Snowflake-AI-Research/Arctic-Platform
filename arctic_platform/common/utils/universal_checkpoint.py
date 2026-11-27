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
"""Load a DeepSpeed ZeRO-1/2 checkpoint at a different DP size through a universal checkpoint.

A ZeRO-1/2 checkpoint is rigid: one optimizer shard per DP rank, so ``engine.load_checkpoint`` refuses it at any other
DP size. Converting it with DeepSpeed's ``ds_to_universal`` and loading the copy with ``checkpoint.load_universal``
restores the fp32 master weights exactly, and the Adam moments and LR scheduler when those are loaded too.

``load_checkpoint_with_dp_resize`` is the collective entry point for a training worker. The other helpers are its
building blocks, for callers that choose where the converted copy goes and which ranks convert it. Torch and
DeepSpeed are imported lazily.

Expert-parallel checkpoints saved in the format of ``expert_parallel_checkpoint``, with each EP rank's experts in a
``{name: tensor}`` file, convert through DeepSpeed's AutoEP support: ``convert_to_universal(..., expert_shards=...)``
lays the checkpoint out as AutoEP saves it. The universal load then gives each EP rank its contiguous block of
experts, so the EP size must not change.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

logger = logging.getLogger(__name__)

UNIVERSAL_TAG_SUFFIX = "_universal"
# First DeepSpeed release that records and loads Ulysses sequence-parallel checkpoints as TP=1.
SEQUENCE_PARALLEL_MIN_DEEPSPEED = "0.20.2"
# How long the other ranks wait for one rank's ds_to_universal. It reads and rewrites the whole checkpoint, which on
# shared storage outlasts DeepSpeed's default process-group timeout.
DEFAULT_CONVERSION_TIMEOUT_S = 4 * 3600

_MODEL_STATES = "mp_rank_00_model_states.pt"
_OPTIM_SHARD_RE = re.compile(r"(?:^|/)(?:bf16_)?zero_pp_rank_(\d+)_mp_rank_\d+_optim_states\.pt$")
_OUTPUT_TAIL_CHARS = 4000
# The fused expert weights DeepSpeed's AutoEP conversion consolidates: [local experts, ...] blocks along dim 0.
_EXPERT_WEIGHT_RE = re.compile(r"^(.+\.experts)\.(w[123])$")
_EXPERT_WEIGHTS = ("w1", "w2", "w3")
_AUTOEP_INPUT_SUFFIX = ".autoep-input"
# ds_to_universal's CLI defaults (4 extract and 2 merge workers) leave most of a GPU node's cores idle, and a process
# started by torchrun inherits OMP_NUM_THREADS=1, which makes every converter worker single-threaded.
_MAX_MERGE_WORKERS = 16
# A DeepSpeed release that does not memory-map the shard holds a whole one per extract worker, plus its fragments.
_EXTRACT_BYTES_PER_SHARD_BYTE = 2


def optimizer_shard_ranks(names: Iterable[str]) -> set[int]:
    """DP ranks with a ZeRO optimizer shard among the file names or paths in ``names``."""
    return {int(m.group(1)) for name in names if (m := _OPTIM_SHARD_RE.search(name))}


def saved_dp_from_files(names: Iterable[str]) -> int | None:
    """DP size a checkpoint tag was saved at, from a listing of its files; ``None`` when it lists no optimizer shard.

    Raises unless the shards are those of DP ranks ``0..N-1``: a listing with gaps does not give the DP size.
    """
    ranks = optimizer_shard_ranks(names)
    if not ranks:
        return None
    if ranks != set(range(len(ranks))):
        missing = sorted(set(range(max(ranks) + 1)) - ranks)
        raise RuntimeError(f"the checkpoint's optimizer shards lack DP rank(s) {missing}")
    return len(ranks)


def resize_needed(saved_dp: int | None, dp_size: int) -> bool:
    """Whether a checkpoint saved at ``saved_dp`` must be converted to load at ``dp_size`` (the ZeRO sequence-DP size).

    A resize reads the whole checkpoint, so every copy of the checkpoint directory needs every file of the tag.
    """
    return saved_dp is not None and saved_dp != dp_size


def checkpoint_copy_key(node_local_storage: bool) -> str:
    """Ranks with equal keys share one copy of the checkpoint directory: a node's ranks, or every rank.

    The caller declares the storage, as DeepSpeed's ``checkpoint.use_node_local_storage`` does; a node-local-looking
    path such as ``/tmp`` may be shared, so it is not inspected.
    """
    return socket.gethostname() if node_local_storage else ""


def is_copy_leader(keys: Sequence[str], rank: int) -> bool:
    """Whether ``rank`` is the lowest rank sharing its copy, given every rank's ``checkpoint_copy_key``.

    The leader prepares and converts its copy for the others: it empties a reused tag directory before a resize
    download, since the conversion reads every file there, and writes the universal copy.
    """
    return keys.index(keys[rank]) == rank


def latest_tag(load_dir: str) -> str | None:
    """The tag DeepSpeed's ``latest`` marker under ``load_dir`` names, or ``None``."""
    latest = os.path.join(load_dir, "latest")
    if not os.path.isfile(latest):
        return None
    with open(latest, encoding="utf-8") as f:
        return f.read().strip() or None


def _int_field(state: Any, key: str) -> int | None:
    value = state.get(key) if isinstance(state, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _saved_model_state(tag_dir: str) -> dict | None:
    import torch

    path = os.path.join(tag_dir, _MODEL_STATES)
    if not os.path.isfile(path):
        return None
    # DeepSpeed's own load unpickles this file in full; mmap keeps the module weights off the host.
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return state if isinstance(state, dict) else None


def saved_dp_world_size(tag_dir: str) -> int | None:
    """DP size a ZeRO-1/2 checkpoint tag was saved at, from its model states; ``None`` when it has none."""
    return _int_field(_saved_model_state(tag_dir), "dp_world_size")


def _has_model_parallelism(state: dict | None) -> bool:
    from deepspeed.checkpoint.constants import CHECKPOINT_PARALLEL_DIMS
    from deepspeed.checkpoint.constants import CHECKPOINT_PP_DEGREE

    if _int_field(state, "mp_world_size") != 1:
        return True
    dims = state.get(CHECKPOINT_PARALLEL_DIMS)
    return isinstance(dims, dict) and dims.get(CHECKPOINT_PP_DEGREE, 1) != 1


def require_complete_checkpoint(tag_dir: str) -> int:
    """Return the saved DP size after checking ``tag_dir`` holds every DP rank's optimizer shard.

    The converter infers the DP size from the shards it finds, so a partial set would convert without error into
    wrong tensors. A node-local checkpoint written by a multi-node job holds only that node's shards.
    """
    saved = saved_dp_world_size(tag_dir)
    if saved is None:
        raise RuntimeError(f"no {_MODEL_STATES} with a dp_world_size under {tag_dir!r}")
    with os.scandir(tag_dir) as entries:
        present = optimizer_shard_ranks(entry.name for entry in entries if entry.is_file())
    missing = sorted(set(range(saved)) - present)
    if missing:
        raise RuntimeError(
            f"{tag_dir!r} was saved at DP size {saved} but lacks the optimizer shards of rank(s) {missing}; "
            "converting it to a universal checkpoint needs every rank's shard."
        )
    return saved


def autoep_layers_metadata(
    param_shapes: Mapping[str, Sequence[int]], expert_names: Iterable[str], ep_size: int
) -> list[dict]:
    """DeepSpeed AutoEP layer metadata for expert weights split over ``ep_size`` ranks along dim 0.

    ``param_shapes`` maps each trainable parameter to its shape on one rank, as a checkpoint's ``param_shapes`` do, and
    ``expert_names`` names the expert-parallel parameters. Each must be a trainable ``<prefix>.experts.w1``, ``w2`` or
    ``w3`` holding the rank's contiguous block of experts, the only layout ``ds_to_universal`` consolidates.
    """
    from deepspeed.checkpoint.constants import AUTOEP_EP_SIZE
    from deepspeed.checkpoint.constants import AUTOEP_EXPERT_KEY_PREFIX
    from deepspeed.checkpoint.constants import AUTOEP_NUM_EXPERTS
    from deepspeed.checkpoint.constants import AUTOEP_NUM_LOCAL_EXPERTS

    layers: dict[str, dict[str, tuple[int, ...]]] = {}
    for name in expert_names:
        match = _EXPERT_WEIGHT_RE.match(name)
        if match is None:
            raise ValueError(
                f"expert-parallel parameter {name!r} is not an experts.w1/w2/w3 weight; ds_to_universal cannot "
                "consolidate it"
            )
        if name not in param_shapes:
            raise ValueError(
                f"expert-parallel parameter {name!r} has no optimizer state; frozen expert weights cannot be "
                "converted to a universal checkpoint"
            )
        layers.setdefault(match.group(1), {})[match.group(2)] = tuple(param_shapes[name])
    if not layers:
        raise ValueError("no expert-parallel parameters to describe")
    metadata = []
    for layer_id, prefix in enumerate(sorted(layers)):
        shapes = layers[prefix]
        if sorted(shapes) != list(_EXPERT_WEIGHTS):
            raise ValueError(f"{prefix} has expert weights {sorted(shapes)}, expected {list(_EXPERT_WEIGHTS)}")
        local = {shape[0] for shape in shapes.values()}
        if len(local) != 1:
            raise ValueError(f"{prefix} expert weights disagree on the local expert count: {shapes}")
        num_local = local.pop()
        metadata.append(
            {
                "moe_layer_id": layer_id,
                AUTOEP_EXPERT_KEY_PREFIX: prefix,
                AUTOEP_NUM_EXPERTS: num_local * ep_size,
                AUTOEP_NUM_LOCAL_EXPERTS: num_local,
                AUTOEP_EP_SIZE: ep_size,
            }
        )
    return metadata


def stage_autoep_input(src_tag_dir: str, staging_dir: str, expert_shards: Sequence[str]) -> list[dict]:
    """Lay out ``src_tag_dir`` in ``staging_dir`` as DeepSpeed's AutoEP saves a ZeRO-1/2 checkpoint.

    ``expert_shards[r]`` is a ``{name: tensor}`` file with expert-parallel rank ``r``'s experts, which the
    checkpoint's own model states hold only for rank 0. ``staging_dir`` gets links to the source files, a copy of the
    model states carrying ``ds_autoep_layers``, and one file per (layer, expert). Returns the layer metadata.
    """
    import torch
    from deepspeed.checkpoint.constants import AUTOEP_EXPERT_KEY_PREFIX
    from deepspeed.checkpoint.constants import AUTOEP_LAYERS_KEY
    from deepspeed.checkpoint.constants import AUTOEP_NUM_EXPERTS
    from deepspeed.checkpoint.constants import AUTOEP_NUM_LOCAL_EXPERTS
    from deepspeed.checkpoint.constants import PARAM_SHAPES

    state = _saved_model_state(src_tag_dir)
    if state is None:
        raise RuntimeError(f"no {_MODEL_STATES} under {src_tag_dir!r}")
    if _has_model_parallelism(state):
        raise RuntimeError(f"{src_tag_dir!r} has tensor or pipeline parallelism; AutoEP conversion needs neither")
    param_shapes = {name: shape for group in state[PARAM_SHAPES] for name, shape in group.items()}
    shards = [torch.load(path, map_location="cpu", mmap=True, weights_only=True) for path in expert_shards]
    if not shards:
        raise ValueError("expert_shards is empty")
    names = set(shards[0])
    for rank, shard in enumerate(shards):
        if set(shard) != names:
            raise RuntimeError(
                f"expert shard {expert_shards[rank]!r} holds {sorted(set(shard) ^ names)} unlike rank 0's shard"
            )
    layers = autoep_layers_metadata(param_shapes, names, len(shards))

    os.makedirs(staging_dir)
    with os.scandir(src_tag_dir) as entries:
        for entry in entries:
            if entry.name != _MODEL_STATES:
                os.symlink(os.path.abspath(entry.path), os.path.join(staging_dir, entry.name))
    state[AUTOEP_LAYERS_KEY] = layers
    torch.save(state, os.path.join(staging_dir, _MODEL_STATES))
    del state
    for layer in layers:
        prefix, num_local = layer[AUTOEP_EXPERT_KEY_PREFIX], layer[AUTOEP_NUM_LOCAL_EXPERTS]
        for weight in _EXPERT_WEIGHTS:
            for rank, shard in enumerate(shards):
                rows = shard[f"{prefix}.{weight}"].shape[0]
                if rows != num_local:
                    raise RuntimeError(
                        f"{expert_shards[rank]!r} holds {rows} experts of {prefix}.{weight}, expected {num_local}"
                    )
        for expert in range(layer[AUTOEP_NUM_EXPERTS]):
            rank, local = divmod(expert, num_local)
            # clone: saving a view would write the whole shard's storage into every expert file.
            tensors = {
                f"{prefix}.{weight}.{expert}": shards[rank][f"{prefix}.{weight}"][local].clone()
                for weight in _EXPERT_WEIGHTS
            }
            name = f"layer_{layer['moe_layer_id']}_expert_{expert}_mp_rank_00_model_states.pt"
            torch.save(tensors, os.path.join(staging_dir, name))
    return layers


def _usable_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _converter_parallelism(input_dir: str) -> tuple[int, int, int]:
    """``ds_to_universal`` extract workers, merge workers and threads per worker for converting ``input_dir``.

    Each extract worker takes one optimizer shard, so more workers than shards, CPUs or host memory allow do not help.
    """
    shards = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if _OPTIM_SHARD_RE.search(name)]
    cpus = _usable_cpus()
    extract = max(1, min(len(shards), cpus))
    memory = _available_memory_bytes()
    if memory is not None and shards:
        largest = max(os.path.getsize(path) for path in shards)
        extract = max(1, min(extract, memory // max(1, _EXTRACT_BYTES_PER_SHARD_BYTE * largest)))
    merge = max(1, min(_MAX_MERGE_WORKERS, cpus))
    threads = max(1, cpus // max(extract, merge))
    return extract, merge, threads


def convert_to_universal(src_tag_dir: str, out_tag_dir: str, *, expert_shards: Sequence[str] | None = None) -> None:
    """Convert one checkpoint tag with ``ds_to_universal`` into ``out_tag_dir``, replacing anything there.

    Runs in a subprocess: the converter forks worker processes, which a process holding CUDA and NCCL state must not.
    The subprocess spreads its workers and their threads over this process's CPUs, whatever ``OMP_NUM_THREADS`` the
    training job runs with. A checkpoint without TP or PP needs no client merge patterns, so the converter may add the
    version-only ``universal_checkpoint_info`` that older DeepSpeed releases do not write. ``expert_shards`` lists
    each EP rank's ``{name: tensor}`` expert file, in EP rank order, for a checkpoint with expert parallelism (see
    ``stage_autoep_input``); the staged input is removed afterwards.
    """
    shutil.rmtree(out_tag_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(out_tag_dir) or ".", exist_ok=True)
    staging = None
    input_dir = src_tag_dir
    if expert_shards is not None:
        staging = out_tag_dir.rstrip("/") + _AUTOEP_INPUT_SUFFIX
        shutil.rmtree(staging, ignore_errors=True)
    try:
        if staging is not None:
            stage_autoep_input(src_tag_dir, staging, expert_shards)
            input_dir = staging
        cmd = [sys.executable, "-m", "deepspeed.checkpoint.ds_to_universal"]
        cmd += ["--input_folder", input_dir, "--output_folder", out_tag_dir]
        if not _has_model_parallelism(_saved_model_state(src_tag_dir)):
            cmd.append("--inject_missing_state")
        extract, merge, threads = _converter_parallelism(input_dir)
        cmd += ["--num_extract_workers", str(extract), "--num_merge_workers", str(merge)]
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        started = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ds_to_universal failed on {src_tag_dir!r} (exit {result.returncode}):\n"
            f"{result.stdout[-_OUTPUT_TAIL_CHARS:]}\n{result.stderr[-_OUTPUT_TAIL_CHARS:]}"
        )
    logger.info(
        "converted %s to universal checkpoint %s in %.1fs (%d extract, %d merge workers, %d threads each)",
        src_tag_dir,
        out_tag_dir,
        time.monotonic() - started,
        extract,
        merge,
        threads,
    )


def expert_parallel_parameter_names(module) -> list[str]:
    """Names of ``module``'s parameters that DeepSpeed shards over an expert-parallel group."""
    return [
        name
        for name, param in module.named_parameters()
        if getattr(param, "allreduce", True) is False and getattr(param, "group_name", None)
    ]


@contextlib.contextmanager
def universal_load(engine) -> Iterator[None]:
    """Turn on DeepSpeed's ``checkpoint.load_universal`` for one load; DeepSpeed reads it only at load time."""
    config = engine._config
    saved = config.load_universal_checkpoint
    config.load_universal_checkpoint = True
    try:
        yield
    finally:
        config.load_universal_checkpoint = saved


def require_resize_support(engine) -> None:
    """Raise unless ``engine`` can load a universal checkpoint converted from another DP size."""
    import deepspeed
    from packaging.version import Version

    stage = engine.zero_optimization_stage()
    if stage not in (1, 2):
        raise RuntimeError(f"universal-checkpoint DP resize supports ZeRO stages 1 and 2, not stage {stage}")
    sequence_parallel = hasattr(engine.mpu, "initialize_sequence_parallel")
    if sequence_parallel and Version(deepspeed.__version__) < Version(SEQUENCE_PARALLEL_MIN_DEEPSPEED):
        raise RuntimeError(
            "universal-checkpoint DP resize with Ulysses sequence parallelism needs "
            f"deepspeed>={SEQUENCE_PARALLEL_MIN_DEEPSPEED}, found {deepspeed.__version__}"
        )


def _world() -> tuple[int, int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def _from_rank0(value):
    import torch.distributed as dist

    if not dist.is_initialized():
        return value
    shared = [value]
    dist.broadcast_object_list(shared, src=0)
    return shared[0]


def _gather(value, group=None) -> list:
    import torch.distributed as dist

    if not dist.is_initialized():
        return [value]
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


@contextlib.contextmanager
def _wait_group(timeout_s: float) -> Iterator[Any]:
    """A gloo group over every rank whose collectives time out after ``timeout_s``. Collective."""
    import torch.distributed as dist

    if not dist.is_initialized():
        yield None
        return
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_s))
    try:
        yield group
    finally:
        dist.destroy_process_group(group)


class CheckpointPeerFailed(RuntimeError):
    """Another rank failed a step of a collective checkpoint load that this rank completed."""


def _raise_if_any_failed(
    error: BaseException | None,
    what: str,
    group=None,
    peer_error: Callable[[str], BaseException] = CheckpointPeerFailed,
) -> None:
    """Collective: re-raise this rank's ``error``, or raise ``peer_error`` naming the ranks that failed."""
    failed = [
        rank for rank, err in enumerate(_gather(None if error is None else repr(error), group)) if err is not None
    ]
    if error is not None:
        raise error
    if failed:
        raise peer_error(f"{what} failed on rank(s) {failed}")


def resize_source_tag(load_dir: str, tag: str | None, *, dp_size: int, saved_dp: int | None = None) -> str | None:
    """The tag to convert when ``load_dir/tag`` was saved at a DP size other than ``dp_size``, else ``None``.

    Collective: rank 0 decides and broadcasts. ``tag=None`` means the ``latest`` tag. ``saved_dp`` is the saved DP size
    when the caller knows it, e.g. ``saved_dp_from_files`` over the listing it downloaded from; otherwise rank 0 reads
    it from the model states.
    """
    rank, _ = _world()
    decision = error = None
    if rank == 0:
        try:
            resolved = tag if tag is not None else latest_tag(load_dir)
            if resolved is not None:
                saved = saved_dp if saved_dp is not None else saved_dp_world_size(os.path.join(load_dir, resolved))
                if resize_needed(saved, dp_size):
                    decision = resolved
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    decision, error = _from_rank0((decision, error))
    if error is not None:
        raise RuntimeError(f"rank 0 could not read the saved DP size of {load_dir!r} tag={tag!r}: {error}")
    return decision


def is_universal_converter(engine, *, node_local_storage: bool | None = None) -> bool:
    """Whether this rank writes the universal copy that the ranks sharing its checkpoint directory read.

    Under node-local storage (``node_local_storage``, by default DeepSpeed's ``checkpoint.use_node_local_storage``)
    every node holds its own copy and each node's lowest rank converts; otherwise one directory is shared by every
    node and rank 0 converts it once. See ``checkpoint_copy_key``. Collective under node-local storage.
    """
    if node_local_storage is None:
        node_local_storage = engine.use_node_local_storage()
    rank, _ = _world()
    if not node_local_storage:
        return rank == 0
    # LOCAL_RANK does not identify a node's ranks when each process sees one GPU, so group by host.
    return is_copy_leader(_gather(checkpoint_copy_key(True)), rank)


def load_checkpoint_with_dp_resize(
    engine,
    load_dir: str,
    tag: str | None = None,
    *,
    scratch_dir: str | None = None,
    node_local_storage: bool | None = None,
    saved_dp: int | None = None,
    load_fn: Callable[..., Any] | None = None,
    peer_error: Callable[[str], BaseException] = CheckpointPeerFailed,
    conversion_timeout_s: float = DEFAULT_CONVERSION_TIMEOUT_S,
    **load_kwargs,
):
    """Load a checkpoint, converting a ZeRO-1/2 checkpoint saved at another DP size. Collective.

    At the saved DP size this is ``load_fn(load_dir, tag, universal=False, **load_kwargs)``. Otherwise the tag is
    converted into ``scratch_dir`` (default ``load_dir``) as ``<tag>_universal_<id>``, a name unique to this load so
    that jobs resizing the same checkpoint do not share a copy; every rank loads it with ``load_fn(scratch_dir,
    <copy tag>, universal=True, **load_kwargs)``, and it is removed, also when the conversion or any rank's load fails.

    - The copy is written by rank 0, or by one rank per node under ``node_local_storage`` (see
      ``is_universal_converter``), and that rank needs every DP rank's optimizer shard under ``load_dir/tag``.
    - ``saved_dp``: see ``resize_source_tag``. The converting rank checks it against the model states.
    - ``load_fn`` defaults to ``load_deepspeed_checkpoint``, for checkpoints saved by ``engine.save_checkpoint`` or in
      the expert-parallel format of ``save_expert_parallel_checkpoint``. A checkpoint of an expert-parallel engine
      converts through its per-EP-rank expert files and keeps the EP size. Another ``load_fn`` must load a universal
      copy as ``load_deepspeed_checkpoint(universal=True)`` does.
    - A rank whose conversion or load fails raises its own error; the others raise ``peer_error(message)``. Ranks wait
      for each other up to ``conversion_timeout_s``.

    Returns what ``load_fn`` returns.
    """
    from .expert_parallel_checkpoint import expert_shard_paths
    from .expert_parallel_checkpoint import has_expert_shards
    from .expert_parallel_checkpoint import load_deepspeed_checkpoint

    if load_fn is None:

        def load_fn(load_dir, tag, *, universal, **kwargs):
            return load_deepspeed_checkpoint(engine, load_dir, tag, universal=universal, **kwargs)

    source = resize_source_tag(load_dir, tag, dp_size=engine.seq_dp_world_size, saved_dp=saved_dp)
    if source is None:
        return load_fn(load_dir, tag, universal=False, **load_kwargs)
    require_resize_support(engine)
    expert_parallel = bool(expert_parallel_parameter_names(engine.module))
    out_root = scratch_dir or load_dir
    out_tag = f"{source}{UNIVERSAL_TAG_SUFFIX}_{_from_rank0(uuid.uuid4().hex[:8])}"
    converter = is_universal_converter(engine, node_local_storage=node_local_storage)
    try:
        # Created before converting: joining a new group is bounded by the default timeout, not ``conversion_timeout_s``.
        with _wait_group(conversion_timeout_s) as group:
            error = None
            if converter:
                try:
                    src = os.path.join(load_dir, source)
                    saved = require_complete_checkpoint(src)
                    if saved_dp is not None and saved != saved_dp:
                        raise RuntimeError(
                            f"{src!r} was saved at DP size {saved}, but its listing has {saved_dp} optimizer shards"
                        )
                    shards = None
                    if expert_parallel:
                        if not has_expert_shards(src):
                            raise RuntimeError(
                                f"{src!r} is in DeepSpeed's MoE checkpoint format, which ds_to_universal cannot "
                                "convert; save expert-parallel checkpoints with save_expert_parallel_checkpoint to "
                                "load them at another DP size"
                            )
                        shards = expert_shard_paths(engine, src)
                    logger.info(
                        "resizing %s from DP %d to DP %d via %s", src, saved, engine.seq_dp_world_size, out_tag
                    )
                    convert_to_universal(src, os.path.join(out_root, out_tag), expert_shards=shards)
                except Exception as exc:
                    error = exc
            _raise_if_any_failed(error, "universal checkpoint conversion", group, peer_error)
            result = error = None
            try:
                result = load_fn(out_root, out_tag, universal=True, **load_kwargs)
            except Exception as exc:
                error = exc
            _raise_if_any_failed(error, "universal checkpoint load", group, peer_error)
            return result
    finally:
        if converter:
            shutil.rmtree(os.path.join(out_root, out_tag), ignore_errors=True)
            # ds_to_universal also points ``latest_universal`` at the copy it wrote.
            marker = os.path.join(out_root, "latest_universal")
            with contextlib.suppress(OSError):
                with open(marker, encoding="utf-8") as f:
                    stale = f.read().strip() == out_tag
                if stale:
                    os.remove(marker)
