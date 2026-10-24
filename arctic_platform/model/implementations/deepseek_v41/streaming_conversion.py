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

"""Out-of-core HF -> PrimeRL conversion for DeepSeek-V4.1 safetensors snapshots.

The conversion chain is replayed on key metadata only. Each output tensor is a list of source
pieces whose bytes are written back to back, so a dim-0 ``Stack`` of per-expert tensors is
the plain concatenation of the experts' bytes. Pass-through pieces are copied as raw byte
ranges in bounded chunks; quantized pieces are dequantized, and packed-expert scales widened to
float32, one source tensor at a time. Peak
host memory is one chunk or one source tensor, never a whole shard, table, or model.
"""

from __future__ import annotations

import json
import logging
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from io import BufferedReader
from pathlib import Path

import torch
from safetensors import safe_open

from arctic_platform.model.implementations.deepseek_v4.conversion_ops import Conditional
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import ConvOp
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import Drop
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import PrefixRename
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import Rename
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import Sequence
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import Stack
from arctic_platform.model.implementations.deepseek_v4.dequantize import dequantize_weight
from arctic_platform.model.implementations.deepseek_v41.converting_deepseek_v41 import RejectUnexplained
from arctic_platform.model.implementations.deepseek_v41.converting_deepseek_v41 import log_deferred_checkpoint_keys
from arctic_platform.model.implementations.deepseek_v41.dequantize import is_packed_expert_scale
from arctic_platform.model.implementations.deepseek_v41.dequantize import quantized_pairs

logger = logging.getLogger(__name__)

INDEX_NAME = "model.safetensors.index.json"
COPY_CHUNK_BYTES = 256 * 1024**2
SHARD_TARGET_BYTES = 32 * 1024**3

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def register_e8m0_safetensors_dtype() -> None:
    """safetensors 0.8 reads ``F8_E8M0`` but its dtype table, used by DCP's HF reader, omits it."""
    from safetensors import torch as safetensors_torch

    safetensors_torch._TYPES.setdefault("F8_E8M0", torch.float8_e8m0fnu)


@dataclass(frozen=True)
class SourceTensor:
    file: Path
    key: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


@dataclass(frozen=True)
class Piece:
    source: SourceTensor
    scale: SourceTensor | None = None
    widen: bool = False


@dataclass
class Planned:
    dtype: str
    shape: tuple[int, ...]
    pieces: list[Piece] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        count = _DTYPE_BYTES[self.dtype]
        for size in self.shape:
            count *= size
        return count


def read_header(path: Path) -> dict[str, SourceTensor]:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    data_start = 8 + length
    return {
        key: SourceTensor(
            file=path,
            key=key,
            dtype=info["dtype"],
            shape=tuple(info["shape"]),
            begin=data_start + info["data_offsets"][0],
            end=data_start + info["data_offsets"][1],
        )
        for key, info in header.items()
    }


def _source_files(source_path: Path) -> list[Path]:
    index_path = source_path / INDEX_NAME
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return [source_path / name for name in sorted(set(weight_map.values()))]
    return sorted(source_path.glob("*.safetensors"))


def _initial_plan(source_path: Path, *, keep_experts: bool = False) -> dict[str, Planned]:
    sources: dict[str, SourceTensor] = {}
    for path in _source_files(source_path):
        for key, tensor in read_header(path).items():
            if key in sources:
                raise ValueError(f"{key} appears in both {sources[key].file.name} and {path.name}")
            sources[key] = tensor
    pairs = quantized_pairs(sources, keep_experts=keep_experts)
    scales = set(pairs.values())
    plan: dict[str, Planned] = {}
    for key, tensor in sources.items():
        if key in scales:
            continue
        scale_key = pairs.get(key)
        if keep_experts and is_packed_expert_scale(key):
            if tensor.dtype != "F8_E8M0":
                raise ValueError(f"{key} is a packed-expert scale but dtype {tensor.dtype}")
            plan[key] = Planned("F32", tensor.shape, [Piece(tensor, widen=True)])
            continue
        if scale_key is None:
            plan[key] = Planned(tensor.dtype, tensor.shape, [Piece(tensor)])
            continue
        shape = tensor.shape
        if tensor.dtype == "I8":
            shape = (*shape[:-1], 2 * shape[-1])
        elif tensor.dtype != "F8_E4M3":
            raise ValueError(f"{key} has a .scale sibling but dtype {tensor.dtype}")
        plan[key] = Planned("BF16", shape, [Piece(tensor, sources[scale_key])])
    return plan


def _stack(plan: dict[str, Planned], op: Stack) -> None:
    if op.dim != 0:
        raise NotImplementedError(f"streaming Stack supports dim 0, got {op.dim} for {op.stacked}")
    items = []
    e = 0
    while op.item.format(e=e) in plan:
        items.append(plan.pop(op.item.format(e=e)))
        e += 1
    if not items:
        return
    first = items[0]
    for item in items[1:]:
        if item.dtype != first.dtype or item.shape != first.shape:
            raise ValueError(f"{op.stacked}: experts disagree on dtype/shape")
    plan[op.stacked] = Planned(
        first.dtype,
        (len(items), *first.shape),
        [piece for item in items for piece in item.pieces],
    )


def _replay(plan: dict[str, Planned], ops: list[ConvOp]) -> None:
    for op in ops:
        if isinstance(op, Sequence):
            _replay(plan, op.ops)
        elif isinstance(op, Conditional):
            _replay(plan, op.then if op.predicate(plan) else op.else_)
        elif isinstance(op, Stack):
            _stack(plan, op)
        elif isinstance(op, (Rename, PrefixRename, Drop, RejectUnexplained)):
            op.hf_to_prime(plan)
        else:
            raise NotImplementedError(f"{type(op).__name__} has no streaming form")


def plan_prime_conversion(source_path: Path, ops: list[ConvOp], *, keep_experts: bool = False) -> dict[str, Planned]:
    plan = _initial_plan(Path(source_path), keep_experts=keep_experts)
    log_deferred_checkpoint_keys(plan)
    _replay(plan, ops)
    return plan


def _assign_shards(plan: dict[str, Planned], target_bytes: int) -> list[list[str]]:
    def locality(name: str) -> tuple[str, int, str]:
        first = plan[name].pieces[0].source
        return (first.file.name, first.begin, name)

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    current_file = None
    for name in sorted(plan, key=locality):
        size = plan[name].nbytes
        source_file = plan[name].pieces[0].source.file
        if current and (current_bytes + size > target_bytes or source_file != current_file):
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
        current_file = source_file
    if current:
        shards.append(current)
    return shards


def _copy_range(destination, source: SourceTensor, handles: dict[Path, BufferedReader], chunk_bytes: int) -> None:
    handle = handles.get(source.file)
    if handle is None:
        handle = handles[source.file] = open(source.file, "rb")
    handle.seek(source.begin)
    remaining = source.end - source.begin
    buffer = bytearray(min(chunk_bytes, max(remaining, 1)))
    view = memoryview(buffer)
    while remaining:
        count = handle.readinto(view[: min(len(buffer), remaining)])
        if not count:
            raise IOError(f"{source.file} ended inside {source.key}")
        destination.write(view[:count])
        remaining -= count


def _write_dequantized(destination, piece: Piece) -> None:
    if piece.scale is None:
        raise RuntimeError(f"{piece.source.key} is marked quantized but has no scale")
    with safe_open(piece.source.file, framework="pt", device="cpu") as weights:
        weight = weights.get_tensor(piece.source.key)
    with safe_open(piece.scale.file, framework="pt", device="cpu") as scales:
        scale = scales.get_tensor(piece.scale.key)
    restored = dequantize_weight(weight, scale).contiguous()
    destination.write(restored.view(torch.uint8).numpy().tobytes())


def _write_widened(destination, piece: Piece) -> None:
    with safe_open(piece.source.file, framework="pt", device="cpu") as tensors:
        widened = tensors.get_tensor(piece.source.key).float().contiguous()
    destination.write(widened.view(torch.uint8).numpy().tobytes())


def _write_shard(path: Path, names: list[str], plan: dict[str, Planned], chunk_bytes: int) -> None:
    header: dict[str, dict] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in names:
        planned = plan[name]
        header[name] = {
            "dtype": planned.dtype,
            "shape": list(planned.shape),
            "data_offsets": [offset, offset + planned.nbytes],
        }
        offset += planned.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    handles: dict[Path, BufferedReader] = {}
    try:
        with open(path, "wb") as destination:
            destination.write(struct.pack("<Q", len(encoded)))
            destination.write(encoded)
            for name in names:
                start = destination.tell()
                for piece in plan[name].pieces:
                    if piece.widen:
                        _write_widened(destination, piece)
                    elif piece.scale is None:
                        _copy_range(destination, piece.source, handles, chunk_bytes)
                    else:
                        _write_dequantized(destination, piece)
                written = destination.tell() - start
                if written != plan[name].nbytes:
                    raise RuntimeError(f"{name}: wrote {written} bytes, planned {plan[name].nbytes}")
    finally:
        for handle in handles.values():
            handle.close()


def write_streaming_prime_cache(
    source_path: Path,
    output_path: Path,
    ops: list[ConvOp],
    *,
    shard_target_bytes: int = SHARD_TARGET_BYTES,
    chunk_bytes: int = COPY_CHUNK_BYTES,
    workers: int = 8,
    keep_experts: bool = False,
) -> dict[str, Planned]:
    """Write converted shards and, last, the HF index that marks the directory complete."""
    plan = plan_prime_conversion(Path(source_path), ops, keep_experts=keep_experts)
    shards = _assign_shards(plan, shard_target_bytes)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    names = [f"model-{index + 1:05d}-of-{len(shards):05d}.safetensors" for index in range(len(shards))]
    total = sum(planned.nbytes for planned in plan.values())
    logger.info(
        "streaming %d tensors (%.1f GiB) into %d shards at %s", len(plan), total / 1024**3, len(shards), output_path
    )

    def write(index: int) -> None:
        _write_shard(output_path / names[index], shards[index], plan, chunk_bytes)
        logger.info("wrote %s (%d tensors)", names[index], len(shards[index]))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(write, range(len(shards))))

    weight_map = {name: names[index] for index, shard in enumerate(shards) for name in shard}
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (output_path / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    return plan
