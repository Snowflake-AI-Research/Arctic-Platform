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

"""Opt-in native broadcast inventories and receiver verification."""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

VERIFY_ENV = "ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY"

_MANIFEST_CHUNK_BYTES = 8 << 20


def enabled() -> bool:
    return os.environ.get(VERIFY_ENV) == "1"


def _manifest_chunks(value, max_elements):
    """Yield bounded views in the tensor's logical row-major order."""
    if value.numel() <= max_elements:
        yield value
        return
    row_elements = math.prod(value.shape[1:])
    if row_elements <= max_elements:
        rows = max(1, max_elements // row_elements)
        for start in range(0, value.shape[0], rows):
            yield value[start : start + rows]
    else:
        for index in range(value.shape[0]):
            yield from _manifest_chunks(value[index], max_elements)


def _equal_element_bytes(left, right):
    """Compare whole element bit patterns, including NaN payloads and signed zero."""
    import torch

    if left.device.type == right.device.type == "cpu":
        import numpy as np

        dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(left.shape[-1])
        if dtype is None:
            equal = (left.numpy() == right.numpy()).all(axis=-1)
        else:
            equal = left.numpy().view(dtype).reshape(-1) == right.numpy().view(dtype).reshape(-1)
        return torch.from_numpy(equal)
    dtype = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(left.shape[-1])
    if dtype is None:
        return (left == right).all(dim=-1)
    return left.view(dtype).reshape(-1) == right.view(dtype).reshape(-1)


def tensor_manifest(value) -> dict:
    """Hash exact logical tensor bytes without a tensor-sized staging allocation."""
    import torch

    if value.layout != torch.strided or value.device.type == "meta":
        raise ValueError("weight-sync manifest requires materialized strided tensors")
    digest = hashlib.sha256()
    elements = max(1, _MANIFEST_CHUNK_BYTES // value.element_size())
    for part in _manifest_chunks(value.detach(), elements):
        flat = part.contiguous().reshape(-1)
        # A singleton can be contiguous with a nonunit stride, which a dtype view rejects.
        if flat.stride(0) != 1:
            flat = flat.clone(memory_format=torch.contiguous_format)
        raw = flat.view(torch.uint8).cpu().numpy()
        digest.update(memoryview(raw).cast("B"))
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "numel": value.numel(),
        "sha256": digest.hexdigest(),
    }


def checkpoint_shard_rows(rows: int, parts: int) -> list[tuple[int, int]]:
    """Row ranges of the ``parts`` shards a checkpoint stores a ``rows``-row table in.

    vLLM's Qwen3.8 n-gram loader places checkpoint shard ``i`` at row ``i * ceil(rows / parts)`` and refuses every
    other shard shape (vllm 0.32.0, ``models/qwen4_exp/nvidia/ngram_embedding.py:450-464``), so a checkpoint a sampler
    loaded stores the table in exactly these ranges.
    """
    size = -(-rows // parts)
    return [(start, min(start + size, rows)) for start in range(0, rows, size)]


def checkpoint_shard_manifests(value, start: int, rows: int, parts: int) -> dict[str, dict | None]:
    """Manifest of each checkpoint shard overlapping ``value``, rows ``start`` onward of a ``rows``-row table.

    Keyed by shard index. A shard ``value`` holds only part of cannot be hashed without rows held elsewhere, so it
    maps to ``None``.
    """
    stop = start + value.shape[0]
    return {
        str(index): tensor_manifest(value[low - start : high - start]) if start <= low and high <= stop else None
        for index, (low, high) in enumerate(checkpoint_shard_rows(rows, parts))
        if low < stop and start < high
    }


def source_files(*objects) -> dict[str, str]:
    """Hash the actual selected callable origins, including partials."""
    result = {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for obj in objects:
        while isinstance(obj, functools.partial):
            obj = obj.func
        path = inspect.getsourcefile(obj)
        if path is None:
            raise ValueError("weight loader has no inspectable source origin")
        actual = Path(path).resolve(strict=True)
        result[str(actual)] = hashlib.sha256(actual.read_bytes()).hexdigest()
    return result


def verify_sources(pins: dict[str, str]) -> None:
    for path, expected in pins.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"weight-sync source changed: {path}")


def maybe_annotate_weight_sync_result(result, evidence):
    if not evidence:
        return result
    if not isinstance(result, dict):
        raise TypeError("actual weight sender did not return a dictionary")
    return {**result, **evidence}


def maybe_send_broadcast_weights(sender, weights, send):
    """Keep the native send and result untouched unless weight-sync evidence is enabled."""
    with maybe_observe_broadcast_send(sender, weights, send) as (stream, evidence):
        result = send(stream)
    return maybe_annotate_weight_sync_result(result, evidence)


def _storage(value):
    return (str(value.device), value.untyped_storage().data_ptr(), value.untyped_storage().nbytes())


class ShadowVerifier:
    """Replay supported ATen loaders into private storages, never into live NaNs.

    Unknown/custom operations and writes outside registered parameters,
    complete persistent integer buffers or newly allocated temporaries fail
    before dispatch. Shared parameter views require one complete backing owner.
    Nonfloating parameters are deliberately unsupported. Only the FP8 quantization operations bound by
    :func:`maybe_bind_fp8_updater` are admitted beside ATen, with their schema-declared writes checked the same way.

    A storage's private backing is allocated when the replay first writes or reads it and is compared, recorded and
    freed as soon as every element has been written, so only the storages being written at once are resident and
    ``max_shadow_bytes`` bounds that resident sum. A completed storage written again is reopened from the live bytes
    it was verified equal to, after rehashing them; completed live bytes that change by any other route are reported
    in ``changed_after_verification``. Inactive partial CUDA shadows are held on CPU between received tensors;
    replay always restores them to the original device. Byte comparisons use bounded CPU chunks.
    """

    def __init__(self, model, *, max_shadow_bytes: int):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_map

        self.model = model
        self.parameters = dict(model.named_parameters())
        self.integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        self.buffer_inventory = self._buffer_inventory()
        self.buffers = {
            name: value
            for name, value in model.named_buffers()
            if name in self.buffer_inventory and self.buffer_inventory[name][-1]
        }
        self.aliases = {name: id(value) for name, value in model.named_parameters(remove_duplicate=False)}
        self.layouts = {
            name: (tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.dtype, value.device)
            for name, value in self.parameters.items()
        }
        self.storage_keys = {}
        self.storage_names: dict[tuple[str, int, int], set[str]] = {}
        storage_owners = {}
        if not self.parameters:
            raise ValueError("weight-sync model has no parameters")
        for name, value in self.parameters.items():
            if (
                value.device.type == "meta"
                or value.layout != torch.strided
                or not value.is_floating_point()
                or value.numel() == 0
                or value.storage_offset() < 0
                or any(stride < 0 for stride in value.stride())
            ):
                raise ValueError(f"unsupported shadow parameter layout: {name}")
            # Increasing strides prove distinct element addresses without a tensor-sized index map.
            span = 1
            for stride, size in sorted(
                (stride, size) for size, stride in zip(value.shape, value.stride()) if size > 1
            ):
                if stride < span:
                    raise ValueError(f"unsupported overlapping shadow parameter layout: {name}")
                span += (size - 1) * stride
            key = _storage(value)
            if (value.storage_offset() + span) * value.element_size() > key[2]:
                raise ValueError(f"unsupported shadow parameter storage bounds: {name}")
            self.storage_keys[name] = key
            self.storage_names.setdefault(key, set()).add(name)
            if value.storage_offset() == 0 and value.numel() * value.element_size() == key[2]:
                if key in storage_owners:
                    raise ValueError("shared parameter storages require one complete backing owner")
                storage_owners[key] = name
        for key, names in self.storage_names.items():
            if key not in storage_owners:
                raise ValueError("unsupported shared parameter storage without a complete backing owner")
            owner = self.parameters[storage_owners[key]]
            if any(self.parameters[name].dtype != owner.dtype for name in names):
                raise ValueError("unsupported reinterpretation of shared parameter storage")
        buffer_storages: dict[tuple[str, int, int], list[str]] = {}
        for name, value in model.named_buffers(remove_duplicate=False):
            if value.layout == torch.strided and value.device.type != "meta":
                key = _storage(value)
                if value.dtype in self.integer_dtypes and name not in self.buffers and key in self.storage_names:
                    raise ValueError(f"unsupported shadow buffer storage or alias: {name}")
                buffer_storages.setdefault(key, []).append(name)
        for name, value in self.buffers.items():
            if (
                value.device.type == "meta"
                or value.layout != torch.strided
                or value.numel() == 0
                or not value.is_contiguous()
                or value.storage_offset() != 0
                or not self.buffer_inventory[name][-1]
            ):
                raise ValueError(f"unsupported shadow buffer layout or persistence: {name}")
            key = _storage(value)
            if (
                value.numel() * value.element_size() != key[2]
                or key in self.storage_names
                or buffer_storages[key] != [name]
            ):
                raise ValueError(f"unsupported shadow buffer storage or alias: {name}")
            self.storage_keys[name] = key
            self.storage_names[key] = {name}
            storage_owners[key] = name
        self.registered = {**self.parameters, **self.buffers}
        self.owners = storage_owners
        if type(max_shadow_bytes) is not int or max_shadow_bytes <= 0:
            raise ValueError("max_shadow_bytes must be a positive integer")
        self.max_shadow_bytes = max_shadow_bytes
        self.sources = source_files(
            model.load_weights,
            TorchDispatchMode,
            tree_map,
            *[value.weight_loader for value in self.parameters.values() if hasattr(value, "weight_loader")],
        )
        self.padding = self._projection_padding()
        for entry in self.padding.values():
            self.sources.update(source_files(type(entry["module"]), type(entry["projection"])))
        verify_sources(self.sources)
        self.native_operations: set = set()
        self.native_schemas: dict[str, str] = {}
        self.backings: dict[tuple[str, int, int], Any] = {}
        self.host_backings: set[tuple[str, int, int]] = set()
        self.shadow_keys: dict[tuple[str, int, int], tuple[str, int, int]] = {}
        self.written_numel: dict[tuple[str, int, int], int] = {}
        self.completed: dict[tuple[str, int, int], dict[str, dict]] = {}
        self.destinations: dict[str, dict[str, int]] = {}
        self.changed: set[str] = set()
        self.resident_bytes = 0
        self.peak_bytes = 0
        self.received: dict[str, dict] = {}
        self.touched: set[str] = set()

    def _fill(self, name):
        """Bytes of the value a never-written element holds; compared bytewise, so float8 needs no NaN kernels."""
        import torch

        value = torch.tensor([float("nan") if name in self.parameters else 0.0]).to(self.registered[name].dtype)
        return value.view(torch.uint8)

    def _view(self, name):
        value = self.registered[name]
        return self.backings[self.storage_keys[name]].as_strided(value.shape, value.stride(), value.storage_offset())

    def _reserve(self, extra):
        if self.resident_bytes + extra > self.max_shadow_bytes:
            raise ValueError(
                f"shadow allocation {self.resident_bytes + extra} exceeds the explicit {self.max_shadow_bytes} byte "
                f"limit with {len(self.backings)} partially written storages resident"
            )

    def _open(self, key, *, from_live):
        import torch

        owner = self.registered[self.owners[key]]
        self._reserve(key[2])
        device = "cpu" if key in self.host_backings else owner.device
        try:
            raw = torch.empty(key[2], dtype=torch.uint8, device=device)
        except torch.OutOfMemoryError:
            if owner.device.type != "cuda":
                raise
            self.host_backings.add(key)
            raw = torch.empty(key[2], dtype=torch.uint8, device="cpu")
        with torch.no_grad():
            if from_live:
                raw.copy_(owner.detach().as_strided((owner.numel(),), (1,), 0).view(torch.uint8))
            else:
                fill = self._fill(self.owners[key]).to(raw.device)
                raw.view(-1, fill.numel()).copy_(fill.expand(key[2] // fill.numel(), -1))
        backing = raw.view(owner.dtype)
        self.backings[key] = backing
        self.shadow_keys[_storage(backing)] = key
        self.resident_bytes += key[2]
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        written = owner.numel() if from_live else self.written_numel.get(key, 0)
        for name in self.storage_names[key].intersection(self.padding):
            start = self.padding[name]["start"]
            with torch.no_grad():
                self._view(name)[start:].copy_(self.parameters[name][start:])
            written += 0 if from_live else self.parameters[name][start:].numel()
        self.written_numel[key] = written
        return backing

    def _reopen(self, key):
        """A completed storage equals its verified live bytes, unless something unobserved changed them."""
        digests = self.completed.pop(key)
        self.changed.update(
            name for name, digest in digests.items() if tensor_manifest(self.registered[name]) != digest
        )
        return self._open(key, from_live=True)

    def _move_backing(self, key, device):
        import torch

        backing = self.backings[key]
        if backing.device == device or (key in self.host_backings and torch.device(device).type == "cuda"):
            return backing
        try:
            moved = backing.to(device)
        except torch.OutOfMemoryError:
            if torch.device(device).type != "cuda":
                raise
            self.host_backings.add(key)
            return backing
        del self.shadow_keys[_storage(backing)]
        self.backings[key] = moved
        self.shadow_keys[_storage(moved)] = key
        return moved

    def _offload_pending(self, held):
        for key, backing in list(self.backings.items()):
            if backing.device.type == "cuda" and _storage(backing) not in held:
                self._move_backing(key, "cpu")

    def _shadow_chunks(self, name):
        """Yield bounded shadow bytes and coverage without reading live tensor bytes."""
        import torch

        shadow = self._view(name)
        elements = max(1, _MANIFEST_CHUNK_BYTES // shadow.element_size())
        fill = self._fill(name)
        offset = 0
        padding_start = self.padding[name]["start"] * math.prod(shadow.shape[1:]) if name in self.padding else None
        for expected in _manifest_chunks(shadow, elements):
            expected = expected.cpu().contiguous().reshape(-1)
            if expected.stride(0) != 1:
                expected = expected.clone(memory_format=torch.contiguous_format)
            expected_bytes = expected.view(torch.uint8).reshape(-1, shadow.element_size())
            uncovered = (
                _equal_element_bytes(expected_bytes, fill)
                if name in self.parameters
                else torch.zeros(expected.numel(), dtype=torch.bool)
            )
            if padding_start is not None:
                uncovered[max(0, padding_start - offset) :] = False
            yield expected_bytes, uncovered
            offset += expected.numel()

    def _comparison_chunks(self, name):
        """Compare exact live bytes with the covered shadow bytes in bounded CPU chunks."""
        import numpy as np
        import torch

        live = self.registered[name].detach()
        elements = max(1, _MANIFEST_CHUNK_BYTES // live.element_size())
        for actual, (expected_bytes, uncovered) in zip(_manifest_chunks(live, elements), self._shadow_chunks(name)):
            actual = actual.cpu().contiguous().reshape(-1)
            if actual.stride(0) != 1:
                actual = actual.clone(memory_format=torch.contiguous_format)
            actual_bytes = actual.view(torch.uint8).reshape(-1, live.element_size())
            equal = _equal_element_bytes(actual_bytes, expected_bytes).numpy()
            yield uncovered, torch.from_numpy(np.logical_not(equal))

    def _uncovered_count(self, name):
        import numpy as np

        return sum(int(np.count_nonzero(uncovered.numpy())) for _, uncovered in self._shadow_chunks(name))

    def _complete(self, key, *, digest=True):
        """Compare, record and free one storage that the loader can no longer leave partially written."""
        import numpy as np

        for name in sorted(self.storage_names[key]):
            shadow = self._view(name)
            if name in self.padding and tensor_manifest(shadow[self.padding[name]["start"] :]) != (
                self.padding[name]["before"]
            ):
                raise RuntimeError(f"shadow loader changed unused HC projection padding: {name}")
            entry = self.destinations.setdefault(name, {"uncovered": 0, "mismatched": 0})
            for uncovered, byte_diff in self._comparison_chunks(name):
                coverage = uncovered.numpy()
                entry["uncovered"] += int(np.count_nonzero(coverage))
                entry["mismatched"] += int(np.count_nonzero(byte_diff.numpy() & ~coverage))
        if digest:
            self.completed[key] = {name: tensor_manifest(self.registered[name]) for name in self.storage_names[key]}
        self._free(key)

    def _free(self, key):
        backing = self.backings.pop(key)
        del self.shadow_keys[_storage(backing)]
        self.written_numel.pop(key, None)
        self.resident_bytes -= key[2]

    def _maybe_complete(self, key):
        owner = self.owners[key]
        if (
            key in self.backings
            and self.written_numel.get(key, 0) >= self.registered[owner].numel()
            and self._uncovered_count(owner) == 0
        ):
            self._complete(key)

    def _projection_padding(self):
        """Only the installed Qwen HC loader's unused trailing rows are padding."""
        result = {}
        origin = "vllm.models.qwen4_exp.nvidia.hyperconnection"
        for prefix, module in self.model.named_modules():
            if type(module).__module__ != origin or type(module).__name__ != "GatedResidual":
                continue
            if type(module) is not importlib.import_module(origin).GatedResidual:
                raise ValueError("HC padding requires the actual installed GatedResidual class")
            if module.use_combine is False:
                continue
            rank, count, hidden, padding = module.lora_rank, module.hc_count, module.hidden_size, module.pad_size
            if (
                module.use_combine is not True
                or any(type(size) is not int or size <= 0 for size in (rank, count, hidden))
                or type(padding) is not int
                or padding != (-(rank + count)) % 16
            ):
                raise ValueError("invalid HC projection padding geometry")
            projection = module.input_mix_weight_down_block_inject
            sizes = [rank, count] + ([padding] if padding else [])
            name = f"{prefix}." if prefix else ""
            name += "input_mix_weight_down_block_inject.weight"
            value = self.parameters.get(name)
            if (
                value is None
                or value is not projection.weight
                or list(projection.output_sizes) != sizes
                or tuple(value.shape) != (sum(sizes), count * hidden)
                or not value.is_contiguous()
                or value.storage_offset() != 0
                or self.storage_names[self.storage_keys[name]] != {name}
            ):
                raise ValueError("invalid HC projection padding storage or output partitions")
            if padding:
                result[name] = dict(
                    module=module,
                    projection=projection,
                    geometry=(rank, count, hidden, padding),
                    start=rank + count,
                    before=tensor_manifest(value[rank + count :]),
                )
        return result

    def _buffer_inventory(self):
        import torch

        persistent = {
            f"{prefix}.{name}" if prefix else name
            for prefix, module in self.model.named_modules(remove_duplicate=False)
            for name in module._buffers
            if name not in module._non_persistent_buffers_set
        }
        return {
            name: (
                id(value),
                _storage(value) if value.layout == torch.strided and value.device.type != "meta" else None,
                tuple(value.shape),
                tuple(value.stride()) if value.layout == torch.strided else None,
                value.storage_offset() if value.layout == torch.strided else None,
                value.dtype,
                value.device,
                name in persistent,
            )
            for name, value in self.model.named_buffers(remove_duplicate=False)
            if value.dtype in self.integer_dtypes
        }

    def _check_inventory(self):
        current = dict(self.model.named_parameters())
        aliases = {name: id(value) for name, value in self.model.named_parameters(remove_duplicate=False)}
        if (
            aliases != self.aliases
            or set(current) != set(self.parameters)
            or any(
                current[name] is not value
                or _storage(value) != self.storage_keys[name]
                or (tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.dtype, value.device)
                != self.layouts[name]
                for name, value in self.parameters.items()
            )
        ):
            raise RuntimeError("weight loader changed the registered parameter identities or storage")
        if self._buffer_inventory() != self.buffer_inventory:
            raise RuntimeError("weight loader changed the registered buffer inventory or identities")
        for name, entry in self.padding.items():
            module = entry["module"]
            rank, count, hidden, padding = entry["geometry"]
            if (
                module.use_combine is not True
                or (module.lora_rank, module.hc_count, module.hidden_size, module.pad_size) != entry["geometry"]
                or module.input_mix_weight_down_block_inject is not entry["projection"]
                or list(entry["projection"].output_sizes) != [rank, count, padding]
                or tensor_manifest(self.parameters[name][entry["start"] :]) != entry["before"]
            ):
                raise RuntimeError(f"weight loader changed unused HC projection padding: {name}")

    @contextmanager
    def observe_load(self, name, value, *, loader, staging=None):
        """``staging`` is a live and a replay callable mapping held staging storages to their eventual destinations.

        A received tensor that only fills staging reaches those destinations; they stay uncovered until a later
        tensor completes the staging and the loader writes them.
        """
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten
        from torch.utils._pytree import tree_map

        if not isinstance(name, str) or not name or name in self.received:
            raise ValueError("received tensor names must be nonempty and unique")
        self._check_inventory()
        verify_sources(self.sources)
        self.sources.update(source_files(loader))
        verify_sources(self.sources)
        entry = tensor_manifest(value)
        verifier = self

        class Observe(TorchDispatchMode):
            def __init__(self, shadow, persistent):
                super().__init__()
                self.shadow = shadow
                # Loader-owned staging that outlives one received tensor, held by this side's loader only.
                self.owned = set(persistent)
                if shadow:
                    self.owned.update(_storage(backing) for backing in verifier.backings.values())
                self.written = set()
                self.keys = set()
                self.opened = {}
                self.staged = set()
                self.reserved = set()

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func.namespace not in {"aten", "prims"} and func not in verifier.native_operations:
                    raise ValueError(f"unobservable native weight-loader operation: {func}")
                flat, _ = tree_flatten((args, kwargs))
                inputs = {_storage(t) for t in flat if isinstance(t, torch.Tensor)}

                def substitute(tensor):
                    if not isinstance(tensor, torch.Tensor):
                        return tensor
                    key = _storage(tensor)
                    if key not in verifier.storage_names or not self.shadow:
                        return tensor
                    backing = verifier.backings.get(key)
                    if backing is None:
                        self.opened[key] = key in verifier.completed
                        if key in verifier.completed:
                            backing = verifier._reopen(key)
                        else:
                            backing = verifier._open(key, from_live=False)
                        self.owned.add(_storage(backing))
                    elif backing.device != tensor.device:
                        backing = verifier._move_backing(key, tensor.device)
                        self.owned.add(_storage(backing))
                    if tensor.dtype != backing.dtype:
                        raise ValueError("reinterpretation of parameter storage is unsupported")
                    return backing.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())

                for index, argument in enumerate(func._schema.arguments):
                    if argument.alias_info is None or not argument.alias_info.is_write:
                        continue
                    actual = args[index] if index < len(args) else kwargs.get(argument.name)
                    tensors, _ = tree_flatten(actual)
                    for tensor in tensors:
                        if not isinstance(tensor, torch.Tensor):
                            continue
                        key = _storage(tensor)
                        target = key if key in verifier.storage_names else None
                        if self.shadow and target is None:
                            target = verifier.shadow_keys.get(key)
                        destination = verifier.storage_names.get(target) if target is not None else None
                        if destination is not None:
                            for buffer_name in destination.intersection(verifier.buffers):
                                buffer = verifier.buffers[buffer_name]
                                source = args[1] if len(args) > 1 else kwargs.get("src")
                                if (
                                    func is not torch.ops.aten.copy_.default
                                    or tensor.dtype != buffer.dtype
                                    or tensor.shape != buffer.shape
                                    or tensor.stride() != buffer.stride()
                                    or tensor.storage_offset() != 0
                                    or not isinstance(source, torch.Tensor)
                                    or source.shape != buffer.shape
                                    or source.dtype != buffer.dtype
                                ):
                                    raise ValueError("integer buffer writes require one complete, exact-dtype copy")
                            self.written.update(destination)
                            self.keys.add(target)
                            if self.shadow:
                                verifier.written_numel[target] = verifier.written_numel.get(target, 0) + tensor.numel()
                            elif target in verifier.completed:
                                verifier._reopen(target)
                            elif target not in verifier.backings and target not in self.reserved:
                                # The replay opens this storage; refuse before the live write, not after it.
                                self.reserved.add(target)
                                verifier._reserve(sum(reserved[2] for reserved in self.reserved))
                        elif key in self.owned:
                            self.staged.add(key)
                        else:
                            raise ValueError("weight loader writes an input, buffer or unowned storage")
                call_args = tree_map(substitute, args)
                call_kwargs = tree_map(substitute, kwargs)
                if self.shadow:
                    operands, _ = tree_flatten((call_args, call_kwargs))
                    host_views = [
                        tensor
                        for tensor in operands
                        if isinstance(tensor, torch.Tensor)
                        and verifier.shadow_keys.get(_storage(tensor)) in verifier.host_backings
                    ]
                    if host_views:
                        # Host shadows can replay metadata views and exact copies, never CUDA arithmetic on CPU.
                        views = {
                            torch.ops.aten.alias.default,
                            torch.ops.aten.as_strided.default,
                            torch.ops.aten.detach.default,
                            torch.ops.aten.expand.default,
                            torch.ops.aten.permute.default,
                            torch.ops.aten.select.int,
                            torch.ops.aten.slice.Tensor,
                            torch.ops.aten.split.Tensor,
                            torch.ops.aten.split_with_sizes.default,
                            torch.ops.aten.squeeze.default,
                            torch.ops.aten.squeeze.dim,
                            torch.ops.aten.t.default,
                            torch.ops.aten.transpose.int,
                            torch.ops.aten.unbind.int,
                            torch.ops.aten.unsqueeze.default,
                            torch.ops.aten.view.default,
                        }
                        if func is torch.ops.aten.copy_.default and call_args[0] is host_views[0]:
                            destination, source, *remaining = call_args
                            if source.dtype != destination.dtype:
                                key = verifier.shadow_keys[_storage(destination)]
                                device = verifier.registered[verifier.owners[key]].device
                                source = source.to(device=device, dtype=destination.dtype)
                            call_args = (destination, source, *remaining)
                        elif func not in views:
                            raise ValueError(f"CUDA shadow arithmetic requires GPU storage: {func}")
                result = func(*call_args, **call_kwargs)
                outputs, _ = tree_flatten(result)
                for tensor in outputs:
                    if isinstance(tensor, torch.Tensor):
                        key = _storage(tensor)
                        if key not in inputs and key not in verifier.storage_names:
                            self.owned.add(key)
                return result

        def pending(held, staged):
            return {name for key, names in held.items() if key in staged for name in names}

        live_staging, replay_staging = staging if staging is not None else (dict, dict)
        with torch.no_grad(), Observe(False, live_staging()) as live:
            yield
        live_pending = pending(live_staging(), live.staged)
        # Each replay receives private input too; loaders must not mutate the wire tensor.
        with torch.no_grad(), Observe(True, replay_staging()) as replay:
            loader(name, value.detach().clone())
        replay_pending = pending(replay_staging(), replay.staged)
        self._check_inventory()
        verify_sources(self.sources)
        if live.written != replay.written or live_pending != replay_pending or tensor_manifest(value) != entry:
            raise RuntimeError("shadow loader destination set or received bytes changed")
        for key in replay.keys | set(replay.opened):
            if key in replay.keys or replay.opened[key]:
                self._maybe_complete(key)
            elif key in self.backings:
                self._free(key)
        entry["destinations"] = sorted(live.written | live_pending)
        self.received[name] = entry
        self.touched.update(live.written | live_pending)
        self._offload_pending(replay_staging())

    def load(self, name, value, *, loader=None):
        """Debug-only convenience for a live load and its private shadow replay."""
        if loader is None:

            def loader(source_name, tensor):
                return self.model.load_weights([(source_name, tensor)])

        with self.observe_load(name, value, loader=loader):
            return loader(name, value)

    def _checkpoint_shards(self, names):
        """Checkpoint-shard manifests of the ``names`` that are vocabulary-parallel tables a checkpoint stores split.

        Such a table's owner declares ``split_ngram_parts``. The first rows of its parameter are its rows
        ``org_vocab_start_index`` to ``org_vocab_end_index``, the ones vLLM loads checkpoint shards into
        (vllm 0.32.0, ``models/qwen4_exp/common/ple.py:80-98``). Its checkpoint name inverts the one
        ``hf_to_vllm_mapper`` prefix rule that produces its name; a table no single rule produces is not reported.
        """
        prefixes = getattr(getattr(self.model, "hf_to_vllm_mapper", None), "orig_to_new_prefix", None) or {}
        result = {}
        for name in names:
            path, _, leaf = name.rpartition(".")
            owner = path.rpartition(".")[0]
            if leaf != "weight" or not owner:
                continue
            table = self.model.get_submodule(path)
            parts = getattr(self.model.get_submodule(owner), "split_ngram_parts", None)
            indices = getattr(table, "shard_indices", None)
            rules = [(old, new) for old, new in prefixes.items() if new is not None and name.startswith(new)]
            if (
                type(parts) is not int
                or parts < 1
                or indices is None
                or getattr(table, "weight", None) is not self.parameters[name]
                or len(rules) != 1
            ):
                continue
            ((old, new),) = rules
            start, stop = indices.org_vocab_start_index, indices.org_vocab_end_index
            result[name] = {
                "checkpoint_name": old + name.removeprefix(new),
                "shards": checkpoint_shard_manifests(
                    self.parameters[name][: stop - start], start, table.org_vocab_size, parts
                ),
            }
        return result

    def finish(self):
        self._check_inventory()
        verify_sources(self.sources)
        for digests in self.completed.values():
            self.changed.update(
                name for name, digest in digests.items() if tensor_manifest(self.registered[name]) != digest
            )
        for key in list(self.backings):
            if self.storage_names[key] & self.touched:
                self._complete(key, digest=False)
            else:
                self._free(key)
        # A destination reached only through staging the loader never wrote out is uncovered in full.
        destinations = {
            name: dict(self.destinations.get(name) or {"uncovered": self.registered[name].numel(), "mismatched": 0})
            for name in sorted(self.touched)
        }
        untouched = {name: value.numel() for name, value in self.parameters.items() if name not in self.touched}
        frozen = self._checkpoint_shards(untouched)
        return {
            "status": "ok",
            "received": self.received,
            "destinations": destinations,
            "unverifiable_destinations": [],
            "non_parameter_destinations": [],
            "changed_after_verification": sorted(self.changed),
            "untouched_parameters": untouched,
            "evidence_sources": self.sources,
            "admitted_native_operations": dict(self.native_schemas),
            "shadow_bytes": {"limit": self.max_shadow_bytes, "peak": self.peak_bytes},
            "preserved_padding": {
                name: dict(start_row=entry["start"], **entry["before"]) for name, entry in self.padding.items()
            },
            **({"checkpoint_shards": frozen} if frozen else {}),
        }


def global_inventory(records: list[dict]) -> dict:
    """Bind full rank coverage and count EP shards once, not once per DP replica."""
    if not isinstance(records, list) or not records:
        raise ValueError("collective inventory requires every actual training rank")
    world = len(records)
    by_rank = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or type(record.get("rank")) is not int
            or not 0 <= record["rank"] < world
            or type(record.get("world_size")) is not int
            or record["world_size"] != world
            or record["rank"] in by_rank
        ):
            raise ValueError("collective rank identities are incomplete or duplicated")
        by_rank[record["rank"]] = record
    result = {}
    for kind in ("parameters", "buffers"):
        first = by_rank[0].get(kind)
        if not isinstance(first, dict) or any(
            not isinstance(record.get(kind), dict) or set(record[kind]) != set(first) for record in records
        ):
            raise ValueError(f"{kind} inventories differ across training ranks")
        inventory = {}
        for name, base in first.items():
            if not isinstance(name, str) or not name or not isinstance(base, dict):
                raise ValueError("inventory tensor names must be nonempty strings")
            groups = set()
            sharded = base.get("ep_ranks") is not None
            for rank in range(world):
                entry = by_rank[rank][kind][name]
                if not isinstance(entry, dict):
                    raise ValueError("inventory tensor records must be dictionaries")
                shape = entry.get("shape")
                if (
                    not isinstance(shape, list)
                    or any(type(size) is not int or size <= 0 for size in shape)
                    or type(entry.get("numel")) is not int
                    or math.prod(shape) != entry["numel"]
                    or not isinstance(entry.get("dtype"), str)
                    or type(entry.get("requires_grad")) is not bool
                    or not isinstance(entry.get("sha256"), str)
                    or len(entry["sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in entry["sha256"])
                ):
                    raise ValueError("invalid actual tensor geometry")
                skipped = entry.get("skip_weight_sync", False)
                if type(skipped) is not bool or (skipped and (kind != "parameters" or entry["requires_grad"])):
                    raise ValueError(f"weight-sync skips require explicitly frozen parameters: {name}")
                if skipped != base.get("skip_weight_sync", False):
                    raise ValueError(f"weight-sync skip classification differs across ranks: {name}")
                if any(entry.get(key) != base.get(key) for key in ("shape", "numel", "dtype", "requires_grad")):
                    raise ValueError(f"rank-local tensor geometry differs: {name}")
                members = entry.get("ep_ranks")
                if (members is not None) != sharded:
                    raise ValueError(f"EP classification differs across ranks: {name}")
                if members is not None:
                    if (
                        not isinstance(members, list)
                        or not members
                        or any(type(member) is not int or not 0 <= member < world for member in members)
                        or rank not in members
                        or len(set(members)) != len(members)
                    ):
                        raise ValueError(f"invalid actual EP group ordering: {name}")
                    groups.add(tuple(members))
            if sharded:
                sizes = {len(group) for group in groups}
                covered: set[int] = set()
                for group in groups:
                    if covered.intersection(group):
                        raise ValueError(f"overlapping or reordered EP group reports: {name}")
                    for member in group:
                        if tuple(by_rank[member][kind][name]["ep_ranks"]) != group:
                            raise ValueError(f"EP group membership disagrees: {name}")
                    covered.update(group)
                if covered != set(range(world)) or len(sizes) != 1:
                    raise ValueError(f"EP groups do not partition the training allocation: {name}")
                # Rank zero's actual EP group owns one complete copy of this tensor.
                selected = by_rank[0][kind][name]["ep_ranks"]
                count = sum(by_rank[rank][kind][name]["numel"] for rank in selected)
            else:
                count = base["numel"]
            inventory[name] = {
                "local_shape": base["shape"],
                "local_numel": base["numel"],
                "global_numel": count,
                "dtype": base["dtype"],
                "requires_grad": base.get("requires_grad"),
                "ep_groups": [list(group) for group in sorted(groups)],
                "skip_weight_sync": base.get("skip_weight_sync", False),
            }
        result[kind] = inventory
    return result


_SEND = ContextVar("native_broadcast_weight_evidence", default=None)

_EXPORT = ContextVar("native_broadcast_export_inventory", default=None)

TRAINING_OPTION = "arctic_platform_weight_sync_verify"

SHADOW_MAX_BYTES_ENV = "ARCTIC_PLATFORM_WEIGHT_SYNC_SHADOW_MAX_BYTES"


def maybe_configure_broadcast_training(ds_config):
    """Consume the opt-in before DeepSpeed sees it in a dedicated trainer process."""
    if TRAINING_OPTION not in ds_config:
        return ds_config
    configured = ds_config[TRAINING_OPTION]
    if type(configured) is not bool:
        raise ValueError(f"{TRAINING_OPTION} must be a boolean")
    result = {key: value for key, value in ds_config.items() if key != TRAINING_OPTION}
    if configured:
        os.environ[VERIFY_ENV] = "1"
    return result


def maybe_configure_broadcast_sampling(sampling, *, verify):
    """Carry verification through the native sampler actor and vLLM child environments."""
    if not verify:
        return sampling
    vllm = deepcopy(sampling.vllm)
    extra_env = vllm.get("extra_env")
    if extra_env is not None and not isinstance(extra_env, dict):
        raise ValueError("weight-sync extra_env must be an object")
    extra_env = dict(extra_env or {})
    if extra_env.get(VERIFY_ENV, "1") != "1":
        raise ValueError("conflicting weight-sync verification environment")
    extra_env[VERIFY_ENV] = "1"
    limits = []
    for value in (extra_env.get(SHADOW_MAX_BYTES_ENV), os.environ.get(SHADOW_MAX_BYTES_ENV)):
        if value is None:
            continue
        try:
            limit = int(value) if isinstance(value, str) else 0
        except ValueError:
            limit = 0
        if limit <= 0:
            raise ValueError(f"{SHADOW_MAX_BYTES_ENV} must be a positive integer byte limit")
        limits.append(limit)
    if len(set(limits)) > 1:
        raise ValueError("conflicting weight-sync shadow byte limits")
    if limits:
        extra_env[SHADOW_MAX_BYTES_ENV] = str(limits[0])
    vllm["extra_env"] = extra_env
    return sampling.model_copy(update={"vllm": vllm}, deep=True)


@contextmanager
def weight_sync_verification_environment():
    """The legacy local client inherits this opt-in; never leak it to later checks."""
    previous = os.environ.get(VERIFY_ENV)
    os.environ[VERIFY_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(VERIFY_ENV, None)
        else:
            os.environ[VERIFY_ENV] = previous


@contextmanager
def maybe_observe_broadcast_send(sender, weights, send):
    """Wrap the stream; the caller still executes its native transport."""
    evidence = {}
    if not enabled():
        yield weights, evidence
        return
    if _SEND.get() is not None:
        raise RuntimeError("nested broadcast weight verification is unsupported")
    state = {"trainer_parameters": None, "trainer_inventory": None}
    sources = source_files(send, type(sender), maybe_observe_broadcast_send)
    manifest = {}
    wire_buffers = {}

    def observed():
        for name, value in weights:
            if not isinstance(name, str) or not name or name in manifest:
                raise ValueError("broadcast tensor names must be nonempty and unique")
            entry = tensor_manifest(value)
            manifest[name] = entry
            if any(_storage(value) == _storage(parameter) for parameter in state.get("skipped_objects", {}).values()):
                raise ValueError("broadcast unexpectedly sent explicitly skipped frozen storage")
            matches = [
                buffer_name
                for buffer_name, buffer in state.get("buffer_objects", {}).items()
                if (
                    value.untyped_storage()._cdata == buffer.untyped_storage()._cdata
                    and value.dtype == buffer.dtype
                    and value.shape == buffer.shape
                    and value.stride() == buffer.stride()
                    and value.storage_offset() == buffer.storage_offset()
                )
            ]
            if len(matches) > 1:
                raise ValueError("exported tensor aliases multiple registered trainer buffers")
            if matches:
                buffer_name = matches[0]
                if buffer_name in wire_buffers.values():
                    raise ValueError(f"registered trainer buffer sent more than once: {buffer_name}")
                source_entry = state["rank_records"][0]["buffers"][buffer_name]
                if any(entry[key] != source_entry[key] for key in ("shape", "dtype", "numel", "sha256")):
                    raise RuntimeError(f"registered trainer buffer changed before sending: {buffer_name}")
                if state["trainer_inventory"]["buffers"][buffer_name]["ep_groups"]:
                    raise ValueError("native buffer evidence does not admit rank-local EP views")
                wire_buffers[name] = buffer_name
            yield name, value
            if tensor_manifest(value) != entry:
                raise RuntimeError("actually sent broadcast tensor changed during transfer")

    token = _SEND.set(state)
    try:
        verify_sources(sources)
        yield observed(), evidence
        verify_sources(sources)
        if not manifest or state["trainer_parameters"] is None:
            raise RuntimeError("broadcast sender has no complete collective trainer inventory")
        sources.update(state["evidence_sources"])
        verify_sources(sources)
        if wire_buffers:
            _verify_trainer_replicas(
                state["rank_records"], state["trainer_inventory"], buffer_names=wire_buffers.values()
            )
        buffers = state["trainer_inventory"].get("buffers", {})
        if state.get("skipped_objects") and state.get("skipped_proof") is None:
            raise RuntimeError("skipped frozen weights have no all-rank export completion proof")
        evidence.update(
            sent_manifest=manifest,
            trainer_parameters=state["trainer_parameters"],
            trainer_buffers={
                "numel": sum(entry["global_numel"] for entry in buffers.values()),
                "count": len(buffers),
                "wire_numel": sum(buffers[name]["global_numel"] for name in wire_buffers.values()),
                "wire_sources": wire_buffers,
            },
            trainer_inventory=state["trainer_inventory"],
            evidence_sources=sources,
        )
        if state.get("skipped_proof") is not None:
            evidence["trainer_skipped_parameters"] = state["skipped_proof"]
    finally:
        _EXPORT.set(None)
        _SEND.reset(token)


def _explicitly_skipped_parameters(model):
    skipped_ids = {
        id(parameter)
        for module in model.modules()
        if getattr(module, "_dss_skip_weight_sync", False) is True
        for parameter in module.parameters(recurse=False)
    }
    return {
        name: value
        for name, value in model.named_parameters()
        if id(value) in skipped_ids or getattr(value, "_dss_skip_weight_sync", False) is True
    }


def _gather_broadcast_evidence(records, local):
    """CPU evidence must not queue NCCL work while another replica still exports GPU tensors."""
    import torch.distributed as dist

    group = dist.new_group(backend="gloo")
    try:
        dist.all_gather_object(records, local, group=group)
    finally:
        dist.destroy_process_group(group)


def maybe_cleanup_broadcast_export(function):
    """Release this generator's evidence even when a non-master export fails or closes."""

    @functools.wraps(function)
    def observed(model, *args, **kwargs):
        if not enabled():
            yield from function(model, *args, **kwargs)
            return
        try:
            yield from function(model, *args, **kwargs)
        finally:
            export = _EXPORT.get()
            if export is not None and export["model"] is model:
                _EXPORT.set(None)

    return observed


def maybe_bind_broadcast_trainer_inventory(model, *, is_master, is_zero3):
    """All exporter ranks bind actual EP holdings; count DP replicas only once."""
    if not enabled():
        return
    import torch.distributed as dist

    from arctic_platform.model.weight_export import iter_model_weights
    from arctic_platform.model.weight_export import weight_export_contract

    if is_zero3:
        raise ValueError("native broadcast evidence does not admit ZeRO3 tensor views")
    if not dist.is_initialized():
        raise RuntimeError("native broadcast evidence requires the actual training process group")
    rank, world = dist.get_rank(), dist.get_world_size()
    sources = source_files(
        maybe_bind_broadcast_trainer_inventory,
        type(model).named_parameters,
        type(model).named_buffers,
        dist.all_gather_object,
        dist.new_group,
        dist.destroy_process_group,
        _gather_broadcast_evidence,
        global_inventory,
        _verify_trainer_replicas,
        _explicitly_skipped_parameters,
        iter_model_weights,
    )
    contract = weight_export_contract(model)
    if contract is not None:
        sources.update(source_files(*(builder for builder in (contract.hf, contract.vllm) if builder is not None)))
    objects = {}
    local_inventory = {"parameters": {}, "buffers": {}}
    error = None
    try:
        if is_master != (rank == 0):
            raise ValueError("exporter master does not match the actual training rank")
        objects = {
            "parameters": dict(model.named_parameters()),
            "buffers": dict(model.named_buffers(remove_duplicate=False)),
        }
        parameter_storages = {_storage(value) for value in objects["parameters"].values()}
        buffer_storages = [_storage(value) for value in objects["buffers"].values()]
        if len(set(buffer_storages)) != len(buffer_storages):
            raise ValueError("registered trainer buffers have ambiguous storage aliases")
        if parameter_storages.intersection(buffer_storages):
            raise ValueError("registered trainer parameters and buffers share storage")
        skipped_ids = {id(value) for value in _explicitly_skipped_parameters(model).values()}
        for module in model.modules():
            if getattr(module, "_dss_skip_weight_sync", False) is True:
                sources.update(source_files(type(module)))
        for kind, tensors in objects.items():
            for name, value in tensors.items():
                if hasattr(value, "ds_id"):
                    raise ValueError("native broadcast evidence does not admit ZeRO3 tensor views")
                members = None
                if getattr(value, "allreduce", True) is False:
                    import deepspeed.utils.groups as groups

                    group_name = getattr(value, "group_name", None)
                    if not isinstance(group_name, str) or not group_name:
                        raise ValueError("expert shard has no actual named process group")
                    group = groups._get_expert_parallel_group(group_name)
                    members = list(dist.get_process_group_ranks(group))
                    sources.update(source_files(groups._get_expert_parallel_group, dist.get_process_group_ranks))
                local_inventory[kind][name] = {
                    **tensor_manifest(value),
                    "requires_grad": value.requires_grad,
                    "ep_ranks": members,
                    "skip_weight_sync": kind == "parameters" and id(value) in skipped_ids,
                }
        if not local_inventory["parameters"]:
            raise ValueError("actual trainer has no registered parameters")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    verify_sources(sources)
    local = {"rank": rank, "world_size": world, **local_inventory, "error": error}
    records = [None] * world
    _gather_broadcast_evidence(records, local)
    errors = [record.get("error") for record in records if isinstance(record, dict) and record.get("error")]
    if errors:
        raise ValueError("collective trainer inventory failed: " + "; ".join(errors))
    inventory = global_inventory(records)
    _verify_trainer_replicas(records, inventory)
    verify_sources(sources)
    skipped_objects = {
        name: value
        for name, value in objects["parameters"].items()
        if inventory["parameters"][name]["skip_weight_sync"]
    }
    _EXPORT.set(
        dict(
            model=model,
            objects=skipped_objects,
            storages={name: _storage(value) for name, value in skipped_objects.items()},
            local=local,
            records=records,
            inventory=inventory,
        )
    )
    state = _SEND.get()
    if is_master:
        if state is None or state["trainer_parameters"] is not None:
            raise RuntimeError("trainer inventory is not bound to exactly one active broadcast")
        state.update(
            trainer_parameters={
                "numel": sum(entry["global_numel"] for entry in inventory["parameters"].values()),
                "count": len(inventory["parameters"]),
            },
            trainer_inventory=inventory,
            buffer_objects=objects["buffers"],
            rank_records=sorted(records, key=lambda record: record["rank"]),
            evidence_sources=sources,
            skipped_objects=skipped_objects,
        )


def _skipped_checkpoint_shards(model, export, rank: int) -> dict:
    """This rank's checkpoint-shard manifests of the skipped tables its checkpoint stores split.

    A Qwen3.8 checkpoint stores each ``*.ple.ple_embedding.ngram_embedding.weight`` table in ``split_ngram_parts``
    shards, the tables its HF export splits (``model/implementations/qwen38/deepspeed_integration.py``). The export
    concatenates the rows of the EP replica holding rank 0 in process-group order, so only that replica's ranks
    report, each the shards overlapping its own rows, which follow the rows the bound inventory records for the members
    before it.
    """
    from arctic_platform.model.weight_export import hf_export_parameter_name
    from arctic_platform.model.weight_export import pretrained_config_of

    config = pretrained_config_of(model)
    parts = getattr(getattr(config, "text_config", config), "split_ngram_parts", None)
    if type(parts) is not int or parts < 1:
        return {}
    held = {record["rank"]: record["parameters"] for record in export["records"]}
    result = {}
    for name, value in export["objects"].items():
        checkpoint_name = hf_export_parameter_name(name)
        members = export["local"]["parameters"][name]["ep_ranks"] or [0]
        if (
            checkpoint_name is None
            or not checkpoint_name.endswith(".ple.ple_embedding.ngram_embedding.weight")
            or 0 not in members
            or rank not in members
        ):
            continue
        rows = [held[member][name]["shape"][0] for member in members]
        result[name] = dict(
            checkpoint_name=checkpoint_name,
            shards=checkpoint_shard_manifests(value, sum(rows[: members.index(rank)]), sum(rows), parts),
        )
    return result


def maybe_complete_broadcast_trainer_inventory(model, *, is_master):
    """Every export rank proves intentionally omitted frozen storage stayed unchanged.

    A rank of the exported EP replica also reports the checkpoint-shard manifests of its rows of a skipped table its
    checkpoint stores split (:func:`_skipped_checkpoint_shards`), so the sampler's unwritten copy can be compared.
    """
    if not enabled():
        return
    import torch.distributed as dist

    export = _EXPORT.get()
    _EXPORT.set(None)
    if export is None or export["model"] is not model:
        raise RuntimeError("export completion has no matching collective trainer inventory")
    if not export["objects"]:
        return
    error = None
    actual = {}
    shards = {}
    try:
        if is_master != (dist.get_rank() == 0):
            raise RuntimeError("skipped frozen export completion changed the master rank")
        current = dict(model.named_parameters())
        skipped = _explicitly_skipped_parameters(model)
        if set(skipped) != set(export["objects"]):
            raise RuntimeError("skipped frozen parameter classification changed")
        for name, value in export["objects"].items():
            if current.get(name) is not value or _storage(value) != export["storages"][name]:
                raise RuntimeError(f"skipped frozen parameter identity changed: {name}")
            actual[name] = tensor_manifest(value)
            before = export["local"]["parameters"][name]
            if value.requires_grad or any(actual[name][key] != before[key] for key in actual[name]):
                raise RuntimeError(f"skipped frozen parameter bytes or layout changed: {name}")
        shards = _skipped_checkpoint_shards(model, export, dist.get_rank())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    local = dict(rank=dist.get_rank(), world_size=dist.get_world_size(), parameters=actual, error=error)
    if shards:
        local["checkpoint_shards"] = shards
    completed = [None] * dist.get_world_size()
    _gather_broadcast_evidence(completed, local)
    if any(
        not isinstance(entry, dict)
        or type(entry.get("rank")) is not int
        or entry.get("world_size") != dist.get_world_size()
        or not isinstance(entry.get("parameters"), dict)
        for entry in completed
    ):
        raise RuntimeError("skipped frozen export completion has invalid rank records")
    by_rank = {entry["rank"]: entry for entry in completed}
    if len(by_rank) != dist.get_world_size() or set(by_rank) != set(range(dist.get_world_size())):
        raise RuntimeError("skipped frozen export completion has incomplete rank coverage")
    for before in export["records"]:
        after = by_rank[before["rank"]]
        if after["error"] is not None:
            raise RuntimeError("skipped frozen export completion failed: " + after["error"])
        if set(after["parameters"]) != set(export["objects"]):
            raise RuntimeError("skipped frozen export completion changed its parameter inventory")
        for name, value in after["parameters"].items():
            if any(value[key] != before["parameters"][name][key] for key in value):
                raise RuntimeError(f"skipped frozen export completion changed rank bytes: {name}")
    if is_master:
        state = _SEND.get()
        if state is None:
            raise RuntimeError("skipped frozen export completion has no active sender")
        state["skipped_proof"] = dict(schema="frozen-weight-sync-skip-v1", before=export["records"], after=completed)
        state["evidence_sources"].update(source_files(maybe_complete_broadcast_trainer_inventory))


def _verify_trainer_replicas(records, inventory, *, buffer_names=()):
    by_rank = {record["rank"]: record for record in records}
    selected = {
        "parameters": inventory["parameters"],
        "buffers": {name: inventory["buffers"][name] for name in buffer_names},
    }
    for kind, tensors in selected.items():
        for name, entry in tensors.items():
            groups = entry["ep_groups"] or [[rank] for rank in sorted(by_rank)]
            for shards in zip(*groups):
                if len({by_rank[rank][kind][name]["sha256"] for rank in shards}) != 1:
                    raise ValueError(f"actual trainer replica bytes disagree: {name}")


def maybe_admit_broadcast_receiver(receiver, *, engine_only):
    if not enabled():
        return
    if not engine_only and receiver.model_config.quantization not in (None, "fp8"):
        raise ValueError("quantized native broadcast shadow evidence supports only the FP8 in-place updater")


def maybe_begin_broadcast_load(receiver, model, engine):
    if not enabled():
        return None
    limit = int(os.environ.get(SHADOW_MAX_BYTES_ENV, str(1 << 30)))
    verifier = ShadowVerifier(model, max_shadow_bytes=limit)
    verifier.sources.update(source_files(maybe_begin_broadcast_load, type(engine).receive_weights))
    verify_sources(verifier.sources)
    return verifier


def maybe_bind_broadcast_loader(verifier, writer):
    if verifier is not None:
        verify_sources(verifier.sources)
        verifier.sources.update(source_files(type(writer)))
        verify_sources(verifier.sources)


def maybe_run_broadcast_load(loader, model, engine, verifier):
    """Pass debug evidence only when enabled, preserving ordinary loader arguments."""
    if verifier is None:
        return loader(model, engine)
    return loader(model, engine, verifier=verifier)


@contextmanager
def maybe_observe_broadcast_tensor(verifier, loader, name, value):
    if verifier is None:
        yield
        return
    with verifier.observe_load(name, value, loader=loader):
        yield


def maybe_load_broadcast_tensor(verifier, loader, name, value):
    with maybe_observe_broadcast_tensor(verifier, loader, name, value):
        return loader(name, value)


FP8_QUANT_OPERATIONS = (
    "_C",
    ("dynamic_scaled_fp8_quant", "static_scaled_fp8_quant", "dynamic_per_token_scaled_fp8_quant"),
)


def maybe_bind_fp8_updater(verifier, updater, make_replay):
    """Bind the live FP8 in-place updater and build its independent replay twin.

    The replay instance is fed every received tensor a second time; its own staging state accumulates fused-module
    shards across tensors, and its parameter writes land in the verifier's private backings.
    """
    if verifier is None:
        return None
    import torch
    from vllm import _custom_ops

    namespace, names = FP8_QUANT_OPERATIONS
    library = getattr(torch.ops, namespace)
    operations = {getattr(library, name).default for name in names if hasattr(library, name)}
    if not operations:
        raise ValueError("no vLLM FP8 quantization operation is registered for the quantized updater")
    for operation in operations:
        arguments = operation._schema.arguments
        if not any(argument.alias_info is not None and argument.alias_info.is_write for argument in arguments):
            raise ValueError(f"FP8 quantization operation declares no written argument: {operation}")
    replay = make_replay()
    if replay is updater or type(replay) is not type(updater) or updater.pending or replay.pending:
        raise ValueError("FP8 shadow replay needs a fresh independent updater of the live updater's class")
    verify_sources(verifier.sources)
    classes = [cls for cls in type(updater).__mro__ if cls is not object]
    verifier.sources.update(source_files(*classes, _custom_ops.scaled_fp8_quant, maybe_bind_fp8_updater))
    verify_sources(verifier.sources)
    verifier.native_operations.update(operations)
    verifier.native_schemas.update({str(operation): str(operation._schema) for operation in operations})
    return replay


def _fp8_staging(verifier, updater):
    def held():
        staged = [(buffer, (f"{module}.weight", f"{module}.weight_scale")) for module, buffer in updater._bufs.items()]
        staged.extend(
            (entry[part], (entry["destination"],))
            for entry in getattr(updater, "_dequant", {}).values()
            for part in ("weight", "scale")
            if part in entry
        )
        return {_storage(buffer): [name for name in names if name in verifier.parameters] for buffer, names in staged}

    return held


@contextmanager
def maybe_observe_fp8_tensor(verifier, updater, replay, name, value):
    if verifier is None:
        yield
        return
    staging = (_fp8_staging(verifier, updater), _fp8_staging(verifier, replay))
    with verifier.observe_load(name, value, loader=replay.feed, staging=staging):
        yield


def maybe_feed_fp8_tensor(verifier, updater, replay, name, value):
    with maybe_observe_fp8_tensor(verifier, updater, replay, name, value):
        return updater.feed(name, value)


def maybe_broadcast_receiver_result(result, verifier, *, rank):
    if verifier is None:
        return result
    return {
        **result,
        "rank": rank,
        "weight_sync_verification": {**verifier.finish(), "rank": rank},
    }


def maybe_collect_broadcast_verification(results, *, engine_only, llm, rank_offset):
    """Preserve native results and reject missing or duplicated sampler ranks."""
    if not enabled() or engine_only:
        return results
    expected_ranks = llm.vllm_config.parallel_config.world_size
    if (
        type(expected_ranks) is not int
        or expected_ranks < 1
        or type(rank_offset) is not int
        or not isinstance(results, list)
        or len(results) != expected_ranks
        or any(
            not isinstance(row, dict) or not isinstance(row.get("weight_sync_verification"), dict) for row in results
        )
        or any(type(row.get("rank")) is not int for row in results)
        or {row["rank"] for row in results} != set(range(rank_offset, rank_offset + expected_ranks))
        or any(
            type(row["weight_sync_verification"].get("rank")) is not int
            or row["weight_sync_verification"]["rank"] != row["rank"]
            for row in results
        )
    ):
        raise RuntimeError("native broadcast did not return every actual sampler rank's verification")
    return [
        {
            **results[0],
            "all_rank_weight_sync_verification": [row["weight_sync_verification"] for row in results],
        },
        *results[1:],
    ]


def maybe_lora_received_manifest(tensors):
    """Hash the received adapter tensors before the LoRA builder casts them or folds scaling into them in place."""
    if not enabled():
        return None
    return {name: tensor_manifest(value) for name, value in tensors.items()}


def _lora_manager_destinations(manager, adapter, module, factor, slot) -> list[str]:
    if module in manager.modules:
        candidates = [(module, slot)]
    elif slot is None:
        candidates = [
            (packed, index)
            for packed, members in getattr(manager, "packed_modules", {}).items()
            if packed in manager.modules
            for index, member in enumerate(members)
            if member == module
        ]
    else:
        candidates = []
    destinations = []
    for name, index in candidates:
        weights = manager._get_lora_layer_weights(adapter, name)
        value = getattr(weights, factor, None) if weights is not None else None
        if index is not None:
            value = value[index] if isinstance(value, list) and index < len(value) else None
        if value is not None:
            destinations.append(f"{name}.{factor}" if index is None else f"{name}.{factor}[{index}]")
    return destinations


def maybe_lora_receiver_result(result, received, built_factors, *, manager, lora_int_id, rank, world_size):
    """Record, per received adapter tensor, the LoRA-manager module factor this rank installed it into.

    A destination is a module of ``manager.modules`` (the modules ``activate_adapter`` hands to ``set_lora``) whose
    installed adapter holds the tensor's factor, either directly or in its slot of a packed module. A tensor the
    builder skipped or overwrote, or whose module the manager does not wrap, has no destination.
    """
    if received is None:
        return result
    adapter = manager.get_adapter(lora_int_id)
    entries = {}
    for name, manifest in received.items():
        built = built_factors.get(name)
        destinations = [] if built is None or adapter is None else _lora_manager_destinations(manager, adapter, *built)
        entries[name] = {**manifest, "destinations": destinations}
    verification = {
        "status": "ok",
        "weight_format": "lora",
        "rank": rank,
        "world_size": world_size,
        "received": entries,
    }
    return {**result, "rank": rank, "weight_sync_verification": verification}


@dataclass
class _LoRABuildEvidence:
    received: dict
    renamed: dict = field(default_factory=dict)
    factors: dict = field(default_factory=dict)


_LORA_BUILD_EVIDENCE: ContextVar[_LoRABuildEvidence | None] = ContextVar("lora_build_evidence", default=None)


def maybe_build_lora_model(builder, **kwargs):
    """Observe the native builder without changing its signature or tensor inputs."""
    if not enabled():
        return builder(**kwargs), None
    evidence = _LoRABuildEvidence(received=maybe_lora_received_manifest(kwargs["tensors"]))
    token = _LORA_BUILD_EVIDENCE.set(evidence)
    try:
        model = builder(**kwargs)
    finally:
        _LORA_BUILD_EVIDENCE.reset(token)
    return model, evidence


def maybe_record_lora_remapping(tensors, prefix, remap):
    evidence = _LORA_BUILD_EVIDENCE.get()
    if evidence is not None:
        evidence.renamed = {name: remap(name, prefix) for name in tensors}


def maybe_record_built_lora_factors(model, mapper, skip_prefixes):
    """Apply the native builder's parser, skip and last-write rules only in an observed build."""
    evidence = _LORA_BUILD_EVIDENCE.get()
    if evidence is None:
        return
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.utils import is_base_embedding_weights
    from vllm.lora.utils import parse_fine_tuned_lora_name

    claimed = {}
    for received, name in evidence.renamed.items():
        evidence.factors[received] = None
        if is_base_embedding_weights(name) or (skip_prefixes and LoRAModel._should_skip_module(name, skip_prefixes)):
            continue
        module, is_lora_a = parse_fine_tuned_lora_name(name, mapper)
        factor = "lora_a" if is_lora_a else "lora_b"
        weights = model.loras.get(module)
        if weights is None or getattr(weights, factor, None) is None:
            continue
        previous = claimed.get((module, factor))
        if previous is not None:
            evidence.factors[previous] = None
        claimed[module, factor] = received
        evidence.factors[received] = (module, factor, None)


def maybe_pack_fused_expert_loras(pack, model):
    evidence = _LORA_BUILD_EVIDENCE.get()
    if evidence is None:
        return pack(model)
    projections = {}
    for name in model.loras:
        for slot, projection in enumerate(("w1", "w2", "w3")):
            suffix = f".experts.{projection}"
            if name.endswith(suffix):
                projections[name] = (name[: -len(f".{projection}")], slot)
            elif name == f"experts.{projection}":
                projections[name] = ("experts", slot)
    packed = pack(model)
    for received, built in evidence.factors.items():
        if built is not None and built[0] in projections:
            module, slot = projections[built[0]]
            evidence.factors[received] = (module, built[1], slot)
    return packed


def maybe_lora_built_receiver_result(result, evidence, **kwargs):
    if evidence is None:
        return result
    return maybe_lora_receiver_result(result, evidence.received, evidence.factors, **kwargs)
