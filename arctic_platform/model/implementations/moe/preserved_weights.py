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

"""Immutable, lazily read source tensors shared by inactive prediction-layer exporters."""

from __future__ import annotations

import json
import os
import stat
from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open


@dataclass(frozen=True)
class PreservedWeight:
    name: str
    path: Path
    identity: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str
    index_identity: tuple[Path, tuple[int, ...]] | None = None


def file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size, metadata.st_mtime_ns)


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint index key: {key}")
        result[key] = value
    return result


def capture_original_weights(
    source_path: Path, expected: set[str] | None, *, prefixes: tuple[str, ...] = ("mtp.",)
) -> list[PreservedWeight]:
    """Inspect headers through held regular-file descriptors, without retaining tensor payloads."""
    preserved = []
    with ExitStack() as stack:
        def hold(path: Path) -> int:
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            stack.callback(os.close, descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"original checkpoint source is not a regular file: {path}")
            if file_identity(path.stat()) != file_identity(metadata):
                raise ValueError(f"original checkpoint source changed while opening: {path}")
            return descriptor

        index_path = source_path / "model.safetensors.index.json"
        index_identity = None
        indexed: dict[Path, set[str]] | None = None
        if os.path.lexists(index_path):
            index_fd = hold(index_path)
            index_identity = (index_path, file_identity(os.fstat(index_fd)))
            with os.fdopen(os.dup(index_fd), encoding="utf-8") as stream:
                index = json.load(stream, object_pairs_hook=unique_json_object)
            mapping = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(mapping, dict) or not all(isinstance(name, str) for name in mapping):
                raise ValueError("original checkpoint index requires a weight_map object")
            names = {name for name in mapping if name.startswith(prefixes)}
            if expected is not None and names != expected:
                raise ValueError(
                    f"original prediction index mismatch: missing={sorted(expected - names)}, "
                    f"unexpected={sorted(names - expected)}"
                )
            indexed = {}
            for name in sorted(names):
                filename = mapping[name]
                if (
                    not isinstance(filename, str) or not filename or "\\" in filename
                    or Path(filename).is_absolute() or ".." in Path(filename).parts
                    or Path(filename).suffix != ".safetensors"
                ):
                    raise ValueError(f"unsafe original prediction shard path for {name}: {filename!r}")
                indexed.setdefault(source_path / filename, set()).add(name)

        names = set()
        for shard in sorted(indexed if indexed is not None else source_path.glob("*.safetensors")):
            # Hold only the current shard, so an unindexed checkpoint does not exhaust file descriptors.
            with ExitStack() as shard_stack:
                descriptor = os.open(shard, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
                shard_stack.callback(os.close, descriptor)
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(f"original checkpoint source is not a regular file: {shard}")
                identity = file_identity(metadata)
                if file_identity(shard.stat()) != identity:
                    raise ValueError(f"original checkpoint source changed while opening: {shard}")
                with safe_open(f"/proc/self/fd/{descriptor}", framework="pt", device="cpu") as reader:
                    shard_names = {name for name in reader.keys() if name.startswith(prefixes)}
                    if indexed is not None and shard_names != indexed[shard]:
                        raise ValueError(f"original prediction shard does not match checkpoint index: {shard}")
                    for name in sorted(shard_names):
                        if name in names:
                            raise ValueError(f"duplicate original prediction tensor: {name}")
                        names.add(name)
                        source = reader.get_slice(name)
                        preserved.append(
                            PreservedWeight(name, shard, identity, tuple(source.get_shape()), source.get_dtype(), index_identity)
                        )
                if file_identity(os.fstat(descriptor)) != identity or file_identity(shard.stat()) != identity:
                    raise ValueError(f"original checkpoint source changed during capture: {shard}")
        if expected is not None and names != expected:
            raise ValueError(
                f"original prediction inventory mismatch: missing={sorted(expected - names)}, "
                f"unexpected={sorted(names - expected)}"
            )
        if index_identity is not None and (
            file_identity(os.fstat(index_fd)) != index_identity[1] or file_identity(index_path.stat()) != index_identity[1]
        ):
            raise ValueError("original prediction index changed during capture")
        for weight in preserved:
            if file_identity(weight.path.stat()) != weight.identity:
                raise ValueError(f"original checkpoint source changed during capture: {weight.path}")
    return preserved


@contextmanager
def open_preserved_weight(weight: PreservedWeight):
    """Pin and check the captured file generation around every individual tensor read."""
    descriptor = os.open(weight.path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        def check():
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError(f"Preserved prediction shard is not a regular file: {weight.path}")
            if (
                file_identity(os.fstat(descriptor)) != weight.identity
                or file_identity(weight.path.stat()) != weight.identity
            ):
                raise RuntimeError(f"Preserved prediction shard changed: {weight.path}")
            if weight.index_identity is not None:
                path, identity = weight.index_identity
                if file_identity(path.stat()) != identity:
                    raise RuntimeError("Preserved prediction index changed")

        check()
        with safe_open(f"/proc/self/fd/{descriptor}", framework="pt", device="cpu") as reader:
            source = reader.get_slice(weight.name)
            if tuple(source.get_shape()) != weight.shape or source.get_dtype() != weight.dtype:
                raise RuntimeError(f"Preserved prediction tensor changed layout or dtype: {weight.name}")
            yield reader
            check()
    finally:
        os.close(descriptor)


def read_preserved_weight(weight: PreservedWeight):
    with open_preserved_weight(weight) as reader:
        tensor = reader.get_tensor(weight.name)
        if tuple(tensor.shape) != weight.shape:
            raise RuntimeError(f"Preserved prediction tensor changed shape: {weight.name}")
    return tensor
