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

"""Opt-in byte manifests for the tensors handed to the native weight sender."""


from __future__ import annotations

import functools
import hashlib
import inspect
import math
import os
from contextlib import contextmanager
from pathlib import Path

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


@contextmanager
def maybe_observe_weight_send(worker, weights):
    """Observe a production-owned send without calling the transport."""
    evidence = {}
    if not enabled():
        yield evidence
        return
    sender = worker._weight_sender
    parameters = list(worker.engine.module.named_parameters())
    if any(hasattr(value, "ds_id") for _, value in parameters):
        raise ValueError("weight-sync evidence does not admit post-gather ZeRO3 tensor views")
    if any(getattr(value, "allreduce", True) is False for _, value in parameters):
        raise ValueError("a single sender cannot prove complete expert-parallel trainer holdings")
    origins = source_files(type(sender).send, type(worker).get_weights)
    manifest = {}
    for name, value in weights:
        if not isinstance(name, str) or not name or name in manifest:
            raise ValueError("weight-sync names must be nonempty and unique")
        manifest[name] = tensor_manifest(value)
    verify_sources(origins)
    yield evidence
    verify_sources(origins)
    if manifest != {name: tensor_manifest(value) for name, value in weights}:
        raise RuntimeError("actually sent tensors changed during weight sync")
    evidence.update(
        sent_manifest=manifest,
        trainer_parameters={
            "numel": sum(value.numel() for _, value in parameters),
            "count": len(parameters),
        },
        evidence_sources=origins,
    )


def maybe_annotate_weight_sync_result(result, evidence):
    if not evidence:
        return result
    if not isinstance(result, dict):
        raise TypeError("actual weight sender did not return a dictionary")
    return {**result, **evidence}
