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

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in byte manifests and dispatcher-observed, isolated shadow loads."""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
from pathlib import Path

VERIFY_ENV = "ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY"


def enabled() -> bool:
    return os.environ.get(VERIFY_ENV) == "1"


def tensor_manifest(value) -> dict:
    import torch

    if value.layout != torch.strided or value.device.type == "meta":
        raise ValueError("weight-sync manifest requires materialized strided tensors")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "numel": value.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
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


def maybe_send_weights(worker, weights):
    """Send the existing list once; evidence is derived from those same tensors."""
    sender = worker._weight_sender
    if sender is None:
        return {"status": "not_initialized"}
    if not enabled():
        return sender.send(weights)
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
    result = sender.send(weights)
    verify_sources(origins)
    if manifest != {name: tensor_manifest(value) for name, value in weights}:
        raise RuntimeError("actually sent tensors changed during weight sync")
    if not isinstance(result, dict):
        raise TypeError("actual weight sender did not return a dictionary")
    return {
        **result,
        "sent_manifest": manifest,
        "trainer_parameters": {
            "numel": sum(value.numel() for _, value in parameters),
            "count": len(parameters),
        },
        "evidence_sources": origins,
    }


def _storage(value):
    return (str(value.device), value.untyped_storage().data_ptr(), value.untyped_storage().nbytes())


class ShadowVerifier:
    """Replay supported ATen loaders into private storages, never into live NaNs.

    Unknown/custom operations and writes outside registered parameters or
    newly allocated temporaries fail before dispatch. Shared parameter storages,
    nonfloating parameters and quantized loaders are deliberately unsupported.
    """

    def __init__(self, model, *, max_shadow_bytes: int):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_map

        self.model = model
        self.parameters = dict(model.named_parameters())
        self.aliases = {name: id(value) for name, value in model.named_parameters(remove_duplicate=False)}
        self.layouts = {
            name: (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
            for name, value in self.parameters.items()
        }
        self.storage_names: dict[tuple[str, int, int], str] = {}
        total = 0
        if not self.parameters:
            raise ValueError("weight-sync model has no parameters")
        for name, value in self.parameters.items():
            if (
                value.device.type == "meta"
                or value.layout != torch.strided
                or not value.is_floating_point()
                or not value.is_contiguous()
                or value.storage_offset() != 0
                or value.numel() == 0
                or value.untyped_storage().nbytes() != value.numel() * value.element_size()
            ):
                raise ValueError(f"unsupported shadow parameter layout: {name}")
            key = _storage(value)
            if key in self.storage_names:
                raise ValueError("shared parameter storages require an explicit supported loader")
            self.storage_names[key] = name
            total += value.numel() * value.element_size()
        if type(max_shadow_bytes) is not int or not 0 < total <= max_shadow_bytes:
            raise ValueError(f"shadow allocation {total} exceeds the explicit {max_shadow_bytes} byte limit")
        self.sources = source_files(
            model.load_weights,
            TorchDispatchMode,
            tree_map,
            *[value.weight_loader for value in self.parameters.values() if hasattr(value, "weight_loader")],
        )
        verify_sources(self.sources)
        self.shadows = {name: torch.full_like(value, float("nan")) for name, value in self.parameters.items()}
        self.shadow_names = {_storage(value): name for name, value in self.shadows.items()}
        self.received: dict[str, dict] = {}
        self.touched: set[str] = set()

    def _check_inventory(self):
        current = dict(self.model.named_parameters())
        aliases = {name: id(value) for name, value in self.model.named_parameters(remove_duplicate=False)}
        if (
            aliases != self.aliases
            or set(current) != set(self.parameters)
            or any(
                current[name] is not value
                or self.storage_names.get(_storage(value)) != name
                or (tuple(value.shape), tuple(value.stride()), value.dtype, value.device) != self.layouts[name]
                for name, value in self.parameters.items()
            )
        ):
            raise RuntimeError("weight loader changed the registered parameter identities or storage")

    def load(self, name, value, *, loader=None):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten
        from torch.utils._pytree import tree_map

        if not isinstance(name, str) or not name or name in self.received:
            raise ValueError("received tensor names must be nonempty and unique")
        self._check_inventory()
        if loader is None:

            def loader(source_name, tensor):
                return self.model.load_weights([(source_name, tensor)])

        verify_sources(self.sources)
        self.sources.update(source_files(loader))
        verify_sources(self.sources)
        entry = tensor_manifest(value)
        verifier = self

        class Observe(TorchDispatchMode):
            def __init__(self, shadow):
                super().__init__()
                self.shadow = shadow
                self.owned = {_storage(tensor) for tensor in verifier.shadows.values()} if shadow else set()
                self.written = set()

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func.namespace not in {"aten", "prims"}:
                    raise ValueError(f"unobservable native weight-loader operation: {func}")
                flat, _ = tree_flatten((args, kwargs))
                inputs = {_storage(t) for t in flat if isinstance(t, torch.Tensor)}

                def substitute(tensor):
                    if not isinstance(tensor, torch.Tensor):
                        return tensor
                    destination = verifier.storage_names.get(_storage(tensor))
                    if destination is None or not self.shadow:
                        return tensor
                    parameter = verifier.parameters[destination]
                    if tensor.dtype != parameter.dtype:
                        raise ValueError("reinterpretation of parameter storage is unsupported")
                    return verifier.shadows[destination].as_strided(
                        tensor.shape, tensor.stride(), tensor.storage_offset()
                    )

                for index, argument in enumerate(func._schema.arguments):
                    if argument.alias_info is None or not argument.alias_info.is_write:
                        continue
                    actual = args[index] if index < len(args) else kwargs.get(argument.name)
                    tensors, _ = tree_flatten(actual)
                    for tensor in tensors:
                        if not isinstance(tensor, torch.Tensor):
                            continue
                        key = _storage(tensor)
                        destination = verifier.storage_names.get(key)
                        if self.shadow:
                            destination = destination or verifier.shadow_names.get(key)
                        if destination is not None:
                            self.written.add(destination)
                        elif key not in self.owned:
                            raise ValueError("weight loader writes an input, buffer or unowned storage")
                call_args = tree_map(substitute, args)
                call_kwargs = tree_map(substitute, kwargs)
                result = func(*call_args, **call_kwargs)
                outputs, _ = tree_flatten(result)
                for tensor in outputs:
                    if isinstance(tensor, torch.Tensor):
                        key = _storage(tensor)
                        if key not in inputs and key not in verifier.storage_names:
                            self.owned.add(key)
                return result

        with torch.no_grad(), Observe(False) as live:
            result = loader(name, value)
        # Each replay receives private input too; loaders must not mutate the wire tensor.
        with torch.no_grad(), Observe(True) as replay:
            loader(name, value.detach().clone())
        self._check_inventory()
        verify_sources(self.sources)
        if live.written != replay.written or tensor_manifest(value) != entry:
            raise RuntimeError("shadow loader destination set or received bytes changed")
        entry["destinations"] = sorted(live.written)
        self.received[name] = entry
        self.touched.update(live.written)
        return result

    def finish(self):
        import torch

        self._check_inventory()
        verify_sources(self.sources)
        destinations = {}
        for name in sorted(self.touched):
            live = self.parameters[name].detach()
            shadow = self.shadows[name]
            uncovered = torch.isnan(shadow)
            byte_diff = (
                live.reshape(-1).view(torch.uint8).reshape(-1, live.element_size())
                != shadow.reshape(-1).view(torch.uint8).reshape(-1, shadow.element_size())
            ).any(dim=1)
            destinations[name] = {
                "uncovered": int(uncovered.sum().item()),
                "mismatched": int((byte_diff & ~uncovered.reshape(-1)).sum().item()),
            }
        return {
            "status": "ok",
            "received": self.received,
            "destinations": destinations,
            "unverifiable_destinations": [],
            "non_parameter_destinations": [],
            "untouched_parameters": {
                name: value.numel() for name, value in self.parameters.items() if name not in self.touched
            },
            "evidence_sources": self.sources,
        }


@functools.lru_cache(maxsize=1)
def receiver_extension():
    """Resolve the optional external dependency only in an enabled sampling worker."""
    from arctic_inference.server.weight_sync.receiver import WeightSyncExtension

    class WeightSyncEvidenceExtension(WeightSyncExtension):
        def sync_weights(self, *args, **kwargs):
            if not enabled():
                return super().sync_weights(*args, **kwargs)
            if self.model_config.quantization:
                raise ValueError("quantized weight-loader shadow evidence is not implemented")
            self._ap_weight_verification = None
            origins = source_files(WeightSyncExtension.sync_weights, WeightSyncExtension._load_batched)
            verify_sources(origins)
            result = super().sync_weights(*args, **kwargs)
            verify_sources(origins)
            if result.get("status") == "engine_ready":
                return result
            if self._ap_weight_verification is None:
                raise RuntimeError("receiver completed without dispatcher-observed verification")
            return {**result, "weight_sync_verification": self._ap_weight_verification}

        def _load_batched(self, model, engine, received_names=None):
            if not enabled():
                return super()._load_batched(model, engine, received_names)
            limit = int(os.environ.get("ARCTIC_PLATFORM_WEIGHT_SYNC_SHADOW_MAX_BYTES", str(1 << 30)))
            verifier = ShadowVerifier(model, max_shadow_bytes=limit)
            verifier.sources.update(source_files(type(engine).receive_weights))
            verify_sources(verifier.sources)
            loaded = 0
            for name, value in engine.receive_weights():
                verifier.load(name, value)
                if received_names is not None:
                    received_names.append(name)
                loaded += 1
            self._ap_weight_verification = verifier.finish()
            return loaded

        def _load_direct(self, model, engine, received_names=None):
            if not enabled():
                return super()._load_direct(model, engine, received_names)
            raise ValueError("direct parameter writer needs a separate supported shadow observer")

        def _load_direct_zero_copy(self, model, engine):
            if not enabled():
                return super()._load_direct_zero_copy(model, engine)
            raise ValueError("zero-copy receive bypasses observable shadow loads")

    return WeightSyncEvidenceExtension


def __getattr__(name):
    if name == "WeightSyncEvidenceExtension":
        return receiver_extension()
    raise AttributeError(name)
