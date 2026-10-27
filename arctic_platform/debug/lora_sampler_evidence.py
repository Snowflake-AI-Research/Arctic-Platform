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

"""Explicit adapter loading over the selected external sampler transport.

Installation and request forwarding are not adapter-storage verification or
full-model coverage. Those require separate evidence; no base-model manifest is
manufactured by this bridge.
"""

from __future__ import annotations

import functools
import importlib

from arctic_platform.debug.weight_sync_evidence import enabled
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import tensor_manifest
from arctic_platform.debug.weight_sync_evidence import verify_sources

EXTENSION = "arctic_platform.debug.lora_sampler_evidence.LoRAWeightSyncExtension"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def engine_kwargs_for_lora(engine_kwargs, *, rank):
    """Bind the explicitly LoRA-enabled engine to this named receiver extension."""
    _require(enabled(), "LoRA sampler evidence must be explicitly enabled")
    _require(type(rank) is int and rank > 0, "actual positive configured LoRA rank required")
    kwargs = dict(engine_kwargs)
    _require(kwargs.get("enable_lora") is True, "the selected engine must enable LoRA explicitly")
    _require(
        type(kwargs.get("max_lora_rank")) is int and kwargs["max_lora_rank"] >= rank,
        "the selected engine must admit the configured adapter rank",
    )
    _require(not kwargs.get("speculative_config"), "speculative adapter sync is not implemented")
    _require(
        kwargs.get("worker_extension_cls")
        in (
            None,
            EXTENSION,
            "arctic_inference.server.weight_sync.WeightSyncExtension",
        ),
        "cannot replace a different selected receiver extension",
    )
    kwargs["worker_extension_cls"] = EXTENSION
    return kwargs


def _bind_sources(admitted_sources, *callables):
    observed = source_files(_bind_sources, *callables)
    _require(
        type(admitted_sources) is dict
        and observed
        and observed.keys() <= admitted_sources.keys()
        and all(admitted_sources[path] == digest for path, digest in observed.items()),
        "every actual adapter callable needs an admitted source identity",
    )
    # Include the whole supplied operation closure, not just its selected callables.
    verify_sources(admitted_sources)
    return dict(admitted_sources)


def cleanup_owned_loras(receiver):
    """Remove only captured owned adapter objects, even after source drift."""
    owned = getattr(receiver, "_ap_owned_lora_installations", {})
    errors = []
    for identity, (manager, model, get_adapter, remove_adapter) in list(owned.items()):
        try:
            current = get_adapter(identity)
            if current is not None:
                _require(current is model, "refusing to remove a different owner's adapter")
                remove_adapter(identity)
            _require(get_adapter(identity) is None, "owned adapter remains after cleanup")
            del owned[identity]
        except BaseException as error:
            errors.append(error)
    if errors:
        raise RuntimeError("owned adapter cleanup failed: " + "; ".join(str(error) for error in errors))
    return {"owned_adapters_empty": not owned}


def receive_and_install_lora(
    receiver, engine, *, lora_int_id, lora_name, lora_config, max_received_bytes, admitted_sources
):
    """Clone the real reusable wire buffers, then use native adapter build/install.

    No base-weight loader or parameter-only shadow judge is used. The returned
    record describes actual wire bytes and installation, not application proof.
    """
    _require(enabled(), "LoRA receiver evidence must be explicitly enabled")
    _require(
        type(lora_int_id) is int and lora_int_id > 0 and type(lora_name) is str and lora_name,
        "positive adapter id and nonempty name required",
    )
    _require(
        type(max_received_bytes) is int and max_received_bytes > 0,
        "an explicit finite received-adapter byte limit is required",
    )
    # The optional inference backend belongs to the selected receiver, not the model factory.
    native = importlib.import_module("arctic_platform.inference.server.weight_sync.receiver")
    _build_lora_model = native._build_lora_model
    _install_lora_adapter = native._install_lora_adapter
    normalize_lora_peft_config = native.normalize_lora_peft_config

    runner = receiver.model_runner
    wrapper = getattr(runner, "lora_manager", None)
    manager = getattr(wrapper, "_adapter_manager", None)
    _require(manager is not None, "actual enabled native LoRA adapter manager required")
    _require(not receiver.model_config.quantization, "quantized adapter evidence is not implemented")
    manager_calls = {
        name: getattr(manager, name) for name in ("add_adapter", "activate_adapter", "get_adapter", "remove_adapter")
    }
    origins = _bind_sources(
        admitted_sources,
        type(engine).receive_weights,
        _build_lora_model,
        _install_lora_adapter,
        normalize_lora_peft_config,
        *manager_calls.values(),
    )
    config = normalize_lora_peft_config(lora_config, location="lora_config")
    _require(
        manager.get_adapter(lora_int_id) is None,
        "existing adapter id cannot be replaced without owned eviction evidence",
    )
    tensors, received = {}, {}
    size = 0
    for name, value in engine.receive_weights():
        _require(type(name) is str and name and name not in received, "adapter wire names must be nonempty and unique")
        _require(".lora_A." in name or ".lora_B." in name, "base-model weights are not adapter weights")
        size += value.numel() * value.element_size()
        _require(size <= max_received_bytes, "adapter bytes exceed the explicit receive limit")
        received[name] = tensor_manifest(value)
        clone = value.detach().to(device="cpu", copy=True).contiguous()
        _require(tensor_manifest(clone) == received[name], "retained adapter copy differs from wire bytes")
        tensors[name] = clone
        verify_sources(origins)
    _require(received, "empty adapter transfer cannot install a policy")
    model = _build_lora_model(
        lora_int_id=lora_int_id,
        tensors=tensors,
        lora_config=config,
        device="cpu",
        dtype=receiver.model_config.dtype,
        model=runner.model,
    )
    verify_sources(origins)
    owned = getattr(receiver, "_ap_owned_lora_installations", None)
    if owned is None:
        owned = receiver._ap_owned_lora_installations = {}
    _require(lora_int_id not in owned, "an owned adapter id is already reserved")
    owned[lora_int_id] = (manager, model, manager_calls["get_adapter"], manager_calls["remove_adapter"])
    try:
        _require(
            all(getattr(manager, name) == method for name, method in manager_calls.items()),
            "selected adapter manager callables changed before installation",
        )
        added, activated = _install_lora_adapter(manager, model, lora_int_id)
        _require(
            all(getattr(manager, name) == method for name, method in manager_calls.items()),
            "selected adapter manager callables changed during installation",
        )
        _require(
            added is True and activated is True and manager_calls["get_adapter"](lora_int_id) is model,
            "actual manager must retain the exact built and activated adapter",
        )
        verify_sources(origins)
    except BaseException as error:
        try:
            cleanup_owned_loras(receiver)
        except BaseException as cleanup_error:
            error.add_note("owned adapter cleanup also failed: " + repr(cleanup_error))
        raise
    return {
        "schema": "explicit-external-lora-installation-v1",
        "status": "adapter_installed",
        "weight_format": "lora",
        "lora_int_id": lora_int_id,
        "lora_name": lora_name,
        "lora_added": added,
        "lora_activated": activated,
        "is_3d_lora_weight": bool(getattr(model, "is_3d_lora_weight", False)),
        "received": received,
        "received_bytes": size,
        "evidence_sources": origins,
        "adapter_storage_verified": False,
        "adapter_application_verified": False,
        "full_parameter_coverage": False,
        "capacity_admitted": False,
        "numerical": None,
    }


def serving_lora_request(installation, *, admitted_sources):
    """Use the installed id without allowing vLLM to reload a placeholder path."""
    _require(enabled(), "LoRA serving evidence must be explicitly enabled")
    _require(
        installation.get("schema") == "explicit-external-lora-installation-v1"
        and installation.get("status") == "adapter_installed"
        and installation.get("lora_added") is True
        and installation.get("lora_activated") is True
        and type(installation.get("lora_int_id")) is int
        and installation["lora_int_id"] > 0
        and type(installation.get("lora_name")) is str
        and installation["lora_name"],
        "a genuine installation record is required before selecting the adapter",
    )
    from vllm.lora.request import LoRARequest

    origins = _bind_sources(admitted_sources, LoRARequest)
    _require(
        installation["evidence_sources"].items() <= origins.items(),
        "serving must retain every installation source identity",
    )
    request = LoRARequest(
        lora_name=installation["lora_name"],
        lora_int_id=installation["lora_int_id"],
        lora_path="arctic-nccl://" + installation["lora_name"],
        load_inplace=False,
        is_3d_lora_weight=installation["is_3d_lora_weight"],
    )
    _require(
        request.lora_int_id == installation["lora_int_id"]
        and request.lora_name == installation["lora_name"]
        and request.load_inplace is False,
        "the actual serving request must select the installed adapter without disk reload",
    )
    verify_sources(origins)
    return request


@functools.lru_cache(maxsize=1)
def receiver_extension():
    """Keep the external receiver superclass and its pairwise NCCL engine."""
    from arctic_inference.server.weight_sync.receiver import WeightSyncExtension

    class LoRAWeightSyncExtension(WeightSyncExtension):
        def close_weight_sync(self):
            error = None
            try:
                result = cleanup_owned_loras(self)
            except BaseException as cleanup_error:
                error = cleanup_error
            try:
                engine_result = super().close_weight_sync()
            except BaseException as engine_error:
                if error is None:
                    raise
                error.add_note("external engine cleanup also failed: " + repr(engine_error))
            if error is not None:
                raise error
            return {**result, "external_engine_cleanup": engine_result}

        def sync_lora_weights(
            self,
            master_addr,
            master_port,
            rank_offset,
            world_size,
            *,
            lora_int_id,
            lora_name,
            lora_config,
            max_received_bytes,
            admitted_sources,
            bucket_size=256 * 1024 * 1024,
            engine_only=False,
        ):
            _require(enabled(), "the explicit LoRA extension must be enabled on the actual TP rank")
            _require(type(bucket_size) is int and bucket_size > 0, "positive native NCCL bucket size required")
            origins = _bind_sources(admitted_sources, type(self)._get_or_create_engine, receive_and_install_lora)
            engine = self._get_or_create_engine(
                master_addr,
                master_port,
                bucket_size,
                False,
                engine_attr="_ws_engine",
                key_attr="_ws_engine_key",
            )
            verify_sources(origins)
            if engine_only:
                return {
                    "status": "engine_ready",
                    "weight_format": "lora",
                    "rank": 1,
                    "world_size": 2,
                    "adapter_storage_verified": False,
                    "capacity_admitted": False,
                }
            return receive_and_install_lora(
                self,
                engine,
                lora_int_id=lora_int_id,
                lora_name=lora_name,
                lora_config=lora_config,
                max_received_bytes=max_received_bytes,
                admitted_sources=origins,
            )

    return LoRAWeightSyncExtension


def __getattr__(name):
    if name == "LoRAWeightSyncExtension":
        return receiver_extension()
    raise AttributeError(name)
