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

"""Own the live receiver's native manager callbacks through evidence failures.

This binding does not derive native slots, invoke a model factory, or admit
storage/application coverage. Captured removal and teardown never depend on a
later source guard succeeding.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field

from arctic_platform.debug.weight_sync_evidence import enabled
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import verify_sources

_MANAGER_METHODS = ("add_adapter", "activate_adapter", "get_adapter", "remove_adapter")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _origins(admitted_sources, *callables):
    observed = source_files(_origins, *callables)
    _require(
        type(admitted_sources) is dict
        and observed
        and observed.keys() <= admitted_sources.keys()
        and all(admitted_sources[path] == digest for path, digest in observed.items()),
        "actual receiver/manager callbacks require admitted source identities",
    )
    verify_sources(admitted_sources)
    return dict(admitted_sources)


@dataclass
class OwnedLoRAReceiverBinding:
    receiver: object
    runner: object
    native_model: object
    wrapper: object
    manager: object
    adapter_id: int
    callbacks: dict
    sources: dict
    adapter: object = None
    closed: bool = False
    events: list = field(default_factory=list)

    def guard(self):
        _require(not self.closed, "receiver binding is already closed")
        verify_sources(self.sources)
        _require(
            self.receiver.model_runner is self.runner
            and self.runner.model is self.native_model
            and self.runner.lora_manager is self.wrapper
            and self.wrapper._adapter_manager is self.manager
            and all(getattr(self.manager, name) == method for name, method in self.callbacks.items()),
            "actual receiver/manager identity or captured callbacks changed",
        )

    def reserve(self, adapter):
        """Own the exact model before any manager method can mutate native state."""
        self.guard()
        _require(self.adapter is None, "this receiver binding already reserved an adapter")
        _require(getattr(adapter, "id", None) == self.adapter_id, "built adapter id differs from the owned id")
        _require(self.callbacks["get_adapter"](self.adapter_id) is None, "cannot replace an existing adapter")
        owned = getattr(self.receiver, "_ap_owned_lora_installations", None)
        if owned is None:
            owned = self.receiver._ap_owned_lora_installations = {}
        _require(type(owned) is dict and self.adapter_id not in owned, "adapter id is already owned")
        owned[self.adapter_id] = (
            self.manager,
            adapter,
            self.callbacks["get_adapter"],
            self.callbacks["remove_adapter"],
        )
        self.adapter = adapter
        self.events.append("reserved")

    def install(self, adapter, *, installer):
        """Call the selected installer once; retain ownership on any rejection."""
        _require(self.adapter is None, "native installation may only be attempted once")
        self.sources = _origins(self.sources, installer)
        self.reserve(adapter)
        added, activated = installer(self.manager, adapter, self.adapter_id)
        self.guard()
        _require(
            added is True and activated is True and self.callbacks["get_adapter"](self.adapter_id) is adapter,
            "native manager did not retain the exact added and activated adapter",
        )
        self.events.append("installed")
        return {
            "schema": "owned-live-lora-manager-installation-v1",
            "adapter_id": self.adapter_id,
            "added": added,
            "activated": activated,
            "captured_manager_callbacks": list(self.callbacks),
            "evidence_sources": dict(self.sources),
            "native_slot_binding_admitted": False,
            "adapter_storage_verified": False,
            "adapter_application_verified": False,
            "capacity_admitted": False,
            "numerical": None,
        }

    def cleanup(self):
        """Use captured owned callbacks even after receiver or source drift."""
        if self.closed:
            return {"owned_adapter_empty": True, "already_closed": True}
        if self.adapter is not None:
            current = self.callbacks["get_adapter"](self.adapter_id)
            _require(current is None or current is self.adapter, "refusing to remove a different owner's adapter")
            if current is self.adapter:
                self.callbacks["remove_adapter"](self.adapter_id)
            _require(self.callbacks["get_adapter"](self.adapter_id) is None, "owned adapter remains after removal")
            owned = getattr(self.receiver, "_ap_owned_lora_installations", {})
            expected = (
                self.manager,
                self.adapter,
                self.callbacks["get_adapter"],
                self.callbacks["remove_adapter"],
            )
            _require(
                type(owned) is dict and owned.get(self.adapter_id) == expected,
                "owned receiver installation ticket changed",
            )
            del owned[self.adapter_id]
        self.closed = True
        self.events.append("cleaned")
        return {"owned_adapter_empty": True, "already_closed": False}


def capture_receiver_manager(receiver, *, adapter_id, admitted_sources):
    """Capture the existing receiver manager, never an imported replacement."""
    _require(enabled(), "LoRA receiver evidence must be explicitly enabled")
    _require(type(adapter_id) is int and adapter_id > 0, "positive actual adapter id required")
    runner = receiver.model_runner
    wrapper = getattr(runner, "lora_manager", None)
    manager = getattr(wrapper, "_adapter_manager", None)
    _require(manager is not None and runner.model is not None, "actual loaded receiver manager/model required")
    callbacks = {name: getattr(manager, name) for name in _MANAGER_METHODS}
    _require(
        all(inspect.ismethod(method) and method.__self__ is manager for method in callbacks.values()),
        "native adapter operations must be bound to the exact live manager",
    )
    sources = _origins(admitted_sources, capture_receiver_manager, type(manager), *callbacks.values())
    binding = OwnedLoRAReceiverBinding(
        receiver,
        runner,
        runner.model,
        wrapper,
        manager,
        adapter_id,
        callbacks,
        sources,
    )
    binding.guard()
    _require(callbacks["get_adapter"](adapter_id) is None, "adapter id is already installed")
    return binding


@contextmanager
def owned_receiver_binding(receiver, *, adapter_id, admitted_sources, teardown):
    """Always attempt captured teardown, including after admission/cleanup errors."""
    _require(callable(teardown), "captured receiver/gateway teardown callback required")
    binding, primary = None, None
    try:
        sources = _origins(admitted_sources, teardown)
        binding = capture_receiver_manager(receiver, adapter_id=adapter_id, admitted_sources=sources)
        yield binding
    except BaseException as error:
        primary = error
        raise
    finally:
        secondary = []
        if binding is not None:
            try:
                binding.cleanup()
            except BaseException as error:
                secondary.append(("owned adapter cleanup", error))
        try:
            teardown()
        except BaseException as error:
            secondary.append(("captured receiver/gateway teardown", error))
        if primary is not None:
            for phase, error in secondary:
                primary.add_note(phase + " also failed: " + repr(error))
        elif secondary:
            phase, error = secondary[0]
            for later_phase, later in secondary[1:]:
                error.add_note(later_phase + " also failed: " + repr(later))
            error.add_note("failure phase: " + phase)
            raise error
