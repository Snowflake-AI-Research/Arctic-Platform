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

"""Router-replay trace: the routing a training rank received from the sampler, and what its MoE routers used.

One trace covers one ``fwd_bwd`` on one rank. It records the per-sample routing popped from the replay cache,
then for every model call the packed ``input_ids`` and ``position_ids``, the ``routed_experts`` tensor handed to
the model, and each router invocation in order: its decoder layer, whether it ran on the forward pass or inside
backward (an activation-checkpoint recompute), whether it replayed supplied routing or computed its own, and the
expert indices it used.
"""

from __future__ import annotations

import re
from typing import Any
from typing import Optional

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_LAYER_ATTR = "_dss_router_replay_layer"

_active: Optional["RouterReplayTrace"] = None


class RouterReplayTrace:
    def __init__(self, model: Any) -> None:
        self.received: dict[str, list] = {}
        self.calls: list[dict] = []
        self.unattributed: list[dict] = []
        self._hook_handles: list[Any] = []
        for name, module in model.named_modules():
            if type(module).__name__ != "TokenChoiceTopKRouter":
                continue
            match = _LAYER_INDEX.search(name)
            if match is not None:
                setattr(module, _LAYER_ATTR, int(match.group(1)))
            if type(module).__module__ != "dss.ray_dss.jobs.models.moe.layers.moe":
                self._hook_handles.append(module.register_forward_hook(self._record_external_router, with_kwargs=True))

    def _record_external_router(self, router: Any, args: tuple, kwargs: dict, output: Any) -> None:
        routed = kwargs.get("routed_experts")
        if routed is None and len(args) >= 3:
            routed = args[2]
        if not isinstance(output, tuple) or len(output) < 2:
            raise TypeError("TokenChoiceTopKRouter must return selected expert indices as output 1")
        self.record_router(router, output[1], replayed=routed is not None)

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def record_received(self, sample_ids, routed_experts_list) -> None:
        import torch

        for sample_id, routed in zip(sample_ids, routed_experts_list, strict=True):
            self.received[str(sample_id)] = routed.to(device="cpu", dtype=torch.int64).tolist()

    def begin_call(self, model_kwargs: dict, *, padding: bool) -> None:
        import torch

        routed = model_kwargs.get("routed_experts")
        self.calls.append(
            {
                "padding": bool(padding),
                "input_ids": model_kwargs["input_ids"].reshape(-1).tolist(),
                "position_ids": model_kwargs["position_ids"].reshape(-1).tolist(),
                "routed_experts": (
                    None
                    if routed is None
                    else routed.reshape(-1, *routed.shape[-2:]).to(device="cpu", dtype=torch.int64).tolist()
                ),
                "router": [],
            }
        )

    def record_router(self, router: Any, indices: Any, *, replayed: bool) -> None:
        import torch

        entry = {
            "layer": getattr(router, _LAYER_ATTR, None),
            "phase": "recompute" if torch._C._current_graph_task_id() != -1 else "forward",
            "replayed": bool(replayed),
            "indices": indices.detach().to(device="cpu", dtype=torch.int64).tolist(),
        }
        (self.calls[-1]["router"] if self.calls else self.unattributed).append(entry)

    def export(self, rank: int) -> dict:
        return {
            "rank": int(rank),
            "received": self.received,
            "calls": self.calls,
            "unattributed_router_calls": self.unattributed,
        }


def start(model: Any) -> RouterReplayTrace:
    global _active
    _active = RouterReplayTrace(model)
    return _active


def active() -> Optional[RouterReplayTrace]:
    return _active


def finish(rank: int) -> Optional[dict]:
    global _active
    trace, _active = _active, None
    if trace is None:
        return None
    trace.close()
    return trace.export(rank)


def record_router(router: Any, indices: Any, *, replayed: bool) -> None:
    if _active is not None:
        _active.record_router(router, indices, replayed=replayed)
