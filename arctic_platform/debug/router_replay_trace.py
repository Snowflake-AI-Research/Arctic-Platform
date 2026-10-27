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

"""Opt-in observation of received routing and completed trainer model calls."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

TRAINING_OPTION = "arctic_platform_router_replay_trace"
TRACE_SCHEMA = "arctic-platform-router-replay-call-v1"
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_LAYER_ATTR = "_dss_router_replay_layer"
_active: RouterReplayTrace | None = None
_publication: dict | None = None


def maybe_configure_router_replay_sampling(sampling, *, verify):
    """Enable capture only for the replay check, preserving the configured cache limits."""
    if not verify:
        return sampling
    inference = deepcopy(sampling.arctic_inference_config or {})
    inference.setdefault("router_replay", {})["enabled"] = True
    return sampling.model_copy(update={"arctic_inference_config": inference}, deep=True)


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
            self._hook_handles.append(module.register_forward_hook(self._record_router, with_kwargs=True))

    def _record_router(self, router: Any, args: tuple, kwargs: dict, output: Any) -> None:
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
            key = str(sample_id)
            if key in self.received:
                raise ValueError(f"duplicate router-replay RX observation for {key}")
            self.received[key] = routed.detach().to(device="cpu", dtype=torch.int64).tolist()

    def begin_call(self, model_kwargs: dict, *, padding: bool) -> None:
        import torch

        routed = model_kwargs.get("routed_experts")
        self.calls.append(
            {
                "padding": bool(padding),
                "input_ids": model_kwargs["input_ids"].detach().reshape(-1).cpu().tolist(),
                "position_ids": model_kwargs["position_ids"].detach().reshape(-1).cpu().tolist(),
                "routed_experts": (
                    None
                    if routed is None
                    else routed.detach().reshape(-1, *routed.shape[-2:]).to(device="cpu", dtype=torch.int64).tolist()
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
    if _active is not None:
        raise RuntimeError("router-replay trace already active")
    _active = RouterReplayTrace(model)
    return _active


def active() -> RouterReplayTrace | None:
    return _active


def finish(rank: int) -> dict | None:
    global _active, _publication
    trace, _active = _active, None
    _publication = None
    if trace is None:
        return None
    trace.close()
    return trace.export(rank)


def record_router(router: Any, indices: Any, *, replayed: bool) -> None:
    if _active is not None:
        _active.record_router(router, indices, replayed=replayed)


def maybe_configure_router_replay_trace(ds_config: dict, model: Any, rank: int) -> dict:
    """Consume a request-scoped option before passing configuration to DeepSpeed."""
    global _publication
    if TRAINING_OPTION not in ds_config:
        return ds_config
    option = ds_config[TRAINING_OPTION]
    if not isinstance(option, dict) or set(option) != {"request_id", "output_dir"}:
        raise ValueError(f"{TRAINING_OPTION} requires request_id and output_dir")
    request_id = option["request_id"]
    if not isinstance(request_id, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", request_id) is None:
        raise ValueError("router-replay request_id must be a nonempty safe identifier")
    root = Path(option["output_dir"]).resolve()
    if not root.is_relative_to("/data-fast"):
        raise ValueError("router-replay evidence must stay under /data-fast")
    if type(rank) is not int or rank < 0:
        raise ValueError("router-replay rank must be a nonnegative integer")
    start(model)
    _publication = {
        "request_id": request_id,
        "output_dir": root / request_id,
        "rank": rank,
        "completed": 0,
    }
    return {key: value for key, value in ds_config.items() if key != TRAINING_OPTION}


def maybe_record_router_replay_received(sample_id: str, routed: Any) -> None:
    """Observe the value actually removed from the trainer's RX cache."""
    if _publication is not None:
        _active.record_received([sample_id], [routed])


def maybe_begin_router_replay_call(engine: Any, model_kwargs: dict, *, backward: bool | str) -> None:
    """Capture every training model call, including duplicated padding schedule calls."""
    if _publication is None or backward is False:
        return
    if getattr(engine, "global_rank", None) != _publication["rank"]:
        raise RuntimeError("router-replay trace rank differs from the executing engine")
    if len(_active.calls) != _publication["completed"]:
        raise RuntimeError("previous router-replay model call has no successful backward")
    _active.begin_call(model_kwargs, padding=False)


def maybe_complete_router_replay_call(engine: Any) -> None:
    """Publish one immutable call only after its actual backward returned successfully."""
    if _publication is None:
        return
    rank = _publication["rank"]
    if getattr(engine, "global_rank", None) != rank:
        raise RuntimeError("router-replay completion rank differs from the executing engine")
    index = _publication["completed"]
    if len(_active.calls) != index + 1:
        raise RuntimeError("router-replay backward has no unique preceding model call")
    root = _publication["output_dir"]
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": TRACE_SCHEMA,
        "request_id": _publication["request_id"],
        "rank": rank,
        "call_index": index,
        "received": _active.received,
        "calls": [_active.calls[index]],
        "unattributed_router_calls": _active.unattributed,
    }
    with (root / f"rank-{rank}-call-{index}.json").open("x") as stream:
        json.dump(record, stream, separators=(",", ":"))
        stream.write("\n")
    _publication["completed"] += 1
