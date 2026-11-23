# Copyright 2026 Snowflake Inc.
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

"""Replay the accepted expert choices for positions that Model Runner V2 executes again."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

# Bytes survive the msgpack round trip through SamplingParams.extra_args.
FORCED_ROUTES_KEY = "dss_router_replay_forced_routes"

# The trainer's score functions (TokenChoiceTopKRouter.forward), applied to float32 logits.
_SCORES = {
    "sigmoid": torch.sigmoid,
    "softmax": lambda logits: F.softmax(logits, dim=1),
    "sqrtsoftplus": lambda logits: F.softplus(logits).sqrt(),
}


class _RouteHistory:
    """The named parent and accepted suffix of one scheduler request, across its admissions."""

    def __init__(self, parent: dict, prefix_len: int, prompt_len: int):
        self.suffix = bytearray()
        self.parent = parent["rows"] if 0 < prefix_len < prompt_len else b""
        self.row_bytes = len(self.parent) // prefix_len if self.parent else 0
        self.dtype = parent.get("dtype")

    def record(self, rows: np.ndarray) -> None:
        if rows.size:
            self.row_bytes = rows[0].nbytes
            self.dtype = rows.dtype.str
            self.suffix.extend(memoryview(rows).cast("B"))

    def from_position(self, start: int) -> dict | None:
        offset = start * self.row_bytes
        if offset >= len(self.parent) + len(self.suffix):
            return None
        if self.suffix:
            first = memoryview(self.parent)[offset:]
            last = memoryview(self.suffix)[max(0, offset - len(self.parent)):]
            rows = b"".join((first, last))
        else:
            rows = self.parent[offset:]
        return {"start": start, "dtype": self.dtype, "rows": rows}


def _patch_scheduler() -> None:
    from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector
    from vllm.v1.core.sched.output import NewRequestData

    if getattr(NewRequestData.from_request, "_arctic_forced_routes", False):
        return
    original_new = NewRequestData.from_request.__func__
    original_output = AuxOutputSchedulerConnector.take_output

    def from_request(cls, request, *args, **kwargs):
        data = original_new(cls, request, *args, **kwargs)
        params = data.sampling_params
        parent = (params.extra_args or {}).get(FORCED_ROUTES_KEY) if params is not None else None
        if isinstance(parent, dict):
            history = getattr(request, "_arctic_route_history", None)
            if history is None:
                history = _RouteHistory(parent, params.routed_experts_prompt_start, data.prompt_len)
                request._arctic_route_history = history
            data.sampling_params = copy(params)
            data.sampling_params.extra_args = {
                **params.extra_args, FORCED_ROUTES_KEY: history.from_position(data.num_computed_tokens)
            }
        return data

    def take_output(self, request, output):
        rows = original_output(self, request, output)
        history = getattr(request, "_arctic_route_history", None)
        if history is not None and rows is not None:  # contiguous, as vLLM's own output assembly assumes
            history.record(rows)
        return rows

    from_request._arctic_forced_routes = True
    NewRequestData.from_request = classmethod(from_request)
    AuxOutputSchedulerConnector.take_output = take_output


class ForcedRoutes:
    """One worker's stored rows by request index, and the forced ids and mask of the step being run."""

    def __init__(self, max_num_reqs: int, row_shape: tuple[int, ...], ids: torch.Tensor):
        # Stored positions [start, end) by request index, as Model Runner V2's RequestState holds per-request state.
        self.start = np.zeros(max_num_reqs, dtype=np.int64)
        self.end = np.zeros(max_num_reqs, dtype=np.int64)
        self.rows: dict[int, np.ndarray] = {}
        self.row_shape = row_shape
        self.ids = ids
        self.mask = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        self.forcing = False

    def add(self, req_index: int, request: Any) -> None:
        forced = (request.sampling_params.extra_args or {}).get(FORCED_ROUTES_KEY)
        if not isinstance(forced, dict):  # vLLM's own API passes only scalars and lists here
            return
        rows = np.frombuffer(forced["rows"], dtype=forced["dtype"]).reshape(-1, *self.row_shape)
        self.start[req_index], self.end[req_index] = forced["start"], forced["start"] + len(rows)
        self.rows[req_index] = rows

    def remove(self, req_index: int) -> None:
        self.end[req_index] = 0
        self.rows.pop(req_index, None)

    def fill(self, batch: InputBatch) -> None:
        """Mark the batch rows that compute a stored position, with the boundaries the routed-experts capture uses
        (``AuxOutputWorkerConnector.prepare_output``), after clearing the rows the previous step marked."""
        if self.forcing:
            self.mask.zero_()
            self.forcing = False
        if not self.rows or not batch.has_prefill:
            return
        idx_mapping = batch.idx_mapping_np
        num_computed = batch.num_computed_tokens_np
        start, end = self.start[idx_mapping], self.end[idx_mapping]
        qsl = batch.query_start_loc_np[: len(idx_mapping) + 1]
        lo = np.maximum(num_computed, start)
        scheduled_end = num_computed + np.diff(qsl)
        hi = np.minimum(scheduled_end, end)
        for i in np.flatnonzero(lo < hi):  # only the requests recomputing stored positions in this step
            at = qsl[i] + lo[i] - num_computed[i]
            stored = self.rows[idx_mapping[i]][lo[i] - start[i] : hi[i] - start[i], : self.ids.shape[1]]
            self.ids[at : at + len(stored)].copy_(torch.from_numpy(stored.astype(np.int32)), non_blocking=True)
            self.mask[at : at + len(stored)] = True
            self.forcing = True
        # Prefill has passed these stored rows; a preempted request is added again with them.
        for i in np.flatnonzero((end > 0) & (scheduled_end >= end)):
            self.remove(idx_mapping[i])


def bind_router(router: Any, layer_id: int, forced: ForcedRoutes) -> None:
    """Route ``forced``'s masked rows to their stored experts. Graph-safe: no host branch reads the mask."""
    # FusedTopKRouter leaves any scaling to the MoE output.
    scale = getattr(router, "routed_scaling_factor", 1.0)
    score = _SCORES[router.scoring_func]
    softmax = router.scoring_func == "softmax"
    compute = type(router)._compute_routing.__get__(router)

    def _compute_routing(hidden_states, router_logits, indices_type, *, input_ids=None):
        weights, ids = compute(hidden_states, router_logits, indices_type, input_ids=input_ids)
        n = ids.shape[0]
        mask = forced.mask[:n, None]
        stored = forced.ids[:n, layer_id].to(ids.dtype)
        indices = stored.long()
        if softmax:
            stored_weights = score(router_logits.float()).gather(1, indices)
        else:
            stored_weights = score(router_logits.gather(1, indices).float())
        if router.renormalize:
            stored_weights = stored_weights / (stored_weights.sum(dim=-1, keepdim=True) + 1e-20)
        if scale != 1.0:
            stored_weights = stored_weights * scale
        return torch.where(mask, stored_weights, weights), torch.where(mask, stored, ids)

    router._compute_routing = _compute_routing


def unsupported(runner: Any) -> str | None:
    """Why stored rows cannot reach every MoE router of ``runner.model``, or None when they can."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCaptureSource
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import FusedTopKBiasRouter
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter

    transfer = runner.vllm_config.kv_transfer_config
    if transfer is not None and transfer.kv_load_failure_policy == "recompute":
        return "KV load failure recovery can rewind an active request without a new admission"
    parallel = runner.vllm_config.parallel_config
    if parallel.data_parallel_size > 1:
        return "data_parallel_size > 1: the router sees other DP ranks' tokens (as with sequence-parallel MoE and DBO)"
    supported = (GroupedTopKRouter, FusedTopKBiasRouter, FusedTopKRouter)
    for module in runner.model.modules():
        if isinstance(module, RoutedExpertsCaptureSource):
            return f"layer {module.layer_id} captures routes directly, outside the MoE router"
        if not isinstance(module, MoERunner):
            continue
        if module._quant_method.is_monolithic:
            return f"layer {module.layer_id} runs a monolithic MoE kernel, which routes inside the kernel"
        router = module.router
        if type(router) not in supported or router.scoring_func not in _SCORES:
            return f"layer {module.layer_id}'s router {type(router).__name__} is not one this hook weights"
        if getattr(router, "num_fused_shared_experts", 0):
            return f"layer {module.layer_id}'s fused shared experts add slots outside the captured top-k"
    return None


def install(runner: Any) -> ForcedRoutes:
    """Bind every MoE router of ``runner.model``; call only when ``unsupported(runner)`` is None."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner

    buffer = runner.aux_output_connector._capturer.device_buffer
    layers = [module for module in runner.model.modules() if isinstance(module, MoERunner)]
    # Stage only the rows MoE routers read; a capture can carry further rows per token.
    ids = buffer.new_zeros(buffer.shape[0], 1 + max(module.layer_id for module in layers), buffer.shape[2])
    forced = ForcedRoutes(runner.max_num_reqs, buffer.shape[1:], ids)
    for module in layers:
        bind_router(module.router, module.layer_id, forced)
    return forced


def patch_model_runner(runner_cls: Any) -> None:
    if getattr(runner_cls.prepare_inputs, "_arctic_forced_routes", False):
        return
    # Until initialize_kv_cache installs the hook on an engine that captures routes, or records why it cannot.
    runner_cls._forced_routes = None
    original_initialize_kv_cache = runner_cls.initialize_kv_cache
    original_add_requests = runner_cls.add_requests
    original_remove_request = runner_cls._remove_request
    original_prepare_inputs = runner_cls.prepare_inputs

    def initialize_kv_cache(self, *args, **kwargs):
        result = original_initialize_kv_cache(self, *args, **kwargs)
        # Ranks share one config, so they agree; the server sends rows only when every rank reports None.
        if self.aux_output_connector is not None:
            self._forced_routes_unsupported = unsupported(self)
            if self._forced_routes_unsupported is None:
                self._forced_routes = install(self)
        return result

    def add_requests(self, scheduler_output):
        original_add_requests(self, scheduler_output)
        if self._forced_routes is not None:
            for new_req_data in scheduler_output.scheduled_new_reqs:
                self._forced_routes.add(self.req_states.req_id_to_index[new_req_data.req_id], new_req_data)

    def _remove_request(self, req_id):
        # add_requests removes each new request id first, before it holds an index.
        if self._forced_routes is not None and req_id in self.req_states.req_id_to_index:
            self._forced_routes.remove(self.req_states.req_id_to_index[req_id])
        return original_remove_request(self, req_id)

    def prepare_inputs(self, *args, **kwargs):
        input_batch = original_prepare_inputs(self, *args, **kwargs)
        if self._forced_routes is not None:
            self._forced_routes.fill(input_batch)
        return input_batch

    prepare_inputs._arctic_forced_routes = True
    runner_cls.initialize_kv_cache = initialize_kv_cache
    runner_cls.add_requests = add_requests
    runner_cls._remove_request = _remove_request
    runner_cls.prepare_inputs = prepare_inputs


def forced_route_support(worker: Any) -> str | None:
    """Why this rank's runner cannot force stored routes, or None when it can."""
    # Model Runner V1 and other runners carry no hook.
    return getattr(worker.model_runner, "_forced_routes_unsupported", "this runner has no routed-experts capture hook")


def ensure_forced_routes_patch() -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    _patch_scheduler()
    patch_model_runner(GPUModelRunner)
    Worker.forced_route_support = forced_route_support
