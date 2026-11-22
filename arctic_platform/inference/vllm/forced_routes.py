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

"""Replay the stored experts on the positions of a continuation's sent rows that Model Runner V2 recomputes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

#: ``SamplingParams.extra_args`` key: ``{"start": first position, "dtype": numpy dtype str, "rows": bytes}``, rows of
#: ``[layers, top_k]`` expert ids. Bytes because vLLM's msgpack decodes an ``Any`` field's ndarray to a bare list.
FORCED_ROUTES_KEY = "dss_router_replay_forced_routes"

# The trainer's score functions (TokenChoiceTopKRouter.forward), applied to float32 logits.
_SCORES = {
    "sigmoid": torch.sigmoid,
    "softmax": lambda logits: F.softmax(logits, dim=1),
    "sqrtsoftplus": lambda logits: F.softplus(logits).sqrt(),
}


class ForcedRoutes:
    """One worker's stored rows by request index, and the forced ids and mask of the step being run."""

    def __init__(self, max_num_reqs: int, max_num_tokens: int, num_layers: int, top_k: int, device: torch.device):
        # Stored positions [start, end) by request index, as Model Runner V2's RequestState holds per-request state.
        self.start = np.zeros(max_num_reqs, dtype=np.int64)
        self.end = np.zeros(max_num_reqs, dtype=np.int64)
        self.rows: dict[int, np.ndarray] = {}
        self.ids = torch.zeros((max_num_tokens, num_layers, top_k), dtype=torch.int32, device=device)
        self.mask = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.forcing = False

    def add(self, req_index: int, request: Any) -> None:
        forced = (request.sampling_params.extra_args or {}).get(FORCED_ROUTES_KEY)
        if not isinstance(forced, dict):  # vLLM's own API passes only scalars and lists here
            return
        rows = np.frombuffer(forced["rows"], dtype=forced["dtype"]).reshape(-1, *self.ids.shape[1:])
        # The splice refuses a capture that reaches the end of the prompt; never force a decoded position.
        if forced["start"] + len(rows) < request.prompt_len:
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
        hi = np.minimum(num_computed + np.diff(qsl), end)
        for i in np.flatnonzero(lo < hi):  # only the requests recomputing stored positions in this step
            at = qsl[i] + lo[i] - num_computed[i]
            stored = self.rows[idx_mapping[i]][lo[i] - start[i] : hi[i] - start[i]]
            self.ids[at : at + len(stored)].copy_(torch.from_numpy(stored.astype(np.int32)), non_blocking=True)
            self.mask[at : at + len(stored)] = True
            self.forcing = True
        # Prefill has passed these stored rows; a preempted request is added again with them.
        for i in np.flatnonzero((end > 0) & (num_computed + np.diff(qsl) >= end)):
            self.remove(idx_mapping[i])


def bind_router(router: Any, layer_id: int, forced: ForcedRoutes) -> None:
    """Route ``forced``'s masked rows to their stored experts. Graph-safe: no host branch reads the mask."""
    # FusedTopKRouter leaves any scaling to the MoE output.
    scale = getattr(router, "routed_scaling_factor", 1.0)
    score = _SCORES[router.scoring_func]
    compute = type(router)._compute_routing.__get__(router)

    def _compute_routing(hidden_states, router_logits, indices_type, *, input_ids=None):
        weights, ids = compute(hidden_states, router_logits, indices_type, input_ids=input_ids)
        n = ids.shape[0]
        mask = forced.mask[:n, None]
        stored = forced.ids[:n, layer_id].to(ids.dtype)
        stored_weights = score(router_logits.float()).gather(1, stored.long())
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

    parallel = runner.vllm_config.parallel_config
    if parallel.data_parallel_size > 1:
        return "data_parallel_size > 1: the router sees other DP ranks' tokens (as with sequence-parallel MoE and DBO)"
    if runner.vllm_config.num_prefill_lookahead_tokens > 1:
        return "multi-module MTP can re-prefill positions before the bounded tail"
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

    capturer = runner.aux_output_connector._capturer
    forced = ForcedRoutes(
        runner.max_num_reqs, capturer.device_buffer.shape[0], *capturer.shape_per_token, runner.device
    )
    for module in runner.model.modules():
        if isinstance(module, MoERunner):
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


def forced_route_blocks(worker: Any) -> tuple[int, int] | str:
    """Report the engine core's resolved cache-hit geometry, or why this worker cannot force routes."""
    runner = worker.model_runner
    reason = getattr(runner, "_forced_routes_unsupported", "this runner has no routed-experts capture hook")
    if reason is not None:
        return reason
    spec = runner.vllm_config.speculative_config
    blocks = 2 if spec is not None and spec.use_eagle_block_drop() else 1
    return runner.kv_cache_config.cache_hit_alignment_tokens, blocks


def ensure_forced_routes_patch() -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    patch_model_runner(GPUModelRunner)
    Worker.forced_route_blocks = forced_route_blocks
