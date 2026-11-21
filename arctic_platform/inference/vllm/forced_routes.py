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

"""Replay the stored experts on the part of a continuation's bounded tail that Model Runner V2 recomputes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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
    """One worker's stored rows per request, and the forced ids and mask of the step being run."""

    def __init__(self, max_num_tokens: int, num_layers: int, top_k: int, device: torch.device) -> None:
        self.rows: dict[str, tuple[int, np.ndarray]] = {}
        self.ids = torch.zeros((max_num_tokens, num_layers, top_k), dtype=torch.int32, device=device)
        self.mask = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)

    def add(self, request: Any) -> None:
        forced = (request.sampling_params.extra_args or {}).get(FORCED_ROUTES_KEY)
        # Public vLLM xargs accept scalars and lists. Only the server emits a mapping with binary rows.
        if not isinstance(forced, dict) or not isinstance(forced.get("rows"), bytes):
            return
        rows = np.frombuffer(forced["rows"], dtype=forced["dtype"]).reshape(-1, *self.ids.shape[1:])
        start = forced["start"]
        # The frontend rejects a capture reaching the end of the effective prompt; never execute its rows.
        if start + len(rows) < request.prompt_len:
            self.rows[request.req_id] = (start, rows)

    def fill(self, req_ids: list[str], query_start_loc: np.ndarray, num_computed: np.ndarray, num_rows: int) -> None:
        """Mark the batch rows that compute a stored position, with the same boundaries the routed-experts capture
        uses (``AuxOutputWorkerConnector.prepare_output``); every other row up to ``num_rows`` is cleared."""
        mask = np.zeros(num_rows, dtype=np.bool_)
        for i, req_id in enumerate(req_ids):
            if req_id not in self.rows:
                continue
            start, rows = self.rows[req_id]
            first = int(num_computed[i])
            lo = max(first, start)
            hi = min(first + int(query_start_loc[i + 1] - query_start_loc[i]), start + len(rows))
            if lo < hi:
                at = int(query_start_loc[i]) + lo - first
                mask[at : at + hi - lo] = True
                stored = torch.from_numpy(rows[lo - start : hi - start].astype(np.int32))
                self.ids[at : at + hi - lo].copy_(stored, non_blocking=True)
        self.mask[:num_rows].copy_(torch.from_numpy(mask), non_blocking=True)


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
    forced = ForcedRoutes(capturer.device_buffer.shape[0], *capturer.shape_per_token, runner.device)
    for module in runner.model.modules():
        if isinstance(module, MoERunner):
            bind_router(module.router, module.layer_id, forced)
    return forced


def patch_model_runner(runner_cls: Any) -> None:
    if getattr(runner_cls.prepare_inputs, "_arctic_forced_routes", False):
        return
    # Until initialize_kv_cache installs the hook on an engine that captures routes, or records why it cannot.
    runner_cls._forced_routes = None
    runner_cls._forced_routes_unsupported = "routed-experts capture is disabled"
    original_initialize_kv_cache = runner_cls.initialize_kv_cache
    original_add_requests = runner_cls.add_requests
    original_remove_request = runner_cls._remove_request
    original_prepare_inputs = runner_cls.prepare_inputs

    def initialize_kv_cache(self, *args, **kwargs):
        result = original_initialize_kv_cache(self, *args, **kwargs)
        if self.aux_output_connector is not None:
            self._forced_routes_unsupported = unsupported(self)
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if tp_size > 1:
                from vllm.distributed.parallel_state import get_tp_group

                reasons: list[str | None] = [None] * tp_size
                torch.distributed.all_gather_object(
                    reasons, self._forced_routes_unsupported, group=get_tp_group().cpu_group
                )
                self._forced_routes_unsupported = next((reason for reason in reasons if reason is not None), None)
            if self._forced_routes_unsupported is None:
                self._forced_routes = install(self)
        return result

    def add_requests(self, scheduler_output):
        original_add_requests(self, scheduler_output)
        if self._forced_routes is not None:
            for new_req_data in scheduler_output.scheduled_new_reqs:
                self._forced_routes.add(new_req_data)

    def _remove_request(self, req_id):
        if self._forced_routes is not None:
            self._forced_routes.rows.pop(req_id, None)
        return original_remove_request(self, req_id)

    def prepare_inputs(self, *args, **kwargs):
        input_batch = original_prepare_inputs(self, *args, **kwargs)
        if self._forced_routes is not None:
            self._forced_routes.fill(
                input_batch.req_ids,
                input_batch.query_start_loc_np,
                input_batch.num_computed_tokens_np,
                input_batch.num_tokens_after_padding,
            )
        return input_batch

    prepare_inputs._arctic_forced_routes = True
    runner_cls.initialize_kv_cache = initialize_kv_cache
    runner_cls.add_requests = add_requests
    runner_cls._remove_request = _remove_request
    runner_cls.prepare_inputs = prepare_inputs


def forced_route_blocks(worker: Any) -> tuple[int, int] | str:
    """Report the engine core's resolved cache-hit geometry, or why this worker cannot force routes."""
    runner = worker.model_runner
    reason = getattr(runner, "_forced_routes_unsupported", "the engine runs Model Runner V1")
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
