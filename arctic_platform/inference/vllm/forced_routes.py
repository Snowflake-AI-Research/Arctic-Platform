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

"""Run a continuation's re-prefilled positions under the routes its previous request stored (Model Runner V2).

A request naming its previous capture (``dss_router_replay_prefix_id``) gets that capture back over the positions it
covers. Its prefix-cache hit stops at a block boundary (one block lower when speculative decoding drops the trailing
block), so the engine recomputes the capture's last positions with its own router, which can pick different experts
than the request that first computed them. The returned routes and the state the rest of the request is sampled on
then disagree.

The server sends those stored rows with the request (``FORCED_ROUTES_KEY``). Each step, the runner marks the batch
rows that compute a stored position, and every MoE router takes the stored experts there, weighted as the trainer's
replay weights them. The capture then records the routes the forward used, so the returned capture is the stored one
by construction. Positions without a stored row route as before.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

#: ``SamplingParams.extra_args`` key: ``{"start": first position, "dtype": numpy dtype str, "rows": bytes}``, rows of
#: ``[layers, top_k]`` expert ids. Bytes because vLLM's msgpack decodes an ``Any`` field's ndarray to a bare list.
FORCED_ROUTES_KEY = "dss_router_replay_forced_routes"

_APPLIED_ATTR = "_arctic_forced_routes"

# The trainer's score functions (TokenChoiceTopKRouter.forward), applied to float32 logits.
_SCORES = {
    "sigmoid": torch.sigmoid,
    "softmax": lambda logits: F.softmax(logits, dim=1),
    "sqrtsoftplus": lambda logits: F.softplus(logits).sqrt(),
}


def _router_scales() -> dict[type, bool]:
    """Routers whose weights gather their scores; True when the router applies routed_scaling_factor itself."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import FusedTopKBiasRouter
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter

    return {GroupedTopKRouter: True, FusedTopKBiasRouter: True, FusedTopKRouter: False}


def replay_weights(router: Any, router_logits: torch.Tensor, ids: torch.Tensor, scaled: bool) -> torch.Tensor:
    """The weights the trainer's replay gives ``ids`` (``TokenChoiceTopKRouter.forward`` with ``routed_experts``).

    The scale is the router's own: a router that leaves routed_scaling_factor to the MoE output gets 1 here too.
    """
    weights = _SCORES[router.scoring_func](router_logits.float()).gather(1, ids.long())
    if router.renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * router.routed_scaling_factor if scaled else weights


class ForcedRoutes:
    """One worker's stored rows per request, and the forced ids and mask of the step being run."""

    def __init__(self, max_num_tokens: int, num_layers: int, top_k: int, device: torch.device) -> None:
        self.rows: dict[str, tuple[int, np.ndarray]] = {}
        self.ids = torch.zeros((max_num_tokens, num_layers, top_k), dtype=torch.int32, device=device)
        self.mask = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)

    def add(self, req_id: str, sampling_params: Any) -> None:
        # Pooling requests carry no sampling params.
        if sampling_params is None or FORCED_ROUTES_KEY not in (sampling_params.extra_args or {}):
            return
        forced = sampling_params.extra_args[FORCED_ROUTES_KEY]
        rows = np.frombuffer(forced["rows"], dtype=forced["dtype"]).reshape(-1, *self.ids.shape[1:])
        self.rows[req_id] = (int(forced["start"]), rows)

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
    scaled = _router_scales()[type(router)]
    compute = type(router)._compute_routing.__get__(router)

    def _compute_routing(hidden_states, router_logits, indices_type, *, input_ids=None):
        weights, ids = compute(hidden_states, router_logits, indices_type, input_ids=input_ids)
        n = ids.shape[0]
        mask = forced.mask[:n, None]
        stored = forced.ids[:n, layer_id].to(ids.dtype)
        stored_weights = replay_weights(router, router_logits, stored, scaled).to(weights.dtype)
        return torch.where(mask, stored_weights, weights), torch.where(mask, stored, ids)

    router._compute_routing = _compute_routing


def install(runner: Any) -> ForcedRoutes:
    """Bind every MoE router of ``runner.model``, or refuse the engine if a stored row could not reach the router."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner

    parallel = runner.vllm_config.parallel_config
    fix = "or start the engine without enable_return_routed_experts"
    if parallel.data_parallel_size > 1:
        raise ValueError(
            "Router replay forces stored routes by batch row, but with data_parallel_size > 1 the router sees "
            f"tokens from other DP ranks (and sequence-parallel MoE or DBO splits them): serve with DP 1 {fix}."
        )
    if parallel.prefill_context_parallel_size > 1:
        raise ValueError(
            "Router replay forces stored routes by batch row, but prefill context parallel partitions the batch: "
            f"serve with prefill_context_parallel_size 1 {fix}."
        )
    capturer = runner.aux_output_connector._capturer
    forced = ForcedRoutes(capturer.device_buffer.shape[0], *capturer.shape_per_token, runner.device)
    supported = _router_scales()
    for module in runner.model.modules():
        if not isinstance(module, MoERunner):
            continue
        router = module.router
        if module._quant_method.is_monolithic:
            unsupported = "monolithic MoE kernel, which routes inside the kernel"
        elif type(router) not in supported or router.scoring_func not in _SCORES:
            unsupported = f"router {type(router).__name__}"
        else:
            bind_router(router, module.layer_id, forced)
            continue
        raise ValueError(
            f"Router replay cannot force stored routes through layer {module.layer_id}'s {unsupported}; it needs a "
            f"modular MoE kernel and a {'/'.join(sorted(c.__name__ for c in supported))} router scoring with "
            f"{'/'.join(sorted(_SCORES))}: choose such a moe_backend {fix}."
        )
    return forced


def patch_model_runner(runner_cls: Any) -> None:
    if getattr(runner_cls.prepare_inputs, _APPLIED_ATTR, False):
        return
    original_init = runner_cls.__init__
    original_initialize_kv_cache = runner_cls.initialize_kv_cache
    original_add_requests = runner_cls.add_requests
    original_remove_request = runner_cls._remove_request
    original_prepare_inputs = runner_cls.prepare_inputs

    def __init__(self, *args, **kwargs):
        self._forced_routes = None
        original_init(self, *args, **kwargs)

    def initialize_kv_cache(self, *args, **kwargs):
        result = original_initialize_kv_cache(self, *args, **kwargs)
        if self.aux_output_connector is not None:
            self._forced_routes = install(self)
        return result

    def add_requests(self, scheduler_output):
        original_add_requests(self, scheduler_output)
        if self._forced_routes is not None:
            for new_req_data in scheduler_output.scheduled_new_reqs:
                self._forced_routes.add(new_req_data.req_id, new_req_data.sampling_params)

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

    for name, fn, orig in (
        ("__init__", __init__, original_init),
        ("initialize_kv_cache", initialize_kv_cache, original_initialize_kv_cache),
        ("add_requests", add_requests, original_add_requests),
        ("_remove_request", _remove_request, original_remove_request),
        ("prepare_inputs", prepare_inputs, original_prepare_inputs),
    ):
        setattr(fn, _APPLIED_ATTR, True)
        fn._orig = orig
        setattr(runner_cls, name, fn)


def ensure_forced_routes_patch() -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    patch_model_runner(GPUModelRunner)
