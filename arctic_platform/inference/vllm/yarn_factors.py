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

"""Per-request YaRN via V2's private mRoPE positions and concatenated tables."""

import torch

from arctic_platform.common.yarn_factors import build_yarn_factors

_PATCHED = False


def install_yarn_tables(model: torch.nn.Module, text_config, factors: list[float], max_model_len: int) -> int:
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    tables = build_yarn_factors(text_config, factors)
    modules = [module for module in model.modules() if isinstance(module, MRotaryEmbedding)]
    if not modules:
        raise ValueError("yarn_factors requires an mRoPE model")
    rows = modules[0].cos_sin_cache.shape[0]
    if max_model_len > rows:
        raise ValueError("yarn_factors max_model_len exceeds the rotary table rows")
    if rows * len(tables.factors) > torch.iinfo(torch.int32).max:
        raise ValueError("yarn_factors table offsets exceed int32")
    caches = {}
    for module in modules:
        old = module.cos_sin_cache
        if old.shape != (rows, tables.inv_freq.shape[1] * 2):
            raise ValueError("yarn_factors requires matching mRoPE table dimensions")
        key = (old.device, old.dtype)
        if key not in caches:
            positions = torch.arange(rows, device=old.device, dtype=torch.float32)
            slots = []
            for inv_freq, scale in zip(tables.inv_freq, tables.attention_scaling):
                freqs = torch.outer(positions, inv_freq.to(old.device))
                scale = scale.to(old.device)
                slots.append(torch.cat((freqs.cos() * scale, freqs.sin() * scale), -1).to(old.dtype))
            caches[key] = torch.cat(slots)
        module.cos_sin_cache = caches[key]
    return rows


def ensure_yarn_factor_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from vllm.v1.worker.gpu.mm.rope import RopeState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    original_init = DefaultModelState.__init__
    original_add = DefaultModelState.add_request
    original_positions = RopeState.init_prefill_positions

    def init(self, vllm_config, model, encoder_cache, device):
        original_init(self, vllm_config, model, encoder_cache, device)
        factors = vllm_config.additional_config.get("yarn_factors", [])
        self._yarn_rows = 0
        if factors:
            if self.rope_state is None:
                raise ValueError("yarn_factors requires an mRoPE model")
            if self.mm_pruner is not None:
                raise ValueError("yarn_factors does not support multimodal position pruning")
            self._yarn_rows = install_yarn_tables(model, self.model_config.hf_text_config, factors, self.max_model_len)
            self._yarn_num_factors = len(set(torch.tensor(factors, dtype=torch.float32).tolist()))

    def add_request(self, req_index, new_req_data):
        if not self._yarn_rows:
            return original_add(self, req_index, new_req_data)
        extra = new_req_data.sampling_params.extra_args or {}
        slot = extra.get("yarn_factor_slot", 0)
        if not isinstance(slot, int) or not 0 <= slot < self._yarn_num_factors:
            raise ValueError("Invalid yarn_factor_slot")
        self.rope_state._yarn_offset = slot * self._yarn_rows
        try:
            return original_add(self, req_index, new_req_data)
        finally:
            self.rope_state._yarn_offset = 0

    def init_prefill_positions(self, req_idx, model, prefill_token_ids, mm_features):
        offset = getattr(self, "_yarn_offset", 0)
        if not offset:
            return original_positions(self, req_idx, model, prefill_token_ids, mm_features)
        positions, delta = model.get_mrope_input_positions(prefill_token_ids, mm_features)
        positions = positions + offset
        delta += offset
        limit = torch.iinfo(torch.int32)
        if (
            positions.min() < limit.min
            or positions.max() > limit.max
            or not limit.min <= delta <= limit.max - self.max_model_len
        ):
            raise ValueError("yarn_factors request positions exceed int32")
        self.prefill_delta.np[req_idx] = delta
        for i in range(self.num_dims):
            self.prefill_positions.stage_write(self.num_dims * req_idx + i, 0, positions[i].tolist())

    DefaultModelState.__init__ = init
    DefaultModelState.add_request = add_request
    RopeState.init_prefill_positions = init_prefill_positions
    _PATCHED = True
