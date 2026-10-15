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

"""Per-token YaRN at the text backbone, before checkpointed decoder layers."""

from contextvars import ContextVar
from functools import wraps

import torch

from arctic_platform.common.yarn_factors import build_yarn_factors


def install_yarn_factors(model: torch.nn.Module, factors: list[float]) -> None:
    backbone = model.model
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    rotary = backbone.rotary_emb
    if not hasattr(rotary, "recomposition_frequencies") or not hasattr(rotary, "mrope_section"):
        raise ValueError("yarn_factors requires a text mRoPE backbone")
    table = build_yarn_factors(backbone.config, factors)
    rotary.register_buffer("yarn_inv_freq", table.inv_freq.to(rotary.inv_freq.device), persistent=False)
    rotary.register_buffer(
        "yarn_attention_scaling",
        table.attention_scaling.to(rotary.inv_freq.device),
        persistent=False,
    )
    rotary.register_buffer(
        "yarn_values",
        torch.tensor(table.factors, dtype=torch.float32, device=rotary.inv_freq.device),
        persistent=False,
    )
    active_factor = ContextVar("yarn_factor", default=None)
    original_backbone = backbone.forward
    original_rotary = rotary.forward

    @wraps(original_backbone)
    def forward(*args, yarn_factor=None, **kwargs):
        token = active_factor.set(yarn_factor)
        try:
            return original_backbone(*args, **kwargs)
        finally:
            active_factor.reset(token)

    @torch.no_grad()
    def rotary_forward(x, position_ids):
        factor = active_factor.get()
        if factor is None:
            return original_rotary(x, position_ids)
        if factor.shape != x.shape[:2]:
            raise ValueError("yarn_factor must have shape [batch, tokens]")
        matches = factor.to(device=x.device, dtype=torch.float32)[..., None] == rotary.yarn_values
        if not matches.any(dim=-1).all():
            raise ValueError("yarn_factor is not in declared yarn_factors")
        slots = matches.to(torch.int64).argmax(dim=-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = position_ids[..., None].float() * rotary.yarn_inv_freq[slots].float()[None]
            scale = rotary.yarn_attention_scaling[slots][None, ..., None].float()
            cos = rotary.recomposition_frequencies(freqs.cos() * scale)
            sin = rotary.recomposition_frequencies(freqs.sin() * scale)
        return cos.to(x.dtype), sin.to(x.dtype)

    backbone.forward = forward
    rotary.forward = rotary_forward
