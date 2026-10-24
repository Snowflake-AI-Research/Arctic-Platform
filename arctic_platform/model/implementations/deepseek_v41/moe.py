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

"""All-routed DeepSeek-V4.1 MoE.

Every backbone layer uses the shared top-k router. There is no hash table. ``expert_bias`` steers
selection for text tokens and ``bias_vl`` steers image-span tokens; neither scales the expert
output. The SwiGLU clamp is the shared asymmetric one (gate from above, up on both sides).
"""

from contextlib import contextmanager

import torch
from torch import nn

from arctic_platform.model.implementations.deepseek_v4.moe import Mxfp4GroupedExperts
from arctic_platform.model.implementations.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config
from arctic_platform.model.implementations.deepseek_v41.dequantize import packs_experts
from arctic_platform.model.implementations.moe.distributed.ep_backend import uses_dispatch_ep
from arctic_platform.model.implementations.moe.layers.moe import MoE
from arctic_platform.model.implementations.moe.layers.moe import MoEArgs


@contextmanager
def _integer_histc_on_cpu():
    """Route expert counts through ``bincount`` where CPU ``histc`` has no integer kernel."""
    original = torch.histc

    def histc(values, bins=100, min=0, max=0, *, out=None):
        if values.is_cuda or values.dtype.is_floating_point:
            return original(values, bins=bins, min=min, max=max, out=out)
        counts = torch.bincount(values.reshape(-1).to(dtype=torch.long), minlength=bins)
        return counts[:bins].to(dtype=torch.float32)

    torch.histc = histc
    try:
        yield
    finally:
        torch.histc = original


class DeepseekV41MoE(MoE):
    def __init__(self, config: DeepseekV41Config, layer_idx: int):
        if config.hidden_act != "silu":
            raise ValueError(f"DeepSeek V4.1 experts implement SiLU, got hidden_act={config.hidden_act!r}")
        # The for-loop expert path is what a CPU forward can differentiate. Grouped mm is CUDA-only.
        super().__init__(
            MoEArgs(
                num_experts=config.n_routed_experts,
                num_shared_experts=1 if config.n_shared_experts > 0 else 0,
                score_func=config.scoring_func,
                route_norm=config.norm_topk_prob,
                route_scale=config.routed_scaling_factor,
                score_before_experts=False,
                top_k=config.num_experts_per_tok,
                load_balance_coeff=1e-3,
                swiglu_limit=config.swiglu_limit,
            ),
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
        )
        if packs_experts(config):
            self.experts = Mxfp4GroupedExperts(
                dim=config.hidden_size,
                hidden_dim=config.moe_intermediate_size,
                num_experts=config.n_routed_experts,
                swiglu_limit=config.swiglu_limit,
            )
            self.experts.set_ep_comm_backend(self.ep_comm_backend)
        self.layer_idx = layer_idx
        self.is_hash = False
        if config.vision_n_layers > 0:
            self.register_buffer("bias_vl", torch.zeros(config.n_routed_experts, dtype=torch.float32), persistent=True)
        else:
            self.bias_vl = None

    def init_weights(self, init_std: float, buffer_device: torch.device) -> None:
        super().init_weights(init_std, buffer_device)
        if not isinstance(self.experts, Mxfp4GroupedExperts):
            nn.init.trunc_normal_(self.experts.w3, mean=0.0, std=0.02)
        if self.bias_vl is not None:
            self.bias_vl.zero_()

    def _selection_bias(self, num_tokens: int, image_mask: torch.Tensor | None) -> torch.Tensor | None:
        bias = self.expert_bias
        if image_mask is None or self.bias_vl is None:
            return bias
        mask = image_mask.reshape(-1).to(dtype=torch.bool)
        if mask.numel() != num_tokens:
            raise ValueError(f"image_mask has {mask.numel()} tokens and the MoE input has {num_tokens}")
        return torch.where(mask.unsqueeze(-1), self.bias_vl, bias)

    def forward(
        self,
        x: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        flat = x.reshape(-1, dim)
        if routed_experts is not None:
            routed_experts = routed_experts.reshape(-1, routed_experts.shape[-1])

        with _integer_histc_on_cpu():
            top_scores, selected, num_tokens_per_expert = self.router(
                flat, self._selection_bias(flat.shape[0], image_mask), routed_experts=routed_experts
            )
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        if uses_dispatch_ep(self.ep_comm_backend) and getattr(self.experts, "_ep_group", None) is not None:
            routed = self._run_deepep_routed_experts(flat, selected, top_scores)
            return routed.reshape(batch, seq_len, dim)

        with _integer_histc_on_cpu():
            top_scores, token_indices, num_tokens_per_expert = self.reorderer(top_scores, selected)
        # Grouped mm is CUDA-only; the for-loop is the same expert math on CPU.
        grouped = self.experts.use_grouped_mm
        self.experts.use_grouped_mm = grouped and flat.is_cuda
        try:
            routed = self._run_routed_experts(
                flat,
                token_indices,
                num_tokens_per_expert.to(dtype=torch.long),
                top_scores,
            )
        finally:
            self.experts.use_grouped_mm = grouped
        out = self.shared_expert(flat) if self.shared_expert is not None else torch.zeros_like(flat)
        routed_indices = token_indices.reshape(-1, 1).expand(-1, dim)
        out = out.scatter_add(dim=0, index=routed_indices, src=routed)
        return out.reshape(batch, seq_len, dim)


__all__ = ["DeepseekV41MoE"]
