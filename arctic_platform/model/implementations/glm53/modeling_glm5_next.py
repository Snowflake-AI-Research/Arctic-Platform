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

from __future__ import annotations

import torch
from torch import Tensor
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.layers.moe import MoE
from arctic_platform.model.implementations.moe.layers.moe import MoEArgs

from .converting_glm5_next import convert_hf_layer_to_prime
from .converting_glm5_next import convert_hf_to_prime
from .converting_glm5_next import convert_prime_layer_to_hf
from .converting_glm5_next import convert_prime_to_hf


class Glm5NextForConditionalGenerationPrimeRL(
    Glm5NextForConditionalGeneration,
    PreTrainedModelPrimeRL,
):
    def __init__(self, config, **_kwargs):
        if getattr(config, "quantization_config", None):
            raise NotImplementedError(
                "Training currently supports the BF16 GLM-5.3-Flash checkpoint. "
                "Use zai-org/GLM-5.3-Flash-BF16; native-FP8 training support is not "
                "implemented yet."
            )
        super().__init__(config)
        text_config = config.text_config
        for layer in self.model.language_model.layers:
            if not isinstance(layer.mlp, Glm5NextTextMoE):
                continue
            layer.mlp = MoE(
                MoEArgs(
                    num_experts=text_config.n_routed_experts,
                    num_shared_experts=text_config.n_shared_experts,
                    score_func=getattr(text_config, "scoring_func", "sigmoid"),
                    route_norm=text_config.norm_topk_prob,
                    route_scale=text_config.routed_scaling_factor,
                    score_before_experts=False,
                    top_k=text_config.num_experts_per_tok,
                    load_balance_coeff=getattr(
                        text_config,
                        "router_aux_loss_coef",
                        1e-3,
                    ),
                    use_grouped_mm=getattr(config, "use_grouped_mm", True),
                    swiglu_limit=text_config.swiglu_limit,
                ),
                dim=text_config.hidden_size,
                hidden_dim=text_config.moe_intermediate_size,
            )
        self._is_vlm = True

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(".mlp.experts.0.gate_proj.weight" in name for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(".mlp.experts.w1" in name for name in state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_prime_to_hf(state_dict)

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_hf_to_prime(state_dict)

    @classmethod
    def convert_layer_to_hf(
        cls,
        state_dict: dict[str, Tensor],
        layer_idx: int,
    ) -> dict[str, Tensor]:
        return convert_prime_layer_to_hf(state_dict, layer_idx)

    @classmethod
    def convert_layer_to_prime(
        cls,
        state_dict: dict[str, Tensor],
        layer_idx: int,
    ) -> dict[str, Tensor]:
        return convert_hf_layer_to_prime(state_dict, layer_idx)

    @classmethod
    def convert_layer_to_vllm_kernel(
        cls,
        state_dict: dict[str, Tensor],
        layer_idx: int,
        quantize_fp8: bool = False,
    ) -> dict[str, Tensor]:
        if quantize_fp8:
            raise NotImplementedError("GLM-5.3-Flash on-the-fly FP8 export is not supported")
        from .vllm_weights import convert_glm5_next_layer_to_vllm

        return convert_glm5_next_layer_to_vllm(state_dict, layer_idx)

    def init_buffers_post_meta(self) -> None:
        rotary = self.model.visual.rotary_pos_emb
        if hasattr(rotary, "compute_axial_rope_parameters"):
            inv_freq, rotary.attention_scaling = rotary.compute_axial_rope_parameters(
                rotary.config,
                rotary.inv_freq.device,
            )
            rotary.inv_freq.copy_(inv_freq)
            rotary.original_inv_freq.copy_(inv_freq)
            return
        inv_freq = 1.0 / (
            rotary.theta
            ** (
                torch.arange(
                    0,
                    rotary.dim,
                    2,
                    dtype=torch.float32,
                    device=rotary.inv_freq.device,
                )
                / rotary.dim
            )
        )
        rotary.inv_freq.copy_(inv_freq)
