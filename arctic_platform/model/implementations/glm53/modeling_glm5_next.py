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
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

from arctic_platform.model.implementations.fp8 import BlockFp8Linear
from arctic_platform.model.implementations.fp8 import fp8_weight_block_size
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.layers.moe import MoE
from arctic_platform.model.implementations.moe.layers.moe import MoEArgs

from .converting_glm5_next import convert_hf_layer_to_prime
from .converting_glm5_next import convert_hf_to_prime
from .converting_glm5_next import convert_prime_layer_to_hf
from .converting_glm5_next import convert_prime_to_hf


def _fp8_modules_to_skip(config) -> set[str]:
    quantization_config = getattr(config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        return set(quantization_config.get("modules_to_not_convert") or ())
    return set(getattr(quantization_config, "modules_to_not_convert", None) or ())


def _replace_native_fp8_linears(model: nn.Module, config, block_size: int) -> None:
    skipped = _fp8_modules_to_skip(config)
    language_model = model.model.language_model
    for name, module in list(language_model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, BlockFp8Linear):
            continue
        full_name = f"model.language_model.{name}"
        checkpoint_name = full_name.replace("model.language_model.", "model.", 1)
        checkpoint_name = checkpoint_name.replace(".mlp.router.gate", ".mlp.gate")
        checkpoint_name = checkpoint_name.replace(
            ".self_attn.forget_gate.f_a_proj",
            ".self_attn.f_a_proj",
        ).replace(
            ".self_attn.forget_gate.f_b_proj",
            ".self_attn.f_b_proj",
        )
        if checkpoint_name.endswith(".mlp.gate") or any(
            checkpoint_name == prefix or checkpoint_name.startswith(f"{prefix}.") for prefix in skipped
        ):
            continue
        if module.bias is not None:
            raise NotImplementedError(f"GLM-5.3 native FP8 linear {checkpoint_name!r} has a bias")
        parent_name, _, child_name = name.rpartition(".")
        parent = language_model.get_submodule(parent_name) if parent_name else language_model
        setattr(
            parent,
            child_name,
            BlockFp8Linear(
                module.in_features,
                module.out_features,
                block_size=block_size,
                device=module.weight.device,
            ),
        )




class Glm5NextTextDecoderLayerPrimeRL(Glm5NextTextDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        routed_experts: Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = hidden_states.dtype

        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
        topk_indices = None
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _, topk_indices = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, MoE):
            routes = routed_experts[:, :, self._replay_layer_idx, :] if routed_experts is not None else None
            hidden_states = self.mlp(hidden_states, routed_experts=routes)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        return hidden_states, topk_indices

class Glm5NextForConditionalGenerationPrimeRL(
    Glm5NextForConditionalGeneration,
    PreTrainedModelPrimeRL,
):
    def __init__(self, config, **_kwargs):
        fp8_block_size = fp8_weight_block_size(config)
        if getattr(config, "quantization_config", None) and fp8_block_size is None:
            raise NotImplementedError("GLM-5.3 training only supports BF16 or fine-grained FP8 checkpoints")
        super().__init__(config)
        text_config = config.text_config
        for layer_idx, layer in enumerate(self.model.language_model.layers):
            layer.__class__ = Glm5NextTextDecoderLayerPrimeRL
            layer._replay_layer_idx = layer_idx
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
                    fp8_block_size=fp8_block_size,
                    swiglu_limit=text_config.swiglu_limit,
                ),
                dim=text_config.hidden_size,
                hidden_dim=text_config.moe_intermediate_size,
            )
        if fp8_block_size is not None:
            _replace_native_fp8_linears(self, config, fp8_block_size)
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
