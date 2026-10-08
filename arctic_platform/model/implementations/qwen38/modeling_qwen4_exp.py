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
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextSparseMoeBlock
from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers

from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.layers.moe import FeedForward
from arctic_platform.model.implementations.moe.layers.moe import MoE
from arctic_platform.model.implementations.moe.layers.moe import MoEArgs

from .converting_qwen4_exp import convert_hf_layer_to_prime
from .converting_qwen4_exp import convert_hf_to_prime
from .converting_qwen4_exp import convert_prime_layer_to_hf
from .converting_qwen4_exp import convert_prime_to_hf


class EPShardedEmbedding(nn.Embedding):
    """Vocabulary-sharded embedding reduced across the expert-parallel group."""

    _dss_shard_on_ep = True
    _dss_skip_weight_sync = True

    def forward(self, input: Tensor) -> Tensor:
        world_size = getattr(self, "_ep_world_size", 1)
        if world_size == 1:
            return F.embedding(input, self.weight)

        rank = self._ep_rank
        rows_per_rank, remainder = divmod(self.num_embeddings, world_size)
        local_rows = rows_per_rank + int(rank < remainder)
        start = rank * rows_per_rank + min(rank, remainder)
        if self.weight.shape[0] != local_rows:
            raise RuntimeError(
                "Qwen3.8 PLE embedding shard has an unexpected shape: "
                f"rank={rank}, expected={local_rows}, actual={self.weight.shape[0]}"
            )

        local_mask = (input >= start) & (input < start + local_rows)
        local_input = (input - start).clamp(min=0, max=local_rows - 1)
        output = F.embedding(local_input, self.weight)
        output = output * local_mask.unsqueeze(-1).to(output.dtype)
        return dist_nn.all_reduce(output, group=self._ep_group)


class Qwen4ExpSparseMoePrimeRL(MoE):
    def __init__(self, config, *, use_grouped_mm: bool):
        super().__init__(
            MoEArgs(
                num_experts=config.num_experts,
                num_shared_experts=0,
                score_func="softmax",
                route_norm=config.norm_topk_prob,
                route_scale=1.0,
                score_before_experts=False,
                top_k=config.num_experts_per_tok,
                use_grouped_mm=use_grouped_mm,
                load_balance_coeff=None,
            ),
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
        )
        self.qwen_shared_expert = FeedForward(
            dim=config.hidden_size,
            hidden_dim=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        routed_experts: Tensor | None = None,
    ) -> Tensor:
        routed_output = super().forward(
            hidden_states,
            routed_experts=routed_experts,
        )
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        shared_output = self.qwen_shared_expert(flat)
        shared_output = torch.sigmoid(self.shared_expert_gate(flat)) * shared_output
        return routed_output + shared_output.view_as(hidden_states)


class Qwen4ExpTextDecoderLayerPrimeRL(Qwen4ExpTextDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        conv_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        ple_input_ids: torch.LongTensor | None = None,
        routed_experts: Tensor | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(
                hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask
            )

        hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, cache_params=past_key_values, attention_mask=conv_mask, **kwargs
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)

        hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        routes = routed_experts[:, :, self._replay_layer_idx, :] if routed_experts is not None else None
        hidden_states = self.mlp(hidden_states, routed_experts=routes)

        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)
        return hidden_states


class Qwen4ExpForConditionalGenerationPrimeRL(
    Qwen4ExpForConditionalGeneration,
    PreTrainedModelPrimeRL,
):
    def __init__(self, config, **_kwargs):
        if getattr(config, "quantization_config", None):
            raise NotImplementedError(
                "Training currently supports the BF16 Qwen3.8-Flash-Next "
                "checkpoint. Native-FP8 training is not implemented."
            )
        super().__init__(config)
        text_config = config.text_config
        use_grouped_mm = getattr(config, "use_grouped_mm", True)
        for layer_idx, layer in enumerate(self.model.language_model.layers):
            layer.__class__ = Qwen4ExpTextDecoderLayerPrimeRL
            layer._replay_layer_idx = layer_idx
            if isinstance(layer.mlp, Qwen4ExpTextSparseMoeBlock):
                layer.mlp = Qwen4ExpSparseMoePrimeRL(
                    text_config,
                    use_grouped_mm=use_grouped_mm,
                )
            if layer.ple is not None:
                embedding = layer.ple.ple_embedding.ngram_embedding
                sharded_embedding = EPShardedEmbedding(
                    embedding.num_embeddings,
                    embedding.embedding_dim,
                    device=embedding.weight.device,
                    dtype=embedding.weight.dtype,
                )
                sharded_embedding.weight.requires_grad_(False)
                sharded_embedding.weight._dss_skip_weight_sync = True
                layer.ple.ple_embedding.ngram_embedding = sharded_embedding
        self._is_vlm = True
        self._requires_hf_weight_sync = True

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(".mlp.experts.gate_up_proj" in name or ".ngram_embedding.shard_" in name for name in state_dict)

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

    def init_buffers_post_meta(self) -> None:
        for rotary in (
            self.model.language_model.rotary_emb,
            self.model.visual.rotary_pos_emb,
        ):
            init_fn = getattr(
                rotary,
                "compute_axial_rope_parameters",
                getattr(rotary, "compute_default_rope_parameters", None),
            )
            if init_fn is None:
                continue
            inv_freq, rotary.attention_scaling = init_fn(
                rotary.config,
                rotary.inv_freq.device,
            )
            rotary.inv_freq.copy_(inv_freq)
            if hasattr(rotary, "original_inv_freq"):
                rotary.original_inv_freq.copy_(inv_freq)

        for module in self.modules():
            if not isinstance(module, Qwen4ExpTextNGramEmbedding):
                continue
            module.layer_multipliers.copy_(
                _build_layer_multipliers(
                    module.unigram_vocab_size,
                    module.ngram_size,
                    module.ple_layer_index,
                    module.seed,
                ).to(module.layer_multipliers.device)
            )
            module.ngram_heads_vocab_sizes.copy_(
                torch.tensor(
                    module.head_vocab_sizes,
                    dtype=torch.long,
                    device=module.ngram_heads_vocab_sizes.device,
                )
            )
            module.ngram_heads_offsets.copy_(
                torch.tensor(
                    module.head_offsets,
                    dtype=torch.long,
                    device=module.ngram_heads_offsets.device,
                )
            )
