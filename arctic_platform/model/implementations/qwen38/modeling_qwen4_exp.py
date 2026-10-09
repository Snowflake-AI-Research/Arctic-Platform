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

import math

import torch
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_mask_to_padding_states
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextSparseMoeBlock
from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers

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


class Qwen4ExpTextNGramEmbeddingPrimeRL(Qwen4ExpTextNGramEmbedding):
    def forward(self, input_ids: torch.Tensor, past_key_values: Cache | None, position_ids: Tensor | None = None) -> torch.Tensor:
        input_ids = input_ids.long()
        # This is a trick to store the previous N=self.context_len `input_ids` - indeed the manipulations are identical to storing
        # a past conv_state, so we can use an additional conv_states inside the Cache for it
        if past_key_values is not None and past_key_values.has_previous_state(self.layer_idx, state_idx=2):
            previous_context = past_key_values.layers[self.layer_idx].conv_states[2].clone()
        else:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        # Store the current input_ids for the next forward
        if past_key_values is not None:
            input_ids_to_cache = input_ids
            # In the case where `input_ids` would be smaller than `self.context_len`, the `update_conv_state` will pad with zeros, whereas
            # here we want to pad with eos, so we do it explicitly
            if (
                not past_key_values.has_previous_state(self.layer_idx, state_idx=2)
                and input_ids.shape[1] < self.context_len
            ):
                input_ids_to_cache = torch.nn.functional.pad(
                    input_ids_to_cache, (self.context_len - input_ids.shape[1], 0), value=self.eos_token_id
                )
            _ = past_key_values.update_conv_state(
                input_ids_to_cache, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len
            )

        # Get full token history
        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]

        if position_ids is not None and past_key_values is None:
            positions = F.pad(position_ids, (self.context_len, 0), value=-1)
            shifted_tokens = [
                torch.where(positions >= shift, tokens, self.eos_token_id) if shift else tokens
                for shift, tokens in enumerate(shifted_tokens)
            ]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]
        # We need explicit device placement here, as the embedding may be skipped from device_map completely (we just need to be
        # careful in the case of offloading to disk)
        execution_device = (
            self.ngram_embedding.weight.device if self.ngram_embedding.weight.device.type != "meta" else None
        )
        return self.ngram_embedding(ngram_ids.to(execution_device)).to(ngram_ids.device).flatten(-2)


class Qwen4ExpTextPLELayerPrimeRL(Qwen4ExpTextPLELayer):
    def _short_conv(self, hidden_states, past_key_values, position_ids=None):
        if position_ids is None or past_key_values is not None:
            return super()._short_conv(hidden_states, past_key_values)
        dilation = self.conv1d.dilation[0]
        kernel_size = self.conv1d.kernel_size[0]
        padded = F.pad(hidden_states.transpose(1, 2), (self.short_conv_state_len, 0))
        windows = padded.unfold(-1, self.short_conv_state_len + 1, 1)[..., ::dilation]
        offsets = torch.arange(kernel_size - 1, -1, -1, device=hidden_states.device) * dilation
        valid = position_ids[..., None] >= offsets
        windows = windows * valid[:, None].to(windows.dtype)
        accumulation_dtype = torch.float32 if windows.dtype in (torch.float16, torch.bfloat16) else windows.dtype
        output = (windows.to(accumulation_dtype) * self.conv1d.weight[:, 0][None, :, None].to(accumulation_dtype)).sum(-1)
        output = output.to(hidden_states.dtype)
        return F.silu(output).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values: Cache | None,
        conv_mask: torch.Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.ple_embedding(input_ids, past_key_values, position_ids=position_ids)
        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        if conv_mask is not None:
            gated_value = apply_mask_to_padding_states(gated_value, conv_mask)
            gated_value_normed = apply_mask_to_padding_states(gated_value_normed, conv_mask)
        output = gated_value + self._short_conv(gated_value_normed, past_key_values, position_ids=position_ids)
        return output


class Qwen4ExpTextDecoderLayerPrimeRL(Qwen4ExpTextDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        conv_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        ple_input_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(
                hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask,
                position_ids=kwargs.pop("ple_position_ids", None),
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
        hidden_states = self.mlp(hidden_states)

        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)
        return hidden_states

class Qwen4ExpModelPrimeRL(Qwen4ExpModel):
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, *args, **kwargs):
        if position_ids is not None:
            kwargs["ple_position_ids"] = position_ids if position_ids.ndim == 2 else position_ids[0]
        return super().forward(input_ids, attention_mask, position_ids, *args, **kwargs)


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
        self.model.__class__ = Qwen4ExpModelPrimeRL
        text_config = config.text_config
        use_grouped_mm = getattr(config, "use_grouped_mm", True)
        for layer in self.model.language_model.layers:
            layer.__class__ = Qwen4ExpTextDecoderLayerPrimeRL
            if isinstance(layer.mlp, Qwen4ExpTextSparseMoeBlock):
                layer.mlp = Qwen4ExpSparseMoePrimeRL(
                    text_config,
                    use_grouped_mm=use_grouped_mm,
                )
            if layer.ple is not None:
                layer.ple.__class__ = Qwen4ExpTextPLELayerPrimeRL
                layer.ple.ple_embedding.__class__ = Qwen4ExpTextNGramEmbeddingPrimeRL
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
