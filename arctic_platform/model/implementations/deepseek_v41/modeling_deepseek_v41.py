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

"""DeepSeek-V4.1 decoder. Shifted mHC, and shared sparse attention threaded between layers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast

from arctic_platform.model.implementations.deepseek_v4.attention import PackedContext
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import apply_hf_to_prime
from arctic_platform.model.implementations.deepseek_v4.conversion_ops import apply_prime_to_hf
from arctic_platform.model.implementations.deepseek_v4.cp import CPContext
from arctic_platform.model.implementations.deepseek_v4.cp import gather_for_cp
from arctic_platform.model.implementations.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from arctic_platform.model.implementations.deepseek_v41.attention import DeepseekV41Attention
from arctic_platform.model.implementations.deepseek_v41.attention import SharedAttentionState
from arctic_platform.model.implementations.deepseek_v41.attention import build_packed
from arctic_platform.model.implementations.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config
from arctic_platform.model.implementations.deepseek_v41.converting_deepseek_v41 import conversion_chain
from arctic_platform.model.implementations.deepseek_v41.converting_deepseek_v41 import log_deferred_checkpoint_keys
from arctic_platform.model.implementations.deepseek_v41.dequantize import dequantize_state_dict_
from arctic_platform.model.implementations.deepseek_v41.dequantize import packs_experts
from arctic_platform.model.implementations.deepseek_v41.engram import DeepseekV41Engram
from arctic_platform.model.implementations.deepseek_v41.engram import EngramHasher
from arctic_platform.model.implementations.deepseek_v41.engram import EngramLayout
from arctic_platform.model.implementations.deepseek_v41.hyperconnections import DeepseekV41HyperConnection
from arctic_platform.model.implementations.deepseek_v41.hyperconnections import collapse
from arctic_platform.model.implementations.deepseek_v41.hyperconnections import identity_pre
from arctic_platform.model.implementations.deepseek_v41.hyperconnections import write_back
from arctic_platform.model.implementations.deepseek_v41.moe import DeepseekV41MoE
from arctic_platform.model.implementations.deepseek_v41.streaming_conversion import register_e8m0_safetensors_dtype
from arctic_platform.model.implementations.deepseek_v41.streaming_conversion import write_streaming_prime_cache
from arctic_platform.model.implementations.layers.norms import RMSNorm
from arctic_platform.model.implementations.layers.norms import RMSNormConfig
from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.layers.lm_head import PrimeLmOutput
from arctic_platform.model.implementations.moe.layers.moe import MoE

# DCP's HF reader resolves the Engram ``F8_E8M0`` scale dtype through safetensors' table.
register_e8m0_safetensors_dtype()


def _seq_lens_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    flat = position_ids.reshape(-1)
    starts = (flat == 0).nonzero(as_tuple=True)[0]
    if starts.numel() == 0 or int(starts[0].item()) != 0:
        raise ValueError("DeepSeek V4.1 position_ids must be 0-based and restart at each document")
    ends = torch.cat([starts[1:], torch.tensor([flat.numel()], dtype=starts.dtype, device=starts.device)])
    return (ends - starts).to(dtype=torch.long)


class DeepseekV41DecoderLayer(GradientCheckpointingLayer):
    """Attention then MoE. Each sublayer is collapsed by the previous sublayer's ``pre``."""

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_idx: int,
        rotary_emb: DeepseekV4RotaryEmbedding,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV41Attention(config, layer_idx, rotary_emb)
        self.mlp = DeepseekV41MoE(config, layer_idx)
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.attn_hc = DeepseekV41HyperConnection(config)
        self.ffn_hc = DeepseekV41HyperConnection(config)
        self.engram = (
            DeepseekV41Engram(config, layer_idx, engram_layout)
            if engram_layout is not None and layer_idx in engram_layout.layer_ids
            else None
        )

    def forward(
        self,
        mhc_states: torch.Tensor,
        pre: torch.Tensor,
        shared: SharedAttentionState | None = None,
        routed_experts: torch.Tensor | None = None,
        *,
        packed: PackedContext,
        engram_hash_ids: torch.Tensor | None = None,
        engram_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, SharedAttentionState]:
        attention_state = SharedAttentionState.empty() if shared is None else shared
        if self.engram is not None:
            if engram_hash_ids is None:
                raise RuntimeError(f"layer {self.layer_idx} Engram needs hash ids")
            mhc_states = self.engram(mhc_states, engram_hash_ids, engram_token_mask)
        attn_pre, attn_post, attn_comb = self.attn_hc.mixes(mhc_states)
        attn_input = self.input_layernorm(collapse(mhc_states, pre))
        attn_output, attention_state = self.self_attn(attn_input, packed, attention_state)
        mhc_states = write_back(mhc_states, attn_output, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.mixes(mhc_states)
        mlp_input = self.post_attention_layernorm(collapse(mhc_states, attn_pre))
        mlp_output = self.mlp(mlp_input, routed_experts=routed_experts)
        return write_back(mhc_states, mlp_output, ffn_post, ffn_comb), ffn_pre, attention_state


KEEP_IN_FP32_MODULES = (
    "attn_hc",
    "ffn_hc",
    "sinks",
    "expert_bias",
    "bias_vl",
    "q_a_norm",
    "kv_norm",
    "compressor.wkv",
    "compressor.wgate",
    "k_norm",
    "input_layernorm",
    "post_attention_layernorm",
    "norm",
)


class DeepseekV41PreTrainedModel(PreTrainedModelPrimeRL):
    config: DeepseekV41Config
    config_class = DeepseekV41Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV41DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": DeepseekV41DecoderLayer}

    @classmethod
    def cp_blocker(cls, config, cp: int) -> str | None:
        del config, cp
        return None

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        init_std = self.config.initializer_range
        if isinstance(module, (DeepseekV41Attention, DeepseekV41HyperConnection)):
            module.init_weights(init_std)
        elif isinstance(module, DeepseekV41MoE):
            module.init_weights(init_std, module.tokens_per_expert.device)
        elif isinstance(module, DeepseekV41Engram):
            module.init_weights()
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            module.init_buffers_post_meta()
        elif isinstance(module, EngramHasher) and module.primes.device.type != "meta":
            module.init_buffers_post_meta()

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return any(module_name in name for module_name in KEEP_IN_FP32_MODULES)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(
            name.endswith("ffn.gate.weight")
            or "ffn.shared_experts." in name
            or ".attn.wq_a." in name
            or name in {"embed.weight", "head.weight", "norm.weight"}
            for name in state_dict
        )

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # A layer group is recognized by the MoE or attention names. The non-layer group has no
        # MoE key, so the embedding and LM head are what mark a converted snapshot.
        return any(
            name.endswith("mlp.router.gate.weight")
            or "mlp.shared_expert." in name
            or name.endswith(".self_attn.q_a_proj.weight")
            or name.endswith("model.embed_tokens.weight")
            or name == "lm_head.weight"
            or name.endswith("model.norm.weight")
            for name in state_dict
        )

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:  # type: ignore[override]
        """Rename a published checkpoint onto this module tree, after dequantizing to bf16.

        Engram embedding rows stay quantized, and so do routed experts the config keeps packed.
        Vision, the aligner, image embeddings, and MTP are logged and dropped. Any other leftover
        key raises.
        """
        log_deferred_checkpoint_keys(state_dict)
        dequantize_state_dict_(state_dict, keep_experts=packs_experts(self.config))
        return apply_hf_to_prime(state_dict, self.conversion_chain(self.config))

    def convert_to_hf(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:  # type: ignore[override]
        return apply_prime_to_hf(state_dict, self.conversion_chain(self.config))

    def stream_convert_to_prime(self, source_path: Path, output_path: Path) -> None:
        """Write the converted cache shard by shard. The full snapshot never sits in memory."""
        write_streaming_prime_cache(
            source_path,
            output_path,
            self.conversion_chain(self.config),
            keep_experts=packs_experts(self.config),
        )

    def init_buffers_post_meta(self) -> None:
        # ``to_empty`` uninitializes non-persistent buffers before checkpoint load. Rotary tables,
        # Engram hash tables, and ``tokens_per_expert`` are rebuilt from the config. The compressed
        # token map is not: it comes from the tokenizer and is installed after this runs.
        for module in self.modules():
            if isinstance(module, DeepseekV4RotaryEmbedding):
                module.init_buffers_post_meta()
            elif isinstance(module, EngramHasher) and module.primes.device.type != "meta":
                module.init_buffers_post_meta()
            elif isinstance(module, MoE) and module.tokens_per_expert.device.type != "meta":
                module.tokens_per_expert.zero_()


class DeepseekV41Model(DeepseekV41PreTrainedModel):
    def __init__(self, config: DeepseekV41Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.engram_hasher = EngramHasher(config) if config.engram_layer_ids else None
        engram_layout = None if self.engram_hasher is None else self.engram_hasher.layout
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(config, layer_idx, self.rotary_emb, engram_layout)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.gradient_checkpointing = False
        self.cp_context = CPContext()
        self.post_init()

    def install_compressed_token_map(self, token_map: Tensor) -> None:
        if self.engram_hasher is None:
            raise RuntimeError("this DeepSeek V4.1 model has no Engram layers")
        self.engram_hasher.install_compressed_token_map(token_map)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor | None = None,
        seq_lens_are_pre_shard: bool = False,
        compressed_token_map: torch.Tensor | None = None,
        compressed_token_ids: torch.Tensor | None = None,
        compressed_pad_id: int | None = None,
    ) -> MoeModelOutputWithPast:
        if self.engram_hasher is not None and input_ids is None:
            raise ValueError("DeepSeek V4.1 Engram needs input_ids; inputs_embeds alone cannot build n-grams")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        cp_rank, cp_world_size = self.cp_context.cp_rank, self.cp_context.cp_world_size
        if seq_lens_are_pre_shard != (cp_world_size > 1):
            raise ValueError(
                f"seq_lens_are_pre_shard={seq_lens_are_pre_shard} disagrees with cp_world_size={cp_world_size}"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if seq_lens is None:
            if position_ids is None:
                raise ValueError("DeepSeek V4.1 needs seq_lens or 0-based position_ids")
            seq_lens = _seq_lens_from_position_ids(position_ids)
        total_tokens = int(seq_lens.sum())
        if total_tokens != inputs_embeds.shape[1] * cp_world_size:
            raise ValueError(
                f"seq_lens covers {total_tokens} tokens and {cp_world_size} rank(s) hold {inputs_embeds.shape[1]} each"
            )

        packed = build_packed(
            self.config,
            seq_lens,
            inputs_embeds.device,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
        )
        if position_ids is not None:
            packed.check_position_ids(position_ids)

        engram_hashes, engram_mask = self._engram_hashes(
            input_ids,
            compressed_token_map=compressed_token_map,
            compressed_token_ids=compressed_token_ids,
            compressed_pad_id=compressed_pad_id,
            seq_lens=seq_lens,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
        )
        mhc_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pre = identity_pre(mhc_states)
        shared = SharedAttentionState.empty()
        for layer_idx, layer in enumerate(self.layers):
            routed = routed_experts[:, :, layer_idx, :].contiguous() if routed_experts is not None else None
            engram_kwargs = {}
            if getattr(layer, "engram", None) is not None:
                if engram_hashes is None or engram_mask is None:
                    raise RuntimeError(f"layer {layer_idx} Engram needs hash ids")
                hash_index = layer.engram.layer_hash_index
                engram_kwargs = {
                    "engram_hash_ids": engram_hashes[:, :, hash_index, :],
                    "engram_token_mask": engram_mask,
                }
            mhc_states, pre, shared = layer(
                mhc_states, pre, shared, routed_experts=routed, packed=packed, **engram_kwargs
            )
        hidden_states = self.norm(collapse(mhc_states, pre))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states)

    def _engram_hashes(
        self,
        input_ids: torch.Tensor | None,
        *,
        compressed_token_map: torch.Tensor | None,
        compressed_token_ids: torch.Tensor | None,
        compressed_pad_id: int | None,
        seq_lens: torch.Tensor,
        cp_rank: int,
        cp_world_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.engram_hasher is None:
            return None, None
        if cp_world_size > 1:
            if self.cp_context.cp_group is None:
                raise RuntimeError("context parallel is enabled but no process group is installed")
            input_ids = gather_for_cp(input_ids, self.cp_context.cp_group)
            if compressed_token_ids is not None:
                compressed_token_ids = gather_for_cp(compressed_token_ids, self.cp_context.cp_group)
        hashes, token_mask = self.engram_hasher(
            input_ids,
            compressed_token_map=compressed_token_map,
            compressed_token_ids=compressed_token_ids,
            compressed_pad_id=compressed_pad_id,
            seq_lens=seq_lens,
        )
        if cp_world_size > 1:
            local = hashes.shape[1] // cp_world_size
            start = cp_rank * local
            hashes = hashes[:, start : start + local]
            token_mask = token_mask[:, start : start + local]
        return hashes, token_mask


class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: DeepseekV41Config):
        super().__init__(config)
        self.model = DeepseekV41Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def install_compressed_token_map(self, token_map: Tensor) -> None:
        self.model.install_compressed_token_map(token_map)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        temperature: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor | None = None,
        seq_lens_are_pre_shard: bool = False,
        compressed_token_map: torch.Tensor | None = None,
        compressed_token_ids: torch.Tensor | None = None,
        compressed_pad_id: int | None = None,
        **kwargs,
    ) -> PrimeLmOutput:
        del kwargs
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
            compressed_token_map=compressed_token_map,
            compressed_token_ids=compressed_token_ids,
            compressed_pad_id=compressed_pad_id,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        sliced = hidden_states[:, slice_indices, :]
        sliced_labels = labels[:, slice_indices] if labels is not None else None
        if type(self.lm_head) is nn.Linear:
            return PrimeLmOutput(logits=self.lm_head(sliced))
        return self.lm_head(sliced, sliced_labels, temperature=temperature)


__all__ = [
    "DeepseekV41DecoderLayer",
    "DeepseekV41ForCausalLM",
    "DeepseekV41Model",
    "DeepseekV41PreTrainedModel",
]
