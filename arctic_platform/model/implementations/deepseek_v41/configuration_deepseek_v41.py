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

"""Configuration for DeepSeek-V4.1.

The published checkpoint is a vision-language wrapper: ``model_type`` is ``deepseek_v41`` and the
text backbone lives under ``text_config``. This class flattens that text block onto the attributes
the backbone reads, and keeps the vision block as ``vision_*`` fields for a later milestone.
"""

from typing import Any

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PretrainedConfig

from arctic_platform.model.implementations.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

# Backbone schedule of deepseek-ai/DeepSeek-V4.1-Flash, plus the three trailing MTP layers.
FLASH_COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
FLASH_KV_SOURCE_LAYER_IDS = [2, 8, 14, 20]
FLASH_INDEX_SOURCE_LAYER_IDS = [2, 8, 14, 20, 24, 28, 32, 36]
_FLASH_NUM_HIDDEN_LAYERS = 40
_FLASH_NEXTN_LAYERS = 3


def _as_int_list(values: list[int] | tuple[int, ...]) -> list[int]:
    return [int(value) for value in values]


@strict
class DeepseekV41Config(PretrainedConfig):
    """Text config for DeepSeek-V4.1, parsed from a nested ``text_config`` or from flat kwargs.

    Defaults match DeepSeek-V4.1-Flash (40 backbone layers, 384 routed experts, no hash router).
    A smaller model is a flat ``DeepseekV41Config(hidden_size=..., num_hidden_layers=...)``; explicit
    arguments win over the same key inside ``text_config``.
    """

    model_type = "deepseek_v41"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }
    _rope_type_labels = ("main", "compress")

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        text = dict(text_config or {})
        text.pop("model_type", None)
        for key, value in text.items():
            kwargs.setdefault(key, value)

        raw_rope = kwargs.pop("rope_parameters", None)
        if raw_rope is None:
            raw_rope = kwargs.pop("rope_scaling", None)
        else:
            kwargs.pop("rope_scaling", None)

        self.vocab_size = kwargs.pop("vocab_size", 129280)
        self.hidden_size = kwargs.pop("hidden_size", 5120)
        self.moe_intermediate_size = kwargs.pop("moe_intermediate_size", 2304)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", _FLASH_NUM_HIDDEN_LAYERS)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 64)
        self.num_key_value_heads = kwargs.pop("num_key_value_heads", 1)
        self.head_dim = kwargs.pop("head_dim", 512)
        self.q_lora_rank = kwargs.pop("q_lora_rank", 1280)
        self.o_lora_rank = kwargs.pop("o_lora_rank", 1024)
        self.o_groups = kwargs.pop("o_groups", 8)
        self.hidden_act = kwargs.pop("hidden_act", "silu")
        self.swiglu_limit = kwargs.pop("swiglu_limit", 10.0)
        self.rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-20)
        self.attention_bias = kwargs.pop("attention_bias", False)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.initializer_range = kwargs.pop("initializer_range", 0.02)
        self.use_cache = kwargs.pop("use_cache", True)
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", 1048576)
        self.rope_theta = kwargs.pop("rope_theta", 10000.0)
        self.compress_rope_theta = kwargs.pop("compress_rope_theta", 160000.0)
        self.sliding_window = kwargs.pop("sliding_window", 128)

        qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", 64)
        partial_rotary_factor = kwargs.pop("partial_rotary_factor", None)
        self.partial_rotary_factor = (
            partial_rotary_factor if partial_rotary_factor is not None else qk_rope_head_dim / self.head_dim
        )
        self.qk_rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

        self.n_routed_experts = kwargs.pop("n_routed_experts", 384)
        self.n_shared_experts = kwargs.pop("n_shared_experts", 1)
        self.num_experts_per_tok = kwargs.pop("num_experts_per_tok", 6)
        self.scoring_func = kwargs.pop("scoring_func", "sqrtsoftplus")
        self.topk_method = kwargs.pop("topk_method", "noaux_tc")
        self.norm_topk_prob = kwargs.pop("norm_topk_prob", True)
        self.routed_scaling_factor = kwargs.pop("routed_scaling_factor", 1.5)

        self.num_nextn_predict_layers = kwargs.pop("num_nextn_predict_layers", _FLASH_NEXTN_LAYERS)
        flash_shape = (
            self.num_hidden_layers == _FLASH_NUM_HIDDEN_LAYERS and self.num_nextn_predict_layers == _FLASH_NEXTN_LAYERS
        )
        compress_ratios = kwargs.pop("compress_ratios", None)
        if compress_ratios is None:
            compress_ratios = (
                FLASH_COMPRESS_RATIOS
                if flash_shape
                else [0] * (self.num_hidden_layers + self.num_nextn_predict_layers)
            )
        self.compress_ratios = _as_int_list(compress_ratios)
        self.kv_source_layer_ids = _as_int_list(
            kwargs.pop("kv_source_layer_ids", FLASH_KV_SOURCE_LAYER_IDS if flash_shape else [])
        )
        self.index_source_layer_ids = _as_int_list(
            kwargs.pop("index_source_layer_ids", FLASH_INDEX_SOURCE_LAYER_IDS if flash_shape else [])
        )
        self.index_n_heads = kwargs.pop("index_n_heads", 32)
        self.index_head_dim = kwargs.pop("index_head_dim", 128)
        self.index_topk = kwargs.pop("index_topk", 512)
        self.candidate_source_layer_id = kwargs.pop("candidate_source_layer_id", 20 if flash_shape else -1)
        self.candidate_topk_blocks = kwargs.pop("candidate_topk_blocks", 2048)
        self.candidate_block_size = kwargs.pop("candidate_block_size", 8)

        self.hc_mult = kwargs.pop("hc_mult", 4)
        self.hc_sinkhorn_iters = kwargs.pop("hc_sinkhorn_iters", 20)
        self.hc_eps = kwargs.pop("hc_eps", 1e-6)

        self.engram_layer_ids = _as_int_list(kwargs.pop("engram_layer_ids", [1, 14] if flash_shape else []))
        self.engram_num_embeddings = _as_int_list(
            kwargs.pop("engram_num_embeddings", [384006168, 384016682] if flash_shape else [])
        )
        self.engram_max_ngram_size = kwargs.pop("engram_max_ngram_size", 4)
        self.engram_vocab_size = kwargs.pop("engram_vocab_size", 16000000)
        self.engram_n_heads = kwargs.pop("engram_n_heads", 8)
        self.engram_head_dim = kwargs.pop("engram_head_dim", 256)
        self.engram_pad_token_id = kwargs.pop("engram_pad_token_id", 2)
        self.engram_compressed_vocab_size = kwargs.pop("engram_compressed_vocab_size", 99092)
        self.engram_quantized = kwargs.pop("engram_quantized", bool(flash_shape))

        self.dspark_block_size = kwargs.pop("dspark_block_size", 5)
        self.dspark_noise_token_id = kwargs.pop("dspark_noise_token_id", 128799)
        self.dspark_target_layer_ids = _as_int_list(
            kwargs.pop("dspark_target_layer_ids", [37, 38, 39] if flash_shape else [])
        )
        self.dspark_markov_rank = kwargs.pop("dspark_markov_rank", 256)
        self.dspark_n_routed_experts = kwargs.pop("dspark_n_routed_experts", 128)
        self.dspark_num_experts_per_tok = kwargs.pop("dspark_num_experts_per_tok", 3)

        vision = dict(vision_config or {})
        vision.pop("model_type", None)
        self.vision_n_layers = kwargs.pop("vision_n_layers", vision.pop("num_hidden_layers", 32))
        self.vision_dim = kwargs.pop("vision_dim", vision.pop("hidden_size", 1024))
        self.vision_n_heads = kwargs.pop("vision_n_heads", vision.pop("num_attention_heads", 16))
        self.vision_inter_dim = kwargs.pop("vision_inter_dim", vision.pop("intermediate_size", 2816))
        self.vision_patch_size = kwargs.pop("vision_patch_size", vision.pop("patch_size", 14))
        self.vision_rope_theta = kwargs.pop("vision_rope_theta", vision.pop("rope_theta", 10000.0))
        self.vision_downsample_ratio = kwargs.pop("vision_downsample_ratio", vision.pop("downsample_ratio", 3))
        self.vision_max_n_token = kwargs.pop("vision_max_n_token", vision.pop("max_image_tokens", 1024))
        self.vision_min_pixels = kwargs.pop("vision_min_pixels", vision.pop("min_pixels", 295936))
        self.vision_max_wh_ratio = kwargs.pop("vision_max_wh_ratio", vision.pop("max_wh_ratio", None))
        for key, value in vision.items():
            kwargs.setdefault(f"vision_{key}", value)

        self.image_token_id = kwargs.pop("image_token_id", 129264)
        tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        pad_token_id = kwargs.pop("pad_token_id", 2)
        bos_token_id = kwargs.pop("bos_token_id", 0)
        eos_token_id = kwargs.pop("eos_token_id", 1)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if raw_rope is None and self.num_hidden_layers == _FLASH_NUM_HIDDEN_LAYERS:
            raw_rope = {
                "rope_type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            }
        self.rope_parameters = DeepseekV4Config._nest_rope_parameters(self, raw_rope)

    def convert_rope_params_to_dict(self, **kwargs):
        # ``rope_parameters`` is assigned after ``super().__init__``, same as V4.
        return kwargs

    def validate_rope(self) -> None:
        DeepseekV4Config.validate_rope(self)

    def validate_architecture(self) -> None:
        if self.num_key_value_heads != 1:
            raise ValueError(f"DeepSeek V4.1 attention is MQA, got num_key_value_heads={self.num_key_value_heads}")
        if self.qk_rope_head_dim > self.head_dim or self.qk_rope_head_dim % 2 != 0:
            raise ValueError(
                f"qk_rope_head_dim ({self.qk_rope_head_dim}) must be an even width of head_dim ({self.head_dim})"
            )
        if self.num_attention_heads % self.o_groups != 0:
            raise ValueError(
                f"o_groups ({self.o_groups}) must divide num_attention_heads ({self.num_attention_heads})"
            )
        expected = self.num_hidden_layers + self.num_nextn_predict_layers
        if len(self.compress_ratios) != expected:
            raise ValueError(
                f"compress_ratios length ({len(self.compress_ratios)}) must equal "
                f"num_hidden_layers + num_nextn_predict_layers ({expected})"
            )
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError(
                f"engram_num_embeddings ({len(self.engram_num_embeddings)}) must match "
                f"engram_layer_ids ({len(self.engram_layer_ids)})"
            )
        for name, layer_ids in (
            ("kv_source_layer_ids", self.kv_source_layer_ids),
            ("index_source_layer_ids", self.index_source_layer_ids),
            ("engram_layer_ids", self.engram_layer_ids),
        ):
            if any(layer_id < 0 or layer_id >= self.num_hidden_layers for layer_id in layer_ids):
                raise ValueError(f"{name} entries must lie in [0, {self.num_hidden_layers}): {layer_ids}")
        if self.candidate_source_layer_id < -1 or self.candidate_source_layer_id >= self.num_hidden_layers:
            raise ValueError(f"candidate_source_layer_id ({self.candidate_source_layer_id}) is outside the backbone")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) must lie in [1, {self.n_routed_experts}]"
            )
        if self.hc_mult < 1 or self.hc_sinkhorn_iters < 1:
            raise ValueError("hc_mult and hc_sinkhorn_iters must be positive")


__all__ = ["DeepseekV41Config"]
