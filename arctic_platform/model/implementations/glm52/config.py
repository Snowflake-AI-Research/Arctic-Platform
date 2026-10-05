"""Lightweight config dataclasses for the carved-out GLM-5.2 loading path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from arctic_platform.model.config import ActivationCheckpointConfig
from arctic_platform.model.implementations.moe.config import DISPATCH_EP_BACKENDS, EPCommBackend

SparseMlaBackend = Literal["flashmla", "ref", "tilelang", "dense"]


@dataclass
class DebugModelConfig:
    num_layers: int | None = None
    random_init: bool = False
    init_seed: int = 42
    first_k_dense_replace: int | None = None
    skip_attention: bool = False
    skip_mlp: bool = False
    hidden_size: int | None = None
    intermediate_size: int | None = None
    moe_intermediate_size: int | None = None


@dataclass
class ModelConfig:
    name: str = "zai-org/GLM-5.2"
    trust_remote_code: bool = True
    vlm: object | None = None

    weight_conversion_cache_dir: str | None = None

    seq_len: int = 2048
    attn: str = "flash_attention_2"
    ac: ActivationCheckpointConfig | None = None
    fsdp_cpu_offload: bool = False

    dp_replicate: int = 1
    ep: int = 1
    ep_comm_backend: EPCommBackend = "deepep"
    sparse_mla_backend: SparseMlaBackend = "ref"
    deepep_num_sms: int = 20
    deepep_token_chunk_size: int | None = None
    cp: int = 1

    impl: Literal["hf", "custom", "auto"] = "auto"
    optimization_dtype: Literal["bfloat16", "float32"] = "float32"
    reduce_dtype: Literal["bfloat16", "float32"] = "float32"
    moe_use_grouped_mm: bool = False

    fused_lm_head_token_chunk_size: int | Literal["auto", "disabled"] = "disabled"
    fp32_lm_head: bool = False

    debug: DebugModelConfig = field(default_factory=DebugModelConfig)
