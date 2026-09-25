"""Lightweight config dataclasses for the carved-out Qwen3.5 loading path.

The runtime-only ``ModelConfig`` retains the fields consumed by the Qwen
implementation. User-controlled checkpoint and offload settings use the
validated Pydantic models from :mod:`arctic_platform.model.config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from arctic_platform.model.config import ActivationCheckpointConfig
from arctic_platform.model.implementations.moe.config import DISPATCH_EP_BACKENDS, EPCommBackend


@dataclass
class DebugModelConfig:
    random_init: bool = False
    num_layers: int | None = None
    gradient_sample_max_numel: int = 0
    full_determinism: bool = False


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-0.6B"
    trust_remote_code: bool = False
    # ``vlm`` is always None on the DSS text-only path; kept for API parity.
    vlm: object | None = None

    # Optional explicit override for the one-time HF<->Prime conversion cache.
    # A ready sibling ``<checkpoint>/prime`` is preferred; otherwise resolution
    # falls back to ``DSS_WEIGHT_CONVERSION_CACHE_DIR`` and then /data-fast.
    # Empty string restores write-to-sibling when no pre-baked cache exists.
    weight_conversion_cache_dir: str | None = None

    seq_len: int = 2048
    attn: str = "flash_attention_2"
    ac: ActivationCheckpointConfig | None = None
    fsdp_cpu_offload: bool = False

    dp_replicate: int = 1
    ep: int = 1
    ep_comm_backend: EPCommBackend = "deepep"
    deepep_num_sms: int = 20
    deepep_token_chunk_size: int | None = None
    cp: int = 1

    impl: Literal["hf", "custom", "auto"] = "auto"
    optimization_dtype: Literal["bfloat16", "float32"] = "float32"
    reduce_dtype: Literal["bfloat16", "float32"] = "float32"
    moe_use_grouped_mm: bool = True

    fused_lm_head_token_chunk_size: int | Literal["auto", "disabled"] = "disabled"
    fp32_lm_head: bool = False

    debug: DebugModelConfig = field(default_factory=DebugModelConfig)
