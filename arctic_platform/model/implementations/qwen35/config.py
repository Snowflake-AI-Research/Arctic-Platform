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
"""Lightweight config dataclasses for the carved-out Qwen3.5 loading path.

These replace prime-rl's pydantic / pydantic-config ``ModelConfig`` and ``ActivationCheckpointConfig`` with
plain dataclasses, carrying only the fields that are actually read on the DSS (DeepSpeed + EP + DeepEP) load path.
Pin-memory validation lives in ``activation_offload_settings.py`` (torch-free); offload runtime wiring imports
that module only through ``offload_settings()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from arctic_platform.model.implementations.gpu.activation_offload_settings import (
    ActivationOffloadSettings,
    validate_pin_memory_bucket_size_mib,
    validate_pin_memory_max_size_gib,
)
from arctic_platform.model.implementations.moe.config import DISPATCH_EP_BACKENDS, EPCommBackend


@dataclass
class ActivationOffloadConfig:
    """Whether/how to stream checkpointed block-boundary activations to CPU (``full`` mode only).

    Only consulted when ``enabled`` is True; see activation_offload.py.
    """

    enabled: bool = False  # stream saved block boundaries to CPU to reclaim GPU memory
    keep_last_n: int = 1  # boundaries to leave resident (needed first in backward)
    use_streams: bool = True
    # Minimum saved-tensor size in bytes to offload; smaller tensors stay on GPU. None uses the
    # offload manager default (1 MiB).
    tensor_size_threshold: int | None = None
    pin_memory_enabled: bool = True
    # "auto" sizes the retained pinned-memory cache from the observed completed-step footprint. A numeric
    # value is a hard retained-cache cap in GiB; 0 evicts returned pinned buffers immediately.
    pin_memory_max_size_gib: float | Literal["auto"] = "auto"
    # Contiguous pinned buffers are rounded up to this MiB boundary for variable-sequence-length reuse.
    pin_memory_bucket_size_mib: int = 64

    def __post_init__(self) -> None:
        self.pin_memory_max_size_gib = validate_pin_memory_max_size_gib(self.pin_memory_max_size_gib)
        self.pin_memory_bucket_size_mib = validate_pin_memory_bucket_size_mib(self.pin_memory_bucket_size_mib)

    def offload_settings(self) -> ActivationOffloadSettings:
        return ActivationOffloadSettings.from_offload_config(self)


@dataclass
class ActivationCheckpointConfig:
    # What the backward pass recomputes per transformer block: "full" recheckpoints each whole block
    # (one saved boundary per block); "selective" checkpoints only the chosen submodules in `targets`.
    mode: Literal["full", "selective"] = "full"
    freq: int = 1
    targets: list[str] = field(default_factory=lambda: ["norm"])
    # CPU-offload of the checkpointed block boundaries; off by default. See activation_offload.py.
    offload_config: ActivationOffloadConfig = field(default_factory=ActivationOffloadConfig)
    # Deterministic MoE routing across whole-block recompute; see router_replay_recompute.py. Required for
    # full-mode AC (+offload) at sp>=4, harmless otherwise, no-op under sampler router-replay.
    router_replay_recompute: bool = True

    def __post_init__(self) -> None:
        # The wire schema passes ac_config as a plain dict (``ActivationCheckpointConfig(**ac_cfg)``),
        # so a nested ``offload_config`` arrives as a dict; coerce it into the dataclass.
        if isinstance(self.offload_config, dict):
            self.offload_config = ActivationOffloadConfig(**self.offload_config)


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
