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

import logging
from functools import wraps

import torch
from pydantic import ConfigDict
from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.config.utils import config

from arctic_platform.inference.patching import ArcticPatch

logger = logging.getLogger(__name__)


@config(config=ConfigDict(extra="forbid"))
class ArcticParallelConfig(ParallelConfig):

    ulysses_sequence_parallel_size: int = 1
    enable_shift_parallel: bool = False
    shift_parallel_threshold: int = 512

    def __post_init__(self, *args, **kwargs):
        if (self.enable_shift_parallel
                and self.ulysses_sequence_parallel_size == 1):
            raise ValueError("ulysses_sequence_parallel_size must be > 1 "
                             "when enable_shift_parallel is True.")
        super().__post_init__(*args, **kwargs)
        # ParallelConfig.__post_init__ sets world_size to pipeline_parallel_size *
        # tensor_parallel_size; override to PP * TP * ulysses_sequence_parallel_size.
        self.world_size = (self.pipeline_parallel_size *
                           self.tensor_parallel_size *
                           self.prefill_context_parallel_size *
                           self.ulysses_sequence_parallel_size)

    def compile_factors(self):
        factors = super().compile_factors()
        factors.update(
            ulysses_sequence_parallel_size=self.ulysses_sequence_parallel_size,
            enable_shift_parallel=self.enable_shift_parallel,
        )
        return factors


@config(config=ConfigDict(extra="forbid"))
class ArcticSpeculativeConfig(SpeculativeConfig):

    method: str | None = None
    disable_by_batch_size: int | None = None
    enable_suffix_decoding: bool = False
    suffix_cache_max_depth: int = 64
    suffix_speculative_tokens: int = 0
    suffix_cache_max_requests: int = 100000
    suffix_max_spec_factor: float = 1.0
    suffix_max_spec_offset: float = 0.0
    suffix_min_token_prob: float = 0.1


class ParallelConfigPatch(ArcticPatch[ParallelConfig]):

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticParallelConfig instead of a
        # ParallelConfig when creating a new instance of the class.
        if cls is ParallelConfig:
            return ArcticParallelConfig.__new__(ArcticParallelConfig, *args,
                                                **kwargs)
        return super(ParallelConfig, cls).__new__(cls)


class SpeculativeConfigPatch(ArcticPatch[SpeculativeConfig]):

    _orig_post_init = SpeculativeConfig.__post_init__

    def __new__(cls, *args, **kwargs):
        if cls is SpeculativeConfig:
            return ArcticSpeculativeConfig.__new__(ArcticSpeculativeConfig,
                                                   *args, **kwargs)
        return super(SpeculativeConfig, cls).__new__(cls)

    def __post_init__(self):
        arctic_method = self.method in ("arctic", "mlp_speculator")
        use_suffix = self.method == "suffix" or (
            self.method is None and self.enable_suffix_decoding
        )
        if (arctic_method or use_suffix) and self.disable_by_batch_size is None:
            self.disable_by_batch_size = 64
        if use_suffix:
            self.method = "arctic_suffix"
            self.enable_suffix_decoding = True
            if self.suffix_speculative_tokens > 0:
                self.num_speculative_tokens = self.suffix_speculative_tokens
            elif self.num_speculative_tokens is None:
                self.num_speculative_tokens = 16
        elif arctic_method:
            if self.draft_tensor_parallel_size is None:
                self.draft_tensor_parallel_size = 1
            if self.enable_suffix_decoding:
                self.suffix_speculative_tokens = self.suffix_cache_max_depth
        self._orig_post_init()


_DFLASH_PATCHED = False


def ensure_dflash_max_position_patch() -> None:
    global _DFLASH_PATCHED
    if _DFLASH_PATCHED:
        return
    original = SpeculativeConfig.__post_init__

    @wraps(original)
    def post_init(self):
        result = original(self)
        if self.method == "dflash":
            # DFlash queries the draft up to num_speculative_tokens past the last target position, unclamped.
            draft = self.draft_model_config.hf_config
            draft.max_position_embeddings = max(
                draft.max_position_embeddings,
                self.target_model_config.max_model_len + self.num_speculative_tokens,
            )
        return result

    SpeculativeConfig.__post_init__ = post_init
    SpeculativeConfigPatch._orig_post_init = post_init

    from vllm.model_executor.layers.rotary_embedding import YaRNScalingRotaryEmbedding
    from vllm.model_executor.models import qwen3_dflash

    get_rope = qwen3_dflash.get_rope

    @wraps(get_rope)
    def get_draft_rope(*args, max_position, **kwargs):
        rope = get_rope(*args, max_position=max_position, **kwargs)
        if isinstance(rope, YaRNScalingRotaryEmbedding) and rope.cos_sin_cache.shape[0] < max_position:
            # YaRN sizes its table original x factor, ignoring max_position; extend it with YaRN's own formula.
            t = torch.arange(max_position, dtype=torch.float32)
            freqs = torch.einsum("i,j -> ij", t, rope._compute_inv_freq(rope.scaling_factor))
            cache = torch.cat((freqs.cos() * rope.mscale, freqs.sin() * rope.mscale), dim=-1)
            rope.cos_sin_cache = cache.to(rope.cos_sin_cache)
        return rope

    qwen3_dflash.get_rope = get_draft_rope
    _DFLASH_PATCHED = True


class VllmConfigPatch(ArcticPatch[VllmConfig]):

    _orig_str = VllmConfig.__str__

    def __str__(self, *args, **kwargs):
        string = self._orig_str(*args, **kwargs)
        string += f", ulysses_sequence_parallel_size={self.parallel_config.ulysses_sequence_parallel_size}"
        string += f", enable_shift_parallel={self.parallel_config.enable_shift_parallel}"
        string += f", shift_parallel_threshold={self.parallel_config.shift_parallel_threshold}"
        return string
