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

"""Opt-in: log each GDN-family metadata build's batch counts and its first layer's resolved kernel settings.

A step's counts choose a linear-attention row's kernel: GLM-5.3's KDA layers, for example, run every non-speculative
row through the chunked prefill kernel whenever the step holds a prefill."""

import logging

# vLLM configures its own logger hierarchy at INFO; an ``arctic_platform`` logger inherits the root WARNING level.
logger = logging.getLogger("vllm.v1.attention.backends.gdn_attn")
# Kernel choices a layer resolves at construction (KDA: GLM-5.3, Kimi; GDN: Qwen); the counts decide the rest.
_LAYER_FIELDS = ("kda_prefill_backend", "gdn_prefill_backend", "gdn_decode_kernel", "enable_packed_recurrent_decode")

_APPLIED = False


def _logged(build):
    def logged_build(self, *args, **kwargs):
        metadata = build(self, *args, **kwargs)
        first = self.layer_names[0]
        layer = self.vllm_config.compilation_config.static_forward_context[first]
        settings = vars(layer)
        logger.info(
            "linear-attention dispatch group=%s layer=%s%s prefills=%d prefill_tokens=%d decodes=%d decode_tokens=%d "
            "spec_decodes=%d spec_decode_tokens=%d",
            first, type(layer).__name__, "".join(f" {key}={settings[key]}" for key in _LAYER_FIELDS if key in settings),
            metadata.num_prefills, metadata.num_prefill_tokens, metadata.num_decodes, metadata.num_decode_tokens,
            metadata.num_spec_decodes, metadata.num_spec_decode_tokens,
        )
        return metadata

    return logged_build


def ensure_dispatch_log() -> None:
    global _APPLIED
    from arctic_platform.inference import envs

    if _APPLIED or not envs.ARCTIC_INFERENCE_LOG_DISPATCH:
        return
    from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    # Kimi-K3's NVIDIA builder replaces build; its ROCm builder calls GDN's, so it logs through it.
    for cls in (GDNAttentionMetadataBuilder, KimiK3KDAMetadataBuilder):
        cls.build = _logged(cls.build)
    _APPLIED = True
