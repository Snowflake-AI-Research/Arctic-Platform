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

"""Log the batch counts of each GDN metadata build and the prefill backend of its first layer.

Which kernel a linear-attention row runs depends on the counts of the step it shares: GLM-5.3's KDA layers, for
example, run every non-speculative row through the chunked prefill kernel whenever the step holds a prefill.
"""

import logging

# vLLM configures its own logger hierarchy at INFO; an ``arctic_platform`` logger inherits the root WARNING level.
logger = logging.getLogger("vllm.v1.attention.backends.gdn_attn")

_APPLIED = False


def describe(builder, metadata) -> str:
    """One line: the build's layer group, its first layer's class and KDA prefill backend, and the batch counts."""
    first_layer = builder.layer_names[0]
    layer = builder.vllm_config.compilation_config.static_forward_context[first_layer]
    # GLM-5.3's KDA layers keep the prefill kernel they resolved at construction; other layers choose theirs in code.
    backend = getattr(layer, "kda_prefill_backend", None)
    return (
        f"linear-attention dispatch group={first_layer} layer={type(layer).__name__} "
        + (f"kda_prefill_backend={backend} " if backend is not None else "")
        + f"prefills={metadata.num_prefills} prefill_tokens={metadata.num_prefill_tokens} "
        f"decodes={metadata.num_decodes} decode_tokens={metadata.num_decode_tokens} "
        f"spec_decodes={metadata.num_spec_decodes} spec_decode_tokens={metadata.num_spec_decode_tokens}"
    )


def ensure_dispatch_log() -> None:
    global _APPLIED
    from arctic_platform.inference import envs

    if _APPLIED or not envs.ARCTIC_INFERENCE_LOG_DISPATCH:
        return
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    build = GDNAttentionMetadataBuilder.build

    def logged_build(self, *args, **kwargs):
        metadata = build(self, *args, **kwargs)
        logger.info(describe(self, metadata))
        return metadata

    GDNAttentionMetadataBuilder.build = logged_build
    _APPLIED = True
