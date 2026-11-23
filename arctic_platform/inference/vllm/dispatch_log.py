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

"""Opt-in: log the batch class and kernel path of every linear-attention metadata build.

A hybrid model's linear-attention layers choose kernels from the step's batch: when any non-speculative row is a
prefill, every non-speculative row, decodes included, runs the chunked prefill kernel; otherwise decodes run the
recurrent kernel, and speculative rows their own path. Which path a token took is not visible in its outputs. With
``ARCTIC_INFERENCE_LOG_DISPATCH=1`` each build logs one line naming it.
"""

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def describe(builder, metadata) -> str:
    """One line: the build's layer group, its batch class and the path its non-speculative rows take."""
    first_layer = builder.layer_names[0]
    layer = builder.vllm_config.compilation_config.static_forward_context[first_layer]
    # KDA layers resolve their own prefill kernel; GDN layers use the builder's.
    prefill_backend = getattr(layer, "kda_prefill_backend", builder.gdn_prefill_backend)
    if metadata.num_prefills > 0:
        path = f"chunked:{prefill_backend}"
    elif metadata.num_decodes > 0:
        path = "recurrent"
    else:
        path = "none"
    return (
        f"linear-attention dispatch group={first_layer} layer={type(layer).__name__} "
        f"prefills={metadata.num_prefills} prefill_tokens={metadata.num_prefill_tokens} "
        f"decodes={metadata.num_decodes} decode_tokens={metadata.num_decode_tokens} "
        f"spec_decodes={metadata.num_spec_decodes} spec_decode_tokens={metadata.num_spec_decode_tokens} "
        f"non_spec_path={path}"
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
