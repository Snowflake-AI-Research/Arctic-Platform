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

"""
Forest Cascade Attention (FCA) backend for ArcticInference.

Activated by passing ``--forest-cascade-attn-configs '{...}'`` to the engine.
When enabled, discovers shared KV prefix groups across requests in each batch
and runs a grouped prefix FA + per-request suffix FA, then merges.

The backend is always registered (it is backward-compatible with the upstream
FlashAttention backend); forest cascade paths only activate at runtime when
the config is present on VllmConfig.
"""

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend


class ForestFlashAttentionBackend(FlashAttentionBackend):
    """FlashAttention backend with forest cascade attention support.

    Inherits all static configuration methods from the upstream backend
    and overrides the factory methods to return FCA-enhanced implementations.
    The FCA code paths are fully gated behind a runtime config check, so
    this is a drop-in replacement even when forest cascade is not enabled.
    """

    # FCA impl performs reshape_and_cache_flash inline in forward(),
    # matching the v0.14 behavior.  In v0.18 the default changed to
    # False (split into do_kv_cache_update), so we explicitly opt-in
    # to the old path here.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls():
        from .flash_attn_forest_cascade import FlashAttentionImpl
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls():
        from .flash_attn_forest_cascade import FlashAttentionMetadataBuilder
        return FlashAttentionMetadataBuilder


def apply_forest_cascade_patches():
    """Replace vLLM's FlashAttentionBackend with the FCA-enhanced variant.

    Always applied — the FCA backend is backward-compatible and only
    activates forest cascade paths when ``--forest-cascade-attn-configs``
    is provided.
    """
    import vllm.v1.attention.backends.flash_attn as target_module
    target_module.FlashAttentionBackend = ForestFlashAttentionBackend


def install_forest_cascade_patches() -> None:
    """Accept ``forest_cascade_attn_configs`` and register the FCA backend.

    Idempotent. Does not load the rest of the Arctic plugin. The caller has
    already checked that the process is on vLLM 0.31.
    """
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.arg_utils import EngineArgs

    from arctic_platform.inference.vllm.args import AsyncEngineArgsPatch
    from arctic_platform.inference.vllm.args import EngineArgsPatch

    if "__new__" not in getattr(AsyncEngineArgs, "_arctic_patches", {}):
        AsyncEngineArgsPatch.apply_patch()
    # create_engine_config is what copies forest_cascade_attn_configs onto VllmConfig.
    if "create_engine_config" not in getattr(EngineArgs, "_arctic_patches", {}):
        EngineArgsPatch.apply_patch()
    apply_forest_cascade_patches()
