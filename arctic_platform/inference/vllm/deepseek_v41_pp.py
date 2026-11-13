# Copyright 2026 Snowflake Inc.
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

"""Serve DeepSeek-V4.1 on a pipeline stage that starts at the last KV source layer.

DeepSeek-V4.1's decoder SWA bounded replay splits each stage's layers at the
last KV source layer ``cut`` (layer 20): layers before it run on every row,
layer ``cut`` and the ones after it replay on each request's trailing window.
The replay starts from the hyper-connection streams (``residual``,
``post_mix``, ``res_mix``) that the layers before ``cut`` produce. vLLM 0.33
allows the replay on any stage that holds ``cut`` and every later layer, which
includes the second stage of the only legal PP2 split (20/20). That stage runs
no layer before ``cut``, receives only ``hidden_states`` and ``pre_mix``, and
fails ``assert residual is not None and post_mix is not None`` in
``_run_layers`` at the first forward.

The fix turns the decoder replay off on such a stage, a state vLLM already has
for other reasons. The sliding-window caches keep bounded replay for prefix
hits (``swa_bounded_replay``); only the eager-prefill trimming of layers 20-39
is lost there, and with router replay on it trims nothing anyway (every row is
kept for the routes). Stages that run a layer before ``cut``, PP1 among them,
are unchanged.

Remove once the pinned vLLM requires ``start_layer < cut`` for the replay.
"""

from __future__ import annotations

from typing import Any

_PATCH_ATTR = "_arctic_deepseek_v41_pp"


def patch_decoder_replay_supported(model_cls: Any, logger: Any) -> None:
    original = model_cls._decoder_replay_supported
    if getattr(original, _PATCH_ATTR, False):
        return

    def _decoder_replay_supported(self, vllm_config, cut: int) -> bool:
        if self.start_layer == cut:
            logger.warning_once(
                "Decoder SWA bounded replay is off: this pipeline stage starts at the "
                "last KV source layer %d, so no layer on it builds the residual "
                "streams the replay starts from.",
                cut,
            )
            return False
        return original(self, vllm_config, cut)

    setattr(_decoder_replay_supported, _PATCH_ATTR, True)
    _decoder_replay_supported._orig = original
    model_cls._decoder_replay_supported = _decoder_replay_supported


def ensure_deepseek_v41_pp_fix() -> None:
    """Patch vLLM's DeepSeek-V4.1 text model (NVIDIA) for a stage starting at ``cut``.

    Called when the DeepSeek-V4.1 entry point is imported, so other models never
    import the DeepSeek-V4.1 kernels.
    """
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        return  # vLLM's ROCm model is a separate class, not patched here.
    from vllm.models.deepseek_v41.nvidia import model

    patch_decoder_replay_supported(model.DeepseekV4Model, model.logger)
