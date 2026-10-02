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

import sys

from vllm.platforms import current_platform

import arctic_platform.inference.envs as envs
from arctic_platform.inference.utils import require_supported_vllm_version


def arctic_inference_plugin():
    if not envs.ARCTIC_INFERENCE_SKIP_VERSION_CHECK:
        require_supported_vllm_version("Arctic Inference plugin")

    from arctic_platform.inference.vllm.dense_prompt_logprobs import (
        ensure_dense_prompt_logprobs_patch,
    )
    from arctic_platform.inference.vllm.router_replay import (
        ensure_router_replay_vllm_patches,
    )
    from arctic_platform.inference.vllm.xgrammar_stop_mask import (
        ensure_xgrammar_stop_mask_fix,
    )

    ensure_router_replay_vllm_patches()
    ensure_xgrammar_stop_mask_fix()
    # Applied before the ARCTIC_INFERENCE_ENABLED branch: a scoring request must
    # be able to opt into dense prompt logprobs whether or not the rest of the
    # Arctic stack is on.
    ensure_dense_prompt_logprobs_patch()

    if not envs.ARCTIC_INFERENCE_ENABLED:
        from arctic_platform.inference.vllm.fp32_lm_head import (
            ensure_fp32_lm_head_vllm_patches,
        )

        ensure_fp32_lm_head_vllm_patches()
        return

    if not envs.ARCTIC_INFERENCE_SKIP_PLATFORM_CHECK:
        if not current_platform.is_cuda():
            raise RuntimeError(
                "Arctic Inference plugin requires the cuda platform!")

    print("\x1b[36;1mArctic Inference plugin is enabled!\x1b[0m",
          file=sys.stderr)

    # Lazy import to avoid potential errors when the plugin is disabled.
    from .patches import apply_arctic_patches
    apply_arctic_patches()
