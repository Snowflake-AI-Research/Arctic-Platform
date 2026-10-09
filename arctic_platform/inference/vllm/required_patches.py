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

"""Required vLLM compatibility patches independent of Arctic optimizations."""


def apply_required_vllm_patches() -> None:
    from arctic_platform.inference.vllm.dense_prompt_logprobs import (
        ensure_dense_prompt_logprobs_patch,
    )
    from arctic_platform.inference.vllm.dflash2_nan_fix import (
        apply_dflash2_nan_fixes,
    )
    from arctic_platform.inference.vllm.mamba_completion_refresh import (
        ensure_mamba_completion_refresh,
    )
    from arctic_platform.inference.vllm.router_replay import (
        ensure_router_replay_vllm_patches,
    )
    from arctic_platform.inference.vllm.spec_decode_grammar import (
        ensure_spec_decode_grammar_fix,
    )
    from arctic_platform.inference.vllm.xgrammar_stop_mask import (
        ensure_xgrammar_stop_mask_fix,
    )

    ensure_router_replay_vllm_patches()
    ensure_xgrammar_stop_mask_fix()
    ensure_dense_prompt_logprobs_patch()
    ensure_spec_decode_grammar_fix()
    apply_dflash2_nan_fixes()
    ensure_mamba_completion_refresh()
