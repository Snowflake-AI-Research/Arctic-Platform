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
def validate_lm_head_fused_ce_config(prl_config: dict) -> None:
    """Reject ``fused_cross_entropy`` combined with ``fused_lm_head_token_chunk_size``.

    Both settings select an lm_head implementation and only one head can be installed. ``fp32_lm_head`` selects
    the arithmetic inside whichever head is installed, so it combines with either.
    """
    fused_cross_entropy = prl_config.get("fused_cross_entropy", "liger")
    if fused_cross_entropy and isinstance(prl_config.get("fused_lm_head_token_chunk_size"), int):
        raise ValueError(
            "PrimeRL MoE DSS config cannot combine fused_cross_entropy with fused_lm_head_token_chunk_size."
        )
