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

"""The rl loss expressed as Cortex's server-side ``grpo`` loss.

Both supported losses are a policy gradient weighted by the detached, upper-capped
importance ratio ``pi_theta / pi_inference`` -- Cortex CISPO with
``is_weight_clip_max`` (``-sg(min(r, cap)) * A * log pi_theta``). With no
``prox_logp_shifted`` in the context the server takes the behavioral (inference)
log-probs as the proximal policy, so the decoupled correction is exactly 1.

IPO's trust region masks tokens whose probability moved more than ``eps``. Cortex
masks probability differences per advantage sign: tokens with ``A >= 0`` whose
probability rose by more than ``eps``, and tokens with ``A < 0`` whose probability
fell by more than ``eps``. IPO also masks the opposite moves, which Cortex keeps.
Masked tokens stay in the token-mean denominator on both sides.
"""

from typing import Any

from prime_rl.configs.trainer import CISPOLossConfig, IPOLossConfig


def grpo_processing(loss: IPOLossConfig | CISPOLossConfig) -> dict[str, Any]:
    config: dict[str, Any] = {"use_cispo_loss": True, "loss_agg_mode": "token-mean"}
    if isinstance(loss, IPOLossConfig):
        config["is_weight_clip_max"] = loss.max_importance_ratio
        config["prob_diff_mask_max_pos"] = loss.eps
        config["prob_diff_mask_max_neg"] = loss.eps
    else:
        config["is_weight_clip_max"] = loss.ratio_high
    return {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": config}
