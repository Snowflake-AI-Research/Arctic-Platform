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
"""Cortex constraints for this harness.

On Cortex the old and new log-probs are re-derived in the same forward, so the
importance ratio is 1 and a second optimizer step would be an uncorrected
off-policy update. KL needs a reference model this path does not run. Resume
from a Cortex checkpoint is not supported. These checks fail before a job is
created.

The loss config is what CodeScout sent from
``src/cortex/trainer.py`` at
https://github.com/18jeffreyma/codescout/blob/abab719e08a55dde78c6da864cd24d84fd47bdf2/src/cortex/trainer.py
SkyRL's Arctic trainer does not forward it, and Cortex would otherwise use
token-mean GRPO with clip 0.2.
"""

from __future__ import annotations

from typing import Any

_LOSS_AGG_MODE = dict(
    token_mean="token-mean",
    sequence_mean="seq-mean-token-mean",
    seq_mean_token_sum_norm="seq-mean-token-sum-norm",
)
_POLICY_LOSS_TYPES = ("regular", "gspo")


def grpo_processing_config(cfg: Any) -> dict[str, Any]:
    """``processing.config`` for Cortex's GRPO loss, from SkyRL's algorithm block."""
    algo = cfg.trainer.algorithm
    loss_config: dict[str, Any] = dict(
        loss_agg_mode=_LOSS_AGG_MODE[algo.loss_reduction],
        eps_clip=float(algo.eps_clip_low),
        eps_clip_higher=float(algo.eps_clip_high),
        entropy_coeff=0.0,
    )
    if algo.policy_loss_type == "gspo":
        loss_config["importance_sampling_level"] = "sequence"
    return loss_config


def validate_openhands_cfg(cfg: Any) -> None:
    """Reject settings Cortex would silently train differently with."""
    algo = cfg.trainer.algorithm
    errors: list[str] = []
    if algo.advantage_estimator != "grpo":
        errors.append(f"trainer.algorithm.advantage_estimator must be 'grpo', got {algo.advantage_estimator!r}")
    if algo.policy_loss_type not in _POLICY_LOSS_TYPES:
        errors.append(
            f"trainer.algorithm.policy_loss_type={algo.policy_loss_type!r} has no Cortex equivalent; "
            f"use one of {_POLICY_LOSS_TYPES}"
        )
    if algo.loss_reduction not in _LOSS_AGG_MODE:
        errors.append(
            f"trainer.algorithm.loss_reduction={algo.loss_reduction!r} has no Cortex equivalent; "
            f"use one of {tuple(_LOSS_AGG_MODE)}"
        )
    if algo.use_kl_loss or algo.use_kl_in_reward:
        errors.append("Cortex cannot compute reference log-probs: set use_kl_loss=false and use_kl_in_reward=false")
    if algo.use_entropy_loss:
        errors.append("trainer.algorithm.use_entropy_loss is not forwarded to Cortex; set it to false")
    if cfg.trainer.update_epochs_per_batch != 1:
        errors.append("trainer.update_epochs_per_batch must be 1: Cortex re-derives old log-probs every step")
    if cfg.trainer.policy_mini_batch_size != cfg.trainer.train_batch_size:
        errors.append(
            "trainer.policy_mini_batch_size must equal trainer.train_batch_size: Cortex has no old log-probs, "
            "so a second optimizer step on the same batch would be an uncorrected off-policy update"
        )
    resume = cfg.trainer.resume_mode
    if resume is not None and resume != "none":
        errors.append("resuming from a checkpoint is not supported on Cortex; set trainer.resume_mode=none")
    placement = cfg.trainer.placement
    arctic = cfg.trainer.arctic_rl
    if placement.colocate_all or arctic.colocate:
        errors.append("Cortex runs training and sampling as separate sub-jobs; set colocate_all=false")
    if arctic.use_zorro:
        errors.append("trainer.arctic_rl.use_zorro must be false: a multi-turn response spans every turn")
    if cfg.generator.step_wise_trajectories:
        errors.append("generator.step_wise_trajectories is not supported on Cortex")
    if len(errors) > 0:
        raise ValueError("Invalid config for OpenHands on Cortex:\n  - " + "\n  - ".join(errors))
