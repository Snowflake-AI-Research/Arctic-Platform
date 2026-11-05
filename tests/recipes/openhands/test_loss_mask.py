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
"""Localization submission parsing and the Cortex config checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openhands_localization.config import grpo_processing_config
from openhands_localization.config import validate_openhands_cfg
from openhands_localization.events import finish_locations
from openhands_localization.events import last_step_is_one_tool_call
from openhands_localization.prompt_text import render_system_prompt


def test_finish_counts_only_a_single_call():
    finish = dict(
        kind="ActionEvent",
        source="agent",
        action=dict(
            kind="LocalizationFinishAction",
            locations=[dict(file="a.py", class_name=None, function_name="f")],
        ),
    )
    assert finish_locations([finish])[0]["file"] == "a.py"
    assert finish_locations([finish, finish]) is None
    assert finish_locations([]) is None


def test_last_step_rejects_trailing_text():
    good = '<tool_call>{"name": "localization_finish"}</tool_call><|im_end|>'
    assert last_step_is_one_tool_call(good)
    assert not last_step_is_one_tool_call(good + "oops<|im_end|>")
    assert not last_step_is_one_tool_call("<tool_call></tool_call><tool_call></tool_call><|im_end|>")


def test_prompt_names_the_registered_tool_and_the_turn_budget():
    text = render_system_prompt(10)
    assert "terminal tool" in text
    assert "bash tool" not in text
    assert "up to 10 turns" in text


def _cfg(**overrides):
    algo = dict(
        advantage_estimator="grpo",
        policy_loss_type="gspo",
        loss_reduction="sequence_mean",
        eps_clip_low=0.0003,
        eps_clip_high=0.0004,
        use_kl_loss=False,
        use_kl_in_reward=False,
        use_entropy_loss=False,
    )
    algo.update(overrides)
    return SimpleNamespace(
        trainer=SimpleNamespace(
            algorithm=SimpleNamespace(**algo),
            update_epochs_per_batch=1,
            policy_mini_batch_size=8,
            train_batch_size=8,
            resume_mode="none",
            placement=SimpleNamespace(colocate_all=False),
            arctic_rl=SimpleNamespace(colocate=False, use_zorro=False),
        ),
        generator=SimpleNamespace(step_wise_trajectories=False),
    )


def test_gspo_config_is_sequence_level():
    config = grpo_processing_config(_cfg())
    assert config["loss_agg_mode"] == "seq-mean-token-mean"
    assert config["importance_sampling_level"] == "sequence"
    assert config["eps_clip"] == 0.0003
    assert config["entropy_coeff"] == 0.0


def test_validate_accepts_the_recipe_and_rejects_a_second_update():
    validate_openhands_cfg(_cfg())
    cfg = _cfg()
    cfg.trainer.update_epochs_per_batch = 2
    with pytest.raises(ValueError, match="update_epochs_per_batch"):
        validate_openhands_cfg(cfg)


def test_validate_rejects_zorro_and_kl():
    cfg = _cfg()
    cfg.trainer.arctic_rl.use_zorro = True
    with pytest.raises(ValueError, match="use_zorro"):
        validate_openhands_cfg(cfg)
    cfg = _cfg()
    cfg.trainer.algorithm.use_kl_loss = True
    with pytest.raises(ValueError, match="use_kl_loss"):
        validate_openhands_cfg(cfg)
