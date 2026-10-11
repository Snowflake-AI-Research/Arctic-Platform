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
"""Loss mask, submission parsing, and the Cortex config checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from arctic_platform.integrations.openhands.config import grpo_processing_config
from arctic_platform.integrations.openhands.config import validate_openhands_cfg
from arctic_platform.integrations.openhands.events import finish_locations
from arctic_platform.integrations.openhands.events import last_step_is_one_tool_call
from arctic_platform.integrations.openhands.loss_mask import assistant_loss_mask
from arctic_platform.integrations.openhands.loss_mask import mask_if_exhausted
from arctic_platform.integrations.openhands.prompt_text import render_system_prompt
from arctic_platform.integrations.openhands.rollout import buffer_after_role
from arctic_platform.integrations.openhands.rollout import stitch_response

START = 1
ASSISTANT = 2


def test_mask_stays_aligned_when_the_response_starts_on_the_turn_marker():
    response = [START, ASSISTANT, 4, 5]
    mask = assistant_loss_mask(response, start_token_id=START, assistant_token_id=ASSISTANT, buffer_succeed=1)
    assert len(mask) == len(response)
    assert mask == [0, 0, 0, 1]


def test_scaffolding_is_masked_and_content_is_not():
    # token before the turn, <|im_start|>, assistant, one buffer token, then content
    response = [9, START, ASSISTANT, 4, 5]
    mask = assistant_loss_mask(response, start_token_id=START, assistant_token_id=ASSISTANT, buffer_succeed=1)
    assert mask == [0, 0, 0, 0, 1]


def test_exhausted_rollout_is_fully_masked():
    assert mask_if_exhausted([1, 1, 0], True) == [0, 0, 0]
    assert mask_if_exhausted([1, 0], False) == [1, 0]


def test_stitch_drops_the_first_prompt_prefix():
    messages = [
        dict(prompt_token_ids=[1, 2], response_token_ids=[3]),
        dict(prompt_token_ids=[1, 2, 3, 4], response_token_ids=[5]),
    ]
    prompt, response = stitch_response(messages)
    assert prompt == [1, 2]
    assert response == [3, 4, 5]


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


def test_instruct_2507_uses_the_short_buffer():
    assert buffer_after_role("Qwen/Qwen3-4B-Instruct-2507") == 1
    assert buffer_after_role("Qwen/Qwen3.5-4B") == 5


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
