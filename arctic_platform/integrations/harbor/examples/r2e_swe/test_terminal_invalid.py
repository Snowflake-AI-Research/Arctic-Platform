"""Terminal-invalid reward override: the part of his recipe Boyi flagged.

prime-rl auto-injects a ``swe_terminal_invalid`` StopConditionFilter with
``retain=True, reward_override=0.0`` over every stop condition its enabled
detectors can emit. These tests pin the derived condition set against the
``terminal_stop_conditions`` implementation in verifiers' gates/config.py.
"""

from __future__ import annotations

import copy

import mini_swe_plus
from mini_swe_plus import GRADING_POLICY, terminal_stop_conditions


def test_matches_his_enabled_detectors() -> None:
    """His policy: format on, duplicate_tool_call on, truncation all on."""
    assert terminal_stop_conditions() == frozenset({
        "format_invalid",
        "repetition",
        "truncation",
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "context_length",
        "harness_timeout",
    })


def test_every_stop_reason_we_observed_is_terminal() -> None:
    """The six stop conditions from the 3-step run all zero the reward.

    format_invalid covers unknown_tool / nonempty_content / missing_tool_call /
    invalid_tool_arguments_schema; duplicate_tool_call emits repetition.
    """
    terminal = terminal_stop_conditions()
    for cond in ("format_invalid", "repetition", "max_output_tokens"):
        assert cond in terminal


def test_disabled_format_drops_only_format_invalid() -> None:
    policy = copy.deepcopy(GRADING_POLICY)
    policy["format"]["enabled"] = False
    conds = terminal_stop_conditions(policy)
    assert "format_invalid" not in conds
    assert "repetition" in conds


def test_repetition_needs_one_of_three_detectors() -> None:
    """char_repeat, duplicate_tool_call and loop_guard all emit "repetition"."""
    policy = copy.deepcopy(GRADING_POLICY)
    policy["duplicate_tool_call"]["enabled"] = False
    assert "repetition" not in terminal_stop_conditions(policy)

    for detector in ("char_repeat", "loop_guard"):
        alt = copy.deepcopy(policy)
        alt[detector]["enabled"] = True
        assert "repetition" in terminal_stop_conditions(alt)


def test_truncation_subfields_are_individually_honoured() -> None:
    policy = copy.deepcopy(GRADING_POLICY)
    policy["truncation"]["max_turns"] = False
    conds = terminal_stop_conditions(policy)
    assert "max_turns" not in conds
    assert "max_output_tokens" in conds

    policy["truncation"]["enabled"] = False
    assert not (terminal_stop_conditions(policy) & {"truncation", "max_output_tokens"})


def test_submission_is_not_a_declared_terminal_condition() -> None:
    """Deliberate: verifiers' terminal_stop_conditions covers only format,
    repetition and truncation, even though submission grading is enabled."""
    assert GRADING_POLICY["submission"]["enabled"] is True
    assert "submission_invalid" not in terminal_stop_conditions()


def test_override_keeps_the_trace_and_its_earned_reward_visible() -> None:
    """retain=True, enforce=False: zero the reward, do not drop the rollout."""
    terminal = terminal_stop_conditions()
    info = {"reward": 1.0, "earned_reward": 1.0, "reward_override": None}
    cond = "format_invalid"
    if cond in terminal:
        info["reward"] = 0.0
        info["reward_override"] = cond

    assert info["reward"] == 0.0
    assert info["earned_reward"] == 1.0, "earned reward stays legible for diagnosis"
    assert info["reward_override"] == "format_invalid"


def test_clean_stop_keeps_its_reward() -> None:
    info = {"reward": 1.0, "earned_reward": 1.0, "reward_override": None}
    if None in terminal_stop_conditions():  # a clean rollout reports no condition
        info["reward"] = 0.0
    assert info["reward"] == 1.0


def test_derived_set_is_not_hardcoded() -> None:
    """Guard against someone replacing the derivation with a literal set."""
    empty = {
        "format": {"enabled": False},
        "char_repeat": {"enabled": False},
        "duplicate_tool_call": {"enabled": False},
        "loop_guard": {"enabled": False},
        "truncation": {"enabled": False},
    }
    assert terminal_stop_conditions(empty) == frozenset()


def test_policy_still_matches_his_toml() -> None:
    """If this fails, our detectors drifted from his rl.toml [reward.swe]."""
    assert GRADING_POLICY["format"] == {
        "enabled": True,
        "require_reasoning": True,
        "max_tool_calls_per_turn": 1,
        "allow_content_between_think_and_tool_call": False,
    }
    assert GRADING_POLICY["char_repeat"]["enabled"] is False
    assert GRADING_POLICY["duplicate_tool_call"] == {"enabled": True, "allowance": 1}
    assert GRADING_POLICY["loop_guard"]["enabled"] is False
    assert mini_swe_plus.GRADING_POLICY["truncation"]["enabled"] is True
