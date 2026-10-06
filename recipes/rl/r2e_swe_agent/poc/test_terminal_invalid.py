"""Terminal-invalid reward override: the part of the reference recipe most easily missed.

prime-rl auto-injects a ``swe_terminal_invalid`` StopConditionFilter with
``retain=True, reward_override=0.0`` over every stop condition its enabled
detectors can emit. These tests pin the derived condition set against the
``terminal_stop_conditions`` implementation in verifiers' gates/config.py.
"""

from __future__ import annotations

import copy

import mini_swe_plus
from mini_swe_plus import GRADING_POLICY, terminal_stop_conditions


def test_matches_the_reference_enabled_detectors() -> None:
    """The reference policy: format on, duplicate_tool_call on, truncation all
    on, plus the SWE harness's own submission gate."""
    assert terminal_stop_conditions() == frozenset({
        "format_invalid",
        "repetition",
        "submission_invalid",
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


def test_submission_is_a_declared_terminal_condition() -> None:
    """The base gating policy covers only format, repetition and truncation,
    but the SWE harness subclass appends its submission gate on top, so an
    enabled submission policy does make the condition terminal."""
    assert GRADING_POLICY["submission"]["enabled"] is True
    assert "submission_invalid" in terminal_stop_conditions()


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
    """If this fails, our detectors drifted from the reference rl.toml [reward.swe]."""
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


def test_submission_gate_is_terminal():
    """The SWE harness adds its own gate on top of the base gating set.

    A trajectory can fix the bug and still be scored zero for not submitting
    with the exact required command, so this condition has to be in the set.
    """
    assert "submission_invalid" in mini_swe_plus.terminal_stop_conditions()


def test_submission_gate_can_be_disabled():
    policy = dict(mini_swe_plus.GRADING_POLICY)
    policy["submission"] = {"enabled": False}
    assert "submission_invalid" not in mini_swe_plus.terminal_stop_conditions(policy)


def test_base_gating_set_still_matches_the_reference() -> None:
    """Differential check against verifiers' own base gating policy.

    Only the base class is loaded here. Importing the SWE harness class itself
    would drag in the whole runtime stack (sandbox runtimes, dialects,
    renderers), which is far more machinery than this assertion needs.
    """
    import importlib.util
    import sys
    import types

    import reference_paths

    v1 = reference_paths.verifiers_v1()
    for name, path in (
        ("verifiers", v1.parent),
        ("verifiers.v1", v1),
        ("verifiers.v1.gates", v1 / "gates"),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod
    if "pydantic_config" not in sys.modules:
        import pydantic

        stub = types.ModuleType("pydantic_config")
        stub.BaseConfig = pydantic.BaseModel
        sys.modules["pydantic_config"] = stub

    def _load(mod_name, path):
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("verifiers.v1.gates.cores", v1 / "gates" / "cores.py")
    config = _load("verifiers.v1.gates.config", v1 / "gates" / "config.py")

    p = GRADING_POLICY
    base = config.GatingPolicyConfig(
        format=p["format"],
        char_repeat=p["char_repeat"],
        duplicate_tool_call=p["duplicate_tool_call"],
        loop_guard=p["loop_guard"],
        truncation=p["truncation"],
    )
    # The harness subclass appends exactly one condition to the base set.
    expected = set(base.terminal_stop_conditions()) | {"submission_invalid"}
    assert set(terminal_stop_conditions()) == expected


def test_reference_harness_still_appends_the_submission_gate() -> None:
    """Guard the assumption the test above bakes in.

    If the reference ever stops appending this condition, the composition
    asserted above is wrong and this fails rather than passing quietly.
    """
    import reference_paths

    src = (reference_paths.harness_dir() / "harness.py").read_text()
    body = src[src.index("def terminal_stop_conditions"):]
    body = body[: body.index("\n    def ", 1)]

    assert "super().terminal_stop_conditions()" in body
    assert "self.submission.enabled" in body
    assert '"submission_invalid"' in body
