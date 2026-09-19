"""The instruction we hand the harness must be the one the reference taskset hands it.

The reference ``R2EGymTaskSet.get_instruction`` returns ``info["problem_statement"]`` and
nothing else; the harness's ``INSTANCE_TEMPLATE`` supplies all scaffolding. Any
wrapper we add lands *inside* the reference ``{{task}}`` slot, so it does not replace the reference
instructions, it competes with them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from r2e_driver import task_prompt

HARNESS = Path(
    "/modeling-code/boyiliu/prime-rl/deps/verifiers/verifiers/v1"
    "/harnesses/mini_swe_agent_plus/program.py"
)


def _instance_template() -> str:
    src = HARNESS.read_text()
    m = re.search(r'INSTANCE_TEMPLATE = """(.*?)"""', src, re.DOTALL)
    assert m, "the reference INSTANCE_TEMPLATE moved; re-check the harness contract"
    return m.group(1)


def _render(statement: str) -> str:
    return _instance_template().replace("{{task}}", task_prompt(
        {"problem_statement": statement}
    )).replace("{{working_dir}}", "/testbed")


def test_instruction_is_the_bare_problem_statement() -> None:
    rec = {"problem_statement": "Title: thing is broken", "instance_id": "x@1"}
    assert task_prompt(rec) == "Title: thing is broken"


def test_no_competing_submit_instruction() -> None:
    """There is no submit tool. Telling the model to "submit" cost us 24 of 96
    rollouts to unknown_tool in the 3-step run."""
    rendered = _render("Title: thing is broken")
    assert not re.search(r"\bsubmit\b\s*\.", rendered.lower()), (
        "a bare 'submit' instruction competes with echo MINI_SWE_AGENT_FINAL_OUTPUT"
    )


def test_submission_command_survives_exactly_once() -> None:
    rendered = _render("Title: thing is broken")
    assert rendered.count("echo MINI_SWE_AGENT_FINAL_OUTPUT") == 1


def test_no_nested_prompt_wrapper() -> None:
    """The reference template already opens <pr_description>; ours must not add a second
    framing inside it."""
    rendered = _render("Title: thing is broken")
    assert "<issue>" not in rendered
    assert rendered.count("<pr_description>") == 1
    assert rendered.count("<instructions>") == 1


def test_statement_is_passed_through_verbatim() -> None:
    """Problem statements carry markdown and code fences; none of it may be
    reformatted or escaped on the way in."""
    statement = "**Title:** `f()` breaks\n\n```python\nimport numpy\n```\n"
    assert task_prompt({"problem_statement": statement}) == statement
    assert statement in _render(statement)


@pytest.mark.parametrize("phrase", [
    "You are a software engineer working in /testbed",
    "Do not modify tests or project configuration files",
    "issue exactly one native tool call",
    "Before submission, run a relevant public test",
])
def test_his_template_still_covers_what_our_wrapper_used_to_say(phrase: str) -> None:
    """Guards the removal: each instruction our wrapper duplicated is present in
    the reference template, so dropping the wrapper loses nothing."""
    assert phrase in _render("Title: thing is broken")
