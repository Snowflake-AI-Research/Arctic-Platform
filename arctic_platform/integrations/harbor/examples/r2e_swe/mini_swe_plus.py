"""Run the ``mini-swe-agent-plus`` harness inside our k3s sandbox.

That harness is a verifiers ``Harness`` object, but its sandbox-side contract is
narrow: stage five stdlib-only modules plus a launcher, then exec the launcher
with connection flags and a grading policy. Reimplementing that contract here
lets us run the exact reference agent code — same protocol validation, same grading,
same submission command — without pulling the verifiers orchestrator, which
expects its own runtime, trace, and taskset objects.

Sourced from the checkout so the agent stays byte-identical to the reference:
    deps/verifiers/verifiers/v1/harnesses/mini_swe_agent_plus/
"""

from __future__ import annotations

import json
import shlex
from typing import Any

import reference_paths

PROGRAM_PATH = "/tmp/verifiers/mini-swe-agent-plus.py"
LAUNCHER_PATH = "/tmp/verifiers/mini-swe-agent-plus-launcher"
GRADING_PATH = "/tmp/verifiers/mini_swe_agent_plus_grading.py"
GATES_CORES_PATH = "/tmp/verifiers/mini_swe_agent_plus_gates_cores.py"
EDITOR_PATH = "/tmp/verifiers/edit_via_str_replace"
PROTOCOL_PATH = "/tmp/verifiers/mini_swe_agent_plus_protocol.py"

STOP_PREFIX = "__MINI_SWE_AGENT_PLUS_STOP__:"
ADVISORY_PREFIX = "__MINI_SWE_AGENT_PLUS_ADVISORY__:"

PROGRAM_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

# The reference run's grading block, copied from that run's own logged config
# so the reward we compute is the one it trained on rather than a
# plausible-looking approximation.
GRADING_POLICY: dict[str, Any] = {
    "enabled": True,
    "format": {
        "enabled": True,
        "require_reasoning": True,
        "max_tool_calls_per_turn": 1,
        "allow_content_between_think_and_tool_call": False,
    },
    "char_repeat": {"enabled": False},
    "duplicate_tool_call": {"enabled": True, "allowance": 1},
    "loop_guard": {"enabled": False},
    "submission": {"enabled": True, "require_exact_command": True},
    "truncation": {
        "enabled": True,
        "response_length": True,
        "max_turns": True,
        "max_input_tokens": True,
        "max_output_tokens": True,
        "max_total_tokens": True,
        "context_length": True,
        "harness_timeout": True,
    },
}

# Mirrors verifiers' TRUNCATION_STOP_FIELDS: policy field -> stop_condition string.
_TRUNCATION_STOP_FIELDS = (
    ("response_length", "truncation"),
    ("max_turns", "max_turns"),
    ("max_input_tokens", "max_input_tokens"),
    ("max_output_tokens", "max_output_tokens"),
    ("max_total_tokens", "max_total_tokens"),
    ("context_length", "context_length"),
    ("harness_timeout", "harness_timeout"),
)


def terminal_stop_conditions(policy: dict[str, Any] | None = None) -> frozenset[str]:
    """Stop conditions whose rollouts prime-rl scores as zero.

    Derived from the enabled detectors rather than hard-coded, mirroring
    ``GradingPolicyConfig.terminal_stop_conditions`` in verifiers, so that
    changing a detector above changes this set with it. prime-rl auto-injects a
    ``swe_terminal_invalid`` filter over exactly these with ``retain=True`` and
    ``reward_override=0.0`` — the trace stays in the batch and keeps its loss
    mask, but trains as a failure. It is not opt-in: the ``zero_reward_on_*``
    knobs under ``[orchestrator.algo]`` are a separate, older path and leaving
    them false does not disable this.
    """
    p = policy if policy is not None else GRADING_POLICY
    out: set[str] = set()
    if p.get("format", {}).get("enabled"):
        out.add("format_invalid")
    # char_repeat, duplicate_tool_call and loop_guard all emit "repetition".
    if any(p.get(k, {}).get("enabled") for k in ("char_repeat", "duplicate_tool_call", "loop_guard")):
        out.add("repetition")
    trunc = p.get("truncation", {})
    if trunc.get("enabled"):
        out.update(cond for field, cond in _TRUNCATION_STOP_FIELDS if trunc.get(field))
    return frozenset(out)


def _sources() -> dict[str, str]:
    """Map sandbox path -> file contents, mirroring ``Harness.setup``."""
    verifiers = reference_paths.verifiers_v1()
    harness = reference_paths.harness_dir()
    return {
        PROGRAM_PATH: (harness / "program.py").read_text(),
        GRADING_PATH: (harness / "grading.py").read_text(),
        # grading.py imports the detector cores as a flat sibling in-sandbox,
        # so they ship from gates/ under a different name than they have here.
        GATES_CORES_PATH: (verifiers / "gates" / "cores.py").read_text(),
        EDITOR_PATH: (harness / "edit_via_str_replace.py").read_text(),
        PROTOCOL_PATH: (harness / "protocol.py").read_text(),
    }


def stage(sandbox, python_command: str = "/testbed/.venv/bin/python") -> None:
    """Write the harness into the sandbox and make it executable.

    ``argv[0]`` must remain a real interpreter path. The reference harness carries a long
    comment about this: aliasing it costs CPython ``sys.executable`` and its
    virtualenv, and on the R2E images the fallback prefix has no stdlib, so the
    interpreter aborts with SIGABRT before the agent ever starts.
    """
    for path, text in _sources().items():
        sandbox.write_file(path, text)
    launcher = (
        "#!/bin/bash\n"
        f"exec {shlex.quote(python_command)} {shlex.quote(PROGRAM_PATH)} \"$@\"\n"
    )
    sandbox.write_file(LAUNCHER_PATH, launcher)
    rc, out = sandbox.exec(f"chmod 0755 {EDITOR_PATH} {LAUNCHER_PATH}", timeout=120)
    if rc != 0:
        raise RuntimeError(f"chmod failed: {out.strip()}")


def run(
    sandbox,
    *,
    base_url: str,
    api_key: str,
    model: str,
    task: str,
    working_dir: str = "/testbed",
    command_timeout_seconds: int = 60,
    grading_policy: dict[str, Any] | None = None,
    timeout: int = 5400,
) -> dict[str, Any]:
    """Run one rollout and return stdout plus the harness's own stop verdict."""
    task_path = "/tmp/verifiers/task.txt"
    sandbox.write_file(task_path, task)

    policy = json.dumps(
        grading_policy if grading_policy is not None else GRADING_POLICY,
        sort_keys=True,
        separators=(",", ":"),
    )
    argv = " ".join(shlex.quote(a) for a in [
        LAUNCHER_PATH,
        f"--base-url={base_url}",
        f"--api-key={api_key}",
        f"--model={model}",
        f"--task-file={task_path}",
        f"--working-dir={working_dir}",
        f"--command-timeout-seconds={command_timeout_seconds}",
        f"--grading-policy={policy}",
    ])
    env = " ".join(f"{k}={shlex.quote(v)}" for k, v in PROGRAM_ENV.items())
    rc, out = sandbox.exec(f"export {env}; {argv}", timeout=timeout)

    stop: dict[str, Any] | None = None
    advisories: list[dict[str, Any]] = []
    for line in out.splitlines():
        for prefix, sink in ((STOP_PREFIX, "stop"), (ADVISORY_PREFIX, "adv")):
            if line.startswith(prefix):
                raw = line.split(":", 1)[1]
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = {"stop_condition": raw, "terminal_invalid": True}
                if sink == "stop":
                    stop = parsed
                else:
                    advisories.append(parsed)
    return {"exit_code": rc, "stdout": out, "stop": stop, "advisories": advisories}
