# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Programmatically build a Harbor task directory (and a dataset of them).

Harbor tasks are on-disk directories with a specific layout (task.toml,
instruction.md, environment/, tests/, solution/). For a demo that runs N
problems through Harbor's real pipeline we generate one task dir per problem.
This module is a small factory for that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


_TASK_TOML = """\
version = "1.0"

[metadata]
a = {a}
b = {b}
op = "{op}"
expected = {expected}

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 60.0

[environment]
build_timeout_sec = 60.0
"""

# HostEnvironment doesn't build an image, but Harbor's Task.is_valid_dir()
# demands an environment/ directory exists. An empty Dockerfile is enough.
_DOCKERFILE_STUB = "FROM scratch\n"

# Harbor's Task validator wants a tests/test.sh; the actual scoring is done
# by ArithmeticVerifier so this script is inert.
_TEST_SH = "#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n"

_SOLUTION_SH = "#!/bin/bash\necho placeholder\n"


def _op_symbol(op: str) -> str:
    return {"add": "+", "mul": "*"}[op]


def _expected(a: int, b: int, op: str) -> int:
    return a + b if op == "add" else a * b


def write_task_dir(
    task_dir: Path,
    a: int,
    b: int,
    op: Literal["add", "mul"] = "mul",
) -> Path:
    """Create a Harbor task dir for a single arithmetic problem."""
    task_dir = Path(task_dir)
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(parents=True, exist_ok=True)

    exp = _expected(a, b, op)
    (task_dir / "task.toml").write_text(
        _TASK_TOML.format(a=a, b=b, op=op, expected=exp)
    )
    (task_dir / "instruction.md").write_text(
        f"What is {a} {_op_symbol(op)} {b}? Reply with only the final integer, nothing else.\n"
    )
    (task_dir / "environment" / "Dockerfile").write_text(_DOCKERFILE_STUB)
    test_sh = task_dir / "tests" / "test.sh"
    test_sh.write_text(_TEST_SH)
    test_sh.chmod(0o755)
    solve_sh = task_dir / "solution" / "solve.sh"
    solve_sh.write_text(_SOLUTION_SH)
    solve_sh.chmod(0o755)
    return task_dir


def sample_problems(
    n: int,
    a_range: tuple[int, int],
    b_range: tuple[int, int],
    rng,
) -> list[tuple[int, int]]:
    """Draw ``n`` problems with each operand from its own range."""
    return [
        (rng.randint(a_range[0], a_range[1]), rng.randint(b_range[0], b_range[1]))
        for _ in range(n)
    ]


def write_dataset(
    dataset_dir: Path,
    problems: list[tuple[int, int]],
    op: Literal["add", "mul"] = "mul",
    prefix: str = "prob",
) -> list[Path]:
    """Create one task dir per problem plus a Harbor ``dataset.toml`` manifest."""
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    task_paths: list[Path] = []
    lines: list[str] = ['version = "1.0"', "", "tasks = ["]
    for i, (a, b) in enumerate(problems):
        name = f"{prefix}_{i:03d}_{a}x{b}"
        td = write_task_dir(dataset_dir / name, a=a, b=b, op=op)
        task_paths.append(td)
        lines.append(f'    {{ path = "./{name}" }},')
    lines.append("]")
    (dataset_dir / "dataset.toml").write_text("\n".join(lines) + "\n")
    return task_paths
