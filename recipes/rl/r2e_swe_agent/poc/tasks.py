"""Tiny SWE-shaped tasks with scriptable rewards.

Same contract as an R2E/SWE-bench instance reduced to its essentials: a repo
state with a defect, an instruction, and a test command whose exit status is the
reward. Small enough that a 1.7B model clears some of them, so the loop produces
reward variance within a GRPO group -- which is all the POC needs, since we are
not chasing convergence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    setup: str
    instruction: str
    test: str

    def prompt(self) -> str:
        """Instruction plus a self-check command.

        Handing the agent its own verification command is what gives a small
        model a path to a non-zero reward, and reward spread within a group is
        what GRPO needs to produce any gradient at all.
        """
        return (
            f"{self.instruction}\n\n"
            f"Verify your work by running exactly:\n"
            f"{self.test} && echo VERIFY_OK\n\n"
            f"The task is complete when that prints VERIFY_OK."
        )


TASKS: list[Task] = [
    Task(
        name="add_sign",
        setup=(
            "mkdir -p /workspace && "
            "printf 'def add(a, b):\\n    return a - b\\n' > /workspace/calc.py"
        ),
        instruction=(
            "The file /workspace/calc.py defines add(a, b), which is supposed to return "
            "the sum of its two arguments but currently returns the difference. "
            "Fix the function body. Do not rename the function."
        ),
        test="cd /workspace && python -c \"from calc import add; assert add(2, 3) == 5; assert add(-1, 1) == 0\"",
    ),
    Task(
        name="greet_format",
        setup=(
            "mkdir -p /workspace && "
            "printf 'def greet(name):\\n    return \"Hi \" + name\\n' > /workspace/greeter.py"
        ),
        instruction=(
            "The file /workspace/greeter.py defines greet(name). It must return "
            "'Hello, <name>!' exactly, but currently returns 'Hi <name>'. Fix it."
        ),
        test="cd /workspace && python -c \"from greeter import greet; assert greet('Ada') == 'Hello, Ada!'\"",
    ),
    Task(
        name="sum_list",
        setup=(
            "mkdir -p /workspace && "
            "printf 'def total(xs):\\n    return 0\\n' > /workspace/agg.py"
        ),
        instruction=(
            "The file /workspace/agg.py defines total(xs), which must return the sum of "
            "the numbers in the list xs. It currently always returns 0. Implement it."
        ),
        test="cd /workspace && python -c \"from agg import total; assert total([1,2,3]) == 6; assert total([]) == 0\"",
    ),
    Task(
        name="max_empty",
        setup=(
            "mkdir -p /workspace && "
            "printf 'def biggest(xs):\\n    return max(xs)\\n' > /workspace/mx.py"
        ),
        instruction=(
            "The file /workspace/mx.py defines biggest(xs). It crashes on an empty list; "
            "it must return None when xs is empty, and the maximum otherwise. Fix it."
        ),
        test="cd /workspace && python -c \"from mx import biggest; assert biggest([1,5,2]) == 5; assert biggest([]) is None\"",
    ),
]
