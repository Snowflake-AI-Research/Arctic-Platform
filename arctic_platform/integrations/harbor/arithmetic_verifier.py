# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-process arithmetic verifier — no container ``exec``, no ``test.sh``.

Reads the completion the agent wrote to the trial's ``agent/`` directory and
compares its last integer to the expected answer stored in the task's
``[metadata]`` block. Returns a Harbor ``VerifierResult``.
"""

from __future__ import annotations

import re

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier


_INT_RE = re.compile(r"-?\d+")


def _strip_thousands_commas(text: str) -> str:
    """Drop commas that separate digit-groups (``715,500`` -> ``715500``) so a
    correct answer written with US thousands separators isn't misread as its
    trailing group. Leaves non-numeric commas alone."""
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)


class ArithmeticVerifier(BaseVerifier):
    async def verify(self) -> VerifierResult:  # type: ignore[override]
        # 1) Read the completion the agent wrote to /logs/agent (host_env root) —
        # Harbor's Trial has already downloaded that dir back to trial_paths.agent_dir.
        completion_path = self.trial_paths.agent_dir / "completion.txt"
        completion = completion_path.read_text() if completion_path.exists() else ""

        # 2) Expected answer lives in task.toml [metadata]. task.config.metadata is
        # a plain dict populated from the parsed task.toml.
        expected = self.task.config.metadata.get("expected")
        if expected is None:
            # Fallback: compute from a/b if present.
            a = self.task.config.metadata.get("a")
            b = self.task.config.metadata.get("b")
            op = self.task.config.metadata.get("op", "mul")
            if a is None or b is None:
                raise ValueError(
                    "Task metadata must set either 'expected' or ('a','b','op'); "
                    f"got {self.task.config.metadata!r}"
                )
            expected = (a + b) if op == "add" else (a * b)

        # 3) Score: last integer in the completion is the model's answer.
        # Normalize thousands-commas first so ``715,500`` is read as ``715500``.
        ints = _INT_RE.findall(_strip_thousands_commas(completion))
        expected_i = int(expected)
        if not ints:
            reward = 0.0
        else:
            answer = int(ints[-1])
            if answer == expected_i:
                reward = 1.0
            else:
                # Dense partial credit by relative closeness. Keeps GRPO
                # advantages non-degenerate when no rollout is exactly right;
                # standard reward-shaping trick for RL on hard-to-solve tasks.
                denom = max(1, abs(expected_i))
                rel_err = abs(answer - expected_i) / denom
                if rel_err <= 0.01:
                    reward = 0.7
                elif rel_err <= 0.05:
                    reward = 0.5
                elif rel_err <= 0.1:
                    reward = 0.3
                elif rel_err <= 0.5:
                    reward = 0.15
                else:
                    reward = 0.05  # any integer output

        # 4) Write the reward file where Harbor's downstream consumers look for it.
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.verifier_dir / "reward.txt").write_text(f"{reward:.4f}")
        return VerifierResult(rewards={"reward": reward})
