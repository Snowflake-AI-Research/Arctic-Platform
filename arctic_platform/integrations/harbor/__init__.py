# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Harbor plugin: train models on Cortex from inside Harbor's own CLI.

Public surface:

* ``CortexRLAgent`` (BaseAgent) and ``HostEnvironment`` (BaseEnvironment)
  — Harbor extensions. Referenced from ``harbor run --agent / --env``
  by full ``module:Class`` import path, or by the short names
  ``arctic-cortex-agent`` and ``arctic-cortex-env`` registered in the
  ``harbor.plugins`` entry-point group (see the top-level
  ``pyproject.toml``).
* ``Rollout``, ``RolloutDataset``, ``PostTrainingConfig``,
  ``TrainingProgress``, ``TrainingRun``, ``InferenceEndpoint`` — the
  post-training RFC's data contract.
* ``ArcticCortexBackend`` — RFC ``PostTrainingBackend`` protocol
  implementation over Cortex Training.
* ``train.cli`` — entry point behind the ``harbor-cortex-train``
  console script.

Scoring uses Harbor's stock ``Verifier`` execing each task's
``tests/test.sh``. See ``README.md`` for the user-facing flow.
"""

from arctic_platform.integrations.harbor.adapter import load_job_dir, pass_at_1
from arctic_platform.integrations.harbor.agent import CortexRLAgent
from arctic_platform.integrations.harbor.backend import ArcticCortexBackend
from arctic_platform.integrations.harbor.env import HostEnvironment
from arctic_platform.integrations.harbor.models import (
    InferenceEndpoint,
    PostTrainingConfig,
    Rollout,
    RolloutDataset,
    TrainingProgress,
    TrainingRun,
)
from arctic_platform.integrations.harbor.task_gen import (
    sample_problems,
    write_dataset,
    write_task_dir,
)

__all__ = [
    "ArcticCortexBackend",
    "CortexRLAgent",
    "HostEnvironment",
    "InferenceEndpoint",
    "PostTrainingConfig",
    "Rollout",
    "RolloutDataset",
    "TrainingProgress",
    "TrainingRun",
    "load_job_dir",
    "pass_at_1",
    "sample_problems",
    "write_dataset",
    "write_task_dir",
]
