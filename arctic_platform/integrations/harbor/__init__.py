# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Harbor plugin: train models on Cortex from inside Harbor's own CLI.

Positioned as a plugin *for Harbor*, distributed with ``arctic_platform``
as a subpackage under ``arctic_platform.integrations.harbor``. Ships:

* Harbor extensions — ``CortexRLAgent`` (BaseAgent) and ``HostEnvironment``
  (BaseEnvironment). Referenced from ``harbor run --agent/--env`` either
  by full ``module:Class`` import path or by the short names registered
  in the ``harbor.plugins`` entry-point group (see the top-level
  ``pyproject.toml``): ``arctic-cortex-agent`` and ``arctic-cortex-env``.
* Data contract — ``Rollout``, ``RolloutDataset``, ``PostTrainingConfig``,
  ``TrainingProgress``, ``TrainingRun``, ``InferenceEndpoint``.
* Backend — ``ArcticCortexBackend`` implements the Harbor post-training
  RFC's ``PostTrainingBackend`` protocol over Cortex Training.
* Driver — ``train.cli`` is the entry point behind the
  ``harbor-cortex-train`` console script.

Scoring uses Harbor's stock ``Verifier``, which uploads and execs each
task's ``tests/test.sh``. No custom ``BaseVerifier`` subclass.
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
