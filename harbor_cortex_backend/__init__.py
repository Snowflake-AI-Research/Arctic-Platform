# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Harbor post-training backend (reference implementation) over Cortex Training.

Users bring a Harbor agent; ``harbor run`` drives training and inference
through the same sampling sub-job. See ``README.md`` and
``rfcs/harbor-post-training-backend.md``.

Public surface:

* Data contract — ``Rollout``, ``RolloutDataset``, ``PostTrainingConfig``,
  ``TrainingProgress``, ``TrainingRun``, ``InferenceEndpoint``.
* Backend — ``ArcticCortexBackend`` implements the RFC's
  ``PostTrainingBackend`` protocol.
* Harbor extensions — ``HostEnvironment`` (BaseEnvironment), ``CortexRLAgent``
  (BaseAgent); referenced from ``harbor run --agent/--env`` by import path.
  Scoring uses Harbor's stock ``Verifier``, which uploads and execs each task's
  ``tests/test.sh``.
* Adapter — ``load_job_dir`` reads a Harbor jobs dir and materializes a
  ``RolloutDataset`` from every trial's ``result.json``.
"""

from arctic_platform.integrations.harbor.adapter import load_job_dir, pass_at_1
from arctic_platform.integrations.harbor.backend import ArcticCortexBackend
from arctic_platform.integrations.harbor.cortex_agent import CortexRLAgent
from arctic_platform.integrations.harbor.host_environment import HostEnvironment
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
