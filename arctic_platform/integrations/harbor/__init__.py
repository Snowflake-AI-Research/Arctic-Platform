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

# NOTE: We deliberately keep this ``__init__`` free of eager imports.
#
# ``CortexRLAgent`` and ``HostEnvironment`` require the ``harbor`` package
# (a Harbor deployment installs it; the arctic base install may not). Eagerly
# importing them here would force every consumer of ``adapter`` /
# ``models`` — including CI, which doesn't ship Harbor — to install Harbor
# just to touch data-plane utilities. Submodules are imported directly:
#
#     from arctic_platform.integrations.harbor.adapter import load_job_dir
#     from arctic_platform.integrations.harbor.agent import CortexRLAgent   # needs harbor
#
# The entry-point group in the top-level ``pyproject.toml`` still uses
# fully-qualified ``module:Class`` strings, so Harbor's plugin loader
# resolves them lazily at CLI parse time — same story.
__all__: list[str] = []
