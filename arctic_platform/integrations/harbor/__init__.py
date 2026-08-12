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

# No eager imports: ``agent.py`` and ``env.py`` require the optional
# ``harbor`` dependency, so consumers of ``adapter`` / ``models`` (including
# CI) can import submodules without installing Harbor. Harbor's own plugin
# loader resolves ``module:Class`` entry points lazily.
__all__: list[str] = []
