# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Arctic RL backend for TRL. The trainer and stub builder import TRL lazily."""

from typing import TYPE_CHECKING

from arctic_platform.integrations.trl.client import ArcticOptimizer
from arctic_platform.integrations.trl.client import ArcticTrainingClient

if TYPE_CHECKING:
    from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker
    from arctic_platform.integrations.trl.trainer import ArcticAsyncGRPOTrainer
    from arctic_platform.integrations.trl.trainer import build_stub_model
    from arctic_platform.integrations.trl.weights import ArcticWeightTransfer
    from arctic_platform.rl.processors.weighted_logprob import weighted_logprob_sum

__all__ = [
    "ArcticAsyncGRPOTrainer",
    "ArcticOptimizer",
    "ArcticRolloutWorker",
    "ArcticTrainingClient",
    "ArcticWeightTransfer",
    "build_stub_model",
    "weighted_logprob_sum",
]


def __getattr__(name: str):
    if name == "ArcticRolloutWorker":
        from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker

        return ArcticRolloutWorker
    if name == "ArcticWeightTransfer":
        from arctic_platform.integrations.trl.weights import ArcticWeightTransfer

        return ArcticWeightTransfer
    if name == "weighted_logprob_sum":
        from arctic_platform.rl.processors.weighted_logprob import weighted_logprob_sum

        return weighted_logprob_sum
    if name in {"ArcticAsyncGRPOTrainer", "build_stub_model"}:
        from arctic_platform.integrations.trl import trainer as trainer_mod

        if name == "ArcticAsyncGRPOTrainer":
            return trainer_mod._trainer_cls()
        return trainer_mod.build_stub_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
