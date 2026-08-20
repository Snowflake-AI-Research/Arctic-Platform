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

"""Arctic RL backend for TRL. Construct the trainer with these objects; ``weighted_logprob_sum`` is lazy."""

from typing import TYPE_CHECKING

from arctic_platform.integrations.trl.client import ArcticOptimizer
from arctic_platform.integrations.trl.client import ArcticTrainingClient

if TYPE_CHECKING:
    from arctic_platform.integrations.trl.loss import weighted_logprob_sum
    from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker
    from arctic_platform.integrations.trl.weights import ArcticWeightTransfer

__all__ = [
    "ArcticOptimizer",
    "ArcticTrainingClient",
    "ArcticRolloutWorker",
    "ArcticWeightTransfer",
    "weighted_logprob_sum",
]


def __getattr__(name: str):
    # Lazy: rollout/weights need async_grpo; loss needs the server.
    if name == "ArcticRolloutWorker":
        from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker

        return ArcticRolloutWorker
    if name == "ArcticWeightTransfer":
        from arctic_platform.integrations.trl.weights import ArcticWeightTransfer

        return ArcticWeightTransfer
    if name == "weighted_logprob_sum":
        from arctic_platform.integrations.trl.loss import weighted_logprob_sum

        return weighted_logprob_sum
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
