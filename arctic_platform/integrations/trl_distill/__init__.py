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

"""CPU-only Arctic adapters for TRL-style async distillation.

New work on Arctic ``main``. Does not import TRL. Does not extend PR #84.

``ArcticAsyncDistillationTrainer`` keeps the driver on CPU. Generate, teacher
score, gather/fwd-bwd, optimizer step, and weight sync run on Arctic.
TRL ``AsyncDistillationTrainer`` still loads a local student; use this trainer
until TRL adds ``training_client=``.
"""

from arctic_platform.integrations.trl_distill.client import ArcticOPDOptimizer
from arctic_platform.integrations.trl_distill.client import ArcticOPDTrainingClient
from arctic_platform.integrations.trl_distill.client import ForwardBackwardOutput
from arctic_platform.integrations.trl_distill.config import ArcticAsyncDistillationConfig
from arctic_platform.integrations.trl_distill.rollout import ArcticOPDRolloutWorker
from arctic_platform.integrations.trl_distill.stub import RemoteStudentStub
from arctic_platform.integrations.trl_distill.trainer import ArcticAsyncDistillationTrainer
from arctic_platform.integrations.trl_distill.trainer import create_arctic_async_distillation_trainer
from arctic_platform.integrations.trl_distill.types import RolloutSample
from arctic_platform.integrations.trl_distill.weights import ArcticOPDWeightTransfer

__all__ = [
    "ArcticAsyncDistillationConfig",
    "ArcticAsyncDistillationTrainer",
    "ArcticOPDOptimizer",
    "ArcticOPDRolloutWorker",
    "ArcticOPDTrainingClient",
    "ArcticOPDWeightTransfer",
    "ForwardBackwardOutput",
    "RemoteStudentStub",
    "RolloutSample",
    "create_arctic_async_distillation_trainer",
]
