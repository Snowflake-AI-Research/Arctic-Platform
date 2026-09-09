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

"""Arctic on-policy distillation — client, config, scoring, and processors.

Complementary to Hugging Face TRL DistillationTrainer, not a drop-in. See
``docs/opd.md``.
"""

from arctic_platform.opd.client import DEFAULT_PROCESSING
from arctic_platform.opd.client import ArcticOPDClient
from arctic_platform.opd.client import create_arctic_opd_client
from arctic_platform.opd.config import ArcticOPDClientConfig
from arctic_platform.opd.processor import apply_opd_global_token_config
from arctic_platform.opd.processor import count_opd_loss_tokens
from arctic_platform.opd.processor import on_policy_distill_loss
from arctic_platform.opd.scoring import score_teacher
from arctic_platform.opd.scoring import score_teacher_topk

__all__ = [
    "ArcticOPDClient",
    "ArcticOPDClientConfig",
    "DEFAULT_PROCESSING",
    "apply_opd_global_token_config",
    "count_opd_loss_tokens",
    "create_arctic_opd_client",
    "on_policy_distill_loss",
    "score_teacher",
    "score_teacher_topk",
]
