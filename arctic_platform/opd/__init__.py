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

"""Arctic on-policy distillation client."""

from arctic_platform.opd.client import DEFAULT_PROCESSING
from arctic_platform.opd.client import ArcticOPDClient
from arctic_platform.opd.client import create_arctic_opd_client
from arctic_platform.opd.config import ArcticOPDClientConfig
from arctic_platform.opd.scoring import score_teacher

__all__ = [
    "ArcticOPDClient",
    "ArcticOPDClientConfig",
    "DEFAULT_PROCESSING",
    "create_arctic_opd_client",
    "score_teacher",
]
