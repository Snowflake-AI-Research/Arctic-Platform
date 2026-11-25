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

"""ToyHybridMoE's job-time options: kernel choices that never change the checkpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict


class ToyOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short_conv_kernel: Literal["loop", "fused"] = "loop"
    """``fused`` runs each packed row through one ``conv1d``; it needs whole rows, so no sequence parallelism."""
