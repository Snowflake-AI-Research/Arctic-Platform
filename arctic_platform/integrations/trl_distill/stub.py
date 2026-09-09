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

"""CPU placeholder so Accelerate/Trainer can ``prepare()`` without a local student."""

from __future__ import annotations

import torch
from torch import nn


class RemoteStudentStub(nn.Module):
    """One dummy CPU parameter. Weights and compute live on Arctic."""

    def __init__(self) -> None:
        super().__init__()
        self._remote_anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("RemoteStudentStub has no local compute; use ArcticOPDTrainingClient")
