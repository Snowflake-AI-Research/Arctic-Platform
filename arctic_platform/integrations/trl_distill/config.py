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

"""CPU-only Arctic async distillation config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArcticAsyncDistillationConfig:
    """Trainer-process knobs. Model compute stays on Arctic."""

    max_completion_length: int = 32
    beta: float = 0.0
    teacher_top_k: int = 8
    temperature: float = 1.0
    teacher_temperature: float = 1.0
    add_tail_bucket: bool = True
    steps: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-6
    weight_sync_steps: int = 1
    pad_token_id: int = 0
    max_seq_len: int | None = None
    repeat_batch: bool = False
