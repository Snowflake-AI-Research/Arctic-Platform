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

"""Identify DeepSpeed BF16_Optimizer's zero-norm assert."""

from __future__ import annotations

import traceback
from typing import Any

# DeepSpeed ``BF16_Optimizer.step`` has no assertion message. Match the raising
# frame's source line so nested asserts under ``step()`` are not swallowed.
_BF16_ZERO_NORM_ASSERT_LINE = "assert all_groups_norm > 0."


def is_bf16_zero_norm_assert(exc: BaseException, optimizer: Any | None) -> bool:
    """True only when *exc* is BF16_Optimizer's zero-norm ``assert``.

    Requires a live ``_global_grad_norm`` of 0 and that the innermost traceback
    frame is ``bf16_optimizer.py:step`` on that exact assertion line.
    """
    gn = getattr(optimizer, "_global_grad_norm", None)
    if gn is None:
        return False
    try:
        if float(gn) != 0.0:
            return False
    except (TypeError, ValueError):
        return False
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return False
    frame = frames[-1]
    filename = str(frame.filename).replace("\\", "/")
    if not filename.endswith("/bf16_optimizer.py") or frame.name != "step":
        return False
    return (frame.line or "").strip() == _BF16_ZERO_NORM_ASSERT_LINE
