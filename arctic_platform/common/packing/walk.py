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

"""One traversal of a batch's values, shared by everything in this package that rewrites a batch."""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping

# Returned by a transform to leave a key out of the result.
DROP = object()


def map_batch_values(
    batch: Mapping[str, Any],
    transform: Callable[[str, Any], Any],
) -> Dict[str, Any]:
    """Rebuild ``batch`` with ``transform(key, value)`` applied to every value that is not itself a dict.

    Sub-dicts are walked with the same transform and their leaves keyed by their own names, because RL carries
    per-token tensors inside a ``context`` sub-dict. Routing every rewrite through one traversal is what keeps a
    transform from reaching the top level and missing ``context``: leaves that disagree about their token width
    survive dispatch and only fail later, at the model call or in a loss reduction.

    A transform returning ``DROP`` omits that key.
    """
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            result[key] = map_batch_values(value, transform)
            continue
        transformed = transform(key, value)
        if transformed is not DROP:
            result[key] = transformed
    return result
