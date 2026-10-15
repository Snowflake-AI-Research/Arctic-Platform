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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint directory helpers (no Ray / DeepSpeed imports)."""

from __future__ import annotations

import os
import re
import shutil


def resolve_checkpoint_save_paths(root: str, step: int | None) -> tuple[str, str]:
    """Return ``(save_dir, prune_root)`` for a training checkpoint save.

    ``prune_root`` is the job checkpoint directory (the parent of ``checkpoint-*``
    children). It is always ``root``, including when ``step`` is omitted.
    """
    if step is not None:
        save_dir = os.path.join(root, f"checkpoint-{int(step)}")
    else:
        save_dir = root
    return save_dir, root


def prune_checkpoint_dirs(parent_dir: str, keep: int) -> int:
    """Keep the newest ``keep`` ``checkpoint-*`` dirs under ``parent_dir``; remove older.

    Returns the number of directories removed. No-op when ``keep <= 0``.
    """
    if keep is None or int(keep) <= 0:
        return 0
    keep = int(keep)
    if not os.path.isdir(parent_dir):
        return 0
    pat = re.compile(r"^checkpoint-(\d+)$")
    found = []
    for name in os.listdir(parent_dir):
        m = pat.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(parent_dir, name)))
    found.sort(key=lambda x: x[0])
    to_remove = found[:-keep] if len(found) > keep else []
    removed = 0
    for _, path in to_remove:
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed
