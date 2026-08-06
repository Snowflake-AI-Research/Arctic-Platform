# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint directory helpers (no Ray / DeepSpeed imports)."""

from __future__ import annotations

import os
import re
import shutil


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
