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

"""Scratch workdir selection for heavyweight correctness runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

CORRECTNESS_TMPDIR_ENV = "DSS_CORRECTNESS_TMPDIR"
_DATA_FAST_ROOT = Path("/data-fast")
_DATA_FAST_TMPDIR = Path("/data-fast/tmp")


def require_data_fast_path(path: str | Path, *, label: str = "correctness storage path") -> Path:
    """Return a canonical fast-storage descendant without creating any directories."""
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        raise ValueError(f"{label} must resolve under /data-fast") from None
    if resolved == _DATA_FAST_ROOT or not resolved.is_relative_to(_DATA_FAST_ROOT):
        raise ValueError(f"{label} must be a resolved descendant of /data-fast")
    if not _DATA_FAST_ROOT.is_dir():
        raise ValueError(f"{label} requires an existing /data-fast directory")
    return resolved


def correctness_workdir(prefix: str) -> Path:
    """Create a heavyweight correctness workdir exclusively on fast storage."""
    if not isinstance(prefix, str) or os.sep in prefix or (os.altsep and os.altsep in prefix):
        raise ValueError("correctness workdir prefix must not contain a directory path")
    override = os.environ.get(CORRECTNESS_TMPDIR_ENV)
    root = require_data_fast_path(
        override if override else _DATA_FAST_TMPDIR,
        label=CORRECTNESS_TMPDIR_ENV if override else "correctness workdir",
    )
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))
