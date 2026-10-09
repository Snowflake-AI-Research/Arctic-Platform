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
_DATA_FAST_TMPDIR = Path("/data-fast/tmp")
_SAFE_FALLBACK_TMPDIR = Path("/tmp")
_UNSAFE_TMPDIR_ROOTS = (Path("/code/users/stas"),)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_under(path: Path, root: Path) -> bool:
    try:
        _resolved(path).relative_to(_resolved(root))
    except ValueError:
        return False
    return True


def _is_unsafe_tmpdir(path: Path) -> bool:
    return any(_is_under(path, root) for root in _UNSAFE_TMPDIR_ROOTS)


def _safe_env_tmpdir() -> Path | None:
    for name in ("TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(name)
        if value and not _is_unsafe_tmpdir(Path(value)):
            return Path(value)
    return None


def correctness_workdir(prefix: str) -> Path:
    """Create a correctness harness workdir away from home-code storage.

    Correctness runs place checkpoint trees below their workdir, so they should not inherit a job
    script's home-directory TMPDIR. An explicit correctness override is still available for operators
    that need a different fast-disk root.
    """
    override = os.environ.get(CORRECTNESS_TMPDIR_ENV)
    if override:
        root = Path(override)
        if _is_unsafe_tmpdir(root):
            raise ValueError(f"{CORRECTNESS_TMPDIR_ENV} must not point under /code/users/stas")
    elif _DATA_FAST_TMPDIR.exists() or _DATA_FAST_TMPDIR.parent.exists():
        root = _DATA_FAST_TMPDIR
    else:
        root = _safe_env_tmpdir() or _SAFE_FALLBACK_TMPDIR
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))
