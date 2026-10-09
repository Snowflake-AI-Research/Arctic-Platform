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

"""Correctness workdirs stay on fast scratch even when generic TMPDIR is unsafe."""

from __future__ import annotations

from pathlib import Path

import pytest

from arctic_platform.correctness.harness import workdir
from arctic_platform.correctness.harness.workdir import CORRECTNESS_TMPDIR_ENV


def _reset_tmp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CORRECTNESS_TMPDIR_ENV, raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.delenv("TEMP", raising=False)
    monkeypatch.delenv("TMP", raising=False)


def test_correctness_workdir_prefers_data_fast_over_unsafe_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_fast_tmp = tmp_path / "data-fast" / "tmp"
    data_fast_tmp.parent.mkdir()
    _reset_tmp_env(monkeypatch)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", data_fast_tmp)
    monkeypatch.setattr(workdir, "_SAFE_FALLBACK_TMPDIR", tmp_path / "fallback")
    monkeypatch.setenv("TMPDIR", "/code/users/stas/ap3tmp")

    created = workdir.correctness_workdir("dss-correctness-")

    assert created.parent == data_fast_tmp
    assert created.name.startswith("dss-correctness-")


def test_correctness_workdir_does_not_fall_back_to_unsafe_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = tmp_path / "safe-tmp"
    _reset_tmp_env(monkeypatch)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", tmp_path / "missing-data-fast" / "tmp")
    monkeypatch.setattr(workdir, "_SAFE_FALLBACK_TMPDIR", fallback)
    monkeypatch.setenv("TMPDIR", "/code/users/stas/ap3tmp")

    created = workdir.correctness_workdir("dss-correctness-")

    assert created.parent == fallback
    assert not str(created).startswith("/code/users/stas/")


def test_correctness_workdir_honors_safe_explicit_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "override"
    _reset_tmp_env(monkeypatch)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", tmp_path / "data-fast" / "tmp")
    monkeypatch.setenv("TMPDIR", "/code/users/stas/ap3tmp")
    monkeypatch.setenv(CORRECTNESS_TMPDIR_ENV, str(override))

    created = workdir.correctness_workdir("dss-correctness-")

    assert created.parent == override


def test_correctness_workdir_rejects_unsafe_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_tmp_env(monkeypatch)
    monkeypatch.setenv(CORRECTNESS_TMPDIR_ENV, "/code/users/stas/ap3tmp")

    with pytest.raises(ValueError, match=CORRECTNESS_TMPDIR_ENV):
        workdir.correctness_workdir("dss-correctness-")
