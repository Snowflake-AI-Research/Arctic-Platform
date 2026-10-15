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

"""Fast-storage policy checked before heavyweight correctness allocation."""

from pathlib import Path

import pytest

from arctic_platform.correctness.harness import workdir


@pytest.fixture
def fast_root(tmp_path, monkeypatch):
    root = tmp_path / "data-fast"
    root.mkdir()
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", root)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", root / "tmp")
    for name in (workdir.CORRECTNESS_TMPDIR_ENV, "TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(name, raising=False)
    return root


@pytest.mark.parametrize("as_string", [False, True])
def test_require_data_fast_path_returns_canonical_path_without_writes(fast_root, as_string):
    destination = fast_root / "new" / ".." / "checkpoints"
    result = workdir.require_data_fast_path(str(destination) if as_string else destination)
    assert result == fast_root / "checkpoints"
    assert not result.exists()
    assert list(fast_root.iterdir()) == []


def test_require_data_fast_path_expands_home_and_relative_paths(fast_root, monkeypatch):
    monkeypatch.setenv("HOME", str(fast_root))
    monkeypatch.chdir(fast_root)
    assert workdir.require_data_fast_path("~/states") == fast_root / "states"
    assert workdir.require_data_fast_path("states") == fast_root / "states"
    assert not (fast_root / "states").exists()


@pytest.mark.parametrize(
    "destination", ["/code/users/stas/bulk", "/code/other/bulk", "/tmp/bulk", "/data-fast-other/bulk"]
)
def test_require_data_fast_path_rejects_other_storage(fast_root, destination):
    with pytest.raises(ValueError, match="checkpoint export.* /data-fast"):
        workdir.require_data_fast_path(destination, label="checkpoint export")
    assert list(fast_root.iterdir()) == []


@pytest.mark.parametrize("kind", ["root", "sibling", "traversal", "relative"])
def test_require_data_fast_path_rejects_root_and_escaping_paths(fast_root, monkeypatch, kind):
    monkeypatch.chdir(fast_root.parent)
    destination = {
        "root": fast_root,
        "sibling": fast_root.with_name("data-fast-other") / "bulk",
        "traversal": fast_root / ".." / "outside" / "bulk",
        "relative": Path("outside/bulk"),
    }[kind]
    with pytest.raises(ValueError, match="/data-fast"):
        workdir.require_data_fast_path(destination)
    assert list(fast_root.iterdir()) == []
    assert not (fast_root.parent / "outside").exists()


def test_require_data_fast_path_rejects_symlink_escape(fast_root):
    outside = fast_root.parent / "outside"
    outside.mkdir()
    (fast_root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="/data-fast"):
        workdir.require_data_fast_path(fast_root / "escape" / "bulk")
    assert list(outside.iterdir()) == []


def test_require_data_fast_path_accepts_internal_symlink_as_canonical_path(fast_root):
    target = fast_root / "target"
    target.mkdir()
    (fast_root / "alias").symlink_to(target, target_is_directory=True)
    assert workdir.require_data_fast_path(fast_root / "alias" / "states") == target / "states"
    assert list(target.iterdir()) == []


def test_require_data_fast_path_rejects_symlinked_fast_root(fast_root, monkeypatch):
    outside = fast_root.parent / "outside"
    outside.mkdir()
    alias = fast_root.parent / "fake-fast"
    alias.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", alias)
    with pytest.raises(ValueError, match="/data-fast"):
        workdir.require_data_fast_path(alias / "bulk")
    assert list(outside.iterdir()) == []


def test_require_data_fast_path_rejects_missing_fast_storage_without_creating_it(fast_root, monkeypatch):
    missing = fast_root.parent / "missing-fast"
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", missing)
    with pytest.raises(ValueError, match="existing /data-fast"):
        workdir.require_data_fast_path(missing / "states")
    assert not missing.exists()


@pytest.mark.parametrize("generic_tmpdir", ["/code/users/stas/tmp", "/tmp", "/home/other/tmp"])
def test_correctness_workdir_ignores_generic_temp_environment(fast_root, monkeypatch, generic_tmpdir):
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, generic_tmpdir)
    first = workdir.correctness_workdir("capture-")
    second = workdir.correctness_workdir("capture-")
    assert first.parent == second.parent == fast_root / "tmp"
    assert first != second
    assert first.is_dir() and second.is_dir()
    assert first.name.startswith("capture-")


def test_correctness_workdir_accepts_only_fast_explicit_override(fast_root, monkeypatch):
    override = fast_root / "captures"
    monkeypatch.setenv(workdir.CORRECTNESS_TMPDIR_ENV, str(override))
    created = workdir.correctness_workdir("capture-")
    assert created.parent == override
    assert created.is_dir()
    assert not (fast_root / "tmp").exists()


@pytest.mark.parametrize("kind", ["code", "tmp", "root", "traversal", "symlink"])
def test_correctness_workdir_rejects_override_before_allocation(fast_root, monkeypatch, kind):
    outside = fast_root.parent / "outside"
    outside.mkdir()
    (fast_root / "escape").symlink_to(outside, target_is_directory=True)
    override = {
        "code": Path("/code/users/stas/bulk"),
        "tmp": Path("/tmp/bulk"),
        "root": fast_root,
        "traversal": fast_root / ".." / "outside" / "bulk",
        "symlink": fast_root / "escape" / "bulk",
    }[kind]
    monkeypatch.setenv(workdir.CORRECTNESS_TMPDIR_ENV, str(override))

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("invalid storage must be rejected before workdir allocation")

    monkeypatch.setattr(workdir.tempfile, "mkdtemp", unexpected_allocation)
    with pytest.raises(ValueError, match=f"{workdir.CORRECTNESS_TMPDIR_ENV}.* /data-fast"):
        workdir.correctness_workdir("capture-")
    assert list(outside.iterdir()) == []
    assert not (fast_root / "tmp").exists()


def test_correctness_workdir_missing_fast_storage_never_falls_back(fast_root, monkeypatch):
    missing = fast_root.parent / "missing-fast"
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", missing)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", missing / "tmp")
    monkeypatch.setenv("TMPDIR", str(fast_root))
    with pytest.raises(ValueError, match="existing /data-fast"):
        workdir.correctness_workdir("capture-")
    assert not missing.exists()
    assert list(fast_root.iterdir()) == []


@pytest.mark.parametrize("prefix", ["../capture-", "/code/users/stas/capture-", "nested/capture-"])
def test_correctness_workdir_rejects_prefix_traversal_before_allocation(fast_root, prefix):
    with pytest.raises(ValueError, match="prefix"):
        workdir.correctness_workdir(prefix)
    assert list(fast_root.iterdir()) == []
