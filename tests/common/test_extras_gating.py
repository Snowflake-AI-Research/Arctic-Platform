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
"""``require_extra`` must accept any extra that provisions the caller, and every
extra it names must actually exist.

Nothing here knows what this project depends on. The parser runs against fictional
``Requires-Dist`` lines, and the two call-site checks compare extra *names* only, so
adding, moving, or renaming a dependency never requires touching this file.

Pure metadata + AST checks: no install, no network, no GPUs.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

from arctic_platform import _extras

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Hatchling flattens ``arctic_platform[...]`` self-references at build time, so each
# extra lists its transitive closure directly. This mirrors the shape of the real
# METADATA; the distributions are invented.
FLATTENED = (
    "unconditional>=1.0",
    "shared-dep; extra == 'lite'",
    "lite-only; extra == 'lite'",
    "shared-dep; extra == 'full'",
    "lite-only; extra == 'full'",
    "full-only[sub,other]>=0.2.0; extra == 'full'",
    "pinned-dep==1.2.3; extra == 'full'",
)

# setuptools and pdm emit the self-reference instead of flattening it.
NESTED = (
    "shared-dep; extra == 'lite'",
    'arctic_platform[lite]; extra == "full"',
    "pinned-dep==1.2.3; extra == 'full'",
)

# A layout in which [full] has stopped carrying [lite]'s packages.
DIVERGED = (
    "lite-only; extra == 'lite'",
    "shared-dep; extra == 'lite'",
    "shared-dep; extra == 'full'",
    "pinned-dep==1.2.3; extra == 'full'",
)
_FULL_INSTALLED = {"shared-dep", "pinned-dep"}


@pytest.fixture
def gate(monkeypatch):
    """Drive the extras module from fictional metadata and a fictional installed set."""

    def configure(requires_dist, installed=()):
        monkeypatch.setattr(_extras, "_requires_dist", lambda: tuple(requires_dist))
        monkeypatch.setattr(_extras, "_installed", lambda name: name in set(installed))
        _extras._provided_by.cache_clear()
        return _extras

    _extras._provided_by.cache_clear()
    yield configure
    _extras._provided_by.cache_clear()


class TestProvidedBy:
    def test_strips_version_specs_and_bracketed_extras(self, gate):
        """A requirement resolves to its bare distribution name."""
        resolved = gate(FLATTENED)._provided_by("full")
        assert resolved == {"shared-dep", "lite-only", "full-only", "pinned-dep"}

    def test_ignores_unconditional_deps_and_other_extras(self, gate):
        """Only lines carrying this extra's marker count."""
        assert gate(FLATTENED)._provided_by("lite") == {"shared-dep", "lite-only"}

    def test_recurses_self_references(self, gate):
        """A self-reference pulls in the referenced extra's requirements."""
        assert gate(NESTED)._provided_by("full") == {"shared-dep", "pinned-dep"}

    def test_unknown_extra_resolves_empty(self, gate):
        """An extra absent from the metadata yields nothing, so its gate stays quiet."""
        assert gate(FLATTENED)._provided_by("nope") == frozenset()


class TestRequireExtra:
    def test_passes_when_named_extra_is_installed(self, gate):
        """A satisfied extra raises nothing."""
        gate(FLATTENED, installed={"shared-dep", "lite-only"}).require_extra("lite")

    def test_accepts_any_named_extra(self, gate):
        """An install satisfying only the second name still passes."""
        gate(DIVERGED, installed=_FULL_INSTALLED).require_extra("lite", "full")

    def test_single_extra_rejects_a_diverged_install(self, gate):
        """Naming one extra is what made a [full]-only install fail spuriously."""
        with pytest.raises(ImportError, match="lite-only"):
            gate(DIVERGED, installed=_FULL_INSTALLED).require_extra("lite")

    def test_reports_missing_packages_and_every_option(self, gate):
        """The error names what is absent and each extra that would supply it."""
        with pytest.raises(ImportError, match=r"lite-only.*\[lite\]' or 'arctic-platform\[full\]"):
            gate(DIVERGED).require_extra("lite", "full")

    def test_absent_metadata_is_a_noop(self, gate):
        """An uninstalled checkout has nothing to check, so the real ImportError surfaces."""
        gate((), installed=()).require_extra("lite", "full")


def _gated_extras(package_root: Path) -> dict[str, set[str]]:
    """Map source file -> the extras named in its ``require_extra(...)`` calls."""
    found: dict[str, set[str]] = {}
    for path in sorted(package_root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != "require_extra":
                continue
            named = {arg.value for arg in node.args if isinstance(arg, ast.Constant)}
            if named:
                found.setdefault(str(path.relative_to(package_root.parent)), set()).update(named)
    return found


def _optional_dependencies() -> dict[str, list[str]]:
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


class TestExtraNamesResolve:
    """Extra names only. Neither check reads a dependency list."""

    def test_every_gated_extra_is_declared(self):
        """A gate naming an extra that pyproject.toml does not define would never fire."""
        gates = _gated_extras(_REPO_ROOT / "arctic_platform")
        assert gates, "no require_extra() call sites found — was the helper renamed?"
        declared = set(_optional_dependencies())
        for site, extras in sorted(gates.items()):
            assert extras <= declared, f"{site} gates on undeclared {sorted(extras - declared)}"

    def test_self_references_resolve(self):
        """An undefined arctic_platform[...] reference installs nothing, silently."""
        optional = _optional_dependencies()
        for extra, deps in optional.items():
            for dep in deps:
                if dep.startswith("arctic_platform["):
                    ref = dep.split("[")[1].rstrip("]")
                    assert ref in optional, f"[{extra}] references undefined [{ref}]"
