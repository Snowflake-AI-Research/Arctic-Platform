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
"""``require_any_dep_group`` must accept any extra that provisions the caller,
and every extra it names must actually exist.

Nothing here knows what this project depends on. The parser runs against fictional
``Requires-Dist`` lines, and the two call-site checks compare extra *names* only, so
adding, moving, or renaming a dependency never requires touching this file.

Pure metadata + AST checks: no install, no network, no GPUs.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

from arctic_platform import _dependency_groups as dependency_groups

_REPO_ROOT = Path(__file__).resolve().parents[1]

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
    """Drive dependency-group checks from fictional metadata and installs."""

    def configure(requires_dist, installed=()):
        monkeypatch.setattr(dependency_groups, "_requires_dist", lambda: tuple(requires_dist))
        monkeypatch.setattr(dependency_groups, "_installed", lambda name: name in set(installed))
        dependency_groups._provided_by.cache_clear()
        return dependency_groups

    dependency_groups._provided_by.cache_clear()
    yield configure
    dependency_groups._provided_by.cache_clear()


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


class TestRequireAnyDepGroup:
    def test_passes_when_named_extra_is_installed(self, gate):
        """A satisfied extra raises nothing."""
        gate(FLATTENED, installed={"shared-dep", "lite-only"}).require_any_dep_group("lite")

    def test_accepts_any_named_extra(self, gate):
        """An install satisfying only the second name still passes."""
        gate(DIVERGED, installed=_FULL_INSTALLED).require_any_dep_group("lite", "full")

    def test_single_extra_rejects_a_diverged_install(self, gate):
        """Naming one extra is what made a [full]-only install fail spuriously."""
        with pytest.raises(ImportError, match="lite-only"):
            gate(DIVERGED, installed=_FULL_INSTALLED).require_any_dep_group("lite")

    def test_reports_missing_packages_and_every_option(self, gate):
        """The error names what is absent and each extra that would supply it."""
        with pytest.raises(ImportError, match=r"lite-only.*\[lite\]' or 'arctic-platform\[full\]"):
            gate(DIVERGED).require_any_dep_group("lite", "full")

    def test_absent_metadata_is_a_noop(self, gate):
        """An uninstalled checkout has nothing to check, so the real ImportError surfaces."""
        gate((), installed=()).require_any_dep_group("lite", "full")


def _gated_extras(package_root: Path) -> dict[str, set[str]]:
    """Map source file -> extras named in ``require_any_dep_group(...)`` calls."""
    found: dict[str, set[str]] = {}
    for path in sorted(package_root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != "require_any_dep_group":
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
        assert gates, "no require_any_dep_group() call sites found — was the helper renamed?"
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

    def test_cortex_install_does_not_pull_the_inference_extra(self):
        """[cortex] and [sft] do not select [inference]. [rl] keeps published arctic-inference."""
        with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
            project = tomllib.load(f)
        include = project["tool"]["hatch"]["build"]["include"]
        assert "arctic_platform" in include
        optional = project["project"]["optional-dependencies"]
        assert "arctic-inference[server,vllm]>=0.3.0" in optional["rl"]
        assert "arctic_platform[inference]" not in optional["rl"]
        hook = project["tool"]["hatch"]["build"]["hooks"]["custom"]
        assert hook["path"] == "hatch_build.py"
        for extra in ("cortex", "sft"):
            for dep in optional[extra]:
                assert "arctic_platform[inference]" not in dep
                assert "vllm" not in dep

    def test_precompiled_ops_are_opt_in(self, monkeypatch):
        """Native extensions compile only when ARCTIC_INFERENCE_PRECOMPILED_OPS is set."""
        sys.path.insert(0, str(_REPO_ROOT))
        from hatch_build import precompiled_ops_requested

        monkeypatch.delenv("ARCTIC_INFERENCE_PRECOMPILED_OPS", raising=False)
        assert not precompiled_ops_requested()
        for value in ("1", "true", "on"):
            monkeypatch.setenv("ARCTIC_INFERENCE_PRECOMPILED_OPS", value)
            assert precompiled_ops_requested()
        monkeypatch.setenv("ARCTIC_INFERENCE_PRECOMPILED_OPS", "0")
        assert not precompiled_ops_requested()

    def test_inference_docs_and_benchmarks_are_excluded_from_the_wheel(self):
        """The copied docs tree sits in the package directory and stays out of the wheel."""
        with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
            project = tomllib.load(f)
        exclude = project["tool"]["hatch"]["build"]["exclude"]
        excluded_dirs = (
            "arctic_platform/inference/benchmark/",
            "arctic_platform/inference/docs/",
            "arctic_platform/inference/projects/",
        )
        for path in excluded_dirs + ("arctic_platform/inference/README.md",):
            assert path in exclude
            assert (_REPO_ROOT / path.rstrip("/")).exists()
        for path in (
            "arctic_platform/inference/csrc",
            "arctic_platform/inference/setup.py",
            "arctic_platform/inference/semi_persistence/scripts",
        ):
            assert (_REPO_ROOT / path).exists()
            assert path not in exclude
            assert f"{path}/" not in exclude
