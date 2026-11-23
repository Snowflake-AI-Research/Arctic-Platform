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

import ast
from pathlib import Path

import pytest

_DSS_COMPARISON_ADAPTERS = {
    "correctness/harness/engine_admission.py",
    "correctness/harness/gateway.py",
    "correctness/harness/lora.py",
    "correctness/harness/native_optimizer.py",
    "correctness/harness/optimizer.py",
    "correctness/harness/storage.py",
}


def _dss_import_violations(source: str, path: Path) -> list[str]:
    # Only the correctness integration can depend on the optional DSS runtime,
    # and its imports must wait until a runtime-dependent function is called.
    optional_adapter = path.as_posix() in _DSS_COMPARISON_ADAPTERS
    violations = []

    def visit(node, inside_function=False):
        inside_function = inside_function or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            imported = []
        violations.extend(
            f"{path}:{node.lineno}: {module}"
            for module in imported
            if (module == "dss" or module.startswith("dss.")) and not (optional_adapter and inside_function)
        )
        for child in ast.iter_child_nodes(node):
            visit(child, inside_function)

    visit(ast.parse(source, filename=str(path)))
    return violations


def test_arctic_platform_does_not_import_dss():
    package_root = Path(__file__).resolve().parents[2] / "arctic_platform"
    violations = [
        violation
        for path in package_root.rglob("*.py")
        for violation in _dss_import_violations(path.read_text(), path.relative_to(package_root))
    ]
    assert violations == []


@pytest.mark.parametrize(
    "path",
    [
        "model/loader.py",
        "common/deepspeed_engine.py",
        "correctness/harness/runner.py",
        *sorted(_DSS_COMPARISON_ADAPTERS),
    ],
)
@pytest.mark.parametrize("statement", ["import dss", "from dss.ray_dss.models import JobConfig"])
@pytest.mark.parametrize("scope", ["module", "function", "async-function"])
def test_dss_dependency_boundary_requires_a_deferred_correctness_adapter(path, statement, scope):
    source = (
        statement
        if scope == "module"
        else f"{'async ' if scope == 'async-function' else ''}def load():\n    {statement}"
    )
    violations = _dss_import_violations(source, Path(path))
    permitted = path in _DSS_COMPARISON_ADAPTERS and scope != "module"
    assert bool(violations) is not permitted


@pytest.mark.parametrize("relative_path", ["model/loader.py", "correctness/harness/runner.py"])
@pytest.mark.parametrize("statement", ["import dss", "import dss.ray_dss", "from dss.ray_dss import models"])
def test_dependency_check_rejects_dss_in_runtime_code(tmp_path, relative_path, statement):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(statement)
    assert _dss_import_violations(path.read_text(), Path(relative_path)) == [
        f"{relative_path}:1: {statement.split()[1]}"
    ]


def test_dependency_check_rejects_correctness_selftest_reference_imports(tmp_path):
    relative_path = Path("correctness/selftest/check_reference.py")
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text("from dss.ray_dss import models")
    assert _dss_import_violations(path.read_text(), relative_path) == [f"{relative_path}:1: dss.ray_dss"]


def test_dependency_check_does_not_exempt_other_selftest_directories(tmp_path):
    relative_path = Path("model/selftest/loader.py")
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text("import dss")
    assert _dss_import_violations(path.read_text(), relative_path) == [f"{relative_path}:1: dss"]


@pytest.mark.parametrize("relative_path", sorted(_DSS_COMPARISON_ADAPTERS))
@pytest.mark.parametrize("function", ["def", "async def"])
def test_dependency_check_allows_only_lazy_comparison_adapter_imports(tmp_path, relative_path, function):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(f"{function} compare():\n    from dss.ray_dss import models\n")
    assert _dss_import_violations(path.read_text(), Path(relative_path)) == []

    path.write_text("from dss.ray_dss import models\n")
    assert _dss_import_violations(path.read_text(), Path(relative_path)) == [f"{relative_path}:1: dss.ray_dss"]


@pytest.mark.parametrize("relative_path", ["common/deepspeed_engine.py", "correctness/harness/runner.py"])
def test_dependency_check_rejects_lazy_dss_imports_outside_comparison_adapters(tmp_path, relative_path):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text("def run():\n    import dss\n")
    assert _dss_import_violations(path.read_text(), Path(relative_path)) == [f"{relative_path}:2: dss"]


@pytest.mark.parametrize(
    "relative_path",
    [
        "debug/other.py",
        "debug/training_extra.py",
        "correctness/harness/gateway_extra.py",
        "correctness/selftest_extra/check_reference.py",
        "debug/training.py",
        "correctness/harness/weight_cache.py",
    ],
)
def test_dependency_check_does_not_expand_the_reference_allowlist(tmp_path, relative_path):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text("def compare():\n    import dss as reference\n")
    assert _dss_import_violations(path.read_text(), Path(relative_path)) == [f"{relative_path}:2: dss"]


@pytest.mark.parametrize("scope", ["if True:", "class Reference:"])
def test_dependency_check_rejects_eager_adapter_imports_in_module_scopes(tmp_path, scope):
    relative_path = Path("correctness/harness/engine_admission.py")
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(f"{scope}\n    import dss\n")
    assert _dss_import_violations(path.read_text(), relative_path) == [f"{relative_path}:2: dss"]


def test_dependency_check_matches_the_dss_package_not_similarly_named_modules(tmp_path):
    relative_path = Path("model/loader.py")
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text("import dss_tools\nfrom another.dss import reference\n")
    assert _dss_import_violations(path.read_text(), relative_path) == []
