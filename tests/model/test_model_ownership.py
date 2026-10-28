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
    "correctness/harness/checkpoint_store.py",
    "correctness/harness/gateway.py",
    "correctness/harness/lora.py",
    "correctness/harness/native_lora.py",
    "correctness/harness/storage.py",
    "debug/training.py",
}


def _dss_import_violations(package_root: Path) -> list[str]:
    violations = []
    for path in package_root.rglob("*.py"):
        relative_path = path.relative_to(package_root)
        # These tests compare AP against DSS; they are not AP runtime dependencies.
        if relative_path.parts[:2] == ("correctness", "selftest"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        # Only these reference adapters may load DSS, and only inside a function.
        reference_imports = set()
        if relative_path.as_posix() in _DSS_COMPARISON_ADAPTERS:
            reference_imports = {
                node
                for function in ast.walk(tree)
                if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
                for node in ast.walk(function)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            }
        for node in ast.walk(tree):
            if node in reference_imports:
                continue
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            violations.extend(
                f"{relative_path}:{node.lineno}: {module}"
                for module in imported
                if module == "dss" or module.startswith("dss.")
            )
    return violations


def test_arctic_platform_does_not_import_dss():
    package_root = Path(__file__).resolve().parents[2] / "arctic_platform"
    assert _dss_import_violations(package_root) == []


@pytest.mark.parametrize("relative_path", ["model/loader.py", "correctness/harness/runner.py"])
@pytest.mark.parametrize("statement", ["import dss", "import dss.ray_dss", "from dss.ray_dss import models"])
def test_dependency_check_rejects_dss_in_runtime_code(tmp_path, relative_path, statement):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(statement)
    assert _dss_import_violations(tmp_path) == [f"{relative_path}:1: {statement.split()[1]}"]


def test_dependency_check_allows_correctness_selftest_reference_imports(tmp_path):
    path = tmp_path / "correctness" / "selftest" / "check_reference.py"
    path.parent.mkdir(parents=True)
    path.write_text("from dss.ray_dss import models")
    assert _dss_import_violations(tmp_path) == []


def test_dependency_check_does_not_exempt_other_selftest_directories(tmp_path):
    path = tmp_path / "model" / "selftest" / "loader.py"
    path.parent.mkdir(parents=True)
    path.write_text("import dss")
    assert _dss_import_violations(tmp_path) == ["model/selftest/loader.py:1: dss"]


@pytest.mark.parametrize("relative_path", sorted(_DSS_COMPARISON_ADAPTERS))
def test_dependency_check_allows_only_lazy_comparison_adapter_imports(tmp_path, relative_path):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text("def compare():\n    from dss.ray_dss import models\n")
    assert _dss_import_violations(tmp_path) == []

    path.write_text("from dss.ray_dss import models\n")
    assert _dss_import_violations(tmp_path) == [f"{relative_path}:1: dss.ray_dss"]


@pytest.mark.parametrize("relative_path", ["common/deepspeed_engine.py", "correctness/harness/runner.py"])
def test_dependency_check_rejects_lazy_dss_imports_outside_comparison_adapters(tmp_path, relative_path):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text("def run():\n    import dss\n")
    assert _dss_import_violations(tmp_path) == [f"{relative_path}:2: dss"]
