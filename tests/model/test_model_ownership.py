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

_OPTIONAL_DEBUG_DSS_IMPORTS = {
    ("debug/training.py", ("participation_coordinator_class", "_spawn_workers"), "dss.ray_dss.jobs.gpu"),
    ("debug/training.py", ("main",), "dss.ray_dss"),
}


def _dss_imports(tree, functions=()):
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _dss_imports(node, (*functions, node.name))
            continue
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            imported = []
        for module in imported:
            if module == "dss" or module.startswith("dss."):
                yield node.lineno, functions, module
        yield from _dss_imports(node, functions)


def test_arctic_platform_does_not_import_dss():
    package_root = Path(__file__).resolve().parents[2] / "arctic_platform"
    violations = []
    optional = set()
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root).as_posix()
        tree = ast.parse(path.read_text(), filename=str(path))
        for line, functions, module in _dss_imports(tree):
            identity = relative, functions, module
            if identity in _OPTIONAL_DEBUG_DSS_IMPORTS:
                optional.add(identity)
            else:
                violations.append(f"{relative}:{line}: {module}")
    assert violations == []
    assert optional == _OPTIONAL_DEBUG_DSS_IMPORTS


@pytest.mark.parametrize(
    ("relative", "code"),
    [
        ("model/factory.py", "import dss"),
        ("common/deepspeed_worker.py", "def load():\n    from dss.ray_dss import training"),
        ("debug/training.py", "from dss.ray_dss import gateway_server"),
        ("debug/training.py", "def unrelated():\n    from dss.ray_dss import gateway_server"),
        ("debug/other.py", "def main():\n    from dss.ray_dss import gateway_server"),
    ],
)
def test_ownership_exceptions_do_not_admit_ordinary_or_eager_dss_imports(relative, code):
    imports = list(_dss_imports(ast.parse(code)))
    assert imports
    assert all((relative, functions, module) not in _OPTIONAL_DEBUG_DSS_IMPORTS for _, functions, module in imports)
