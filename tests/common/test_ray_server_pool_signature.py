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

"""Call sites in ray_server.py must match ReplicaPool's real signature.

A MagicMock pool accepts any keyword, so this walks the source instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_POOL = _ROOT / "arctic_platform" / "inference" / "server" / "replica_pool.py"
_SERVER = _ROOT / "arctic_platform" / "common" / "ray_server.py"
_CHECKED = ("initialize", "sleep", "wake_up")


def _method_params(path: Path, class_name: str, method: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method:
                return {arg.arg for arg in item.args.args + item.args.kwonlyargs} - {"self"}
    raise AssertionError(f"{class_name}.{method} not found in {path}")


def test_ray_server_pool_calls_match_replica_pool():
    params = {name: _method_params(_POOL, "ReplicaPool", name) for name in _CHECKED}
    tree = ast.parse(_SERVER.read_text())
    mismatches: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in params:
            continue
        unknown = sorted(kw.arg for kw in node.keywords if kw.arg is not None and kw.arg not in params[node.func.attr])
        if unknown:
            mismatches.append(f"{node.func.attr} line {node.lineno}: {unknown}")
    assert mismatches == []


def test_wake_inference_does_not_take_restore_weights():
    tree = ast.parse(_SERVER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "wake_inference":
            names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            assert "restore_weights" not in names
    weight_sync = (_ROOT / "arctic_platform" / "common" / "utils" / "weight_sync.py").read_text()
    assert "restore_weights" not in weight_sync
