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

"""Preserve the shared MoE argument constructor while adding precision options."""

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

OLD_ARGUMENTS = (
    ("num_experts", 8),
    ("num_shared_experts", 1),
    ("score_func", "softmax"),
    ("route_norm", True),
    ("route_scale", 1.25),
    ("score_before_experts", False),
    ("top_k", 2),
    ("use_grouped_mm", False),
    ("load_balance_coeff", 0.125),
    ("fp8_block_size", 128),
    ("swiglu_limit", 7.0),
)


@pytest.fixture
def args_class():
    root = Path(
        os.environ.get(
            "QWEN35_SOURCE_ROOT",
            Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations",
        )
    )
    path = root / "moe/layers/moe.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MoEArgs")
    namespace = {"dataclass": dataclass, "Literal": Literal}
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace["MoEArgs"]


@pytest.mark.parametrize("count", [6, 7, len(OLD_ARGUMENTS)])
def test_existing_positional_arguments_keep_their_meaning(args_class, count):
    supplied = OLD_ARGUMENTS[:count]
    args = args_class(*(value for _, value in supplied))
    assert {name: getattr(args, name) for name, _ in supplied} == dict(supplied)


def test_precision_keyword_does_not_shift_existing_positional_arguments(args_class):
    args = args_class(*(value for _, value in OLD_ARGUMENTS), route_weights_dtype="input")
    assert {name: getattr(args, name) for name, _ in OLD_ARGUMENTS} == dict(OLD_ARGUMENTS)
    assert args.route_weights_dtype == "input"
