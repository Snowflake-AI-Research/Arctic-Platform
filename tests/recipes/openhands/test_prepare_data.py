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
"""SWE-smith rows become the recipe's train and validation parquets."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from prepare_data import VALIDATION_ROWS  # noqa: E402
from prepare_data import build  # noqa: E402


def _raw(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        dict(
            instance_id=[f"repo.{i}" for i in range(n)],
            file_changes=[[dict(file=f"f{i}.py")] for i in range(n)],
            repo=["swesmith/repo"] * n,
            base_commit=[None] * n,
            problem_statement=["  "] + [f"bug {i}" for i in range(1, n)],
            patch=["diff"] * n,
        )
    )


def test_blank_problem_statements_are_dropped_and_100_held_out():
    train, validation = build(_raw(VALIDATION_ROWS + 11))
    assert len(validation) == VALIDATION_ROWS
    assert len(train) == 10
    assert "repo.0" not in set(train.instance_id) | set(validation.instance_id)


def test_rows_carry_prompt_target_and_use_patch():
    train, _ = build(_raw(VALIDATION_ROWS + 5))
    row = train.iloc[0]
    assert list(row.prompt) == [dict(role="user", content=row.problem_statement)]
    assert row.target == row.file_changes
    assert row.use_patch
    assert row.base_commit is None


def test_split_is_deterministic():
    first, _ = build(_raw(VALIDATION_ROWS + 20))
    second, _ = build(_raw(VALIDATION_ROWS + 20))
    assert list(first.instance_id) == list(second.instance_id)
