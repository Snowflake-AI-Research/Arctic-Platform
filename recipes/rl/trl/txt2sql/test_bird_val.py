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

"""CPU tests for verl-style BIRD val metric keys and sql_reward_detailed."""

from __future__ import annotations

import os
import sqlite3
import tempfile

import bird_task
from bird_val import VAL_AUX_EXEC
from bird_val import VAL_AUX_FORMAT
from bird_val import VAL_AUX_N
from bird_val import VAL_AUX_SCORE
from bird_val import VAL_CORE_REWARD
from bird_val import to_verl_val_metrics


def _make_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO t (id, name) VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()


def test_sql_reward_detailed_keys():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "toy.sqlite")
        _make_db(db)
        gold = "SELECT id FROM t ORDER BY id"
        details = bird_task.sql_reward_detailed(
            completions=["```sql\nSELECT id FROM t ORDER BY id\n```"],
            ground_truth=[gold],
            db_path=[db],
            data_source=["bird"],
        )
        assert len(details) == 1
        assert set(details[0]) == {"score", "format_correct", "execution_success"}
        assert details[0]["score"] == 1.0
        assert details[0]["execution_success"] == 1.0
        floats = bird_task.sql_reward(
            completions=["```sql\nSELECT id FROM t ORDER BY id\n```"],
            ground_truth=[gold],
            db_path=[db],
        )
        assert floats == [1.0]


def test_verl_key_names_and_means():
    details = [
        {"score": 1.0, "format_correct": 1.0, "execution_success": 1.0},
        {"score": 0.1, "format_correct": 1.0, "execution_success": 0.0},
        {"score": 0.0, "format_correct": 0.0, "execution_success": 0.0},
    ]
    metrics = to_verl_val_metrics(details, n_prompts=3, dropped_overlong=1, truncated=0, considered=4, time_s=1.5)
    assert abs(metrics[VAL_CORE_REWARD] - (1.1 / 3)) < 1e-9
    assert metrics[VAL_AUX_SCORE] == metrics[VAL_CORE_REWARD]
    assert abs(metrics[VAL_AUX_EXEC] - (1.0 / 3)) < 1e-9
    assert abs(metrics[VAL_AUX_FORMAT] - (2.0 / 3)) < 1e-9
    assert metrics[VAL_AUX_N] == 3.0
    assert metrics["val-aux/bird/dropped_overlong"] == 1.0
    assert metrics["val-aux/bird/time_s"] == 1.5


def test_empty_details_zero_means():
    metrics = to_verl_val_metrics([], n_prompts=0, dropped_overlong=5, considered=5)
    assert metrics[VAL_CORE_REWARD] == 0.0
    assert metrics[VAL_AUX_N] == 0.0
    assert metrics["val-aux/bird/dropped_overlong"] == 5.0
