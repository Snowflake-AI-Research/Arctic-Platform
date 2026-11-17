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

"""The BIRD gold-SQL cache must not remember a timeout."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[2] / "recipes" / "rl" / "verl" / "txt2sql" / "bird_reward.py"
    spec = importlib.util.spec_from_file_location("bird_reward_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_timeout_then_success_reruns_the_gold_query():
    module = _load()
    calls = {"n": 0}

    def _execute(db_path, sql, timeout):
        del db_path, sql, timeout
        calls["n"] += 1
        if calls["n"] == 1:
            return TimeoutError("slow")
        return frozenset({(1,)})

    module._execute_with_timeout = _execute
    module._GOLD_RESULTS.clear()
    first = module._cached_gold_result("/tmp/db.sqlite", "select 1", 1.0)
    assert isinstance(first, TimeoutError)
    second = module._cached_gold_result("/tmp/db.sqlite", "select 1", 1.0)
    assert second == frozenset({(1,)})
    assert calls["n"] == 2
    third = module._cached_gold_result("/tmp/db.sqlite", "select 1", 1.0)
    assert third == second
    assert calls["n"] == 2


def test_gold_cache_drops_the_oldest_success():
    module = _load()
    module._GOLD_CACHE_MAX = 1
    module._GOLD_RESULTS.clear()
    seen = {"n": 0}

    def _execute(db_path, sql, timeout):
        del db_path, timeout
        seen["n"] += 1
        return frozenset({(sql,)})

    module._execute_with_timeout = _execute
    module._cached_gold_result("/tmp/db.sqlite", "select 1", 1.0)
    module._cached_gold_result("/tmp/db.sqlite", "select 2", 1.0)
    assert len(module._GOLD_RESULTS) == 1
    module._cached_gold_result("/tmp/db.sqlite", "select 1", 1.0)
    assert seen["n"] == 3
