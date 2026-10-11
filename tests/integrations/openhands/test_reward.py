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
"""Localization reward. The SWE-smith row shape is file_changes[].changes."""

from __future__ import annotations

from arctic_platform.integrations.openhands.reward import f1
from arctic_platform.integrations.openhands.reward import localization_reward
from arctic_platform.integrations.openhands.reward import parse_locations

_INSTANCE = dict(
    file_changes=[
        dict(
            file="src/dotenv/cli.py",
            changes=dict(
                edited_modules=["src/dotenv/cli.py:cli"],
                edited_entities=["src/dotenv/cli.py:cli"],
            ),
        )
    ]
)


def test_missing_submission_scores_zero():
    reward, detail = localization_reward(None, _INSTANCE)
    assert reward == 0.0
    assert detail["file_reward"] == 0.0


def test_exact_function_scores_three():
    locations = [dict(file="src/dotenv/cli.py", class_name=None, function_name="cli")]
    reward, detail = localization_reward(locations, _INSTANCE)
    assert detail["file_reward"] == 1.0
    assert detail["module_reward"] == 1.0
    assert detail["entity_reward"] == 1.0
    assert reward == 3.0


def test_file_only_scores_one():
    locations = [dict(file="src/dotenv/cli.py", class_name=None, function_name=None)]
    reward, detail = localization_reward(locations, _INSTANCE)
    assert detail["file_reward"] == 1.0
    assert detail["module_reward"] == 0.0
    assert detail["entity_reward"] == 0.0
    assert reward == 1.0


def test_empty_file_rejects_the_submission():
    assert parse_locations([dict(file="", class_name=None, function_name="cli")]) == (set(), set(), set())


def test_class_method_ids():
    files, modules, entities = parse_locations(
        [dict(file="src/a.py", class_name="Parser", function_name="parse_data")]
    )
    assert files == {"src/a.py"}
    assert modules == {"src/a.py:Parser"}
    assert entities == {"src/a.py:Parser.parse_data"}


def test_empty_ground_truth_scores_zero():
    assert f1({"a.py"}, set()) == 0.0
