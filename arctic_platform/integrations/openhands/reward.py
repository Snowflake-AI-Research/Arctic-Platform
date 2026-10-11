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
"""File, module, and function F1 for a localization submission.

Adapted from ``src/rewards/file_localization/`` in
https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2

The score is the sum of three F1s, each in ``[0, 1]``, so the maximum is 3.
A rollout that did not submit exactly one ``localization_finish`` scores 0.
Duplicate predictions collapse to a set, which is what the cited scorer does.
"""

from __future__ import annotations

from typing import Any


def _present(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def f1(predicted: set[str], truth: set[str]) -> float:
    """Set F1. An empty ground-truth set scores 0."""
    if len(truth) == 0:
        return 0.0
    overlap = len(predicted & truth)
    precision = overlap / len(predicted) if len(predicted) > 0 else 0.0
    recall = overlap / len(truth)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def parse_locations(locations: list[dict]) -> tuple[set[str], set[str], set[str]]:
    """Split a finish submission into file, module, and entity ids.

    An empty file path rejects the whole submission. A module is
    ``file:Class`` when a class is named, otherwise ``file:function``. An
    entity is ``file:Class.function`` when both are named, otherwise
    ``file:function``.
    """
    files: list[str] = []
    modules: list[str] = []
    entities: list[str] = []
    for location in locations:
        file_path = location.get("file")
        if not _present(file_path):
            return set(), set(), set()
        class_name = location.get("class_name")
        function_name = location.get("function_name")
        files.append(file_path)
        if _present(class_name):
            modules.append(f"{file_path}:{class_name}")
        elif _present(function_name):
            modules.append(f"{file_path}:{function_name}")
        if _present(class_name) and _present(function_name):
            entities.append(f"{file_path}:{class_name}.{function_name}")
        elif _present(function_name):
            entities.append(f"{file_path}:{function_name}")
    return set(files), set(modules), set(entities)


def _ground_truth(instance: dict) -> tuple[set[str], set[str], set[str]]:
    files: list[str] = []
    modules: list[str] = []
    entities: list[str] = []
    for change in instance.get("file_changes") or []:
        file_path = change.get("file")
        if _present(file_path):
            files.append(file_path)
        edits = change.get("changes") or {}
        for module in edits.get("edited_modules") or []:
            if _present(module):
                modules.append(module)
        for entity in edits.get("edited_entities") or []:
            if _present(entity):
                entities.append(entity)
    return set(files), set(modules), set(entities)


def localization_reward(
    structured_locations: list[dict] | None,
    instance: dict,
    *,
    file_weight: float = 1.0,
    module_weight: float = 1.0,
    entity_weight: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Sum of weighted file, module, and entity F1. ``None`` locations score 0."""
    if structured_locations is None:
        detail = dict(file_reward=0.0, module_reward=0.0, entity_reward=0.0, reward=0.0)
        return 0.0, detail
    predicted = parse_locations(structured_locations)
    truth = _ground_truth(instance)
    scores = [f1(pred, gold) for pred, gold in zip(predicted, truth)]
    weights = [file_weight, module_weight, entity_weight]
    total = sum(score * weight for score, weight in zip(scores, weights))
    detail = dict(
        file_reward=scores[0],
        module_reward=scores[1],
        entity_reward=scores[2],
        reward=total,
    )
    return total, detail
