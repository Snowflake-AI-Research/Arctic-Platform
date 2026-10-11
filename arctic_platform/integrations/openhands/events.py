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
"""Read a localization submission out of an OpenHands conversation.

Adapted from ``get_structured_locations`` and ``sanity_check_last_step`` in
https://github.com/18jeffreyma/codescout/blob/abab719e08a55dde78c6da864cd24d84fd47bdf2/src/generator/code_search_generator.py

Events may be OpenHands objects or the dicts ``model_dump`` produces. The
submission counts only when ``localization_finish`` was called exactly once,
and the last sampled turn contains exactly one well-formed tool call.
"""

from __future__ import annotations

from typing import Any


def _action(event: Any) -> Any:
    if isinstance(event, dict):
        return event.get("action") or {}
    return getattr(event, "action", None)


def _kind(value: Any, field: str) -> str | None:
    if isinstance(value, dict):
        kind = value.get(field)
        return kind if isinstance(kind, str) else None
    kind = getattr(value, field, None)
    if isinstance(kind, str):
        return kind
    return type(value).__name__ if value is not None and field == "kind" else None


def _is_finish(event: Any) -> bool:
    if isinstance(event, dict):
        source = event.get("source")
        event_kind = event.get("kind")
    else:
        source = getattr(event, "source", None)
        event_kind = getattr(event, "kind", None)
        if event_kind is None:
            event_kind = type(event).__name__
    action = _action(event)
    action_kind = _kind(action, "kind")
    return source == "agent" and event_kind == "ActionEvent" and action_kind == "LocalizationFinishAction"


def _locations_of(event: Any) -> list[dict]:
    action = _action(event)
    if isinstance(action, dict):
        locations = action.get("locations") or []
    else:
        locations = getattr(action, "locations", None) or []
    parsed = []
    for location in locations:
        if isinstance(location, dict):
            parsed.append(
                dict(
                    file=location.get("file"),
                    class_name=location.get("class_name"),
                    function_name=location.get("function_name"),
                )
            )
        else:
            parsed.append(
                dict(
                    file=getattr(location, "file", None),
                    class_name=getattr(location, "class_name", None),
                    function_name=getattr(location, "function_name", None),
                )
            )
    return parsed


def finish_locations(events: list[Any]) -> list[dict] | None:
    """Locations from the single finish call, or ``None`` when it was not exactly once."""
    finishes = [event for event in events if _is_finish(event)]
    if len(finishes) != 1:
        return None
    return _locations_of(finishes[0])


def last_step_is_one_tool_call(text: str) -> bool:
    """The last sampled turn is one tool call and nothing after it.

    More than one ``<tool_call>``, a missing ``<|im_end|>``, or any
    non-whitespace between ``</tool_call>`` and ``<|im_end|>`` rejects the
    submission. Searching correctly does not count until this holds.
    """
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        return False
    if text.count("<|im_end|>") != 1:
        return False
    between = text.split("</tool_call>", 1)[1].split("<|im_end|>", 1)[0]
    return between.strip() == ""
