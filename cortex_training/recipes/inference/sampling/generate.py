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

"""Tokenize chat prompts and generate with a Cortex Training sampling job."""

from __future__ import annotations

from typing import Any


def render_chat(renderer: Any, messages: list[dict[str, str]]) -> list[int]:
    conversation = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role not in ("user", "assistant", "system"):
            role = "user"
        conversation.append({"role": role, "content": str(message.get("content") or "")})
    if not conversation:
        raise ValueError("cannot render an empty chat")
    token_ids = [int(token) for token in renderer.build_generation_prompt(conversation).to_ints()]
    if not token_ids:
        raise ValueError("renderer produced an empty generation prompt")
    return token_ids


def render_user_prompt(renderer: Any, prompt: str) -> list[int]:
    return render_chat(renderer, [{"role": "user", "content": prompt}])


def completion_text(result: Any) -> str:
    """Extract generated text from a Cortex Training / dss-platform generate item."""
    if not isinstance(result, dict):
        return str(result or "")
    for key in ("text", "completion", "generated_text", "output_text"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return str(result.get("text") or "")
