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
"""Localization prompts. The shell tool is named ``terminal``."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment
from jinja2 import FileSystemLoader

_PROMPTS = Path(__file__).resolve().parent / "prompts"


def render_system_prompt(max_turns: int) -> str:
    return _environment().get_template("system.j2").render(max_turns=max_turns)


def render_user_prompt(instance: dict, working_dir: str) -> str:
    return _environment().get_template("user.j2").render(instance=instance, working_dir=working_dir)


def _environment() -> Environment:
    return Environment(loader=FileSystemLoader(_PROMPTS), autoescape=False)
