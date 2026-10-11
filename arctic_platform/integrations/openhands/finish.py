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
"""The submit tool. Calling it ends the rollout.

Adapted from ``src/tools/localization_finish.py`` in
https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2

The reward is zero unless this tool is called exactly once. Searching the
repository does not score until then.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from openhands.sdk import Action
from openhands.sdk import Observation
from openhands.sdk import ToolDefinition
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.tool import ToolAnnotations
from openhands.sdk.tool import ToolExecutor
from pydantic import BaseModel
from pydantic import Field
from rich.text import Text

if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation

TOOL_DESCRIPTION = """Submit your final code localization results.

Use this tool when you have identified all relevant files, classes, and functions that need to be modified to address the issue described in the problem statement.

Provide a structured list of locations. Each location must have:
- file: Path to the file relative to the root of the repository (required)
- class_name: Class name (optional)
- function_name: Function/method name (optional)

You must submit a list of locations that require modification and for each location you must follow the below rules in your output:
1. If the required modifications belong to a specific function that belongs to a class, provide the file path, class name, and function name.
2. If the required modification belongs to a function that is not part of any class, provide the file path and function name.
3. If the required modification does not belong to any specific class or a function (e.g. global variables, imports, new class, new global function etc.), it is sufficient to provide only the file path.
4. If the required modification belongs to a class (e.g. adding a new method to a class, changing the class inheritance), provide the file path and class name. If you are modifying the __init__ method of a class, you should provide the function name as well.

IMPORTANT:
1. If multiple different edits need to be edited in the same file, you should create separate entries for each edit, specifying the same file path but different class/function names as applicable. Each entry should compulsorily include the file path.
2. Do NOT include duplicate entries in your output for which the file, class, and function names are all identical.
3. Ensure that the file paths are accurate and relative to the root of the repository without any leading "./" or "/". All locations must be valid and exist in the codebase and this applies to class and function names as well.
4. Aim for high precision (all returned locations are relevant) and high recall (no relevant locations missed).
5. The agent will terminate its execution after you call this tool.
"""


class CodeLocation(BaseModel):
    """One place in the repository. The file is required."""

    file: str = Field(description="Path to the file, relative to the repository root")
    class_name: str | None = Field(default=None, description="Class name, when the edit is inside a class")
    function_name: str | None = Field(default=None, description="Function or method name, when the edit is inside one")


class LocalizationFinishAction(Action):
    """Submit the locations that need to change."""

    locations: list[CodeLocation] = Field(description="Locations to modify. Each entry's file path is required.")

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append(f"Submitting {len(self.locations)} location(s)\n")
        for index, location in enumerate(self.locations, 1):
            content.append(f"  {index}. {location.file}")
            if location.class_name is not None:
                content.append(f" {location.class_name}")
            if location.function_name is not None:
                content.append(f".{location.function_name}")
            content.append("\n")
        return content


class LocalizationFinishObservation(Observation):
    """The conversation ends after this observation."""

    @property
    def visualize(self) -> Text:
        return Text()


class LocalizationFinishExecutor(ToolExecutor):
    def __call__(
        self,
        action: LocalizationFinishAction,
        conversation: BaseConversation | None = None,
    ) -> LocalizationFinishObservation:
        locations = [
            dict(file=location.file, class_name=location.class_name, function_name=location.function_name)
            for location in action.locations
        ]
        if conversation is not None:
            conversation.state.execution_status = ConversationExecutionStatus.FINISHED
        return LocalizationFinishObservation.from_text(text=json.dumps(locations, indent=2))


class LocalizationFinishTool(ToolDefinition[LocalizationFinishAction, LocalizationFinishObservation]):
    """Submit localization results and end the conversation."""

    @classmethod
    def create(cls, conv_state, **params) -> Sequence[LocalizationFinishTool]:
        if len(params) > 0:
            raise ValueError("LocalizationFinishTool doesn't accept parameters")
        return [
            cls(
                name="localization_finish",
                action_type=LocalizationFinishAction,
                observation_type=LocalizationFinishObservation,
                description=TOOL_DESCRIPTION,
                executor=LocalizationFinishExecutor(),
                annotations=ToolAnnotations(
                    title="localization_finish",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]
