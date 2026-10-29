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
"""OpenHands ``terminal`` tool executor backed by a Harbor environment.

The agent keeps the stock ``TerminalTool`` schema and description, but every
command runs through ``BaseEnvironment.exec()`` of the trial's environment
(Snowflake sandbox, Docker, Modal, host, ...), so the agent and the verifier
see the same filesystem and the agent never runs commands in the training
driver's process.

Each command runs to completion in a fresh shell. The working directory
carries over between commands; exported variables and background jobs do not.

OpenHands serializes tool specs, so a rollout's executor is looked up by id
(:func:`terminal_tool`) instead of being passed in the spec.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from openhands.sdk import Tool
from openhands.sdk.tool import ToolExecutor
from openhands.sdk.tool import register_tool
from openhands.tools.terminal.definition import TerminalAction
from openhands.tools.terminal.definition import TerminalObservation
from openhands.tools.terminal.definition import TerminalTool
from openhands.tools.terminal.metadata import CmdOutputMetadata

TOOL_KEY = "harbor_terminal"
CWD_MARKER = "__HARBOR_TERMINAL_CWD__"
DEFAULT_TIMEOUT_S = 180
# Extra wait on the driver side beyond the command's own timeout, for the
# environment's transport to report back.
_RESULT_SLACK_S = 60


def wrap_command(command: str) -> str:
    """``command``, then print the shell's final working directory and keep the command's exit code."""
    return f'{command}\n__harbor_rc=$?\nprintf "\\n{CWD_MARKER}%s\\n" "$(pwd)"\nexit $__harbor_rc'


def split_cwd(stdout: str) -> tuple[str, str | None]:
    """``(stdout without the marker line, reported cwd or None)``."""
    idx = stdout.rfind(CWD_MARKER)
    if idx < 0:
        return stdout, None
    cwd = stdout[idx + len(CWD_MARKER) :].strip() or None
    output = stdout[:idx]
    return (output[:-1] if output.endswith("\n") else output), cwd


class HarborTerminalExecutor(ToolExecutor[TerminalAction, TerminalObservation]):
    """Runs ``terminal`` actions with ``environment.exec()`` on the trial's event loop."""

    def __init__(
        self,
        environment: Any,
        loop: asyncio.AbstractEventLoop,
        working_dir: str,
        *,
        default_timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._environment = environment
        self._loop = loop
        self._initial_cwd = working_dir
        self._cwd = working_dir
        self._default_timeout_s = int(default_timeout_s)

    @property
    def working_dir(self) -> str:
        return self._cwd

    def __call__(self, action: TerminalAction, conversation: Any = None) -> TerminalObservation:  # noqa: ARG002
        if action.reset:
            self._cwd = self._initial_cwd
            return TerminalObservation.from_text(
                text=(
                    "Terminal session has been reset. All previous environment variables and session state have been"
                    " cleared."
                ),
                command="[RESET]",
                exit_code=0,
            )
        if action.is_input:
            return TerminalObservation.from_text(
                text="Each command runs to completion, so there is no running process to send input to.",
                command=action.command,
                exit_code=1,
                is_error=True,
            )
        if not action.command.strip():
            return TerminalObservation.from_text(
                text="No command was given.", command=action.command, exit_code=1, is_error=True
            )

        timeout = int(action.timeout) if action.timeout else self._default_timeout_s
        future = asyncio.run_coroutine_threadsafe(
            self._environment.exec(wrap_command(action.command), cwd=self._cwd, timeout_sec=timeout),
            self._loop,
        )
        try:
            result = future.result(timeout + _RESULT_SLACK_S)
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent as the command's output
            future.cancel()
            timed_out = "timed out" in str(exc).lower() or isinstance(exc, TimeoutError)
            metadata = CmdOutputMetadata(exit_code=-1 if timed_out else 1, working_dir=self._cwd)
            return TerminalObservation.from_text(
                text=f"Command timed out after {timeout} seconds." if timed_out else f"Command failed: {exc}",
                command=action.command,
                metadata=metadata,
                exit_code=metadata.exit_code,
                timeout=timed_out,
                is_error=True,
            )

        stdout, cwd = split_cwd(result.stdout or "")
        if cwd:
            self._cwd = cwd
        stderr = result.stderr or ""
        output = stdout + (("\n" if stdout and not stdout.endswith("\n") else "") + stderr if stderr else "")
        exit_code = int(result.return_code)
        metadata = CmdOutputMetadata(exit_code=exit_code, working_dir=self._cwd)
        metadata.suffix = f"\n[The command completed with exit code {exit_code}.]"
        return TerminalObservation.from_text(
            command=action.command,
            text=output,
            metadata=metadata,
            exit_code=exit_code,
        )


_EXECUTORS: dict[str, HarborTerminalExecutor] = {}
_EXECUTORS_LOCK = threading.Lock()


def _create_terminal(conv_state: Any, executor_id: str) -> list[TerminalTool]:
    with _EXECUTORS_LOCK:
        executor = _EXECUTORS[executor_id]
    return list(TerminalTool.create(conv_state, executor=executor))


register_tool(TOOL_KEY, _create_terminal)


@contextmanager
def terminal_tool(executor: HarborTerminalExecutor) -> Iterator[Tool]:
    """A ``terminal`` tool spec bound to ``executor`` while the context is open."""
    executor_id = uuid.uuid4().hex
    with _EXECUTORS_LOCK:
        _EXECUTORS[executor_id] = executor
    try:
        yield Tool(name=TOOL_KEY, params={"executor_id": executor_id})
    finally:
        with _EXECUTORS_LOCK:
            _EXECUTORS.pop(executor_id, None)
