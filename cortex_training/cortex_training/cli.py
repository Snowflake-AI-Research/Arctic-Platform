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

"""Public ``cortex-training`` command entry point."""

from __future__ import annotations

import sys
from typing import Any
from typing import Callable
from typing import TextIO

import cortex_training._cli as _implementation
from cortex_training.tui.__main__ import run as _run_tui

_TUI_FLAG_OPTIONS = {"--no-verify-ssl"}
_TUI_VALUE_OPTIONS = {
    "--config",
    "--base-url",
    "--host",
    "--pat",
    "--database",
    "--schema",
    "--endpoint",
    "--poll-interval",
    "--poll-timeout",
}


def _tui_argv(argv: list[str]) -> list[str] | None:
    prefix = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "tui":
            return prefix + argv[index + 1 :]
        if argument in _TUI_FLAG_OPTIONS:
            prefix.append(argument)
            index += 1
            continue
        if argument in _TUI_VALUE_OPTIONS:
            if index + 1 >= len(argv):
                return None
            prefix.extend(argv[index : index + 2])
            index += 2
            continue
        if any(argument.startswith(option + "=") for option in _TUI_VALUE_OPTIONS):
            prefix.append(argument)
            index += 1
            continue
        return None
    return None


def build_parser():
    return _implementation.build_parser(prog="cortex-training", include_tui=True)


def parse_args(argv: list[str] | None = None):
    return _implementation.parse_args(
        argv,
        prog="cortex-training",
        include_tui=True,
    )


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[Any], Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    tui_argv = _tui_argv(effective_argv)
    if tui_argv is not None:
        return _run_tui(tui_argv, prog="cortex-training tui")
    return _implementation.main(
        effective_argv,
        prog="cortex-training",
        include_tui=True,
        client_factory=client_factory,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
