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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""List maintained diagnostics or invoke one existing module entry point."""

from __future__ import annotations

import argparse
import runpy
import sys
from typing import Sequence

DIAGNOSTICS = {
    "lm_head_chunking": "Regression with the chunked LM head disabled",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arctic_platform.correctness.diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list diagnostic names without importing them")
    run = commands.add_parser("run", help="run one diagnostic with its existing arguments", add_help=False)
    run.add_argument("name", choices=sorted(DIAGNOSTICS))
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "list":
        for name, description in sorted(DIAGNOSTICS.items()):
            print(f"{name}\t{description}")
        return 0

    module = f"arctic_platform.correctness.diagnostics.{args.name}"
    original_argv = sys.argv
    sys.argv = [module, *args.arguments]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as error:
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        raise
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
