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

"""Resolve Cortex Training connection arguments and launch the read-only TUI.

Connection handling is delegated to ``cortex_training._cli`` so the TUI shares the
SDK's auth surface: it honours ``cortex-training login`` state, ``--config`` /
``CORTEX_TRAINING_CONFIG``, the ``CORTEX_TRAINING_*`` / ``SNOWFLAKE_*`` env vars, and explicit
``--base-url`` (local/mock) or ``--host`` + ``--pat`` flags, exactly like the
CLI. Run ``cortex-training login --config config.json`` once and then just
``cortex-training tui JOB_ID``.
"""

from __future__ import annotations

import argparse
import os
import sys


def _build_arg_parser(*, prog: str = "cortex-training tui") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Read-only TUI for Cortex Training logs.",
    )
    p.add_argument(
        "job_id", nargs="?", help="Job (session) id to open directly. If omitted, the TUI shows a job picker."
    )
    p.add_argument("--sub-job-id", help="Sub-job id used to scope log sources and routing.")
    p.add_argument(
        "--config",
        dest="config",
        default=os.environ.get("CORTEX_TRAINING_CONFIG"),
        help=(
            "Path to a reusable Cortex Training config or credential JSON file "
            "(same format as cortex-training; e.g. resolve.json)."
        ),
    )
    p.add_argument("--base-url", help="Direct base URL (local/mock use).")
    p.add_argument("--host", help="Snowflake account host for PAT auth.")
    p.add_argument("--pat", help="Programmatic access token.")
    p.add_argument("--database", help="Database containing the endpoint.")
    # Left as None so config-file / env values can fill them; the CLI resolver
    # applies the PUBLIC / cortex-training defaults when nothing is set.
    p.add_argument("--schema", default=None)
    p.add_argument("--endpoint", default=None)
    p.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Minimum seconds between log polls per source (request-rate floor).",
    )
    p.add_argument("--poll-timeout", type=float, default=1800.0)
    p.add_argument("--no-verify-ssl", action="store_true")
    return p


def run(argv=None, *, prog: str = "cortex-training tui") -> int:
    parser = _build_arg_parser(prog=prog)
    args = parser.parse_args(argv)

    try:
        from cortex_training.tui.app import CortexTrainingLogTUI
    except ImportError as exc:
        print(
            f"The TUI requires the optional 'textual' dependency: pip install cortex-training ({exc})",
            file=sys.stderr,
        )
        return 2

    import cortex_training._cli as cli

    # Reuse the CLI's resolution: --config / login state / env / defaults.
    args = cli._resolve_args(args)
    try:
        args = cli._normalize_connection_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.database:
        parser.error("provide --database or set CORTEX_TRAINING_DATABASE/SNOWFLAKE_DATABASE")
    if args.base_url is None and (args.host is None or args.pat is None):
        parser.error(
            "no connection configured: run 'cortex-training login --config config.json', "
            "set CORTEX_TRAINING_CONFIG, pass --config resolve.json, or pass "
            "--base-url (local/mock) or --host + --pat"
        )

    client = cli.build_client(args, cli._load_cortex_training_client_class())

    # job_id is optional: given, we jump straight to that job's logs; omitted,
    # the TUI opens the job picker (list_jobs) so you can choose one.
    CortexTrainingLogTUI(client, args.job_id, sub_job_id=args.sub_job_id, poll_interval=args.poll_interval).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
