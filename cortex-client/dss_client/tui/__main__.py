"""`neutrino-tui` entry point: resolve connection args, build the SDK client,
and launch the read-only TUI.

Connection handling is delegated to ``dss_neutrino_cli`` so the TUI shares the
SDK's auth surface: it honours ``dss-neutrino login`` state, ``--config`` /
``NEUTRINO_CONFIG``, the ``NEUTRINO_*`` / ``SNOWFLAKE_*`` env vars, and explicit
``--base-url`` (local/mock) or ``--host`` + ``--pat`` flags, exactly like the
CLI. Run ``dss-neutrino login --config config.json`` once and then just
``neutrino-tui JOB_ID``.
"""

from __future__ import annotations

import argparse
import os
import sys


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neutrino-tui",
        description="Read-only TUI for Neutrino logs.",
    )
    p.add_argument("job_id", nargs="?", help="Job (session) id to open directly. If omitted, the TUI shows a job picker.")
    p.add_argument("--sub-job-id", help="Sub-job id used to scope log sources and routing.")
    p.add_argument(
        "--config",
        dest="config",
        default=os.environ.get("NEUTRINO_CONFIG"),
        help=(
            "Path to a reusable Neutrino CLI config or credential JSON file "
            "(same format as dss-neutrino; e.g. resolve.json)."
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
    p.add_argument("--poll-interval", type=float, default=1.0,
                   help="Minimum seconds between log polls per source (request-rate floor).")
    p.add_argument("--poll-timeout", type=float, default=1800.0)
    p.add_argument("--no-verify-ssl", action="store_true")
    return p


def run(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        from dss_client.tui.app import NeutrinoLogTUI
    except ImportError as exc:
        print(
            "The TUI requires the optional 'textual' dependency: "
            f"pip install 'dss-client[tui]' ({exc})",
            file=sys.stderr,
        )
        return 2

    import dss_neutrino_cli as cli

    # Reuse the CLI's resolution: --config / login state / env / defaults.
    args = cli._resolve_args(args)
    try:
        args = cli._normalize_connection_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.database:
        parser.error("provide --database or set NEUTRINO_DATABASE/SNOWFLAKE_DATABASE")
    if args.base_url is None and (args.host is None or args.pat is None):
        parser.error(
            "no connection configured: run 'dss-neutrino login --config config.json', "
            "set NEUTRINO_CONFIG, pass --config resolve.json, or pass "
            "--base-url (local/mock) or --host + --pat"
        )

    client = cli.build_client(args, cli._load_neutrino_client_class())

    # job_id is optional: given, we jump straight to that job's logs; omitted,
    # the TUI opens the job picker (list_jobs) so you can choose one.
    NeutrinoLogTUI(
        client, args.job_id, sub_job_id=args.sub_job_id, poll_interval=args.poll_interval
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
