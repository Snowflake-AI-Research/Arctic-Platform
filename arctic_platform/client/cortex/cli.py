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
"""``cortex`` -- manage Cortex training jobs from the shell.

Control-plane commands go through `CortexJobs`. Data-plane commands (fwd-bwd,
step, generate, sync-weights) reconnect a real `ArcticClient` to the job via
`CortexJobs.attach`, so the CLI and library run the same code path rather than
maintaining a second implementation of the wire.

The connection comes from `cortex.connection.resolve`: explicit flags win, then
``--config``, then whatever ``cortex login`` remembered, then ``CORTEX_*`` env.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from arctic_platform._dependency_groups import require_any_dep_group

require_any_dep_group("cli")

import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from arctic_platform.client.config import CortexConfig  # noqa: E402
from arctic_platform.client.cortex import connection  # noqa: E402
from arctic_platform.client.cortex.jobs import CortexJobs  # noqa: E402
from arctic_platform.client.cortex.logs import CortexLogs  # noqa: E402

app = typer.Typer(
    name="cortex",
    help="Manage Cortex training jobs.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()
err_console = Console(stderr=True)

# Connection flags live on the top-level callback so they apply to every command.
_STATE: dict[str, Any] = {}

JOB_ENV = "CORTEX_JOB"


@app.callback()
def main_options(
    config: str = typer.Option(None, "--config", help="Connection JSON file.", envvar=connection.CONFIG_ENV),
    host: str = typer.Option(None, help="Snowflake account host for PAT auth."),
    pat: str = typer.Option(None, help="Programmatic access token."),
    base_url: str = typer.Option(None, help="Direct base URL for a local/mock server; skips PAT auth."),
    database: str = typer.Option(None, help="Database holding the endpoint."),
    schema: str = typer.Option(None, help="Schema holding the endpoint."),
    endpoint: str = typer.Option(None, help="REST endpoint name."),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON instead of tables."),
) -> None:
    _STATE.update(
        as_json=as_json,
        overrides=dict(host=host, pat=pat, base_url=base_url, database=database, schema=schema, endpoint=endpoint),
        config_path=config,
    )


def _config() -> CortexConfig:
    try:
        return connection.resolve(config_path=_STATE.get("config_path"), **_STATE.get("overrides", {}))
    except ValueError as exc:
        raise typer.BadParameter(
            f"{exc}\nRun 'cortex login --config config.json', set CORTEX_CONFIG, or pass --host/--pat/--database."
        ) from exc


def _jobs() -> CortexJobs:
    return CortexJobs(_config())


def _as_json() -> bool:
    return bool(_STATE.get("as_json"))


def _emit(value: Any) -> None:
    """Print a JSON payload. Every command can fall back to this with --json."""
    console.print_json(json.dumps(value, sort_keys=True, default=str))


def _read_json_file(path: str) -> dict:
    raw = sys.stdin.read() if path == "-" else Path(path).expanduser().read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return parsed


def _job_arg(job_id: str | None) -> str:
    import os

    resolved = job_id or os.environ.get(JOB_ENV)
    if not resolved:
        raise typer.BadParameter(f"provide a JOB_ID or set {JOB_ENV}")
    return resolved


def _client(job_id: str):
    """An `ArcticClient` reattached to an existing job.

    Never `shutdown()` this: the CLI did not create the job, and shutdown cancels it.
    """
    from arctic_platform.client import ArcticClient

    return ArcticClient(_jobs().attach(job_id))


# ── connection ───────────────────────────────────────────────────────────────
@app.command()
def login(config: str = typer.Option(..., "--config", help="Connection JSON file to remember.")) -> None:
    """Remember a connection file so later commands need no flags."""
    path = connection.login(config)
    if _as_json():
        _emit({"config_path": str(path), "logged_in": True})
        return
    console.print(f"[green]Logged in[/] using [bold]{path}[/]")


@app.command()
def logout() -> None:
    """Forget the remembered connection file."""
    if _as_json():
        _emit({"logged_out": connection.logout()})
        return
    console.print("[green]Logged out[/]" if connection.logout() else "[yellow]Not logged in[/]")


@app.command()
def config() -> None:
    """Show the resolved connection, with the PAT masked."""
    _emit(connection.redacted(_config()))


# ── jobs ─────────────────────────────────────────────────────────────────────
@app.command("list")
def list_jobs(status: str = typer.Option(None, "--status", help="Filter by job status.")) -> None:
    """List jobs, oldest first so the newest land next to the prompt."""
    jobs = _oldest_first(_jobs().list(status=status))
    if _as_json():
        _emit({"jobs": jobs})
        return
    if not jobs:
        console.print("[yellow]No jobs found[/]")
        return

    table = Table(title=f"Cortex jobs{f' ({status})' if status else ''}")
    for column in ("job id", "status", "created", "sub-jobs", "gpus"):
        table.add_column(column, overflow="fold")
    for job in jobs:
        sub_jobs = job.get("sub_jobs") or []
        table.add_row(
            str(job.get("job_id", "")),
            _status_markup(job.get("status")),
            str(job.get("created_at", "")),
            ", ".join(_short_type(s.get("job_type")) for s in sub_jobs) or "-",
            str(sum(_sub_job_gpus(s) for s in sub_jobs) or "-"),
        )
    console.print(table)


@app.command()
def get(job_id: str = typer.Argument(None)) -> None:
    """Show one job."""
    _emit(_jobs().get(_job_arg(job_id)))


@app.command()
def submit(
    json_file: str = typer.Argument(..., help="CreateJob JSON, or '-' for stdin."),
    job_id: str = typer.Option(None, "--job-id", help="Set or override the top-level job_id."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the body without sending it."),
    wait_running: bool = typer.Option(False, "--wait", help="Wait until the job is running."),
) -> None:
    """Submit a job from a SnowAPI CreateJob body."""
    body = _read_json_file(json_file)
    if job_id is not None:
        body = {**body, "job_id": job_id}
    if dry_run:
        _emit(body)
        return

    jobs = _jobs()
    response = jobs.submit(body)
    created = response.get("job_id") or body.get("job_id")
    if wait_running:
        if not created:
            raise ValueError("submit --wait needs a job_id in the create response")
        response = jobs.wait(str(created))
    _emit(response)


@app.command()
def cancel(job_id: str = typer.Argument(None)) -> None:
    """Cancel a job."""
    resolved = _job_arg(job_id)
    _jobs().cancel(resolved)
    if _as_json():
        _emit({"cancelled": True, "job_id": resolved})
        return
    console.print(f"[green]Cancelled[/] {resolved}")


@app.command()
def wait(job_id: str = typer.Argument(None)) -> None:
    """Wait until a job is running."""
    _emit(_jobs().wait(_job_arg(job_id)))


@app.command()
def capacity() -> None:
    """Show the account's reserved GPU capacity and current usage."""
    reserved = _jobs().capacity()
    if _as_json():
        _emit(reserved.model_dump())
        return

    table = Table(title="GPU capacity")
    table.add_column("reservation")
    table.add_column("reserved", justify="right")
    table.add_column("in use", justify="right")
    table.add_column("available", justify="right")
    color = "green" if reserved.available_gpus > 0 else "red"
    table.add_row(
        "yes" if reserved.has_reservation else "no",
        str(reserved.reserved_gpus),
        str(reserved.in_use_gpus),
        f"[{color}]{reserved.available_gpus}[/]",
    )
    console.print(table)


@app.command()
def checkpoints(job_id: str = typer.Argument(None)) -> None:
    """List a job's checkpoints."""
    found = _jobs().checkpoints(_job_arg(job_id))
    if _as_json():
        _emit({"checkpoints": found})
        return
    if not found:
        console.print("[yellow]No checkpoints found[/]")
        return

    table = Table(title="Checkpoints")
    for column in ("checkpoint id", "type", "step", "created"):
        table.add_column(column, overflow="fold")
    for checkpoint in found:
        table.add_row(
            str(checkpoint.get("checkpoint_id", "")),
            str(checkpoint.get("checkpoint_type", "")),
            str(checkpoint.get("step", "")),
            str(checkpoint.get("created_at", "")),
        )
    console.print(table)


# ── data plane (reconnects to a running job) ─────────────────────────────────
@app.command("fwd-bwd")
def fwd_bwd(
    job_id: str = typer.Argument(None),
    json_file: str = typer.Argument(..., help="fwd-bwd spec JSON, or '-' for stdin."),
) -> None:
    """Run one forward-backward pass from a readable spec."""
    from arctic_platform.client.cortex import batch as batch_spec

    spec = _read_json_file(json_file)
    payload = batch_spec.build_fwd_bwd_batch(spec)
    result = _client(_job_arg(job_id)).fwd_bwd(payload)
    _emit({"job_id": _job_arg(job_id), "result": result})


@app.command()
def step(
    job_id: str = typer.Argument(None),
    lr: float = typer.Option(None, "--lr", help="Learning rate override; server default when unset."),
) -> None:
    """Run one optimizer step."""
    resolved = _job_arg(job_id)
    _emit({"job_id": resolved, "learning_rate": lr, "result": _client(resolved).step(learning_rate=lr)})


@app.command()
def generate(
    job_id: str = typer.Argument(None),
    json_file: str = typer.Argument(..., help="generate spec JSON, or '-' for stdin."),
) -> None:
    """Generate completions from a readable spec."""
    from arctic_platform.client.cortex import batch as batch_spec

    spec = _read_json_file(json_file)
    resolved = _job_arg(job_id)
    results = _client(resolved).generate(**batch_spec.read_generate_spec(spec))
    _emit({"job_id": resolved, "results": results})


@app.command()
def load(
    job_id: str = typer.Argument(None),
    checkpoint_id: str = typer.Argument(..., help="Checkpoint id to load."),
    source_job_id: str = typer.Option(None, "--source-job-id", help="Load from another job's checkpoint store."),
    target_sub_job_id: str = typer.Option(
        None, "--target-sub-job-id", help="Training sub-job to load into, e.g. JOB:training:0."
    ),
    no_poll: bool = typer.Option(False, "--no-poll", help="Print the request id without waiting."),
) -> None:
    """Load a checkpoint into a running job."""
    resolved = _job_arg(job_id)
    jobs = _jobs()
    request_id = jobs.load(resolved, checkpoint_id, source_job_id=source_job_id, target_sub_job_id=target_sub_job_id)
    response: dict[str, Any] = {"job_id": resolved, "checkpoint_id": checkpoint_id, "request_id": request_id}
    if not no_poll:
        response["result"] = jobs.poll_request(resolved, request_id)
    _emit(response)


@app.command("sync-weights")
def sync_weights(
    job_id: str = typer.Argument(None),
    source_sub_job_id: str = typer.Option(None, "--source-sub-job-id", help="Defaults to the training sub-job."),
    target_sub_job_id: list[str] = typer.Option(
        None, "--target-sub-job-id", help="Repeatable. Defaults to the sampling sub-job."
    ),
    weight_format: str = typer.Option(None, "--weight-format", help="vllm, hf, or lora (adapter-only)."),
) -> None:
    """Sync training weights into the sampling engine(s)."""
    resolved = _job_arg(job_id)
    result = _client(resolved).sync_weights(
        weight_format=weight_format,
        source_sub_job_id=source_sub_job_id,
        target_sub_job_ids=list(target_sub_job_id) if target_sub_job_id else None,
    )
    _emit({"job_id": resolved, "result": result})


# ── logs ─────────────────────────────────────────────────────────────────────
@app.command()
def logs(
    job_id: str = typer.Argument(None),
    sub_job_id: str = typer.Option(None, "--sub-job", help="Sub-job whose logs to read."),
    follow: bool = typer.Option(True, "--follow/--no-follow", help="Keep tailing after draining the log."),
    plain: bool = typer.Option(False, "--plain", help="Stream to stdout instead of opening the log browser."),
    poll_interval: float = typer.Option(1.0, "--poll-interval", help="Minimum seconds between polls per source."),
) -> None:
    """Browse or tail a job's logs live."""
    resolved = _job_arg(job_id) if (job_id or not sys.stdin.isatty()) else None
    jobs = _jobs()
    reader = CortexLogs(jobs, poll_interval=poll_interval)

    # The browser needs a terminal and textual; piping or --plain gets a plain stream.
    if plain or _as_json() or not sys.stdout.isatty():
        if resolved is None:
            raise typer.BadParameter("streaming logs needs a JOB_ID")
        _stream_plain(reader, resolved, sub_job_id, follow)
        return

    from arctic_platform.client.cortex.tui import run_log_browser

    run_log_browser(jobs, reader, resolved, sub_job_id=sub_job_id)


def _stream_plain(reader: CortexLogs, job_id: str, sub_job_id: str | None, follow: bool) -> None:
    from arctic_platform.client.cortex.tui.format import format_log_entry

    try:
        for entry in reader.stream_logs(job_id, sub_job_id=sub_job_id, follow=follow):
            console.print(format_log_entry(entry), highlight=False, soft_wrap=True)
    except KeyboardInterrupt:  # a tail ends when the reader walks away
        pass


@app.command("download-logs")
def download_logs(
    job_id: str = typer.Argument(None),
    output_dir: str = typer.Option(None, "--output-dir", help="Defaults to the current directory."),
) -> None:
    """Download the job's archived log files, grouped by sub-job."""
    resolved = _job_arg(job_id)
    jobs = _jobs()
    saved = CortexLogs(jobs).download_execution_logs(resolved, output_dir)
    if _as_json():
        _emit({"job_id": resolved, "logs": saved})
        return
    if not saved:
        console.print("[yellow]No log files found[/]")
        return
    for log in saved:
        console.print(f"[green]saved[/] {log['saved_path']}")


# ── rendering helpers ────────────────────────────────────────────────────────
def _short_type(job_type: Any) -> str:
    return str(job_type or "").lower().removeprefix("job_type_") or "?"


def _sub_job_gpus(sub_job: dict) -> int:
    config = sub_job.get("training_config") or sub_job.get("inference_config") or {}
    return int(config.get("n_gpus") or 0)


def _status_markup(status: Any) -> str:
    short = str(status or "").lower().removeprefix("job_state_")
    color = {
        "running": "green",
        "done": "blue",
        "failed": "red",
        "cancelled": "yellow",
        "canceled": "yellow",
    }.get(short, "cyan")
    return f"[{color}]{short or '?'}[/]"


def _oldest_first(jobs: list[dict]) -> list[dict]:
    """Sort by creation so the newest row lands nearest the prompt."""
    from datetime import datetime
    from datetime import timezone

    def created(job: Any) -> float | None:
        raw = str((job or {}).get("created_at") or "").strip() if isinstance(job, dict) else ""
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()

    dated = [(created(job), index, job) for index, job in enumerate(jobs)]
    if not any(stamp is not None for stamp, _, _ in dated):
        return list(reversed(jobs))  # no timestamps: the server's newest-first, flipped
    return [job for _, _, job in sorted(dated, key=lambda row: (row[0] is None, row[0] or 0.0, row[1]))]


def main() -> None:
    """Console-script entry point: render expected failures, don't traceback at them."""
    try:
        app()
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        err_console.print(f"[red]error:[/] {_explain(exc)}")
        raise typer.Exit(1) from exc
    except ModuleNotFoundError as exc:
        err_console.print(
            f"[red]error:[/] missing dependency '{exc.name or 'unknown'}'. "
            "Install the CLI extra: pip install 'arctic_platform[cli]'"
        )
        raise typer.Exit(1) from exc


def _explain(exc: BaseException) -> str:
    """The message, plus the Snowflake request id when the server gave us one."""
    response = getattr(exc, "response", None)
    request_id = getattr(response, "headers", {}).get("x-snowflake-request-id") if response is not None else None
    return f"{exc}\nsnowflake request id: {request_id}" if request_id else str(exc)


if __name__ == "__main__":
    main()
