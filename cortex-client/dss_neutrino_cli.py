"""Command line interface for Neutrino job management."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

_CONFIG_KEYS = {
    "base_url",
    "host",
    "pat",
    "database",
    "schema",
    "endpoint",
    "poll_interval",
    "poll_timeout",
    "no_verify_ssl",
    "verify_ssl",
}

_CONFIG_ALIASES = {
    "url": "base_url",
    "db": "database",
}

_LOGIN_STATE_ENV = "DSS_NEUTRINO_LOGIN_FILE"
_NEUTRINO_CLIENT_MODULE_NAME = "neutrino_client_for_cli"


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _normalize_host(host: str) -> str:
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host.rstrip("/")


def _has_url_scheme(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _load_neutrino_client_module():
    """Load neutrino_client.py directly to avoid package-level torch imports."""
    if _NEUTRINO_CLIENT_MODULE_NAME in sys.modules:
        return sys.modules[_NEUTRINO_CLIENT_MODULE_NAME]

    module_path = (
        Path(__file__).resolve().parent / "dss_client" / "neutrino_client.py"
    )
    spec = importlib.util.spec_from_file_location(
        _NEUTRINO_CLIENT_MODULE_NAME, module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Neutrino client from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_neutrino_client_class():
    module = _load_neutrino_client_module()
    return module.NeutrinoClient


def _load_forward_backward_payload_builder():
    module = _load_neutrino_client_module()
    return module.build_forward_backward_payload


def _created_epoch(raw: Any) -> float | None:
    if raw is None:
        return None
    created_at = str(raw).strip()
    if not created_at:
        return None
    if created_at.endswith("Z"):
        created_at = created_at[:-1] + "+00:00"
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.timestamp()


def _jobs_latest_last(jobs: list[Any]) -> list[Any]:
    """Render jobs oldest-first so the newest entries land at the terminal."""
    dated_jobs = [
        (
            _created_epoch(job.get("created_at")) if isinstance(job, dict) else None,
            index,
            job,
        )
        for index, job in enumerate(jobs)
    ]
    if any(created is not None for created, _, _ in dated_jobs):
        return [
            job
            for _, _, job in sorted(
                dated_jobs,
                key=lambda item: (
                    item[0] is None,
                    item[0] if item[0] is not None else float("inf"),
                    item[1],
                ),
            )
        ]
    return list(reversed(jobs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dss-neutrino",
        description="Manage Neutrino jobs through the SNOWAPI endpoint.",
    )
    parser.add_argument(
        "--config",
        default=_env("NEUTRINO_CONFIG"),
        help="Path to a reusable Neutrino CLI config JSON file.",
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for a local/mock SnowAPI server. Skips PAT auth.",
    )
    parser.add_argument(
        "--host",
        help="Snowflake account host for PAT auth.",
    )
    parser.add_argument(
        "--pat",
        help="Programmatic access token.",
    )
    parser.add_argument(
        "--database",
        help="Database containing the cortex-training endpoint.",
    )
    parser.add_argument(
        "--schema",
        help="Schema containing the cortex-training endpoint. Defaults to PUBLIC.",
    )
    parser.add_argument(
        "--endpoint",
        help="REST endpoint name. Defaults to cortex-training.",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification for PAT-authenticated requests.",
    )
    parser.add_argument("--poll-interval", type=float)
    parser.add_argument("--poll-timeout", type=float)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON instead of pretty JSON.",
    )
    parser.add_argument(
        "--job",
        "--job-id",
        dest="job",
        help="Job id for data-plane commands such as fwd-bwd.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser(
        "submit",
        help="Submit a Neutrino job from a CreateJob JSON file.",
    )
    submit.add_argument(
        "json_file",
        help="Path to CreateJob JSON, or '-' for stdin.",
    )
    submit.add_argument("--job-id", help="Set or override the top-level job_id.")
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the CreateJob body without sending it.",
    )
    submit.add_argument(
        "--wait",
        action="store_true",
        help="Wait until the submitted job reaches running.",
    )

    get = subparsers.add_parser("get", help="Fetch one Neutrino job.")
    get.add_argument("job_id")

    checkpoints = subparsers.add_parser(
        "checkpoints",
        help="List checkpoints for one Neutrino job.",
    )
    checkpoints.add_argument("job_id")

    list_jobs = subparsers.add_parser("list", help="List Neutrino jobs.")
    list_jobs.add_argument("--status", help="Optional status filter.")

    subparsers.add_parser(
        "capacity",
        help="Show reserved GPU capacity and current usage for the caller account.",
    )

    cancel = subparsers.add_parser("cancel", help="Cancel one Neutrino job.")
    cancel.add_argument("job_id")

    wait = subparsers.add_parser("wait", help="Wait until one job reaches running.")
    wait.add_argument("job_id")

    fwd_bwd = subparsers.add_parser(
        "fwd-bwd",
        help="Run one forward-backward request from a readable payload JSON file.",
    )
    fwd_bwd.add_argument(
        "json_file",
        help="Path to fwd-bwd JSON, or '-' for stdin.",
    )

    step = subparsers.add_parser(
        "step",
        help="Run one optimizer step for a training job.",
    )
    step.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer step. Defaults to 1e-4.",
    )

    load = subparsers.add_parser(
        "load",
        help="Load a checkpoint into an already-created Neutrino job.",
    )
    load.add_argument("checkpoint_id", help="Checkpoint id/tag to load.")
    load.add_argument(
        "--source-job-id",
        help="Load the checkpoint from another job's checkpoint store.",
    )
    load.add_argument(
        "--no-poll",
        action="store_true",
        help="Print the request id without polling for completion.",
    )

    generate = subparsers.add_parser(
        "generate",
        help="Run one generate request from a readable payload JSON file.",
    )
    generate.add_argument(
        "json_file",
        help="Path to generate JSON, or '-' for stdin.",
    )

    weight_sync = subparsers.add_parser(
        "weight-sync",
        help="Sync weights from a training sub-job to one or more sampling sub-jobs.",
    )
    weight_sync.add_argument(
        "--source-sub-job-id",
        help="Training sub-job id. Defaults to JOB_ID:training:0.",
    )
    weight_sync.add_argument(
        "--target-sub-job-id",
        action="append",
        dest="target_sub_job_ids",
        help=(
            "Sampling sub-job id. Can be repeated. "
            "Defaults to JOB_ID:sampling:0."
        ),
    )
    weight_sync.add_argument(
        "--operation-sub-job-id",
        help=(
            "Sub-job id used to route the operation envelope. "
            "Defaults to the source training sub-job id."
        ),
    )
    weight_sync.add_argument(
        "--operation-sub-job-type",
        help="Sub-job type used to route the operation envelope.",
    )
    weight_sync.add_argument(
        "--no-poll",
        action="store_true",
        help="Print the request id without polling for completion.",
    )

    download_log = subparsers.add_parser(
        "download-log",
        help="Download all log files for a Neutrino job's experiment run.",
    )
    download_log.add_argument("job_id")
    download_log.add_argument(
        "--output-dir",
        dest="output_dir",
        help=(
            "Directory to write log files into, grouped as "
            "<output_dir>/<sub_job_id>/<filename>. Created if missing. "
            "Defaults to the current working directory."
        ),
    )

    login = subparsers.add_parser(
        "login",
        help="Remember a Neutrino config file for future commands.",
    )
    login.add_argument(
        "--config",
        required=True,
        dest="login_config",
        help="Path to the Neutrino CLI config JSON file to remember.",
    )

    return parser


def _login_state_path() -> Path:
    override = _env(_LOGIN_STATE_ENV)
    if override:
        return Path(override).expanduser()
    config_home = _env("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "dss-neutrino" / "login.json"


def _read_login_config_path() -> str | None:
    path = _login_state_path()
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dss-neutrino login state {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"invalid dss-neutrino login state {path}: expected object")
    config_path = parsed.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        raise ValueError(
            f"invalid dss-neutrino login state {path}: missing config_path"
        )
    return config_path


def _write_login_config_path(config_path: str) -> str:
    path = Path(config_path).expanduser()
    _load_config(str(path))
    saved_path = str(path.resolve())
    state_path = _login_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config_path": saved_path}
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return saved_path


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    path_obj = Path(path).expanduser()
    try:
        parsed = json.loads(path_obj.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in config {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("config JSON must be an object")

    if "connection" in parsed:
        connection = parsed["connection"]
        if not isinstance(connection, dict):
            raise ValueError("config connection must be an object")
        parsed = connection

    config: dict[str, Any] = {}
    unknown = []
    for key, value in parsed.items():
        normalized = _CONFIG_ALIASES.get(key, key)
        if normalized not in _CONFIG_KEYS:
            unknown.append(key)
            continue
        config[normalized] = value
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown config key(s): {names}")
    return config


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _config_str(config: dict[str, Any], key: str) -> str | None:
    if key not in config or config[key] is None:
        return None
    if not isinstance(config[key], str):
        raise ValueError(f"config {key} must be a string")
    return config[key]


def _config_float(config: dict[str, Any], key: str) -> float | None:
    if key not in config or config[key] is None:
        return None
    if isinstance(config[key], bool) or not isinstance(config[key], (int, float)):
        raise ValueError(f"config {key} must be a number")
    return float(config[key])


def _config_bool(config: dict[str, Any], key: str) -> bool | None:
    if key not in config or config[key] is None:
        return None
    if not isinstance(config[key], bool):
        raise ValueError(f"config {key} must be a boolean")
    return config[key]


def _has_connection(
    base_url: str | None,
    host: str | None,
    pat: str | None,
    database: str | None,
) -> bool:
    return bool(database and (base_url or (host and pat)))


def _select_config(args: argparse.Namespace, *, load_login: bool) -> dict[str, Any]:
    if args.config:
        return _load_config(args.config)

    if not load_login:
        return {}

    base_url = _coalesce(args.base_url, _env("NEUTRINO_BASE_URL"))
    host = _coalesce(args.host, _env("NEUTRINO_HOST", "SNOWFLAKE_HOST"))
    pat = _coalesce(args.pat, _env("NEUTRINO_PAT", "SNOWFLAKE_PAT"))
    database = _coalesce(
        args.database,
        _env("NEUTRINO_DATABASE", "SNOWFLAKE_DATABASE"),
    )
    if _has_connection(base_url, host, pat, database):
        return {}

    login_config = _read_login_config_path()
    if login_config is None:
        return {}
    args.config = login_config
    return _load_config(login_config)


def _resolve_args(
    args: argparse.Namespace,
    *,
    load_login: bool = True,
) -> argparse.Namespace:
    config = _select_config(args, load_login=load_login)
    args.base_url = _coalesce(
        args.base_url,
        _config_str(config, "base_url"),
        _env("NEUTRINO_BASE_URL"),
    )
    args.host = _coalesce(
        args.host,
        _config_str(config, "host"),
        _env("NEUTRINO_HOST", "SNOWFLAKE_HOST"),
    )
    args.pat = _coalesce(
        args.pat,
        _config_str(config, "pat"),
        _env("NEUTRINO_PAT", "SNOWFLAKE_PAT"),
    )
    args.database = _coalesce(
        args.database,
        _config_str(config, "database"),
        _env("NEUTRINO_DATABASE", "SNOWFLAKE_DATABASE"),
    )
    args.schema = _coalesce(
        args.schema,
        _config_str(config, "schema"),
        _env("NEUTRINO_SCHEMA", "SNOWFLAKE_SCHEMA"),
        "PUBLIC",
    )
    args.endpoint = _coalesce(
        args.endpoint,
        _config_str(config, "endpoint"),
        _env("NEUTRINO_ENDPOINT"),
        "cortex-training",
    )
    args.poll_interval = _coalesce(
        args.poll_interval,
        _config_float(config, "poll_interval"),
        0.5,
    )
    args.poll_timeout = _coalesce(
        args.poll_timeout,
        _config_float(config, "poll_timeout"),
        1800.0,
    )
    if not args.no_verify_ssl:
        no_verify_ssl = _config_bool(config, "no_verify_ssl")
        verify_ssl = _config_bool(config, "verify_ssl")
        if no_verify_ssl is not None:
            args.no_verify_ssl = no_verify_ssl
        elif verify_ssl is not None:
            args.no_verify_ssl = not verify_ssl
    return args


def _normalize_connection_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.base_url is None or _has_url_scheme(args.base_url):
        return args
    if args.pat and args.host is None:
        args.host = args.base_url
        args.base_url = None
        return args
    raise ValueError(
        "base_url must start with http:// or https://. For Snowflake PAT auth, "
        "use host instead of base_url."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "login":
        return args

    dry_run = args.command == "submit" and args.dry_run
    args = _resolve_args(args, load_login=not dry_run)
    if dry_run:
        return args
    args = _normalize_connection_args(args)
    if not args.database:
        parser.error("provide --database or set NEUTRINO_DATABASE/SNOWFLAKE_DATABASE")
    if args.base_url is None and (args.host is None or args.pat is None):
        parser.error("provide --base-url for local/mock use, or both --host and --pat")
    return args


def build_client(args: argparse.Namespace, neutrino_client_cls):
    kwargs = {
        "database": args.database,
        "schema": args.schema,
        "endpoint": args.endpoint,
        "poll_interval": args.poll_interval,
        "poll_timeout": args.poll_timeout,
    }
    if args.base_url:
        return neutrino_client_cls(base_url=args.base_url, **kwargs)
    return neutrino_client_cls.from_pat(
        host=_normalize_host(args.host),
        pat=args.pat,
        verify_ssl=not args.no_verify_ssl,
        **kwargs,
    )


def _read_json_object(path: str, stdin: TextIO) -> dict[str, Any]:
    if path == "-":
        raw = stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("job JSON must be an object")
    return parsed


def _validate_create_job_body(body: dict[str, Any]) -> None:
    sub_job_configs = body.get("sub_job_configs")
    if not isinstance(sub_job_configs, list) or not sub_job_configs:
        raise ValueError("job JSON must contain a non-empty sub_job_configs list")


def _print_json(value: Any, stdout: TextIO, *, compact: bool) -> None:
    if compact:
        json.dump(value, stdout, separators=(",", ":"), sort_keys=True)
    else:
        json.dump(value, stdout, indent=2, sort_keys=True)
    stdout.write("\n")


def _format_error(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    parts = [str(exc)]
    request_id = getattr(response, "headers", {}).get("x-snowflake-request-id")
    if request_id:
        parts.append(f"snowflake request id: {request_id}")

    body = (getattr(response, "text", "") or "").strip()
    if body:
        if len(body) > 4000:
            body = body[:4000] + "...<truncated>"
        parts.append(f"response body: {body}")
    return "\n".join(parts)


def _cmd_submit(
    args: argparse.Namespace,
    client,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    body = _read_json_object(args.json_file, stdin)
    _validate_create_job_body(body)
    if args.job_id is not None:
        body = dict(body)
        body["job_id"] = args.job_id

    if args.dry_run:
        _print_json(body, stdout, compact=args.compact)
        return 0

    response = client.create_job_from_body(body)
    if args.wait:
        job_id = response.get("job_id") or body.get("job_id")
        if not job_id:
            raise ValueError(
                "submit --wait requires the create response to include job_id"
            )
        response = client.wait_for_job(str(job_id))
    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_login(args: argparse.Namespace, stdout: TextIO) -> int:
    config_path = _write_login_config_path(args.login_config)
    _print_json(
        {"config_path": config_path, "logged_in": True},
        stdout,
        compact=args.compact,
    )
    return 0


def _json_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _cmd_fwd_bwd(
    args: argparse.Namespace,
    client,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    if not args.job:
        raise ValueError("provide --job JOB_ID for fwd-bwd")

    spec = _read_json_object(args.json_file, stdin)
    poll = _json_bool(spec.get("poll", True), "fwd-bwd poll")
    build_payload = _load_forward_backward_payload_builder()
    payload = build_payload(spec)

    request_id = client.forward_backward(args.job, payload)
    response = {
        "job_id": args.job,
        "payload_size_bytes": len(payload),
        "request_id": request_id,
    }
    if poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_step(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for step")

    request_id = client.step(args.job, learning_rate=args.lr)
    response = {
        "job_id": args.job,
        "learning_rate": args.lr,
        "request_id": request_id,
        "result": client.poll_request(args.job, request_id),
    }
    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_load(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for load")

    request_id = client.load(
        args.job,
        checkpoint_id=args.checkpoint_id,
        source_job_id=args.source_job_id,
    )
    response = {
        "checkpoint_id": args.checkpoint_id,
        "job_id": args.job,
        "request_id": request_id,
    }
    if args.source_job_id is not None:
        response["source_job_id"] = args.source_job_id
    if not args.no_poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_generate(
    args: argparse.Namespace,
    client,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for generate")

    spec = _read_json_object(args.json_file, stdin)
    payload = spec.get("payload", spec)
    if not isinstance(payload, dict):
        raise ValueError("generate payload must be an object")

    poll = _json_bool(spec.get("poll", True), "generate poll")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("generate JSON must contain a non-empty prompts list")

    sampling_params = payload.get("sampling_params")
    if sampling_params is not None:
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts):
                raise ValueError("generate sampling_params list length must match prompts length")
            if any(
                item is not None and not isinstance(item, dict)
                for item in sampling_params
            ):
                raise ValueError("generate sampling_params list items must be objects or null")
        elif not isinstance(sampling_params, dict):
            raise ValueError("generate sampling_params must be an object or list")

    strict = payload.get("strict")
    if strict is not None:
        strict = _json_bool(strict, "generate strict")

    request_id = client.generate(
        args.job,
        prompts=prompts,
        sampling_params=sampling_params,
        routing_key=payload.get("routing_key"),
        strict=strict,
    )
    response = {
        "job_id": args.job,
        "prompt_count": len(prompts),
        "request_id": request_id,
    }
    if poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_weight_sync(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for weight-sync")

    source_sub_job_id = args.source_sub_job_id or f"{args.job}:training:0"
    target_sub_job_ids = args.target_sub_job_ids or [f"{args.job}:sampling:0"]
    request_id = client.weight_sync(
        args.job,
        source_sub_job_id=source_sub_job_id,
        target_sub_job_ids=target_sub_job_ids,
        sub_job_id=args.operation_sub_job_id,
        sub_job_type=args.operation_sub_job_type,
    )
    response = {
        "job_id": args.job,
        "request_id": request_id,
        "source_sub_job_id": source_sub_job_id,
        "target_sub_job_ids": target_sub_job_ids,
    }
    if not args.no_poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_download_log(args: argparse.Namespace, client, stdout: TextIO) -> int:
    logs = client.fetch_execution_logs(args.job_id)
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else Path.cwd()
    saved = []
    for log in logs:
        file_path = out_dir / (log["sub_job_id"] or "unknown") / log["filename"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(log["content"], encoding="utf-8")
        saved.append(
            {
                "sub_job_id": log["sub_job_id"],
                "filename": log["filename"],
                "s3_uri": log["s3_uri"],
                "saved_path": str(file_path),
            }
        )
    _print_json({"job_id": args.job_id, "logs": saved}, stdout, compact=args.compact)
    return 0


def _run(
    args: argparse.Namespace,
    client_factory: Callable[[argparse.Namespace], Any],
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    if args.command == "login":
        return _cmd_login(args, stdout)

    if args.command == "submit" and args.dry_run:
        return _cmd_submit(args, None, stdout, stdin)

    client = client_factory(args)
    if args.command == "submit":
        return _cmd_submit(args, client, stdout, stdin)
    if args.command == "get":
        _print_json(client.get_job(args.job_id), stdout, compact=args.compact)
        return 0
    if args.command == "checkpoints":
        _print_json(
            {"checkpoints": client.list_checkpoints(args.job_id)},
            stdout,
            compact=args.compact,
        )
        return 0
    if args.command == "list":
        jobs = client.list_jobs(status=args.status)
        _print_json({"jobs": _jobs_latest_last(jobs)}, stdout, compact=args.compact)
        return 0
    if args.command == "capacity":
        _print_json(client.get_capacity(), stdout, compact=args.compact)
        return 0
    if args.command == "cancel":
        client.cancel_job(args.job_id)
        _print_json(
            {"cancelled": True, "job_id": args.job_id},
            stdout,
            compact=args.compact,
        )
        return 0
    if args.command == "wait":
        _print_json(client.wait_for_job(args.job_id), stdout, compact=args.compact)
        return 0
    if args.command == "fwd-bwd":
        return _cmd_fwd_bwd(args, client, stdout, stdin)
    if args.command == "step":
        return _cmd_step(args, client, stdout)
    if args.command == "load":
        return _cmd_load(args, client, stdout)
    if args.command == "generate":
        return _cmd_generate(args, client, stdout, stdin)
    if args.command == "weight-sync":
        return _cmd_weight_sync(args, client, stdout)
    if args.command == "download-log":
        return _cmd_download_log(args, client, stdout)
    raise ValueError(f"unknown command: {args.command}")


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[argparse.Namespace], Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    stdin = stdin or sys.stdin

    try:
        args = parse_args(argv)
        dry_run = args.command == "submit" and args.dry_run
        no_client = dry_run or args.command == "login"
        if client_factory is None and not no_client:
            neutrino_client_cls = _load_neutrino_client_class()
            client_factory = lambda parsed_args: build_client(
                parsed_args,
                neutrino_client_cls,
            )
        elif client_factory is None:
            client_factory = lambda _parsed_args: None
        return _run(args, client_factory, stdout, stdin)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {_format_error(exc)}", file=stderr)
        return 1
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(
            f"error: missing Python dependency '{missing}'. "
            "Install the missing package, or install this repo first, "
            "for example: python3 -m pip install -e .",
            file=stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
