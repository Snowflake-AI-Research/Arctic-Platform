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
"""Two ways to read a job's output, because they come from different places.

*Live* logs are a control-plane read: ``/operation`` with ``tail-logs``, answered
inline (no request to poll) by the sub-job's zone-manager pod, whose stdout already
carries worker output via Ray's ``log_to_driver``. Cursor-paged, so a tail is just
"fetch, yield, advance, back off".

*Archived* logs live in the experiment run's Snowflake stage, reachable only with
temporary S3 credentials: experiment-run -> ``SYSTEM$GET_VSTAGE_WRITE_CREDS`` ->
list -> get. That path needs boto3, imported lazily so it stays optional.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from arctic_platform.client.cortex.jobs import CortexJobs

_MAX_INTERVAL = 6.0
_BACKOFF = 1.25


class CortexLogs:
    """Log access for one job's connection. Read-only."""

    def __init__(self, jobs: CortexJobs, *, poll_interval: float = 1.0) -> None:
        self.jobs = jobs
        self.session = jobs.session
        self.poll_interval = poll_interval

    # ── live tail ────────────────────────────────────────────────────────────
    def tail_logs(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_lines: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """One page of a sub-job's logs from ``cursor``.

        Returns ``{"entries": [...], "next_cursor": str, "eof": bool}``; an empty
        cursor reads from the start.
        """
        payload: dict[str, Any] = {}
        if cursor is not None:
            payload["cursor"] = cursor
        if max_lines is not None:
            payload["max_lines"] = max_lines
        return self._operation(job_id, "tail-logs", payload, sub_job_id, sub_job_type)

    def tail_events(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_events: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """One page of the session's scheduling/zone events, served by the ZMD itself."""
        payload: dict[str, Any] = {}
        if cursor is not None:
            payload["cursor"] = cursor
        if max_events is not None:
            payload["max_events"] = max_events
        return self._operation(job_id, "tail-events", payload, sub_job_id, sub_job_type)

    def stream_logs(
        self,
        job_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        cursor: str | None = None,
        max_lines: int | None = None,
        follow: bool = True,
    ) -> Iterator[dict]:
        """Yield log entries, draining the log then (with ``follow``) tailing it live."""
        return self._pages(
            lambda cur: self.tail_logs(
                job_id, cursor=cur, max_lines=max_lines, sub_job_id=sub_job_id, sub_job_type=sub_job_type
            ),
            "entries",
            follow=follow,
            cursor=cursor,
        )

    def stream_events(
        self,
        job_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        cursor: str | None = None,
        follow: bool = True,
    ) -> Iterator[dict]:
        return self._pages(
            lambda cur: self.tail_events(job_id, cursor=cur, sub_job_id=sub_job_id, sub_job_type=sub_job_type),
            "events",
            follow=follow,
            cursor=cursor,
        )

    def _operation(
        self,
        job_id: str,
        operation_type: str,
        payload: dict,
        sub_job_id: str | None,
        sub_job_type: str | None,
    ) -> dict:
        body: dict[str, Any] = {"operation_type": operation_type, "payload": payload}
        if sub_job_id is not None:
            body["sub_job_id"] = sub_job_id
        if sub_job_type is not None:
            body["sub_job_type"] = sub_job_type
        return self.session.send("POST", f"{self.session.prefix}/{job_id}/operation", json=body)

    def _pages(self, fetch, entries_key: str, *, follow: bool, cursor: str | None) -> Iterator[dict]:
        """Drive a cursor-paged tail: fetch a page, yield it, advance, back off when idle."""
        delay = self.poll_interval
        while True:
            page = fetch(cursor)
            cursor = page.get("next_cursor", cursor)
            entries = page.get(entries_key) or []
            yield from entries
            if entries:
                delay = self.poll_interval  # made progress; poll again promptly
                continue
            if page.get("eof", False) and not follow:
                return
            time.sleep(delay)
            delay = min(delay * _BACKOFF, _MAX_INTERVAL)

    # ── archived logs ────────────────────────────────────────────────────────
    def fetch_execution_logs(self, job_id: str) -> list[dict[str, str]]:
        """Every log file under the job's experiment-run stage.

        One entry per object beneath a ``_logs/<sub_job_id>/`` subtree, as
        ``{sub_job_id, filename, s3_uri, content}``.
        """
        run = self.jobs.experiment_run(job_id)
        try:
            run_uri = f"snow://experiment/{run['experiment_name']}/versions/{run['experiment_run_name']}/"
        except KeyError as exc:
            raise ValueError(f"experiment-run response missing field: {exc.args[0]}") from exc

        creds = _parse_s3_stage_credentials(self._sql_scalar(f"SELECT SYSTEM$GET_VSTAGE_WRITE_CREDS('{run_uri}')"))
        bucket = creds["bucket"]
        prefix = f"{creds['prefix']}/" if creds["prefix"] else ""

        s3 = _s3_client(creds)
        results: list[dict[str, str]] = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents") or []:
                key = item.get("Key")
                if not isinstance(key, str):
                    continue
                _, sep, after_logs = key.partition("/_logs/")
                if not sep:
                    continue
                sub_job_id, _, filename = after_logs.partition("/")
                if not sub_job_id or not filename:
                    continue
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                results.append(
                    {
                        "sub_job_id": sub_job_id,
                        "filename": filename,
                        "s3_uri": f"s3://{bucket}/{key}",
                        "content": body.decode("utf-8"),
                    }
                )
        return results

    def download_execution_logs(self, job_id: str, out_dir: str | Path | None = None) -> list[dict[str, str]]:
        """Write the archived logs to ``<out_dir>/<sub_job_id>/<filename>``.

        Grouping by sub-job keeps same-named siblings (``execution.jsonl``,
        ``server.log``) from colliding. Returns the saved entries without contents.
        """
        root = Path(out_dir).expanduser() if out_dir else Path.cwd()
        saved = []
        for log in self.fetch_execution_logs(job_id):
            path = root / (log["sub_job_id"] or "unknown") / log["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(log["content"], encoding="utf-8")
            saved.append({k: log[k] for k in ("sub_job_id", "filename", "s3_uri")} | {"saved_path": str(path)})
        return saved

    def _sql_scalar(self, statement: str) -> Any:
        """Run a synchronous SQL statement and return ``data[0][0]`` (a SYSTEM$ scalar)."""
        cx = self.jobs.config
        rows = self.session.send(
            "POST",
            f"{self.session.base_url}/api/v2/statements",
            json={"statement": statement, "database": cx.database, "schema": cx.schema_},
        ).get("data")
        if not rows or not isinstance(rows, list) or not isinstance(rows[0], list) or not rows[0]:
            raise ValueError(f"SQL query returned no rows: {statement}")
        return rows[0][0]


def _s3_client(creds: dict[str, str]):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("downloading archived logs needs boto3: pip install 'arctic_platform[cli]'") from exc

    return boto3.client(
        "s3",
        aws_access_key_id=creds["access_key_id"],
        aws_secret_access_key=creds["secret_access_key"],
        aws_session_token=creds["session_token"],
        region_name=creds["region"] or None,
    )


def _parse_s3_stage_credentials(raw_value: Any) -> dict[str, str]:
    stage = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    if not isinstance(stage, dict):
        raise ValueError("stage credentials response is not a JSON object")
    if (stage.get("locationType") or "").upper() != "S3":
        raise NotImplementedError(f"log download only supports S3 stages; got {stage.get('locationType')!r}")

    location = (stage.get("location") or "").removeprefix("s3://").strip("/")
    bucket, _, prefix = location.partition("/")
    if not bucket:
        raise ValueError(f"stage credentials missing bucket: {stage.get('location')!r}")
    creds = stage.get("creds")
    if not isinstance(creds, dict):
        raise ValueError("stage credentials missing AWS creds")
    try:
        return {
            "bucket": bucket,
            "prefix": prefix,
            "region": stage.get("region") or "",
            "access_key_id": creds["AWS_KEY_ID"],
            "secret_access_key": creds["AWS_SECRET_KEY"],
            "session_token": creds.get("AWS_TOKEN"),
        }
    except KeyError as exc:
        raise ValueError(f"stage credentials missing AWS field: {exc.args[0]}") from exc
