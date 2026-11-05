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
"""Service-side probe of the model audit: one Cortex Training job, driven only through ``cortex_training``.

Not collected by pytest. ``--body`` is the create-job body; rows are the audit rows of ``--ckpt``'s config. Every
row is a batch row of its own (the service plans its own packing), padded on the right. Writes one ``torch.save``
dict to ``--out``; exported checkpoints are downloaded into the directory of the same name.

Phases, each run when the body has the sub-job it needs:

- trainer forward: all rows in one request and again; with ``--forwards all`` also in reverse order, each alone,
  and requests whose last row has 1..8 tokens. A data-parallel job rejects requests with fewer rows than ranks, so
  the short-row requests carry at least one row per rank, the rows are sent alone only without data parallelism,
  and the layout jobs send only the full batch;
- step: a weights-only export, one forward/backward and step at ``--lr`` (the step reports per-parameter gradient
  norms), then weights-only and resumable exports;
- sampler: prompt log-probs of each row alone, again, and of all rows in one request; with a training sub-job, then
  for each ``--sync-format`` a weight sync and the rows again. Every format after the first syncs a state one more
  step on (exported first), so a sync that changes nothing cannot pass by reusing the previous one's weights.

Connection: ``AP_AUDIT_CLIENT_CONNECTION`` names a Snowflake connection profile; otherwise ``CORTEX_TRAINING_HOST``,
``CORTEX_TRAINING_PAT``, ``CORTEX_TRAINING_DATABASE`` and ``CORTEX_TRAINING_SCHEMA`` (default ``PUBLIC``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_lib  # noqa: E402

_WEIGHT_SUFFIXES = (".safetensors", ".json")


def connect():
    from cortex_training.client import CortexTrainingClient

    connection = os.environ.get("AP_AUDIT_CLIENT_CONNECTION")
    if connection:
        client = CortexTrainingClient.from_configs(connection)
    else:
        # An explicit host, so a default connection profile on the machine cannot redirect the audit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            client = CortexTrainingClient.from_pat(
                host=os.environ["CORTEX_TRAINING_HOST"],
                pat=os.environ["CORTEX_TRAINING_PAT"],
                database=os.environ["CORTEX_TRAINING_DATABASE"],
                schema=os.environ.get("CORTEX_TRAINING_SCHEMA", "PUBLIC"),
            )
    client.poll_timeout = 1800.0
    return client


def _batch(rows: list[list[int]]) -> dict[str, torch.Tensor]:
    width = max(len(r) for r in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    pos = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        n = len(row)
        ids[i, :n] = torch.tensor(row)
        mask[i, :n] = 1
        pos[i, :n] = torch.arange(n)
        labels[i, : n - 1] = torch.tensor(row[1:], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask, "position_ids": pos, "labels": labels}


def _per_row(rows: list[list[int]], logprobs) -> list[torch.Tensor]:
    """Row ``i``'s log-probs ``[len(row)]``: index ``t`` scores token ``t + 1``, the last entry is 0."""
    out = []
    for i, row in enumerate(rows):
        lp = torch.as_tensor(logprobs[i], dtype=torch.float32).flatten()[: len(row)].clone()
        lp[len(row) - 1 :] = 0.0
        out.append(lp)
    return out


def _prompt_logprob(entry, token: int) -> float:
    info = entry.get(token, entry.get(str(token)))
    return float(info["logprob"] if isinstance(info, dict) else info)


def _without_urls(value):
    if isinstance(value, dict):
        return {k: _without_urls(v) for k, v in value.items() if k != "url"}
    if isinstance(value, list):
        return [_without_urls(v) for v in value]
    return value


class Job:
    def __init__(self, client, body: dict, log):
        self.client = client
        self.log = log
        self.id = client.create_job_from_body(body)["job_id"]
        log(f"created job {self.id}")
        self.info = client.wait_for_job(self.id)
        log(f"job {self.id} is {self.info.get('status')}")
        self.subs = {sj["job_type"].lower(): sj for sj in self.info.get("sub_jobs", [])}

    def _wait(self, request_id):
        return self.client.poll_request(self.id, request_id)

    def forward(self, rows: list[list[int]]) -> list[torch.Tensor]:
        frame = {"args": (), "kwargs": _batch(rows), "processing": {"post": ["compute_logprobs"], "loss_fn": None}}
        from cortex_training import wire

        reply = self.client.forward(self.id, wire.dumps(frame), sub_job_type="training")
        if isinstance(reply, str):
            reply = self._wait(reply)
        elif "request_id" in reply:
            reply = self._wait(reply["request_id"])
        return _per_row(rows, reply["logprobs"])

    def forward_backward(self, rows: list[list[int]]) -> dict:
        from cortex_training import wire

        batch = _batch(rows)
        valid = (batch["labels"] != -100).float()
        frame = {
            "args": (),
            "kwargs": batch,
            "context": {"loss_mask": valid},
            "processing": {
                "post": ["compute_logprobs"],
                "loss_fn": "causal_cross_entropy",
                "config": {"batch_num_tokens": float(valid.sum())},
            },
        }
        return self._wait(self.client.forward_backward(self.id, wire.dumps(frame)))

    def step(self, lr: float) -> dict:
        return self._wait(self.client.step(self.id, lr))

    def export(self, checkpoint_type: str, dest: Path, suffixes: tuple[str, ...] | None) -> dict:
        """Save, export and download the new checkpoint; ``suffixes`` limits which files are fetched."""
        known = {c["checkpoint_id"] for c in self.client.list_checkpoints(self.id)}
        saved = self._wait(self.client.save(self.id, checkpoint_type=checkpoint_type))
        new = [c for c in self.client.list_checkpoints(self.id) if c["checkpoint_id"] not in known]
        assert len(new) == 1, f"expected one new checkpoint after save, found {new}"
        exported = self.client.export_checkpoint(self.id, new[0]["checkpoint_id"])
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        wanted = [i for i in exported["files"] if suffixes is None or i["filename"].endswith(suffixes)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            fetched = list(pool.map(lambda i: self._fetch(i["url"], dest / Path(i["filename"]).name), wanted))
        self.log(f"{checkpoint_type} checkpoint {new[0]['checkpoint_id']}: fetched {len(fetched)} file(s) to {dest}")
        return {"saved": _without_urls(saved), "files": _without_urls(exported["files"]), "path": str(dest)}

    @staticmethod
    def _fetch(url: str, target: Path) -> str:
        urllib.request.urlretrieve(url, target)
        return target.name

    def generate(self, rows: list[list[int]]) -> list[torch.Tensor]:
        params = {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0, "logprobs": 0}
        reply = self._wait(self.client.generate(self.id, rows, params))
        scored = []
        for row, result in zip(rows, reply["results"]):
            lp = torch.full((len(row),), float("nan"))
            for t in range(1, len(row)):
                lp[t - 1] = _prompt_logprob(result["prompt_logprobs"][t], row[t])
            scored.append(lp)
        return scored

    def weight_sync(self, weight_format: str) -> dict:
        source, target = self.subs["training"]["sub_job_id"], self.subs["sampling"]["sub_job_id"]
        return self._wait(self.client.weight_sync(self.id, source, [target], weight_format=weight_format))

    def cancel(self) -> None:
        self.client.cancel_job(self.id)


def _status(fn) -> str:
    try:
        lps = fn()
        return "ok" if all(bool(torch.isfinite(lp[:-1]).all()) for lp in lps) else "non-finite log-probs"
    except Exception as err:  # noqa: BLE001
        return f"{type(err).__name__}: {str(err).strip().splitlines()[-1][:300] if str(err).strip() else ''}"


def run(profile, ckpt: str, body: dict, lr: float, sync_formats: list[str], forwards: str, out_dir: Path, log) -> dict:
    _, rows, pad = audit_lib.probe_inputs(profile, ckpt)
    training = next((s for s in body["sub_job_configs"] if s["job_type"] == "training"), None)
    dp = audit_lib.data_parallel_size(training) if training else 1
    if len(rows) < dp:
        raise ValueError(f"{len(rows)} audit rows cannot fill a request to {dp} data-parallel ranks")
    job = Job(connect(), body, log)
    result: dict = {"job": {"id": job.id, **{k: v for k, v in job.subs.items()}}, "rows": rows}
    try:
        if "training" in job.subs:
            fwd = {
                "labels": audit_lib.pack_rows(rows, pad)[2],
                "packed_logprobs": audit_lib.pack_logprobs(rows, job.forward(rows), pad),
                "repeat_logprobs": audit_lib.pack_logprobs(rows, job.forward(rows), pad),
            }
            if forwards == "all":
                reversed_rows = list(reversed(rows))
                fwd["reversed_logprobs"] = audit_lib.pack_logprobs(reversed_rows, job.forward(reversed_rows), pad)
                if dp == 1:
                    fwd["isolated_logprobs"] = [job.forward([row])[0] for row in rows]
                company = rows[1 : max(dp, 2)]
                fwd["short_rows"] = {
                    n: _status(lambda n=n: job.forward(company + [rows[0][:n]])) for n in audit_lib.SHORT_ROW_LENGTHS
                }
            result["forward"] = fwd
            log("trainer forwards done")
            exports = {"initial": job.export("weights-only", out_dir / "initial", _WEIGHT_SUFFIXES)["path"]}
            loss = job.forward_backward(rows)
            reply = job.step(lr)
            result["step"] = {"lr": lr, "loss": loss.get("avg_loss"), "fwd_bwd": _without_urls(loss), "reply": reply}
            exports["after_step"] = job.export("weights-only", out_dir / "after_step", _WEIGHT_SUFFIXES)["path"]
            exports["resumable"] = job.export("resumable", out_dir / "resumable", None)["path"]
            result["exports"] = exports
            log(f"step done: grad_norm={reply.get('grad_norm')}")
        if "sampling" in job.subs:
            sampler = {"alone": [job.generate([row])[0] for row in rows]}
            sampler["alone_repeat"] = [job.generate([row])[0] for row in rows]
            sampler["batched"] = job.generate(rows)
            result["sampler"] = sampler
            log("sampler scores done")
            for i, fmt in enumerate(sync_formats if "training" in job.subs else []):
                state = "after_step"
                if i:
                    job.forward_backward(rows)
                    job.step(lr)
                    state = f"sync_{fmt}"
                    result["exports"][state] = job.export("weights-only", out_dir / state, _WEIGHT_SUFFIXES)["path"]
                sampler[f"sync_{fmt}_state"] = state
                try:
                    sampler[f"sync_{fmt}_rpc"] = _without_urls(job.weight_sync(fmt))
                except Exception as err:  # noqa: BLE001
                    sampler[f"sync_{fmt}_error"] = f"{type(err).__name__}: {str(err).strip()[:2000]}"
                    continue
                sampler[f"after_sync_{fmt}"] = [job.generate([row])[0] for row in rows]
                log(f"sync {fmt} done")
    finally:
        try:
            job.cancel()
            log(f"cancelled job {job.id}")
        except Exception as err:  # noqa: BLE001
            log(f"cancel of job {job.id} failed: {err}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--ckpt", required=True, help="local checkpoint whose config.json defines the rows")
    parser.add_argument("--body", required=True, help="JSON create-job body")
    parser.add_argument("--lr", type=float, default=None, help="step learning rate; defaults to the audit lr")
    parser.add_argument("--sync-format", action="append", default=[], help="weight-sync format, repeatable")
    parser.add_argument("--forwards", choices=("all", "batch"), default="all")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    profile = audit_lib.load_profile(args.profile)
    out = Path(args.out)
    lr = profile.audit_lr if args.lr is None else args.lr
    try:
        body = json.loads(args.body)
        result = run(profile, args.ckpt, body, lr, args.sync_format, args.forwards, out.with_suffix(""), log)
        result["status"] = {"ok": True}
    except Exception as err:
        result = {
            "status": {"ok": False, "error": f"{type(err).__name__}: {err}", "traceback": traceback.format_exc()}
        }
        print(result["status"]["traceback"], file=sys.stderr, flush=True)
    torch.save(result, out)


if __name__ == "__main__":
    main()
