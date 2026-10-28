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

"""Collect completed router-replay evidence from the owning allocation."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import socket
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

from arctic_platform.debug.training import TRACE_SCHEMA
from arctic_platform.debug.training import RouterReplayTrace  # noqa: F401 - compatibility export
from arctic_platform.debug.training import active  # noqa: F401
from arctic_platform.debug.training import finish  # noqa: F401
from arctic_platform.debug.training import record_router  # noqa: F401
from arctic_platform.debug.training import start  # noqa: F401

_RECEIPT = "ROUTER_REPLAY_TRACE_CHUNK "
_CHUNK_BYTES = 4096


def _node_trace_receipt(output_dir: str, request_id: str, token: str) -> dict:
    from .storage import require_data_fast_path

    root = require_data_fast_path(output_dir, label="router-replay evidence") / request_id
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", request_id) is None:
        raise ValueError("invalid router-replay request identity")
    if root.resolve() != root:
        raise ValueError("router-replay request directory cannot be a symlink")
    records = []
    for path in sorted(root.glob("*.json")):
        if re.fullmatch(r"rank-\d+-call-\d+\.json", path.name) is None or path.is_symlink():
            raise ValueError(f"unexpected router-replay artifact: {path.name}")
        raw = path.read_bytes()
        record = json.loads(raw)
        if record.get("schema") != TRACE_SCHEMA or record.get("request_id") != request_id:
            raise ValueError("router-replay artifact schema or request differs")
        records.append({"filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "record": record})
    return {"token": token, "host": socket.gethostname(), "request_id": request_id, "records": records}


def _receipt_chunks(receipt: dict) -> Iterator[dict]:
    # pdsh truncates oversized lines; keep each independently framed transfer well below its line buffer.
    raw = json.dumps(receipt, separators=(",", ":")).encode()
    metadata = {
        "token": receipt["token"],
        "host": receipt["host"],
        "request_id": receipt["request_id"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "count": (len(raw) + _CHUNK_BYTES - 1) // _CHUNK_BYTES,
    }
    for index, offset in enumerate(range(0, len(raw), _CHUNK_BYTES)):
        yield {**metadata, "index": index, "data": base64.b64encode(raw[offset : offset + _CHUNK_BYTES]).decode()}


def _decode_trace_chunks(output: str, *, hosts: list[str], request_id: str, token: str) -> list[dict]:
    transfers = {}
    for line in output.splitlines():
        prefix, separator, raw = line.partition(_RECEIPT)
        if not separator:
            continue
        host = prefix.strip().removesuffix(":").strip()
        chunk = json.loads(raw)
        size, count, index = chunk.get("size"), chunk.get("count"), chunk.get("index")
        if (
            host not in hosts
            or chunk.get("token") != token
            or chunk.get("request_id") != request_id
            or not isinstance(chunk.get("host"), str)
            or not chunk["host"]
            or type(size) is not int
            or size < 1
            or type(count) is not int
            or count != (size + _CHUNK_BYTES - 1) // _CHUNK_BYTES
            or type(index) is not int
            or not 0 <= index < count
            or not isinstance(chunk.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", chunk["sha256"]) is None
            or not isinstance(chunk.get("data"), str)
            or len(chunk["data"]) > 4 * ((_CHUNK_BYTES + 2) // 3)
        ):
            raise ValueError("router-replay collector received an invalid allocation chunk")
        metadata = (chunk["host"], size, count, chunk["sha256"])
        previous, parts = transfers.setdefault(host, (metadata, {}))
        if metadata != previous or index in parts:
            raise ValueError("router-replay collector received mixed or duplicate allocation chunks")
        try:
            part = base64.b64decode(chunk["data"], validate=True)
        except ValueError as exc:
            raise ValueError("router-replay collector received a corrupted chunk") from exc
        if len(part) != min(_CHUNK_BYTES, size - index * _CHUNK_BYTES):
            raise ValueError("router-replay collector chunk size differs")
        parts[index] = part
    missing = [host for host in hosts if host not in transfers or len(transfers[host][1]) != transfers[host][0][2]]
    if missing:
        raise RuntimeError(f"router-replay collection failed: missing or incomplete={missing}")
    receipts = []
    for host in hosts:
        (node, size, count, digest), parts = transfers[host]
        raw = b"".join(parts[index] for index in range(count))
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("router-replay collector receipt checksum differs")
        receipt = json.loads(raw)
        if receipt.get("host") != node or receipt.get("token") != token or receipt.get("request_id") != request_id:
            raise ValueError("router-replay collector receipt identity differs")
        receipts.append(receipt)
    if len({receipt["host"] for receipt in receipts}) != len(hosts):
        raise ValueError("router-replay receipts describe duplicate executing nodes")
    return receipts


def _merge_trace_receipts(receipts: list[dict], *, request_id: str, ranks: int, calls: int) -> list[dict]:
    by_rank = {rank: {} for rank in range(ranks)}
    rank_hosts = {}
    for receipt in receipts:
        if receipt.get("request_id") != request_id:
            raise ValueError("router-replay receipt request differs")
        for artifact in receipt["records"]:
            record = artifact["record"]
            rank, call = record.get("rank"), record.get("call_index")
            if (
                record.get("schema") != TRACE_SCHEMA
                or record.get("request_id") != request_id
                or type(rank) is not int
                or rank not in by_rank
                or type(call) is not int
                or call not in range(calls)
                or artifact["filename"] != f"rank-{rank}-call-{call}.json"
            ):
                raise ValueError("router-replay artifact identity or call count differs")
            if call in by_rank[rank]:
                raise ValueError("duplicate router-replay rank/call artifact")
            if rank in rank_hosts and rank_hosts[rank] != receipt["host"]:
                raise ValueError("router-replay rank artifacts span multiple nodes")
            rank_hosts[rank] = receipt["host"]
            if len(record.get("calls", [])) != 1:
                raise ValueError("router-replay artifact must contain exactly one completed call")
            by_rank[rank][call] = record
    traces = []
    for rank, records in by_rank.items():
        if set(records) != set(range(calls)):
            raise ValueError(f"router-replay rank {rank} lacks the exact {calls} completed model calls")
        received = records[0]["received"]
        if any(record["received"] != received for record in records.values()):
            raise ValueError("router-replay RX observations changed between completed calls")
        traces.append(
            {
                "rank": rank,
                "received": received,
                "calls": [records[index]["calls"][0] for index in range(calls)],
                "unattributed_router_calls": [
                    entry for record in records.values() for entry in record.get("unattributed_router_calls", [])
                ],
            }
        )
    return traces


def collect_traces(option: dict, response: dict, *, ranks: int) -> list[dict]:
    """Read completed, node-local artifacts before shutting down the owning RL client."""
    from .dss_driver import _allocation_hostfile
    from .storage import _ds_ssh
    from .storage import parse_hostfile
    from .storage import require_data_fast_path

    calls = (response.get("metrics") or {}).get("rl_model_calls")
    if type(calls) is not int or calls < 1 or type(ranks) is not int or ranks < 1:
        raise ValueError("router-replay requires positive actual rl_model_calls and configured training ranks")
    root = require_data_fast_path(option["output_dir"], label="router-replay evidence")
    request_id, token = option["request_id"], uuid.uuid4().hex
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", request_id) is None:
        raise ValueError("invalid router-replay request identity")
    hostfile = _allocation_hostfile()
    if hostfile is None:
        receipts = [_node_trace_receipt(str(root), request_id, token)]
    else:
        original_hostfile = hostfile.read_bytes()
        hosts = [entry.host for entry in parse_hostfile(hostfile)]
        if not hosts or len(hosts) != len(set(hosts)):
            raise ValueError("router-replay requires unique allocation hosts")
        source = Path(__file__).resolve()
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        script = (
            "import hashlib, json, sys\nfrom pathlib import Path\nsys.path.insert(0,"
            f" {str(source.parents[3])!r})\nsource = Path({str(source)!r})\nassert"
            f" hashlib.sha256(source.read_bytes()).hexdigest() == {source_sha!r}\nfrom"
            " arctic_platform.correctness.harness.router_replay_trace import _node_trace_receipt,"
            f" _receipt_chunks\nreceipt = _node_trace_receipt({str(root)!r}, {request_id!r}, {token!r})\nfor chunk in"
            f" _receipt_chunks(receipt):\n    print({_RECEIPT!r} + json.dumps(chunk, separators=(',', ':')),"
            " flush=True)\n"
        )
        encoded = base64.b64encode(script.encode()).decode()
        done = subprocess.run(
            [_ds_ssh(), "-f", str(hostfile), f"echo {encoded} | base64 -d | {shlex.quote(sys.executable)} -"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if done.returncode or hostfile.read_bytes() != original_hostfile:
            raise RuntimeError(f"router-replay collection failed: rc={done.returncode}, allocation hostfile changed")
        receipts = _decode_trace_chunks(done.stdout, hosts=hosts, request_id=request_id, token=token)
    traces = _merge_trace_receipts(receipts, request_id=request_id, ranks=ranks, calls=calls)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{request_id}-receipts.json").open("x") as stream:
        json.dump(receipts, stream, separators=(",", ":"))
        stream.write("\n")
    return traces
