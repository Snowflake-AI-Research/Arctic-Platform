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

"""Fill every allocation node's converted-weight cache before a local Cortex training job is created.

DSS gives a training job ``training_create_timeout_seconds`` (600 s by default) to become ready. A cold HF->PrimeRL
conversion of a large MoE checkpoint spends most of that before the optimizer exists, so the job fails at the deadline
on a node that has never converted the checkpoint. One process per node converts first; the job's own load then finds
a ready node-local cache and only reads it.

The preparation builds the ``ModelSpec`` the DSS worker builds -- the same model path, the same loader options from
DSS's own ``prime_rl_config``, the same expert-parallel width -- and calls the loader's ``prepare_weights`` hook, which
runs the conversion half of the loader's ``load_dcp_from_hf``. The cache path therefore comes from the same function
in both places; the per-node receipt records it so the worker's ``Reusing existing HF→PrimeRL conversion cache`` line
can be compared against it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RECEIPT_TAG = "AP_WEIGHT_CACHE_RECEIPT "
# The node process resolves the checkpoint and the cache root from these, exactly as a training worker would.
FORWARDED_ENV = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "DSS_ALLOW_REMOTE_MODELS",
    "DSS_WEIGHT_CONVERSION_CACHE_DIR",
    "DSS_WEIGHT_CONVERSION_CACHE_SCOPE",
)
NODE_TIMEOUT_SECONDS = 3600


def training_weight_request(sub_job: dict) -> dict | None:
    """Return the per-node preparation request for a Cortex training sub-job, or None when it converts nothing.

    Only the MoE route converts (DSS ``model_options.moe_route``: expert parallelism above 1).
    """
    from dss.model_paths import resolve_model_path
    from dss.ray_dss.jobs.debug.config import without_training_debug_options
    from dss.ray_dss.jobs.gpu.sp.config import ep_size_of
    from dss.ray_dss.jobs.gpu.sp.config import resolve_attn_implementation
    from dss.ray_dss.jobs.gpu.sp.config import sp_size_of
    from dss.ray_dss.jobs.gpu.training import prime_rl_config
    from dss.ray_dss.model_options import moe_route
    from dss.ray_dss.models import TrainingConfig

    training = TrainingConfig.model_validate(sub_job["training_config"]).model_dump()
    ep_size = ep_size_of(training)
    if training.get("model_provider") != "arctic" or not moe_route(ep_size):
        return None
    attn = resolve_attn_implementation(training)
    return {
        "model_name": sub_job.get("local_model_dir") or resolve_model_path(sub_job["model_name"]),
        "ep_size": ep_size,
        "sp_size": sp_size_of(training),
        "loader_options": without_training_debug_options(dict(prime_rl_config(sub_job, training, attn))),
        "tiled_mlp_token_chunk_size": training.get("tiled_mlp_token_chunk_size"),
        "peft_config": training.get("peft_config"),
        "world_size": int(sub_job["training_config"]["n_gpus"]),
    }


def prepare_node(request: dict, *, rank: int) -> dict:
    """Convert on this node, as the process with global ``rank`` and local rank 0 would during the load."""
    from arctic_platform.model.loader import prepare_model_weights
    from arctic_platform.model.moe_spec import moe_model_spec

    spec = moe_model_spec(
        model_name=request["model_name"],
        ep_size=request["ep_size"],
        sp_size=request["sp_size"],
        loader_options=request["loader_options"],
        tiled_mlp_token_chunk_size=request["tiled_mlp_token_chunk_size"],
        peft_config=request["peft_config"],
    )
    started = time.monotonic()
    path = prepare_model_weights(spec, world_size=request["world_size"], rank=rank, local_rank=0)
    return {
        "loader": spec.loader,
        "cache_path": None if path is None else str(path),
        "seconds": time.monotonic() - started,
    }


def _node_script(request: dict, ranks: dict[str, int]) -> str:
    source = Path(__file__).resolve()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    return (
        "import hashlib, json, socket, subprocess, sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(source.parents[3])!r})\n"
        f"source = Path({str(source)!r})\n"
        f"assert hashlib.sha256(source.read_bytes()).hexdigest() == {source_sha!r}, 'weight_cache.py differs'\n"
        "from arctic_platform.correctness.harness.weight_cache import prepare_node\n"
        f"ranks = {json.dumps(ranks)}\n"
        "addresses = subprocess.run(['hostname', '-I'], capture_output=True, text=True).stdout.split()\n"
        "names = {socket.gethostname(), socket.getfqdn(), *addresses}\n"
        "rank = [value for key, value in ranks.items() if key in names]\n"
        "assert len(rank) == 1, f'node matches {len(rank)} hostfile entries: {sorted(names)}'\n"
        f"receipt = prepare_node(json.loads({json.dumps(request)!r}), rank=rank[0])\n"
        "receipt = {'host': socket.gethostname(), 'rank': rank[0], **receipt}\n"
        f"print({RECEIPT_TAG!r} + json.dumps(receipt), flush=True)\n"
    )


def prepare_allocation_weights(sub_job: dict) -> list[dict]:
    """Fill the converted-weight cache on every node of the local gateway's allocation; return one receipt per node.

    Node ``i`` of the hostfile prepares as global rank ``i * slots``, so only the first node writes a cache that is
    shared rather than node-local, matching the job's own rank-0 writer.
    """
    from arctic_platform.correctness.harness.gateway import _allocation_hostfile
    from arctic_platform.correctness.harness.storage import _ds_ssh
    from arctic_platform.correctness.harness.storage import parse_hostfile

    request = training_weight_request(sub_job)
    if request is None:
        return []
    hostfile = _allocation_hostfile()
    if hostfile is None:
        receipt = {"host": os.uname().nodename, "rank": 0, **prepare_node(request, rank=0)}
        print(RECEIPT_TAG + json.dumps(receipt), flush=True)
        return [receipt]
    entries = parse_hostfile(hostfile)
    ranks, first = {}, 0
    for entry in entries:
        ranks[entry.host] = first
        first += entry.slots
    env = " ".join(f"{key}={shlex.quote(os.environ[key])}" for key in FORWARDED_ENV if key in os.environ)
    encoded = base64.b64encode(_node_script(request, ranks).encode()).decode()
    command = f"echo {encoded} | base64 -d | env {env} {shlex.quote(sys.executable)} -"
    print(f"preparing converted weights for {request['model_name']} on {len(entries)} node(s)", flush=True)
    done = subprocess.run(
        [_ds_ssh(), "-f", str(hostfile), command], capture_output=True, text=True, timeout=NODE_TIMEOUT_SECONDS
    )
    receipts = [json.loads(line.split(RECEIPT_TAG, 1)[1]) for line in done.stdout.splitlines() if RECEIPT_TAG in line]
    if done.returncode or len(receipts) != len(entries):
        raise RuntimeError(
            f"converted-weight preparation failed: rc={done.returncode}, {len(receipts)} of {len(entries)} nodes "
            f"reported\n{done.stdout[-4000:]}\n{done.stderr[-4000:]}"
        )
    for receipt in receipts:
        print(RECEIPT_TAG + json.dumps(receipt), flush=True)
    return receipts


def maybe_prepare_allocation_weights(client_config: Any) -> list[dict]:
    """Prepare weights for a client config that creates a training job; return the node receipts."""
    if not getattr(client_config, "training_gpus", 0):
        return []
    return prepare_allocation_weights(client_config._cortex_training_sub_job())
