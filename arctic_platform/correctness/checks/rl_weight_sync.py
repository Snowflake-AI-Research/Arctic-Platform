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

"""Weight sync from the training zone into the sampling zone, compared element by element.

One RL step at the config's learning rate moves the trainer's weights off the checkpoint both zones loaded,
so a sync that drops or corrupts a tensor leaves the sampler holding something the trainer does not. The sync
then runs with the gateway's defaults, and three records are compared:

- the trainer's manifest: shape, dtype, element count and sha256 of every tensor it sends, and the number of
  elements it holds across its expert-parallel shards (``debug.weight_sync_manifest``);
- each sampler rank's record of every tensor it received, hashed the same way, with the parameters the loader
  wrote it into (``ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY=1`` in the sampling workers' environment);
- each sampler rank's shadow replay: every received tensor is loaded a second time into NaN-filled copies of
  its destination parameters, so a NaN left behind is an element no tensor wrote, and a written element that
  differs from the live parameter is one the live load did not keep.

The check fails when the trainer sends fewer elements than it holds, when a rank's received set differs from
the sent set by name, shape, dtype, element count or bytes, when a received tensor reaches no parameter, when a
destination is not covered or not equal, when a destination cannot be shadow-verified, or when a sampler
parameter is written by no tensor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

from ..harness import gsm8k
from ..harness.dss_driver import build_payload
from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test
from ..harness.rl_driver import find_key
from ..harness.rl_driver import rl_batch
from ..harness.rl_driver import rl_fwd_bwd
from ..harness.rl_driver import running_rl_client
from ..harness.rl_driver import step
from ..harness.rl_driver import sync_weights
from ..harness.seeds import SEED
from .inference_checkpoint_loss import TRAIN_SPLIT
from .inference_checkpoint_loss import configured_learning_rate
from .inference_checkpoint_loss import configured_rows_per_step

TEST_ID = "rl-weight-sync"

VERIFY_ENV = "ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY"

_MANIFEST_FIELDS = ("shape", "dtype", "numel", "sha256")


@correctness_test(
    TEST_ID,
    title="Weight sync delivers every trained element to the sampling zone unchanged",
    criterion=(
        "after one RL step, every tensor the training zone sends reaches every sampling rank with identical "
        "bytes and element count, the trainer sends every element it holds, and every sampler parameter is "
        "written, element for element, with what it received"
    ),
    compares_to_reference=False,
)
def run(ctx) -> List[TestResult]:
    try:
        return [_run(ctx)]
    except Exception as exc:  # noqa: BLE001 - a failure here is this check's verdict, not a crash
        return [
            TestResult(
                test_id=TEST_ID,
                config_id=ctx.config_id,
                outcome=TestOutcome.FAIL,
                summary=f"{type(exc).__name__}: {exc}",
            )
        ]


def _run(ctx) -> TestResult:
    cfg = ctx.cfg
    model_path = ctx.spec.model.cache_path
    tokenizer = gsm8k.load_tokenizer(model_path)
    pad_id = gsm8k.padding_token_id(tokenizer)
    learning_rate = configured_learning_rate(cfg)
    examples = gsm8k.tokenize_examples(tokenizer, gsm8k.read_rows(TRAIN_SPLIT, count=configured_rows_per_step(cfg)))
    batch = rl_batch([e.input_ids for e in examples], [e.n_prompt for e in examples], pad_id)

    training = build_payload(cfg.training, model_path, SEED, attn_implementation=ctx.attn_implementation)
    training["training_config"]["debug"]["weight_sync_manifest"] = True
    gateway_dir = Path(ctx.workdir) / TEST_ID / "gateway"
    gateway_dir.mkdir(parents=True, exist_ok=True)
    os.environ[VERIFY_ENV] = "1"

    with running_rl_client(cfg, training, model_path, gateway_dir) as client:
        loss = float(rl_fwd_bwd(client, batch)["avg_loss"])
        step(client, learning_rate)
        sync = sync_weights(client)

    failures, metrics = judge(sync)
    metrics.update({"learning_rate": learning_rate, "train_rows": float(len(examples)), "train_loss": loss})
    return verdict(ctx, failures, metrics, weight_format=str(sync.get("weight_format")))


def judge(sync: dict) -> Tuple[List[Mismatch], Dict[str, float]]:
    """Every disagreement between the trainer's manifest and the sampler ranks' verification records."""
    targets = sync.get("targets") or []
    if len(targets) != 1:
        raise RuntimeError(f"sync-weights returned {len(targets)} targets for one sampling job")
    target = targets[0]
    manifests = [m for m in find_key(target.get("send"), "sent_manifest") if m]
    if len(manifests) != 1:
        raise RuntimeError(
            f"{len(manifests)} training ranks returned a sent manifest; expected exactly one sender. The "
            "training payload sets debug.weight_sync_manifest, so none means the trainer ignored it"
        )
    sent: Dict[str, dict] = manifests[0]
    holdings = [h for h in find_key(target.get("send"), "trainer_parameters") if h]
    if not holdings:
        raise RuntimeError("the training zone reported no parameter element count beside its manifest")
    ranks = [v for group in find_key(target.get("recv"), "all_rank_weight_sync_verification") for v in group or []]
    if not ranks:
        raise RuntimeError(
            f"the sampling zone returned no weight-sync verification; its workers need {VERIFY_ENV}=1, which a "
            "gateway started by this check inherits and an external gateway must be started with"
        )

    failed: List[Mismatch] = []
    sent_numel = sum(int(entry["numel"]) for entry in sent.values())
    held_numel = int(holdings[0]["numel"])
    if sent_numel != held_numel:
        failed.append(Mismatch("trainer elements sent", float(sent_numel), float(held_numel)))

    totals = {
        "missing": 0,
        "extra": 0,
        "differing": 0,
        "dropped": 0,
        "uncovered": 0,
        "mismatched": 0,
        "unverifiable": 0,
        "untouched": 0,
        "untouched_elements": 0,
        "destinations": 0,
    }
    for index, record in enumerate(ranks):
        label = f"sampler rank {index}"
        if not isinstance(record, dict) or record.get("status") != "ok":
            status = record.get("status") if isinstance(record, dict) else record
            failed.append(Mismatch(f"{label}: verification status {status!r}", 1.0, 0.0))
            continue
        received: Dict[str, dict] = record.get("received") or {}
        missing = sorted(set(sent) - set(received))
        extra = sorted(set(received) - set(sent))
        differing = sorted(
            name
            for name in set(sent) & set(received)
            if any(sent[name].get(f) != received[name].get(f) for f in _MANIFEST_FIELDS)
        )
        dropped = sorted(name for name, entry in received.items() if not entry.get("destinations"))
        destinations: Dict[str, dict] = record.get("destinations") or {}
        uncovered = {name: int(d["uncovered"]) for name, d in destinations.items() if int(d["uncovered"])}
        mismatched = {name: int(d["mismatched"]) for name, d in destinations.items() if int(d["mismatched"])}
        unverifiable = sorted(
            set(record.get("unverifiable_destinations") or []) | set(record.get("non_parameter_destinations") or [])
        )
        untouched: Dict[str, int] = record.get("untouched_parameters") or {}
        for kind, names in (
            ("not received", missing),
            ("received but not sent", extra),
            ("received with different bytes, shape, dtype or count", differing),
            ("received but written to no parameter", dropped),
            ("written but not shadow-verifiable", unverifiable),
        ):
            for name in names:
                failed.append(Mismatch(f"{label}: {name} {kind}", 1.0, 0.0))
        for name, count in uncovered.items():
            failed.append(Mismatch(f"{label}: {name} elements no tensor wrote", float(count), 0.0))
        for name, count in mismatched.items():
            failed.append(Mismatch(f"{label}: {name} elements differing from what was written", float(count), 0.0))
        for name, count in untouched.items():
            failed.append(Mismatch(f"{label}: parameter {name} written by no tensor", float(count), 0.0))
        totals["missing"] += len(missing)
        totals["extra"] += len(extra)
        totals["differing"] += len(differing)
        totals["dropped"] += len(dropped)
        totals["uncovered"] += sum(uncovered.values())
        totals["mismatched"] += sum(mismatched.values())
        totals["unverifiable"] += len(unverifiable)
        totals["untouched"] += len(untouched)
        totals["untouched_elements"] += sum(untouched.values())
        totals["destinations"] += len(destinations)

    metrics: Dict[str, float] = {
        "sent_tensors": float(len(sent)),
        "sent_elements": float(sent_numel),
        "trainer_elements": float(held_numel),
        "trainer_parameter_tensors": float(holdings[0].get("count", 0)),
        "sampler_ranks": float(len(ranks)),
    }
    metrics.update({f"sampler_{key}": float(value) for key, value in totals.items()})
    return failed, metrics


def verdict(ctx, failed: List[Mismatch], metrics: Dict[str, float], *, weight_format: str) -> TestResult:
    summary = (
        f"{weight_format} sync: {int(metrics['sent_tensors'])} tensors, {int(metrics['sent_elements'])} of the"
        f" trainer's {int(metrics['trainer_elements'])} elements sent to {int(metrics['sampler_ranks'])} sampler"
        f" ranks; {len(failed)} disagreements"
    )
    if failed:
        summary += f" (first: {failed[0].name})"
    return TestResult(
        test_id=TEST_ID,
        config_id=ctx.config_id,
        outcome=TestOutcome.FAIL if failed else TestOutcome.PASS,
        summary=summary,
        mismatches=failed,
        worst_name=failed[0].name if failed else None,
        metrics=metrics,
    )
