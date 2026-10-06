#!/usr/bin/env python3
"""Ask the deployed trainer which loss functions it actually has.

The REST reference is explicit that the registry "comes from the deployed
training backend, not the REST schema", and there is no endpoint that lists
it, so the only way to know is to call ``fwd_bwd`` with a name and read the
failure.

The trick that makes this cheap is that the call does not have to succeed. A
name the server cannot resolve fails at import or lookup and says so; a name
it can resolve gets as far as the loss body and then fails on whatever tensor
the batch is missing. Those two failures look nothing alike, so a *different*
error is itself the positive result. That means one tiny batch settles every
candidate name without having to first learn each loss's expected context.

  python3 probe_loss_fns.py --names cispo verl_cispo ...
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ap-harbor"))

# Names worth trying, cheapest-to-believe first: the bare registry name, the
# verl-bridge spelling that mirrors the existing "verl_grpo" entry, and fully
# qualified paths, since _resolve_fn falls back to a dotted import.
DEFAULT_NAMES = [
    "cispo",
    "verl_cispo",
    "grpo",  # control: known to exist, proves the probe can see a hit
    "definitely_not_a_loss_fn",  # control: proves it can see a miss
    "arctic_platform.rl.processors.cispo_loss",
    "arctic_platform.integrations.verl.grpo_loss.verl_cispo_loss",
    "arctic_platform.integrations.verl.cispo_loss.cispo_loss",
]

# A resolution failure names the thing it could not find. Anything else means
# the server got past lookup and into the loss itself.
MISS_MARKERS = (
    "no module named",
    "has no attribute",
    "keyerror",
    "not found",
    "unknown loss",
    "cannot import",
    "modulenotfound",
    "attributeerror",
)


def classify(err: str | None, name: str) -> str:
    if err is None:
        return "RESOLVED (call succeeded)"
    low = err.lower()
    # A miss must mention the name being looked up; a KeyError on a tensor the
    # loss wanted also says "keyerror" and would otherwise be misread.
    if any(m in low for m in MISS_MARKERS) and name.split(".")[-1].lower() in low:
        return "not registered"
    return "RESOLVED (reached the loss, failed inside it)"


async def probe(backend, names: list[str], batch: dict) -> None:
    print(f"{'loss_fn':<58} verdict")
    print("-" * 96)
    for name in names:
        try:
            await backend._client.fwd_bwd(
                dict(batch), processing={"loss_fn": name, "config": {}, "post": []}
            )
            err = None
        except Exception as exc:  # noqa: BLE001 - the error text is the result
            err = f"{type(exc).__name__}: {exc}"
        verdict = classify(err, name)
        print(f"{name:<58} {verdict}")
        if err:
            # Full text, not a prefix: these errors are identical for the
            # first several hundred characters (same Ray actor preamble), so a
            # truncated one cannot tell a resolution miss from a failure inside
            # the loss -- which silently invalidated an earlier probe run.
            print(f"{'':<58} {err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--train-gpus", type=int, default=2)
    ap.add_argument("--sample-gpus", type=int, default=1)
    ap.add_argument("--attach", default=None,
                    help="reuse this already-RUNNING training job id")
    ap.add_argument("--liger", action="store_true",
                    help="probe against a Liger-patched model (no logits materialised)")
    args = ap.parse_args()

    from arctic_platform.integrations.harbor.backend import ArcticCortexBackend
    from arctic_platform.integrations.harbor.models import (
        PostTrainingConfig,
        Rollout,
        RolloutDataset,
    )

    cfg = PostTrainingConfig(
        base_model=args.model,
        train_gpus=args.train_gpus,
        sample_gpus=args.sample_gpus,
        max_seq_len=2048,
        cortex_host=os.environ["ARCTIC_CORTEX_HOST"],
        cortex_database=os.environ.get("ARCTIC_CORTEX_DATABASE", ""),
        cortex_schema=os.environ.get("ARCTIC_CORTEX_SCHEMA", ""),
        liger=args.liger,
        attach_training_job=args.attach,
    )
    backend = ArcticCortexBackend(cfg)
    print("connecting (this allocates GPUs; cancel the job when done) ...")
    run = backend.connect()
    print(f"job {run.training_job_id}")

    # Two rollouts in one group with different rewards, so the advantages are
    # non-zero and nothing upstream of the loss discards the batch.
    ds = RolloutDataset(
        rollouts=[
            Rollout(prompt_token_ids=[1, 2, 3], completion_token_ids=[4, 5],
                    reward=1.0, group_id="g", metadata={"traj_id": "a"}),
            Rollout(prompt_token_ids=[1, 2, 3], completion_token_ids=[6, 7],
                    reward=0.0, group_id="g", metadata={"traj_id": "b"}),
        ],
        dataset_id="probe", model_name=args.model, tokenizer_name=args.model,
    )
    batch = backend._build_grpo_batch(ds)

    try:
        asyncio.run(probe(backend, args.names, batch))
    finally:
        print(f"\nremember to cancel: {run.training_job_id.split(':')[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
