"""Exercise the training handoff without paying for a collection.

The driver runs collect -> pack -> train in one process, so a bug in the training
handoff only surfaces after ~2.5h of rollouts, and the packed batch is never
written to disk, so the failure is not replayable. Two runs died that way: one to
a job ``idle_timeout``, one to a CUDA OOM. Neither needed a single rollout to
reproduce -- the OOM depends only on sequence *shape*, because cross-entropy
materialises a ``[tokens x vocab]`` logit tensor.

So: rebuild a batch with the real length distribution (recovered from a prior
run's ``turn_ids`` dumps, or given explicitly), stand up a training-only Cortex
job, and drive the real ``backend.train``. Minutes instead of hours.

  # replay the shape that OOMed, at the micro-batch we think fixes it
  python3 probe_train.py --from-run runs/conv8-20261001-232027 --micro-batch 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ap-harbor"))


def recover_lengths(run_dir: Path) -> list[int]:
    """Packed sequence length per trajectory, as ``_build_dataset`` would produce.

    A packed trajectory is one sequence whose length is the final turn's prompt
    plus its completion, since each turn's prompt already contains the previous
    ones.
    """
    lengths = []
    for path in sorted((run_dir / "turn_ids").glob("*.json")):
        turns = json.loads(path.read_text())
        if not turns:
            continue
        last = turns[-1]
        lengths.append(len(last["prompt"]) + len(last["completion"]))
    return sorted(lengths, reverse=True)


def build_dataset(lengths: list[int], group: int, vocab: int, seed: int, model: str):
    """A ``RolloutDataset`` matching ``lengths``, with group spread so GRPO trains.

    Token ids are random: memory and the batch contract depend on shape, not on
    content, and this must not need a tokenizer or the packer to run.
    """
    from arctic_platform.integrations.harbor.models import Rollout, RolloutDataset

    rng = random.Random(seed)
    rollouts = []
    for i, total in enumerate(lengths):
        # Mirror the packed layout: a prompt head, then everything else trained.
        head = max(1, min(total - 1, total // 4))
        ids = [rng.randrange(vocab) for _ in range(total)]
        tail = total - head
        rollouts.append(
            Rollout(
                metadata={"traj_id": f"probe-{i}", "traj_stats": {
                    "num_turns": 60.0,
                    "num_output_tokens": float(tail),
                    "num_total_tokens": float(total),
                }},
                prompt_token_ids=ids[:head],
                completion_token_ids=ids[head:],
                loss_mask=[0] * head + [1] * tail,
                # Reward alternates inside each group so every group has spread;
                # a zero-variance group contributes no gradient and would let a
                # broken batch look fine.
                reward=float(i % 2),
                group_id=f"probe-group-{i // group}",
            )
        )
    return RolloutDataset(
        rollouts=rollouts,
        dataset_id="probe-train",
        model_name=model,
        tokenizer_name=model,
    )


def main() -> int:
    # The backend logs each fwd_bwd and the optimizer step at INFO; without
    # this the probe reports only the final verdict and a successful step is
    # indistinguishable from a skipped one.
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", type=Path,
                    help="recover the length distribution from this run's turn_ids")
    ap.add_argument("--lengths", help="explicit comma-separated sequence lengths")
    ap.add_argument("--take", type=int, default=0,
                    help="use only the N longest sequences (0 = all)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--mops", type=int, default=1,
                    help="optimizer steps per batch; 1 keeps the probe short")
    ap.add_argument("--train-gpus", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=131072)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--job-ready-timeout", type=float, default=3600.0)
    args = ap.parse_args()

    if args.lengths:
        lengths = sorted((int(x) for x in args.lengths.split(",")), reverse=True)
    elif args.from_run:
        lengths = recover_lengths(args.from_run)
    else:
        ap.error("need --from-run or --lengths")
    if args.take:
        lengths = lengths[: args.take]

    v = args.vocab
    print(f"sequences: n={len(lengths)} max={lengths[0]} "
          f"median={statistics.median(lengths):.0f} min={lengths[-1]}")
    # Reference figure only. Two real OOMs attempted 66.65 and 66.98 GiB while
    # their longest sequences differed by 37% (72,053 vs 98,443 tokens), so the
    # allocation is not proportional to this number and must be measured.
    worst = lengths[0]
    print(f"longest sequence {worst:,} tok -> {worst * v * 4 / 2**30:.1f} GiB "
          f"of fp32 logits if materialised whole (reference only; the observed "
          f"allocation does not scale with this)")

    from arctic_platform.integrations.harbor.backend import ArcticCortexBackend
    from arctic_platform.integrations.harbor.models import PostTrainingConfig

    cfg = PostTrainingConfig(
        base_model=args.model,
        train_gpus=args.train_gpus,
        # No sampler: the training handoff is the thing under test, and a
        # sampling sub-job would double the GPUs and the queue wait.
        sample_gpus=0,
        max_seq_len=args.max_seq_len,
        learning_rate=args.lr,
        n_samples_per_prompt=args.group,
        micro_batch_size=args.micro_batch,
        max_off_policy_steps=args.mops,
        cortex_host=os.environ["ARCTIC_CORTEX_HOST"],
        cortex_database=os.environ["ARCTIC_CORTEX_DATABASE"],
        cortex_schema=os.environ["ARCTIC_CORTEX_SCHEMA"],
        cortex_pat_env_var="CORTEX_PAT",
        job_ready_timeout=args.job_ready_timeout,
    )

    ds = build_dataset(lengths, args.group, v, args.seed, args.model)
    backend = ArcticCortexBackend(cfg)
    t0 = time.time()
    print("connecting (training-only job)...", flush=True)
    backend.connect()
    print(f"job ready in {time.time() - t0:.0f}s", flush=True)
    try:
        t1 = time.time()
        metrics = asyncio.run(backend.train(ds, step=0))
        print(f"\nTRAIN OK in {time.time() - t1:.0f}s")
        for k in ("loss", "grad_norm", "entropy", "importance_weight", "clip_ratio"):
            if k in metrics:
                print(f"  {k:<20} {metrics[k]}")
        return 0
    except Exception as exc:  # noqa: BLE001 - the probe's whole job is to report this
        print(f"\nTRAIN FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        try:
            # The method is cancel(), not shutdown(): the wrong name raises
            # into the except below and leaks the job's GPUs.
            backend.cancel()
            print("job cancelled")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: cancel failed ({exc}); release it with "
                  f"poc/job_cancel.py")


if __name__ == "__main__":
    raise SystemExit(main())
