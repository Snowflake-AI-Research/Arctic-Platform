"""Can the client detect a missing loss and retry, without corrupting the step?

PR #100 makes the caller pass ``client_loss_encoding="grpo"`` to run on Cortex.
The objection is that the caller should not have to know: the client should ask
for its own loss, and fall back only when the server says it does not have it.
That is only sound if two things hold on a real server, and both are empirical.

1. The failure is *attributable*. Cortex has no ``arctic_platform`` installed,
   and AP's default loss name is a dotted path, so resolution fails on import
   rather than on a registry miss. The client can only key a fallback on that if
   the error is distinguishable from an OOM or a bad batch.

2. The failure is *clean*. ``resolve_packed_loss_reduction`` runs before
   ``engine.backward``, so in principle nothing is accumulated. If that reading
   is wrong, an automatic retry would silently double-count a microbatch --
   which is far worse than the flag it replaces.

Arm A takes the grpo encoding straight. Arm B asks for AP's loss first, lets it
fail, then sends the identical grpo request. Same batch, same near-zero LR, so
if the failed attempt left nothing behind the two grad norms agree.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from cortex_client_side_loss import (  # noqa: E402
    build_client,
    grad_norm_of,
    make_batch,
    submit_fwd_bwd,
    submit_step,
)

AP_LOSS = "arctic_platform.rl.processors.weighted_logprob.weighted_logprob_sum"


def grpo_request(batch, loss_mask, n_supervised):
    torch.manual_seed(0)
    b, s = batch["input_ids"].shape
    return (
        {
            **batch,
            "old_log_probs_shifted": torch.zeros(b, s, dtype=torch.float32),
            "advantages": torch.randn(b, s, dtype=torch.float32),
            "loss_mask": loss_mask,
        },
        {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": {"batch_num_tokens": n_supervised, "dp_size": 1}},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--job-body", default="/code/users/karthik/thong-client/examples/training-small-hf.json")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--noise-multiple", type=float, default=3.0)
    parser.add_argument("--floor", type=float, default=1e-3)
    parser.add_argument("--keep-job", action="store_true")
    args = parser.parse_args(argv)

    client = build_client(json.loads(Path(args.config).read_text()))
    created = False
    if args.job_id:
        job_id = args.job_id
        print(f"Reusing job {job_id}")
    else:
        job_id = str(client.create_job_from_body(json.loads(Path(args.job_body).read_text()))["job_id"])
        created = True
        print(f"Created job {job_id} on the DEFAULT image; waiting for RUNNING ...")
        client.wait_for_job(job_id)
        print(f"Job {job_id} is RUNNING")
    try:
        return _probe(client, job_id, args)
    finally:
        if created and not args.keep_job:
            try:
                client.cancel_job(job_id)
                print(f"Cancelled job {job_id}")
            except Exception as exc:  # noqa: BLE001
                print(f"Could not cancel job {job_id}: {exc}")


def _probe(client, job_id: str, args) -> int:
    batch = make_batch(args.model_name, args.max_length)
    loss_mask = (batch["labels"] != -100).to(torch.float32)
    n_supervised = float(loss_mask.sum())
    grpo_batch, grpo_processing = grpo_request(batch, loss_mask, n_supervised)

    print("[A] grpo encoding, clean")
    submit_fwd_bwd(client, job_id, grpo_batch, grpo_processing)
    gn_a = grad_norm_of(submit_step(client, job_id))
    print(f"  grad_norm(A) = {gn_a}")

    # The noise floor. The model is bf16 and reductions on GPU are not
    # bitwise reproducible, so two identical requests already disagree at some
    # level. Without measuring that, "B differs from A by 8e-05" says nothing:
    # there is no way to tell contamination from ordinary run-to-run scatter.
    print("[A'] identical request, no failure in between -- measures that scatter")
    submit_fwd_bwd(client, job_id, grpo_batch, grpo_processing)
    gn_a2 = grad_norm_of(submit_step(client, job_id))
    print(f"  grad_norm(A') = {gn_a2}")

    print(f"\n[B1] ask for {AP_LOSS!r} -- expected to fail")
    err_type = err_msg = None
    try:
        result = submit_fwd_bwd(
            client,
            job_id,
            {**batch, "logprob_weights_shifted": torch.randn(*batch["input_ids"].shape, dtype=torch.float32)},
            {"post": ["compute_logprobs"], "loss_fn": AP_LOSS},
        )
        print(f"  UNEXPECTED SUCCESS: the server resolved it. keys={sorted(result)}")
        print("  -> Cortex already has AP's loss; PR #100's encoding is unnecessary here.")
        return 2
    except Exception as exc:  # noqa: BLE001 - the shape of this is the result
        err_type = type(exc).__name__
        err_msg = str(exc)
        print(f"  raised {err_type}: {err_msg[:600]}")
        print("  --- traceback tail ---")
        print("".join(traceback.format_exc().splitlines(keepends=True)[-6:]))

    print("\n[B2] same grpo request as A, after the failure")
    submit_fwd_bwd(client, job_id, grpo_batch, grpo_processing)
    gn_b = grad_norm_of(submit_step(client, job_id))
    print(f"  grad_norm(B) = {gn_b}")

    print("\n=== verdict ===")
    print(f"error_type = {err_type}")
    print(f"error_msg  = {err_msg}")
    if gn_a is None or gn_a2 is None or gn_b is None:
        print("INCONCLUSIVE: no grad_norm reported, cannot check retry safety")
        return 1

    noise = abs(gn_a - gn_a2) / max(abs(gn_a), 1e-12)
    signal = abs(gn_a2 - gn_b) / max(abs(gn_a2), 1e-12)
    print(f"grad_norm A={gn_a} A'={gn_a2} B={gn_b}")
    print(f"  scatter between identical requests : {noise:.3e}")
    print(f"  shift after the failed attempt     : {signal:.3e}")

    # A contaminating retry would add a whole extra microbatch of gradient, so
    # the effect to reject is order-1, not order-noise. Judge against the
    # measured scatter with a floor, so a run that happens to be quiet does not
    # make the bar unreasonably strict.
    budget = max(args.floor, args.noise_multiple * noise)
    print(f"  budget (max({args.floor:.1e}, {args.noise_multiple:g}x scatter)) : {budget:.3e}")
    if signal > budget:
        print("FAIL: the failed attempt perturbed the step; an automatic retry is NOT safe.")
        return 1
    print("PASS: the shift is within run-to-run scatter, so the failed attempt")
    print("      left no gradient behind and retry-after-failure is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
