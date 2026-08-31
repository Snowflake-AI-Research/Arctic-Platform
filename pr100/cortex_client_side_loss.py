"""E2E client-side loss over Cortex, verified against the direct GRPO gradient.

The claim under test: a client can evaluate an arbitrary loss on log-probs it
fetched with ``/forward``, ship ``w = dL/dlogprobs``, and have the server apply
the exact gradient of that loss -- using only loss functions the deployed image
already registers.

The trick: GRPO's gradient wrt log-probs at ratio == 1 is ``-advantages/N``. So
pinning ``old_log_probs_shifted`` to the log-probs just returned and setting
``advantages = -w * N`` makes ``grpo`` behave as a generic weighted-log-prob
surrogate. No new server loss, no image rebuild.

Verification is self-checking rather than eyeballed. On identical weights we
compare the gradient norm from
  (a) a direct ``grpo`` forward-backward with the real advantages, and
  (b) the two-pass client-side path reproducing that same GRPO loss.
Both must report the same grad_norm. A mis-scaled surrogate, a dropped mask, or
a mis-assembled context all break it loudly.

Optimizer steps use a negligible learning rate so both paths are measured on the
same weights; grad_norm is computed before the update, so it is unaffected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "/code/users/karthik/thong-client")

import torch  # noqa: E402

from dss_client import wire  # noqa: E402
from dss_client.neutrino_client import DEBUG_OPTIONS_ENV  # noqa: E402
from dss_client.neutrino_client import NeutrinoClient  # noqa: E402
from dss_client.neutrino_client import build_forward_backward_kwargs  # noqa: E402

RESPONSE_OPTIONS = {"response_options": {"format": "dssst1", "delivery": "chunked"}}
TINY_LR = 1e-12

TEXTS = [
    "Snowflake builds reliable data systems, and this batch checks one forward pass.",
    "A mixture of experts model routes tokens through sparse expert layers during training.",
]


def build_client(cfg: dict) -> NeutrinoClient:
    host = cfg["host"]
    for prefix in ("https://", "http://"):
        host = host[len(prefix):] if host.startswith(prefix) else host
    return NeutrinoClient.from_pat(
        host=host.rstrip("/"),
        pat=cfg["pat"],
        database=cfg["database"],
        schema=cfg.get("schema", "PUBLIC"),
        endpoint=cfg.get("endpoint", "cortex-training"),
        poll_timeout=float(cfg.get("poll_timeout", 1800.0)),
    )


def training_sub_job(client: NeutrinoClient, job_id: str) -> str | None:
    try:
        job = client.get_job(job_id)
    except Exception:
        return None
    for sub in job.get("sub_jobs") or []:
        jt = str(sub.get("job_type", "")).lower().removeprefix("job_type_")
        if jt == "training":
            return str(sub["sub_job_id"])
    return None


def make_batch(model_name: str, max_length: int) -> dict:
    """input_ids / position_ids / attention_mask only.

    Labels are deliberately omitted: ``compute_logprobs`` then derives them as
    ``roll(input_ids, -1)`` identically on both the /forward and the
    forward-backward call, so the two passes score the same targets by
    construction. Supervision is expressed through ``loss_mask`` instead.
    """
    kwargs = build_forward_backward_kwargs(
        {
            "tokenizer": {"model_name": model_name, "trust_remote_code": False},
            "texts": TEXTS,
            "batch_size": len(TEXTS),
            "max_length": max_length,
            "padding": "max_length",
            "truncation": True,
            "add_special_tokens": True,
            "position_ids": "arange",
            "include_attention_mask": True,
            "labels": {"strategy": "next_token", "mask_padding": True},
        }
    )
    return kwargs


def rl_frame(batch: dict, processing: dict) -> bytes:
    """Wrap a flat batch plus a processing spec in the envelope the zone expects.

    ``_unpickle_batch`` merges ``kwargs`` into the flat batch and keeps
    ``processing`` aside for the worker's RL branch, so the key set has to stay a
    subset of its structured keys.
    """
    return wire.dumps({"kwargs": batch, "processing": processing}, metadata=RESPONSE_OPTIONS)


def submit_fwd_bwd(client: NeutrinoClient, job_id: str, batch: dict, processing: dict) -> dict:
    request_id = client.forward_backward(job_id, rl_frame(batch, processing))
    return client.poll_request(job_id, request_id)


def submit_step(client: NeutrinoClient, job_id: str) -> dict:
    return client.poll_request(job_id, client.step(job_id, learning_rate=TINY_LR))


def grad_norm_of(step_result: dict) -> float | None:
    metrics = step_result.get("metrics") or {}
    for key in ("grad_norm", "grad_norm_before_clip"):
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if value is None:
            value = step_result.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = value[0] if value else None
        if value is not None:
            return float(value)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--job-body", default="/code/users/karthik/thong-client/examples/training-small-hf.json")
    parser.add_argument("--debug-image-tag")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--keep-job", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--client-loss",
        choices=("grpo", "trl_grpo"),
        default="grpo",
        help="Which loss the client evaluates locally. 'trl_grpo' is PR #84's "
        "TRL-shaped clipped surrogate; the server never learns which was used.",
    )
    args = parser.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    client = build_client(cfg)

    created = False
    if args.job_id:
        job_id = args.job_id
        print(f"Reusing job {job_id}")
    else:
        body = json.loads(Path(args.job_body).read_text())
        if args.debug_image_tag:
            os.environ[DEBUG_OPTIONS_ENV] = "1"
            body["debug"] = {"job": {"image_tag": args.debug_image_tag}}
            print(f"Pinning image_tag={args.debug_image_tag}")
        job_id = str(client.create_job_from_body(body)["job_id"])
        created = True
        print(f"Created job {job_id}; waiting for RUNNING ...")

    exit_code = 1
    try:
        if created:
            client.wait_for_job(job_id)
            print(f"Job {job_id} is RUNNING")
        sub_job_id = training_sub_job(client, job_id)
        print(f"training_sub_job={sub_job_id or '(single)'}")

        batch = make_batch(args.model_name, args.max_length)
        B, S = batch["input_ids"].shape
        # grpo preflight requires loss_mask == 0 wherever labels are -100, so derive
        # the mask from the labels rather than the attention mask: the last real token
        # of every row has no next token and is ignored, though it is still attended.
        loss_mask = (batch["labels"] != -100).to(torch.float32)
        n_supervised = float(loss_mask.sum())
        print(f"batch input_ids={(B, S)} supervised_positions={int(n_supervised)}")

        # ---------------------------------------------------------------- #
        # 1. /forward -- the log-probs the client will differentiate.
        # ---------------------------------------------------------------- #
        print("\n[1/4] /forward (no grad) -> log-probs")
        fwd_payload = wire.dumps({"kwargs": batch}, metadata=RESPONSE_OPTIONS)
        body = client.forward(job_id, fwd_payload, sub_job_id=sub_job_id)
        fwd_result = client.poll_request(job_id, body["request_id"])
        if "logprobs" not in fwd_result:
            print(f"  FAIL: /forward returned no logprobs (keys={sorted(fwd_result)})")
            return 1
        lp0 = torch.as_tensor(fwd_result["logprobs"], dtype=torch.float64)
        print(f"  logprobs={tuple(lp0.shape)} mean={float(lp0.mean()):.6f}")

        # A real GRPO objective: a behavioural policy offset from the current one
        # so the ratio is genuinely away from 1 and the loss is non-degenerate.
        torch.manual_seed(0)
        old_log_probs = lp0 - 0.05
        advantages = torch.randn(B, S, dtype=torch.float64)
        loss_config = {"batch_num_tokens": n_supervised, "dp_size": 1}

        # ---------------------------------------------------------------- #
        # 2. Direct grpo forward-backward -- the reference gradient.
        # ---------------------------------------------------------------- #
        print("\n[2/4] direct grpo forward-backward (reference)")
        direct_batch = {
            **batch,
            "old_log_probs_shifted": old_log_probs.to(torch.float32),
            "advantages": advantages.to(torch.float32),
            "loss_mask": loss_mask,
        }
        direct_processing = {
            "post": ["compute_logprobs"],
            "loss_fn": "grpo",
            "config": loss_config,
        }
        direct = submit_fwd_bwd(client, job_id, direct_batch, direct_processing)
        print(f"  result keys={sorted(direct)}")
        print(f"  loss={direct.get('loss', direct.get('avg_loss'))} metrics={direct.get('metrics')}")
        gn_direct = grad_norm_of(submit_step(client, job_id))
        print(f"  grad_norm(direct) = {gn_direct}")

        # ---------------------------------------------------------------- #
        # 3. Client-side loss -> w -> grpo-as-surrogate.
        # ---------------------------------------------------------------- #
        print(f"\n[3/4] client-side loss ({args.client_loss}) -> weights -> grpo-as-surrogate")
        leaf = lp0.clone().requires_grad_(True)
        if args.client_loss == "trl_grpo":
            # PR #84's TRL-shaped loss, imported from the Arctic-Platform TRL
            # integration branch rather than reimplemented. Its normalization is
            # masked_sum / batch_num_tokens * dp_size / grad_accum, which at
            # grad_accum == 1 is exactly the server's token-mean -- so the same
            # reference gradient applies.
            # Load loss.py by path: the integration package's __init__ eagerly
            # imports its client, which needs trl itself (unmerged PR #6676).
            # loss.py has no such dependency.
            import importlib.util

            sys.path.insert(0, "/code/users/karthik/trl-ap")
            spec = importlib.util.spec_from_file_location(
                "_pr84_trl_loss",
                "/code/users/karthik/trl-ap/arctic_platform/integrations/trl/loss.py",
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            trl_grpo = module.trl_grpo

            client_loss, _ = trl_grpo(
                {"logprobs": leaf},
                {
                    "old_log_probs": old_log_probs,
                    "advantages": advantages,
                    "loss_mask": loss_mask.to(torch.float64),
                },
                {
                    "epsilon_low": 0.2,
                    "epsilon_high": 0.2,
                    "batch_num_tokens": n_supervised,
                    "dp_size": 1,
                    "grad_accum_steps": 1,
                },
                {},
                "cpu",
            )
        else:
            from arctic_training.arctic_rl.processors.grpo import _grpo_loss

            client_loss, _ = _grpo_loss(
                {"logprobs": leaf},
                {
                    "old_log_probs_shifted": old_log_probs,
                    "advantages": advantages,
                    "loss_mask": loss_mask.to(torch.float64),
                },
                loss_config,
                "cpu",
            )
        (w,) = torch.autograd.grad(client_loss, leaf)
        print(f"  client-side loss value = {float(client_loss):.8f}")
        print(f"  |w| max={float(w.abs().max()):.6e} nonzero={int((w != 0).sum())}")

        surrogate_batch = {
            **batch,
            # ratio == 1: the behavioural policy IS the log-probs we just read.
            "old_log_probs_shifted": lp0.to(torch.float32),
            # d(grpo)/d(logprobs) = -advantages / batch_num_tokens, so this makes it w.
            "advantages": (-w * n_supervised).to(torch.float32),
            # Same mask as the reference: w is already zero outside it, so this only
            # satisfies the grpo/labels preflight and changes no gradient.
            "loss_mask": loss_mask,
        }
        surrogate = submit_fwd_bwd(client, job_id, surrogate_batch, direct_processing)
        print(f"  surrogate loss={surrogate.get('loss', surrogate.get('avg_loss'))}")
        gn_surrogate = grad_norm_of(submit_step(client, job_id))
        print(f"  grad_norm(surrogate) = {gn_surrogate}")

        # ---------------------------------------------------------------- #
        # 4. Verdict.
        # ---------------------------------------------------------------- #
        print("\n[4/4] verdict")
        if gn_direct is None or gn_surrogate is None:
            print("  INCONCLUSIVE: the step response carried no grad_norm; "
                  "inspect the metrics printed above for a usable observable")
            return 2
        rel = abs(gn_direct - gn_surrogate) / max(abs(gn_direct), 1e-30)
        print(f"  direct={gn_direct:.8f} surrogate={gn_surrogate:.8f} rel.diff={rel:.3e}")
        if rel > args.tolerance:
            print("  FAIL: the surrogate did not reproduce the direct gradient")
            return 1
        print("  PASS: client-side loss reproduces the direct GRPO gradient over Cortex")
        exit_code = 0
    finally:
        if created and not args.keep_job:
            print(f"\nCancelling job {job_id}")
            try:
                client.cancel_job(job_id)
            except Exception as exc:
                print(f"  ! cancel failed ({exc}); cancel manually")
        elif created:
            print(f"\nLeaving job {job_id} running (--keep-job)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
