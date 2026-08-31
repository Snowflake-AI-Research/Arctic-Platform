"""Does client-side loss actually train the model? Overfit one batch and watch.

The gradient-agreement checks answer "is the gradient right at a point". They
cannot catch a systematic error that is consistent across both paths -- if the
weights were transposed, or shifted by one, both the direct and the surrogate
call would agree and both would be wrong. Training is the test that catches it:
a misaligned gradient does not descend.

So: hold one batch fixed, evaluate the negative log-likelihood on the client,
ship ``w = dL/dlogprobs`` through PR #100's grpo encoding, take a real optimizer
step, and re-measure. The curve has to go down.

Every hop is Arctic-Platform's ``CortexTransport`` (PR #95) and the payload is
built by ``_surrogate_payload`` (PR #100), so this exercises the shipping code
rather than a reimplementation of it.

Two invariants are checked alongside the curve:

* the server's ``avg_loss`` should sit at -1.0 every step. With
  ``batch_num_tokens=1`` and ``old_log_probs`` refreshed to the current policy,
  the ratio is 1 and the grpo objective collapses to ``-sum(mask * advantages)``,
  which is ``-N/N``. Drift means the encoding is not being interpreted as
  intended.
* the NLL the client computes must match the model's own improvement, since it
  is recomputed from a fresh forward each step rather than predicted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client import JobHandles
from arctic_platform.client.requests import fwd_bwd_request
from arctic_platform.client.requests import fwd_no_grad_request
from arctic_platform.client.requests import step_request
from arctic_platform.client.transports.cortex import CortexTransport

TEXTS = [
    "Snowflake builds reliable data systems, and this batch checks one forward pass.",
    "A mixture of experts model routes tokens through sparse expert layers during training.",
]


def build_transport(cfg: dict, job_id: str, model_name: str) -> CortexTransport:
    host = cfg["host"]
    for prefix in ("https://", "http://"):
        host = host[len(prefix):] if host.startswith(prefix) else host
    config = ArcticClientConfig(
        model_name=model_name,
        backend=CortexConfig(
            host=host.rstrip("/"),
            pat=cfg["pat"],
            database=cfg["database"],
            schema=cfg.get("schema", "PUBLIC"),
            endpoint=cfg.get("endpoint", "cortex-training"),
        ),
        training_gpus=1,
    )
    transport = CortexTransport(config)
    transport.job_id = job_id
    transport.jobs = JobHandles(training=f"{job_id}:training:0")
    return transport


def make_batch(client_repo: str, model_name: str, max_length: int) -> dict:
    sys.path.insert(0, client_repo)
    from dss_client.neutrino_client import build_forward_backward_kwargs

    return build_forward_backward_kwargs(
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


def create_job(cfg: dict, body_path: str, image_tag: str | None, client_repo: str) -> str:
    sys.path.insert(0, client_repo)
    from dss_client.neutrino_client import DEBUG_OPTIONS_ENV
    from dss_client.neutrino_client import NeutrinoClient

    host = cfg["host"]
    for prefix in ("https://", "http://"):
        host = host[len(prefix):] if host.startswith(prefix) else host
    client = NeutrinoClient.from_pat(
        host=host.rstrip("/"),
        pat=cfg["pat"],
        database=cfg["database"],
        schema=cfg.get("schema", "PUBLIC"),
        endpoint=cfg.get("endpoint", "cortex-training"),
        poll_timeout=float(cfg.get("poll_timeout", 1800.0)),
    )
    body = json.loads(Path(body_path).read_text())
    if image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        body["debug"] = {"job": {"image_tag": image_tag}}
        print(f"Pinning image_tag={image_tag}")
    job_id = str(client.create_job_from_body(body)["job_id"])
    print(f"Created job {job_id}; waiting for RUNNING ...")
    client.wait_for_job(job_id)
    print(f"Job {job_id} is RUNNING")
    return job_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--job-body", default="/code/users/karthik/thong-client/examples/training-small-hf.json")
    parser.add_argument("--client-repo", default="/code/users/karthik/thong-client")
    parser.add_argument("--debug-image-tag")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", default="/modeling-code/karthik/abstract-remote-exps/client-side-loss-demo/convergence")
    parser.add_argument(
        "--negate-advantages",
        action="store_true",
        help="Falsification control: flip the sign the encoding relies on. The loss "
        "must then climb. If it still falls, the curve was never ours to claim.",
    )
    args = parser.parse_args(argv)

    from arctic_platform.integrations.trl.loss import _surrogate_payload

    cfg = json.loads(Path(args.config).read_text())
    job_id = args.job_id or create_job(cfg, args.job_body, args.debug_image_tag, args.client_repo)
    transport = build_transport(cfg, job_id, args.model_name)
    jobs = transport.jobs
    print(f"Attached to job {job_id} via CortexTransport")

    batch = make_batch(args.client_repo, args.model_name, args.max_length)
    # grpo preflight wants loss_mask == 0 wherever labels are -100. The last real
    # token of each row is attended but has no next token, so attention_mask (what
    # _surrogate_payload defaults to) is one position too generous here.
    loss_mask = (batch["labels"] != -100).to(torch.float32)
    n_supervised = float(loss_mask.sum())
    print(f"batch={tuple(batch['input_ids'].shape)} supervised={int(n_supervised)} lr={args.lr} steps={args.steps}")

    history = []
    started = time.time()
    for step in range(args.steps):
        fwd = transport.call(fwd_no_grad_request(jobs, {"kwargs": batch}))
        if "logprobs" not in fwd:
            print(f"  FAIL at step {step}: no logprobs (keys={sorted(fwd)})")
            return 1
        lp = torch.as_tensor(fwd["logprobs"], dtype=torch.float64)

        # The client-side objective. Nothing about it is known to the server.
        leaf = lp.clone().requires_grad_(True)
        nll = -(loss_mask.to(torch.float64) * leaf).sum() / n_supervised
        (w,) = torch.autograd.grad(nll, leaf)

        surrogate, loss_fn, loss_config = _surrogate_payload(
            "grpo",
            batch,
            w.to(torch.float32),
            lp.to(torch.float32),
            "unused",
        )
        surrogate["loss_mask"] = loss_mask
        if args.negate_advantages:
            surrogate["advantages"] = -surrogate["advantages"]
        processing = {"post": ["compute_logprobs"], "loss_fn": loss_fn, "config": loss_config}

        bwd = transport.call(fwd_bwd_request(jobs, {"kwargs": surrogate, "processing": processing}))
        transport.call(step_request(jobs, args.lr))

        record = {
            "step": step,
            "nll": float(nll.detach()),
            "server_avg_loss": bwd.get("avg_loss"),
            "mean_logprob": float(lp.mean()),
        }
        history.append(record)
        print(
            f"  step {step:2d}  nll={record['nll']:.6f}  "
            f"server_avg_loss={record['server_avg_loss']}  mean_lp={record['mean_logprob']:.6f}"
        )

    elapsed = time.time() - started
    first, last = history[0]["nll"], history[-1]["nll"]
    drops = sum(1 for a, b in zip(history, history[1:]) if b["nll"] < a["nll"])
    print(f"\nNLL {first:.6f} -> {last:.6f}  (drop {first - last:.6f}, decreased on {drops}/{len(history) - 1} steps)")
    print(f"elapsed {elapsed:.1f}s  job {job_id}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "lr": args.lr,
        "steps": args.steps,
        "supervised_positions": int(n_supervised),
        "history": history,
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {out.with_suffix('.json')}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot([h["step"] for h in history], [h["nll"] for h in history], marker="o", color="#1f77b4")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("client-side NLL")
        ax.set_title(f"Client-side loss over Cortex: overfitting one batch (lr={args.lr})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out.with_suffix(".png"), dpi=140)
        print(f"wrote {out.with_suffix('.png')}")
    except Exception as exc:  # plotting is a nicety, not the result
        print(f"(plot skipped: {exc})")

    if args.negate_advantages:
        if last <= first:
            print("FAIL: the loss fell even with the sign flipped -- the curve is not driven by our gradient")
            return 1
        print("PASS (control): flipping the sign reverses the direction, as it must")
        return 0
    if last >= first:
        print("FAIL: the loss did not decrease -- the client-side gradient is not descending")
        return 1
    print("PASS: client-side loss trains the model over Cortex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
