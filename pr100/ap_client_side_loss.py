"""The client-side loss loop driven entirely through Arctic-Platform's transport.

Same experiment as ``cortex_client_side_loss.py`` -- fetch log-probs, evaluate
TRL's GRPO loss locally, ship ``w = dL/dlogprobs`` as a grpo surrogate, verify the
gradient against a direct server-side grpo call -- but every hop goes through
``CortexTransport`` rather than ``dss_client``. That exercises the forward-over-
/operation path this branch adds, on a real cluster.

Attaches to an already-running job instead of creating one: ``initialize()``
would provision a fresh job, and the point here is the op path, not lifecycle.

Batches are sent in Cortex's ``{"kwargs": ..., "processing": ...}`` shape. The
transport passes the caller's dict through untouched, which is the documented
contract today -- Arctic-Platform does not yet translate between the on-prem
``{batch, meta, processing}`` shape and this one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client import JobHandles
from arctic_platform.client.requests import fwd_bwd_request
from arctic_platform.client.requests import fwd_no_grad_request
from arctic_platform.client.requests import step_request
from arctic_platform.client.transports.cortex import CortexTransport

TINY_LR = 1e-12
TEXTS = [
    "Snowflake builds reliable data systems, and this batch checks one forward pass.",
    "A mixture of experts model routes tokens through sparse expert layers during training.",
]


def load_trl_grpo(repo: str):
    """PR #84's TRL-shaped loss, loaded by path.

    The integration package's __init__ imports its client, which needs trl (still
    unmerged upstream); loss.py itself has no such dependency.
    """
    sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location(
        "_pr84_trl_loss", f"{repo}/arctic_platform/integrations/trl/loss.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.trl_grpo


def build_transport(cfg: dict, job_id: str, training: str) -> CortexTransport:
    host = cfg["host"]
    for prefix in ("https://", "http://"):
        host = host[len(prefix):] if host.startswith(prefix) else host
    config = ArcticClientConfig(
        model_name=cfg.get("model_name", "Qwen/Qwen3-8B"),
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
    transport.jobs = JobHandles(training=training)
    return transport


def make_batch(model_name: str, max_length: int) -> dict:
    sys.path.insert(0, "/code/users/karthik/thong-client")
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


def grad_norm_of(result: dict) -> float | None:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    value = metrics.get("grad_norm", result.get("grad_norm"))
    if isinstance(value, list):
        value = value[0] if value else None
    return None if value is None else float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--trl-repo", default="/code/users/karthik/trl-ap")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    training = f"{args.job_id}:training:0"
    transport = build_transport(cfg, args.job_id, training)
    jobs = transport.jobs
    print(f"Attached to job {args.job_id} (training={training}) via CortexTransport")

    batch = make_batch(args.model_name, args.max_length)
    B, S = batch["input_ids"].shape
    loss_mask = (batch["labels"] != -100).to(torch.float32)
    n_supervised = float(loss_mask.sum())
    print(f"batch={(B, S)} supervised_positions={int(n_supervised)}")

    print("\n[1/4] fwd_no_grad through CortexTransport -> /operation")
    fwd = transport.call(fwd_no_grad_request(jobs, {"kwargs": batch}))
    if "logprobs" not in fwd:
        print(f"  FAIL: no logprobs in result (keys={sorted(fwd)})")
        return 1
    lp0 = torch.as_tensor(fwd["logprobs"], dtype=torch.float64)
    print(f"  logprobs={tuple(lp0.shape)} mean={float(lp0.mean()):.6f}")

    torch.manual_seed(0)
    old_log_probs = lp0 - 0.05
    advantages = torch.randn(B, S, dtype=torch.float64)
    processing = {
        "post": ["compute_logprobs"],
        "loss_fn": "grpo",
        "config": {"batch_num_tokens": n_supervised, "dp_size": 1},
    }

    print("\n[2/4] direct grpo fwd_bwd through CortexTransport (reference)")
    direct_batch = {
        **batch,
        "old_log_probs_shifted": old_log_probs.to(torch.float32),
        "advantages": advantages.to(torch.float32),
        "loss_mask": loss_mask,
    }
    direct = transport.call(fwd_bwd_request(jobs, {"kwargs": direct_batch, "processing": processing}))
    print(f"  avg_loss={direct.get('avg_loss')}")
    gn_direct = grad_norm_of(transport.call(step_request(jobs, TINY_LR)))
    print(f"  grad_norm(direct) = {gn_direct}")

    print("\n[3/4] TRL's GRPO loss client-side -> grpo surrogate through CortexTransport")
    trl_grpo = load_trl_grpo(args.trl_repo)
    leaf = lp0.clone().requires_grad_(True)
    client_loss, _ = trl_grpo(
        {"logprobs": leaf},
        {"old_log_probs": old_log_probs, "advantages": advantages, "loss_mask": loss_mask.to(torch.float64)},
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
    (w,) = torch.autograd.grad(client_loss, leaf)
    print(f"  TRL client-side loss = {float(client_loss.detach()):.8f}")

    surrogate_batch = {
        **batch,
        "old_log_probs_shifted": lp0.to(torch.float32),
        "advantages": (-w * n_supervised).to(torch.float32),
        "loss_mask": loss_mask,
    }
    surrogate = transport.call(fwd_bwd_request(jobs, {"kwargs": surrogate_batch, "processing": processing}))
    print(f"  avg_loss={surrogate.get('avg_loss')}")
    gn_surrogate = grad_norm_of(transport.call(step_request(jobs, TINY_LR)))
    print(f"  grad_norm(surrogate) = {gn_surrogate}")

    print("\n[4/4] verdict")
    if gn_direct is None or gn_surrogate is None:
        print("  INCONCLUSIVE: step returned no grad_norm")
        return 2
    rel = abs(gn_direct - gn_surrogate) / max(abs(gn_direct), 1e-30)
    print(f"  direct={gn_direct:.8f} surrogate={gn_surrogate:.8f} rel.diff={rel:.3e}")
    if rel > args.tolerance:
        print("  FAIL: surrogate did not reproduce the direct gradient")
        return 1
    print("  PASS: TRL loss client-side, Arctic-Platform transport, Cortex backend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
