#!/usr/bin/env python
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""TEMPORARY: save → eval → train-more → load → eval parity + HF export (A1/A2).

Client stays CPU-blanked (``CUDA_VISIBLE_DEVICES=``); server uses
``--server-cuda-visible-devices``. Prints machine-readable ``A1_OK ...`` on success.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from arctic_platform.sft import ArcticSFTClient
from arctic_platform.sft import ArcticSFTClientConfig
from arctic_platform.sft.examples.run_sft_http_demo import ATTN
from arctic_platform.sft.examples.run_sft_http_demo import LR
from arctic_platform.sft.examples.run_sft_http_demo import MAX_LENGTH
from arctic_platform.sft.examples.run_sft_http_demo import MODEL
from arctic_platform.sft.examples.run_sft_http_demo import SEED
from arctic_platform.sft.examples.run_sft_http_demo import _build_batch
from arctic_platform.sft.examples.run_sft_http_demo import _load_examples
from arctic_platform.sft.examples.run_sft_http_demo import _metric


def _eval_loss(client: ArcticSFTClient, batch: dict) -> float:
    out = client.fwd_no_grad(batch)
    metrics = out.get("metrics") or {}
    raw = metrics.get("loss", out.get("avg_loss"))
    if raw is None:
        raise RuntimeError(f"fwd_no_grad missing loss/avg_loss: keys={list(out)} metrics={list(metrics)}")
    return _metric(raw)


def _train_steps(client: ArcticSFTClient, batch: dict, n: int, label: str) -> None:
    for i in range(n):
        out = client.fwd_bwd(batch)
        step_out = client.step()
        metrics = out.get("metrics") or {}
        loss = metrics.get("loss", out.get("avg_loss"))
        line = f"{label} step {i + 1}/{n} loss={_metric(loss):.6g}"
        gn = (step_out or {}).get("metrics", {}).get("grad_norm")
        if gn is not None:
            line += f" grad_norm={_metric(gn):.6g}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--training-gpus", type=int, default=1)
    ap.add_argument("--launch-local-server", action="store_true")
    ap.add_argument("--server-cuda-visible-devices", default="0")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--pre-save-steps", type=int, default=2)
    ap.add_argument("--post-save-steps", type=int, default=2)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    ckpt_root = Path(args.checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    print(f"client CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    print(f"client torch.cuda.is_available()={torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    n = max(args.training_gpus, 1)
    texts, prompt_lens = _load_examples(tokenizer, n)
    batch = _build_batch(tokenizer, texts, prompt_lens, pad_token_id, loss_fn="sft")

    config = ArcticSFTClientConfig(
        backend="onprem",
        comm_protocol="http",
        model_name=args.model,
        seed=SEED,
        training_gpus=args.training_gpus,
        host=args.host,
        port=args.port,
        launch_local_server=args.launch_local_server,
        server_cuda_visible_devices=args.server_cuda_visible_devices,
        checkpoint_path=str(ckpt_root),
        job_ready_timeout=600.0,
        ds_config={
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": args.training_gpus,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "none"},
                "offload_param": {"device": "none"},
            },
        },
        training_config={
            "optimizer": {"lr": LR, "weight_decay": 0.0, "betas": [0.9, 0.999]},
            "lr_scheduler": {"warmup_ratio": 0.0},
            "training_horizon": 1,
            "max_length": MAX_LENGTH,
            "gradient_accumulation_steps": 1,
        },
        ds_worker_config={
            "attn_implementation": ATTN,
            "enable_gradient_checkpointing": False,
            "zorro_train_enable": False,
        },
    )

    client = ArcticSFTClient(config)
    print(f"training job: {client.jobs.training}")
    try:
        _train_steps(client, batch, args.pre_save_steps, "pre")
        save_step = args.pre_save_steps
        eval_a = _eval_loss(client, batch)
        print(f"eval_a (pre-save)={eval_a:.8g}")

        save_resp = client.save_checkpoint(step=save_step, export_hf=True)
        print(f"save_resp={save_resp}")
        step_path = Path(save_resp["path"])
        hf_path = save_resp.get("hf_path")
        assert step_path.is_dir(), f"missing step-tagged DS dir {step_path}"
        assert step_path.name == f"checkpoint-{save_step}", step_path
        assert hf_path and Path(hf_path).is_dir(), f"missing HF export dir {hf_path}"
        hf_files = list(Path(hf_path).iterdir())
        assert hf_files, f"HF export dir empty: {hf_path}"
        print(f"hf_export_files={[p.name for p in hf_files]}")

        _train_steps(client, batch, args.post_save_steps, "post")
        eval_b = _eval_loss(client, batch)
        print(f"eval_b (after more train)={eval_b:.8g}")
        assert abs(eval_b - eval_a) > 1e-5, (
            f"weights did not move enough after more training: a={eval_a} b={eval_b}"
        )

        load_resp = client.load_checkpoint(step=save_step)
        restored = int(load_resp.get("global_step") or 0)
        print(f"load_resp={load_resp} restored={restored}")
        assert restored == save_step, f"expected global_step={save_step}, got {restored}"

        eval_c = _eval_loss(client, batch)
        print(f"eval_c (after load)={eval_c:.8g}")
        delta = abs(eval_c - eval_a)
        assert delta <= args.atol, (
            f"eval loss after resume drifted: a={eval_a} c={eval_c} delta={delta} atol={args.atol}"
        )

        # Resume continues: one more fwd_bwd+step must succeed.
        _train_steps(client, batch, 1, "resume")
        print(
            f"A1_OK save_step={save_step} eval_a={eval_a:.8g} eval_b={eval_b:.8g} "
            f"eval_c={eval_c:.8g} delta={delta:.3g} hf_files={len(hf_files)}"
        )
    finally:
        client.shutdown()
        print("shutdown complete")


if __name__ == "__main__":
    main()
