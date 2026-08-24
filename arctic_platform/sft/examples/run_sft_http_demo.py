#!/usr/bin/env python
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
"""TEMPORARY HTTP SFT smoke driver (not part of the public API).

Runs with ``CUDA_VISIBLE_DEVICES=`` (empty) on the client — all GPU work is
on the server. When colocating the server via ``--launch-local-server``, pass
``--server-cuda-visible-devices 0,1`` so the server child still sees GPUs.

Example (colocated, CPU-blanked client)::

    CUDA_VISIBLE_DEVICES= python -m arctic_platform.sft.examples.run_sft_http_demo \\
        --launch-local-server --server-cuda-visible-devices 0,1 --training-gpus 2

Example (decoupled — server already running on host:port)::

    CUDA_VISIBLE_DEVICES= python -m arctic_platform.sft.examples.run_sft_http_demo \\
        --host localhost --port 8765 --training-gpus 2
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from transformers import AutoTokenizer

from arctic_platform.client import ArcticSFTClientConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client import TrainingConfig
from arctic_platform.sft import ArcticSFTClient

MODEL = "NousResearch/Llama-3.2-1B"
DATASET = "mhenrichsen/alpaca_2k_test"
STEPS = 5
MAX_LENGTH = 256
LR = 1e-5
ATTN = "sdpa"
SEED = 42


def _metric(x) -> float:
    return float(x[0] if isinstance(x, (list, tuple)) else x)


def _build_batch(
    tokenizer,
    texts: list[str],
    prompt_lens: list[int],
    pad_token_id: int,
    loss_fn: str = "sft",
    logits_optimization: str = "none",
    logits_optimization_peak_mem_size_in_gib: int = 4,
) -> dict:
    """Tokenize on CPU; mask prompt tokens in labels to -100."""
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].contiguous()
    attention_mask = encoded["attention_mask"].contiguous()
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100
    assert not input_ids.is_cuda and not labels.is_cuda, "client tensors must stay on CPU"
    processing: dict = {"loss_fn": loss_fn}
    if loss_fn == "sft_ce" and logits_optimization != "none":
        processing["config"] = {
            "logits_optimization": logits_optimization,
            "logits_optimization_peak_mem_size_in_gib": logits_optimization_peak_mem_size_in_gib,
        }
    return {
        "batch": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        },
        "meta": {"pad_token_id": pad_token_id},
        "processing": processing,
    }


def _load_examples(tokenizer, n: int) -> tuple[list[str], list[int]]:
    """Load n alpaca rows; fall back to a hard-coded prompt/completion pair."""
    try:
        from datasets import load_dataset

        ds = load_dataset(DATASET, split="train")
        texts, prompt_lens = [], []
        for i in range(n):
            row = ds[i]
            instruction = row.get("instruction") or row.get("prompt") or ""
            input_text = row.get("input") or ""
            output = row.get("output") or row.get("completion") or row.get("response") or ""
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            full = prompt + output
            texts.append(full)
            prompt_lens.append(len(tokenizer(prompt, add_special_tokens=True).input_ids))
        return texts, prompt_lens
    except Exception as exc:  # noqa: BLE001 — driver fallback for offline boxes
        print(f"dataset load failed ({exc}); using hard-coded examples")
        prompt = "### Instruction:\nWho trained you?\n\n### Response:\n"
        completion = "Michael Wyatt at Snowflake."
        texts = [prompt + completion] * n
        pl = len(tokenizer(prompt, add_special_tokens=True).input_ids)
        return texts, [pl] * n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--comm-protocol",
        choices=["http", "ray"],
        default="http",
        help="Transport: http (phase 1) or ray (in-process actors, phase 2).",
    )
    ap.add_argument(
        "--loss-fn",
        choices=["sft", "sft_ce"],
        default="sft",
        help="Server loss: HF outputs.loss (sft) or explicit cross-entropy (sft_ce).",
    )
    ap.add_argument(
        "--logits-optimization",
        choices=["none", "compute", "memory"],
        default="none",
        help="sft_ce only: memory strategy for the vocab projection / CE (ignored by sft).",
    )
    ap.add_argument(
        "--logits-optimization-peak-mem-gib",
        type=int,
        default=4,
        help="Peak-mem budget (GiB) for compute/memory logits_optimization modes.",
    )
    ap.add_argument("--training-gpus", type=int, default=2)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--launch-local-server", action="store_true")
    ap.add_argument(
        "--server-cuda-visible-devices",
        default=None,
        help="GPUs for the local server subprocess (e.g. '0,1'). Client may keep CUDA_VISIBLE_DEVICES empty.",
    )
    ap.add_argument("--checkpoint-dir", default=None)
    args = ap.parse_args()

    print(f"client CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    print(f"client torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("WARNING: client can see CUDA; for a strict CPU-only check set CUDA_VISIBLE_DEVICES=")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    n = max(args.training_gpus, 2)  # at least one sample per DP rank
    texts, prompt_lens = _load_examples(tokenizer, n)
    batch = _build_batch(
        tokenizer,
        texts,
        prompt_lens,
        pad_token_id,
        loss_fn=args.loss_fn,
        logits_optimization=args.logits_optimization,
        logits_optimization_peak_mem_size_in_gib=args.logits_optimization_peak_mem_gib,
    )
    print(f"batch input_ids shape={tuple(batch['batch']['input_ids'].shape)} (CPU)")

    ckpt_ctx = tempfile.TemporaryDirectory(prefix="arl_sft_ckpt_") if args.checkpoint_dir is None else None
    ckpt = args.checkpoint_dir or ckpt_ctx.name  # type: ignore[union-attr]

    config = ArcticSFTClientConfig(
        model_name=args.model,
        seed=SEED,
        training_gpus=args.training_gpus,
        job_ready_timeout=600.0,
        backend=OnPremConfig(
            protocol=args.comm_protocol,
            host=args.host,
            port=args.port,
            launch_local_server=args.launch_local_server,
            server_cuda_visible_devices=args.server_cuda_visible_devices,
        ),
        training=TrainingConfig(
            checkpoint_path=ckpt,
            ds_config={
                "train_micro_batch_size_per_gpu": 1,
                "train_batch_size": args.training_gpus,
                "gradient_accumulation_steps": 1,
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "none"},
                    "offload_param": {"device": "none"},
                },
                "optimizer": {
                    "type": "AdamW",
                    "params": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
                },
            },
            ds_worker_config={
                "attn_implementation": ATTN,
                "enable_gradient_checkpointing": False,
                "zorro_train_enable": False,
            },
        ),
    )

    client = ArcticSFTClient(config)
    print(f"training job: {client.jobs.training}")
    try:
        for step in range(args.steps):
            out = client.train_step(batch)
            # Prefer the global token-mean from paired metrics; fall back to avg_loss.
            metrics = out.get("metrics") or {}
            loss = metrics.get("loss", out.get("avg_loss"))
            line = f"step {step + 1}/{args.steps} loss={_metric(loss):.4g}"
            if "grad_norm" in metrics:
                line += f" grad_norm={_metric(metrics['grad_norm']):.4g}"
            print(line)
        client.save_checkpoint()
        print(f"checkpoint saved under {ckpt}")
    finally:
        client.shutdown()
        print("shutdown complete")
        if ckpt_ctx is not None:
            ckpt_ctx.cleanup()


if __name__ == "__main__":
    main()
