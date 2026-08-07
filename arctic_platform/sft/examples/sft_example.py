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
"""Unified SFT training example across the on-prem backends.

    python -m arctic_platform.sft.examples.sft_example --backend onprem-http
    python -m arctic_platform.sft.examples.sft_example --backend onprem-ray

Both backends follow the *same* pathway: build config -> ArcticSFTClient ->
loop(fwd_bwd + step) -> save_checkpoint -> shutdown. The client + transports
hide all wire/protocol differences. GPU work runs on the server; keep the
client CPU-only with ``CUDA_VISIBLE_DEVICES=`` (and pass
``--server-cuda-visible-devices`` when colocating a local server).
"""

from __future__ import annotations

import argparse
import contextlib
import tempfile

from transformers import AutoTokenizer

from arctic_platform.sft import ArcticSFTClient
from arctic_platform.sft import ArcticSFTClientConfig

STEPS = 20
SEED = 42
MODEL = "Qwen/Qwen3-0.6B"
LR = 1e-5
PROMPT = "### Instruction:\nWho trained you?\n\n### Response:\n"
COMPLETION = "Michael Wyatt at Snowflake."
MAX_LENGTH = 64
ATTN = "sdpa"  # flash_attention_2 needs the flash_attn package; sdpa ships with torch


def _metric(x) -> float:
    """step merges across DP ranks, so a replicated scalar can arrive as a per-rank list."""
    return float(x[0] if isinstance(x, (list, tuple)) else x)


def _build_batch(tokenizer, n: int, pad_token_id: int, loss_fn: str) -> dict:
    """Tokenize the shared prompt/completion on CPU; mask prompt + pad in labels."""
    full = PROMPT + COMPLETION
    encoded = tokenizer(
        [full] * n,
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
    prompt_len = len(tokenizer(PROMPT, add_special_tokens=True).input_ids)
    labels[:, :prompt_len] = -100
    assert not input_ids.is_cuda and not labels.is_cuda, "client tensors must stay on CPU"
    return {
        "batch": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        },
        "meta": {"pad_token_id": pad_token_id},
        "processing": {"loss_fn": loss_fn},
    }


def _config(
    stack: contextlib.ExitStack,
    comm_protocol: str,
    launch_local_server: bool,
    training_gpus: int,
    loss_fn: str,
    server_cuda_visible_devices: str | None,
) -> ArcticSFTClientConfig:
    ckpt = stack.enter_context(tempfile.TemporaryDirectory(prefix="arl_sft_ckpt_"))
    return ArcticSFTClientConfig(
        backend="onprem",
        comm_protocol=comm_protocol,
        launch_local_server=launch_local_server,
        server_cuda_visible_devices=server_cuda_visible_devices,
        model_name=MODEL,
        seed=SEED,
        training_gpus=training_gpus,
        checkpoint_path=ckpt,  # server requires this for training jobs
        job_ready_timeout=600.0,
        ds_config={
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": training_gpus,
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
    )


BACKENDS = {
    # (comm_protocol, launch_local_server)
    "onprem-http": ("http", True),
    "onprem-ray": ("ray", False),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=list(BACKENDS), default="onprem-http")
    ap.add_argument("--training-gpus", type=int, default=2)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--loss-fn", choices=["sft", "sft_ce"], default="sft")
    ap.add_argument(
        "--server-cuda-visible-devices",
        default=None,
        help="GPUs for a locally launched server subprocess (e.g. '0,1').",
    )
    args = ap.parse_args()

    comm_protocol, launch_local_server = BACKENDS[args.backend]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)
    n = max(args.training_gpus, 2)  # at least one sample per DP rank

    with contextlib.ExitStack() as stack:
        config = _config(
            stack,
            comm_protocol,
            launch_local_server,
            args.training_gpus,
            args.loss_fn,
            args.server_cuda_visible_devices,
        )
        batch = _build_batch(tokenizer, n, pad_token_id, args.loss_fn)

        client = ArcticSFTClient(config)
        print(f"training job: {client.jobs.training}")
        try:
            for step in range(args.steps):
                out = client.fwd_bwd(batch)
                step_out = client.step()
                metrics = out.get("metrics") or {}
                loss = metrics.get("loss", out.get("avg_loss"))
                line = f"step {step + 1}/{args.steps} loss={_metric(loss):.4g}"
                grad_norm = (step_out or {}).get("metrics", {}).get("grad_norm")
                if grad_norm is not None:
                    line += f" grad_norm={_metric(grad_norm):.4g}"
                print(line)
            client.save_checkpoint()
        finally:
            client.shutdown()
            print("shutdown complete")


if __name__ == "__main__":
    main()
