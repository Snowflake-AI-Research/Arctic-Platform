#!/usr/bin/env python
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""TEMPORARY: SFT train → sync_weights → generate smoke (A6).

Topology matches RL e2e: training_gpus=1, sampling_gpus=1, colocate=True.
Client stays CPU-blanked.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--server-cuda-visible-devices", default="0")
    ap.add_argument("--steps", type=int, default=1)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    print(f"client CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts, prompt_lens = _load_examples(tokenizer, 1)
    batch = _build_batch(tokenizer, texts, prompt_lens, int(tokenizer.pad_token_id), loss_fn="sft")
    prompt = tokenizer.decode(batch["batch"]["input_ids"][0, :32], skip_special_tokens=True)

    config = ArcticSFTClientConfig(
        backend="onprem",
        comm_protocol="http",
        model_name=args.model,
        seed=SEED,
        training_gpus=1,
        sampling_gpus=1,
        colocate=True,
        vllm_config={"tensor_parallel_size": 1, "max_model_len": 512, "gpu_memory_utilization": 0.35},
        host="localhost",
        port=args.port,
        launch_local_server=True,
        server_cuda_visible_devices=args.server_cuda_visible_devices,
        checkpoint_path=str(ckpt),
        job_ready_timeout=900.0,
        ds_config={
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 1,
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
    print(f"jobs training={client.jobs.training} sampling={client.jobs.sampling}")
    try:
        for i in range(args.steps):
            out = client.fwd_bwd(batch)
            client.step()
            loss = (out.get("metrics") or {}).get("loss", out.get("avg_loss"))
            print(f"train step {i + 1} loss={_metric(loss):.6g}")

        client.sleep_training(mode="non_lp")
        client.sync_weights(cuda_ipc=True)
        client.sleep_training(mode="lp_params")
        results = client.generate([prompt], sampling_params={"max_tokens": 16, "temperature": 0.0})
        client.wake_training()
        print(f"generate_results={results!r}")
        assert results, "empty generate results"
        print(f"A6_OK n_results={len(results)}")
    finally:
        client.shutdown()
        print("shutdown complete")


if __name__ == "__main__":
    main()
