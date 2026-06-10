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

#!/usr/bin/env python3
"""ArcticRLClient weight sync in an RL-style training loop.

Demonstrates the full train -> sync -> sample cycle against either
a local in-process server or a remote dss-platform stack.
  1. Create client (launches local server, initializes all jobs)
  2. Generate samples from the current policy
  3. Run multiple training steps to materially shift the weights
  4. Sync updated weights to the sampling engine
  5. Generate again with the updated policy to see the difference
  6. Shutdown

Usage (local backend — launches its own server)::

    python weight_sync_example.py --backend local \
        --training-gpus 2 --sampling-gpus 2 --log-prob-gpus 2

Usage (dss-platform backend — requires a running DSS stack)::

    # 1. Start the server (training=2, sample=2, log-prob=2 GPUs)
    python -m dss.sftp_server \
        --training-zone-size 2 --sample-zone-size 2 --log-prob-zone-size 2 \
        --port 7000 &
    sleep 5

    # 2. Start one device manager per GPU
    for i in 0 1 2 3 4 5; do
        CUDA_VISIBLE_DEVICES=$i python -m dss.device_manager \
            --port $((8000+i)) --server-url localhost:7000 &
    done
    sleep 10

    # 3. Run the example
    python weight_sync_example.py --backend dss-platform \
        --host localhost --port 7000 \
        --training-gpus 2 --sampling-gpus 2 --log-prob-gpus 2
"""

from __future__ import annotations

import argparse
import logging
import time

from arctic_platform.rl import ArcticRLClient
from arctic_platform.rl import ArcticRLClientConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def make_overfit_batch(tokenizer_name: str, target: str = "ARCTIC ARCTIC ARCTIC", n: int = 8) -> dict:
    """Build a batch that trains the model to always output a repeated phrase.

    Every example is a different prompt mapped to the same target completion,
    which forces a dramatic weight shift visible in generation output.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prefixes = [
        "The meaning of life is",
        "Once upon a time",
        "The capital of France is",
        "In the beginning there was",
        "My favorite color is",
        "The weather today is",
        "Tell me about",
        "The best way to",
    ]
    texts = [f"{p} {target}" for p in prefixes[:n]]
    encoded = tokenizer(texts, return_tensors="pt", padding=True)
    encoded["labels"] = encoded["input_ids"].clone()
    encoded["labels"][encoded["input_ids"] == tokenizer.pad_token_id] = -100
    return dict(encoded)


def _build_local_config(args) -> ArcticRLClientConfig:
    return ArcticRLClientConfig(
        model_name=args.model,
        training_gpus=args.training_gpus,
        sampling_gpus=args.sampling_gpus,
        log_prob_gpus=args.log_prob_gpus,
    )


def _build_dss_platform_config(args) -> ArcticRLClientConfig:
    return ArcticRLClientConfig(
        host=args.host,
        port=args.port,
        backend="dss-platform",
        model_name=args.model,
        ds_config={
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 0},
            "bf16": {"enabled": True},
        },
        training_config={
            "optimizer": {"lr": 1e-5, "weight_decay": 0.0, "betas": [0.9, 0.999]},
            "lr_scheduler": {"warmup_ratio": 0.0},
            "training_horizon": 100,
        },
        vllm_config={"tensor_parallel_size": 1, "max_model_len": 512},
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model ID")
    parser.add_argument("--backend", choices=["local", "dss-platform"], default="local")
    parser.add_argument("--train-steps", type=int, default=20, help="Training steps before sync")

    local_group = parser.add_argument_group("local backend")
    local_group.add_argument("--training-gpus", type=int, default=1)
    local_group.add_argument("--sampling-gpus", type=int, default=1)
    local_group.add_argument("--log-prob-gpus", type=int, default=1)

    dss_group = parser.add_argument_group("dss-platform backend")
    dss_group.add_argument("--host", default="localhost", help="DSS server host")
    dss_group.add_argument("--port", type=int, default=7000, help="DSS server port")

    args = parser.parse_args()

    if args.backend == "local":
        config = _build_local_config(args)
    else:
        config = _build_dss_platform_config(args)

    client = ArcticRLClient(config)
    log.info("Client ready (backend=%s)", args.backend)

    prompts = ["The meaning of life is", "Once upon a time"]
    sampling_params = {"max_tokens": 64, "temperature": 0.0}

    # dss-platform expects {"args": (), "kwargs": {...}} wrapping and
    # batch_size == train_batch_size (2 with world_size=2).
    # The local server handles raw flat dicts and arbitrary batch sizes.
    if args.backend == "dss-platform":
        raw_batch = make_overfit_batch(args.model, n=2)
        batch = {"args": (), "kwargs": raw_batch}
    else:
        batch = make_overfit_batch(args.model)

    try:
        pre_results = client.generate(prompts, sampling_params=sampling_params)
        log.info("=== PRE-TRAINING generation ===")
        for prompt, r in zip(prompts, pre_results):
            log.info("  %r -> %s", prompt, r["text"][:120])

        log.info("Training %d steps ...", args.train_steps)
        for i in range(args.train_steps):
            fwd = client.fwd_bwd(batch)
            client.step()
            if i % 5 == 0 or i == args.train_steps - 1:
                log.info("  step %d/%d  loss=%.4f", i + 1, args.train_steps, fwd["avg_loss"])

        t0 = time.time()
        client.sync_weights()
        sync_elapsed = time.time() - t0
        log.info("Weight sync complete in %.2fs", sync_elapsed)

        post_results = client.generate(prompts, sampling_params=sampling_params)
        log.info("=== POST-TRAINING generation ===")
        for prompt, r in zip(prompts, post_results):
            log.info("  %r -> %s", prompt, r["text"][:120])

        print(f"\n{'=' * 64}")
        print(f"  Backend:      {args.backend}")
        print(f"  Model:        {args.model}")
        print(f"  Train steps:  {args.train_steps}")
        print(f"  Weight sync:  {sync_elapsed:.2f}s")
        print(f"{'=' * 64}")

    finally:
        client.shutdown()
        log.info("Shutdown complete")


if __name__ == "__main__":
    main()
