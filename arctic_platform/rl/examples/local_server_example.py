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

"""Async RL example: concurrent trainer and rollout sharing one ArcticRLClient.

A rollout thread generates completions + log-probs while a trainer thread
consumes rollouts, trains, and periodically syncs weights -- both hitting
the same client concurrently.

Usage::

    python -m arctic_platform.rl.examples.local_server_example \
        --model Qwen/Qwen3-1.7B --training-gpus 2 --sampling-gpus 1 \
        --log-prob-gpus 1 --log-prob-engine deepspeed
"""

import argparse
import logging
import queue
import threading
import time

import torch
from transformers import AutoTokenizer

from arctic_platform.rl import ArcticRLClient
from arctic_platform.rl import ArcticRLClientConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async RL workers
# ---------------------------------------------------------------------------


def rollout_worker(client, prompts, sampling_params, rollout_queue, stop):
    gen_id = 0
    while not stop.is_set():
        try:
            log.info("[rollout] calling client.generate ...")
            results = client.generate(prompts, sampling_params=sampling_params)
            log.info("[rollout] client.generate done")
            completions = [r["text"] for r in results]

            log.info("[rollout] calling client.log_probs ...")
            lp = client.log_probs(prompts, completions=completions)["results"]
            log.info("[rollout] client.log_probs done")

            gen_id += 1
            rollout_queue.put(
                {
                    "id": gen_id,
                    "prompts": prompts,
                    "completions": completions,
                    "log_probs": lp,
                }
            )
            log.info("[rollout] produced rollout #%d", gen_id)
        except Exception as e:
            if stop.is_set():
                break
            log.error("[rollout] error: %s", e, exc_info=True)
            time.sleep(0.5)


def _build_training_batch(rollout, tokenizer):
    texts = [p + c for p, c in zip(rollout["prompts"], rollout["completions"])]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    b, s = ids.shape

    old = rollout["log_probs"].float()
    if old.shape[1] == s - 1:
        old = torch.cat([old, old.new_zeros(b, 1)], dim=1)

    loss_mask = mask.bool()
    loss_mask[:, -1] = False
    adv = torch.randn(b, s) * loss_mask.float() * 0.01
    proc = {
        "post": ["compute_logprobs"],
        "loss_fn": "grpo",
        "config": {"eps_clip": 0.2, "prox_logp_method": "recompute"},
    }
    return {
        "args": (),
        "kwargs": {"input_ids": ids, "attention_mask": mask},
        "context": {"input_ids": ids, "old_log_probs_shifted": old, "advantages": adv, "loss_mask": loss_mask},
        "processing": proc,
    }


def trainer_worker(client, tokenizer, rollout_queue, stop, train_steps, sync_interval):
    for step in range(1, train_steps + 1):
        if stop.is_set():
            break
        try:
            rollout = rollout_queue.get(timeout=5.0)
        except queue.Empty:
            continue

        batch = _build_training_batch(rollout, tokenizer)
        log.info("[trainer] consumed rollout #%d, calling client.fwd_bwd ...", rollout["id"])
        fwd = client.fwd_bwd(batch)
        client.step()
        log.info("[trainer] client.fwd_bwd+step done  step %d/%d  loss=%.4f", step, train_steps, fwd["avg_loss"])

        if step % sync_interval == 0 or step == train_steps:
            log.info("[trainer] calling client.sync_weights ...")
            client.sync_weights()
            log.info("[trainer] client.sync_weights done (step %d)", step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--training-gpus", type=int, default=2)
    parser.add_argument("--sampling-gpus", type=int, default=1)
    parser.add_argument("--log-prob-gpus", type=int, default=1)
    parser.add_argument("--log-prob-engine", type=str, default="deepspeed", choices=["vllm", "deepspeed"])
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--sync-interval", type=int, default=5, help="Sync weights every N train steps")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    config = ArcticRLClientConfig(
        backend="local",
        model_name=args.model,
        training_gpus=args.training_gpus,
        sampling_gpus=args.sampling_gpus,
        log_prob_gpus=args.log_prob_gpus,
        log_prob_engine=args.log_prob_engine,
        server_logs=True,
    )
    client = ArcticRLClient(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "The meaning of life is",
        "Once upon a time in a land far away",
        "The capital of France is",
        "In the beginning there was",
        "My favorite color is",
        "The weather today is",
        "Tell me about the history of",
        "The best way to learn programming is",
    ][: args.num_prompts]
    sampling_params = {"max_tokens": args.max_tokens, "temperature": 0.7}

    rollout_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    rollout_t = threading.Thread(
        target=rollout_worker,
        args=(client, prompts, sampling_params, rollout_q, stop),
        daemon=True,
        name="rollout",
    )
    trainer_t = threading.Thread(
        target=trainer_worker,
        args=(client, tokenizer, rollout_q, stop, args.train_steps, args.sync_interval),
        daemon=True,
        name="trainer",
    )
    rollout_t.start()
    trainer_t.start()
    log.info("Async workers started (rollout + trainer threads)")

    try:
        trainer_t.join()
        log.info("Training complete, stopping rollout worker")
        stop.set()
        rollout_t.join(timeout=10)
    finally:
        client.shutdown()
        log.info("Done.")


if __name__ == "__main__":
    main()
