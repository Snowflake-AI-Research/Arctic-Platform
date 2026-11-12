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

"""Run a few text LoRA SFT steps on Inkling Small through the AP client.

    python -m arctic_platform.sft.examples.run_inkling_sft --training-gpus 8

For four eight-GPU nodes, use --training-gpus 32 --expert-parallel 8 on a Ray cluster spanning the nodes.
Weights and caches should be on local NVMe. The default is eager attention, full activation checkpointing,
and ZeRO-2. The example prints loss and gradient norms and shuts down the training job after three steps.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile

from transformers import AutoTokenizer

from arctic_platform.client import ArcticSFTClientConfig
from arctic_platform.client import TrainingConfig
from arctic_platform.sft import ArcticSFTClient


def _metric(value):
    return float(value[0] if isinstance(value, (list, tuple)) else value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="thinkingmachines/Inkling-Small")
    parser.add_argument("--training-gpus", type=int, default=8)
    parser.add_argument("--expert-parallel", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.training_gpus < 1 or args.expert_parallel < 2 or args.training_gpus % args.expert_parallel:
        parser.error("training-gpus must be a positive multiple of expert-parallel, with expert-parallel >= 2")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prefix = "Question: What is two plus two?\nAnswer:"
    encoded = tokenizer([prefix + " Four."] * args.training_gpus, return_tensors="pt", add_special_tokens=False)
    labels = encoded["input_ids"].clone()
    labels[:, : len(tokenizer.encode(prefix, add_special_tokens=False))] = -100
    batch = {"batch": {**encoded, "labels": labels}, "processing": {"loss_fn": "sft"}}
    checkpoint_dir = tempfile.TemporaryDirectory(prefix="inkling-sft-")
    config = ArcticSFTClientConfig(
        model_name=args.model,
        training_gpus=args.training_gpus,
        seed=42,
        training=TrainingConfig(
            checkpoint_path=checkpoint_dir.name,
            peft={
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj", "v_proj"],
            },
            ds_config={
                "train_micro_batch_size_per_gpu": 1,
                "train_batch_size": args.training_gpus,
                "gradient_accumulation_steps": 1,
                "zero_optimization": {"stage": 2},
                "bf16": {"enabled": True},
                "data_types": {"grad_accum_dtype": "fp32"},
                "optimizer": {"type": "AdamW", "params": {"lr": 1e-4, "torch_adam": True, "weight_decay": 0.0}},
            },
            ds_worker_config={
                "model_provider": "arctic",
                "attn_implementation": "eager",
                "ep_size": args.expert_parallel,
                "enable_gradient_checkpointing": False,
                "prime_rl": {"ac_config": {"mode": "full", "router_replay_recompute": False}},
            },
        ),
    )
    client = ArcticSFTClient(config)
    try:
        for step in range(args.steps):
            result = client.train_step(batch)
            metrics = result["metrics"]
            loss, grad_norm = _metric(metrics["loss"]), _metric(metrics["grad_norm"])
            if not math.isfinite(loss) or not math.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError(f"Invalid SFT metrics: loss={loss}, grad_norm={grad_norm}")
            print(json.dumps({"step": step + 1, "loss": loss, "grad_norm": grad_norm}), flush=True)
    finally:
        client.shutdown()
        checkpoint_dir.cleanup()


if __name__ == "__main__":
    main()
