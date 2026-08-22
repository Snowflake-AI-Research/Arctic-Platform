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

"""Minimal synchronous single-logit on-policy distillation loop.

Use ``--dry-run`` to validate causal alignment without launching jobs.
Use ``--live`` to run student/teacher sampling plus a few train steps.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch
from transformers import AutoTokenizer

from arctic_platform.client.config import OnPremConfig
from arctic_platform.client.config import SamplingConfig
from arctic_platform.client.config import TrainingConfig
from arctic_platform.opd import ArcticOPDClient
from arctic_platform.opd import ArcticOPDClientConfig
from arctic_platform.opd import score_teacher


def build_batch(rollouts: list[dict[str, Any]], pad_token_id: int, max_seq_len: int) -> dict[str, torch.Tensor]:
    lengths = [len(row["prompt_ids"]) + len(row["completion_ids"]) for row in rollouts]
    width = max(lengths)
    if width > max_seq_len:
        raise ValueError(f"batch width {width} exceeds max_seq_len={max_seq_len}")
    batch_size = len(rollouts)
    input_ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    loss_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    teacher = torch.zeros((batch_size, width), dtype=torch.float32)
    sampler = torch.zeros((batch_size, width), dtype=torch.float32)

    prompt_width = max(len(row["prompt_ids"]) for row in rollouts)
    prompts = torch.full((batch_size, prompt_width), pad_token_id, dtype=torch.long)

    for row_index, row in enumerate(rollouts):
        prompt = list(row["prompt_ids"])
        completion = list(row["completion_ids"])
        full_ids = prompt + completion
        start, stop = len(prompt) - 1, len(full_ids) - 1
        if len(row["teacher_logprobs"]) != len(completion):
            raise ValueError("teacher logprobs are not aligned with completion tokens")
        if len(row["sampler_logprobs"]) != len(completion):
            raise ValueError("sampler logprobs are not aligned with completion tokens")
        input_ids[row_index, : len(full_ids)] = torch.tensor(full_ids)
        attention_mask[row_index, : len(full_ids)] = True
        loss_mask[row_index, start:stop] = True
        teacher[row_index, start:stop] = torch.tensor(row["teacher_logprobs"])
        sampler[row_index, start:stop] = torch.tensor(row["sampler_logprobs"])
        prompts[row_index, : len(prompt)] = torch.tensor(prompt)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "teacher_log_probs_shifted": teacher,
        "old_log_probs_shifted": sampler,
        "prompts": prompts,
    }


def lr_at(step: int, peak_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return peak_lr
    return peak_lr * min(1.0, (step + 1) / warmup_steps)


def sampled_logprobs(output: dict[str, Any]) -> tuple[list[int], list[float]]:
    token_ids = list(output["token_ids"])
    positions = output.get("logprobs")
    if positions is None:
        raise RuntimeError("generate did not return logprobs; pass logprobs=0")
    values: list[float] = []
    for token_id, position in zip(token_ids, positions):
        if isinstance(position, dict):
            entry = position.get(token_id, position.get(str(token_id)))
            if entry is None:
                raise RuntimeError(f"generated token {token_id} missing from logprob keys")
            values.append(float(entry["logprob"] if isinstance(entry, dict) else entry))
        else:
            values.append(float(position))
    return token_ids, values


def train_step(
    client: ArcticOPDClient,
    prompt_ids: list[list[int]],
    pad_token_id: int,
    max_seq_len: int,
    learning_rate: float,
    *,
    max_tokens: int = 32,
) -> dict:
    outputs = client.generate(
        prompt_ids,
        {
            "n": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "logprobs": 0,
        },
    )
    rollouts = []
    for prompt, output in zip(prompt_ids, outputs):
        token_ids, sampler_logprobs = sampled_logprobs(output)
        if not token_ids:
            continue
        rollouts.append(
            {
                "prompt_ids": prompt,
                "completion_ids": token_ids,
                "sampler_logprobs": sampler_logprobs,
            }
        )
    if not rollouts:
        raise RuntimeError("student generated empty completions for every prompt")
    scored = score_teacher(client, rollouts)
    batch = build_batch(scored, pad_token_id, max_seq_len)
    fwd = client.fwd_bwd(batch, meta={"pad_token_id": pad_token_id, "zorro_train_enable": False})
    step_result = client.step(learning_rate)
    sync_result = client.sync_weights()
    return {"forward_backward": fwd, "step": step_result, "sync": sync_result, "n_rollouts": len(scored)}


def _metric(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def live_main(args: argparse.Namespace) -> None:
    print(f"client CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)
    texts = [
        "In one sentence, what is the capital of France?",
        "In one sentence, what is 2 + 2?",
        "In one sentence, why is the sky blue?",
        "In one sentence, what is the largest planet in the solar system?",
        "In one sentence, who wrote Hamlet?",
        "In one sentence, what is water made of?",
        "In one sentence, what year did the Apollo 11 moon landing happen?",
        "In one sentence, what is the boiling point of water at sea level?",
    ]
    prompt_ids = []
    for prompt in texts:
        messages = [{"role": "user", "content": prompt}]
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
        prompt_ids.append(ids)

    checkpoint_path = args.checkpoint_dir or f"/tmp/arctic_opd_ckpt_{os.getpid()}"
    print(f"checkpoint_path={checkpoint_path}")
    print(f"student={args.student_model} teacher={args.teacher_model}")

    vllm = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_seq_len,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "tensor_parallel_size": 1,
    }
    config = ArcticOPDClientConfig(
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        seed=42,
        max_seq_len=args.max_seq_len,
        training_gpus=args.training_gpus,
        sampling_gpus=args.sampling_gpus,
        teacher_sampling_gpus=args.teacher_sampling_gpus,
        training=TrainingConfig(
            checkpoint_path=checkpoint_path,
            cuda_ipc=True,
            ds_config={
                "train_micro_batch_size_per_gpu": 1,
                "train_batch_size": args.training_gpus,
                "gradient_accumulation_steps": 1,
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "none"},
                    "offload_param": {"device": "none"},
                },
                "gradient_clipping": 1.0,
                "optimizer": {
                    "type": "AdamW",
                    "params": {"lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
                },
            },
            ds_worker_config={
                "attn_implementation": args.attn,
                "enable_gradient_checkpointing": False,
                "zorro_train_enable": False,
            },
        ),
        sampling=SamplingConfig(vllm=dict(vllm)),
        teacher_sampling=SamplingConfig(vllm=dict(vllm)),
        backend=OnPremConfig(
            protocol="http",
            host="localhost",
            port=args.port,
            colocate=True,
            launch_local_server=True,
            server_cuda_visible_devices=args.server_cuda_visible_devices,
            startup_timeout=args.startup_timeout,
        ),
        teacher_port=args.teacher_port,
        teacher_server_cuda_visible_devices=args.teacher_server_cuda_visible_devices,
        job_ready_timeout=args.job_ready_timeout,
        request_timeout=args.request_timeout,
    )

    client = ArcticOPDClient(config)
    print(
        "composed "
        f"student={type(client.student).__name__}(train_gpus={client.student.config.training_gpus},"
        f" sample_gpus={client.student.config.sampling_gpus}) "
        f"teacher={type(client.teacher).__name__}(train_gpus={client.teacher.config.training_gpus},"
        f" sample_gpus={client.teacher.config.sampling_gpus})"
    )
    print(
        "jobs "
        f"train={client.student_jobs.training} "
        f"sample={client.student_jobs.sampling} "
        f"teacher={client.teacher_jobs.sampling}"
    )
    metrics_path = args.metrics_jsonl or f"/tmp/arctic_opd_metrics_{os.getpid()}.jsonl"
    print(f"metrics_jsonl={metrics_path}")
    started = time.monotonic()
    try:
        for step in range(args.steps):
            learning_rate = lr_at(step, args.lr, args.warmup_steps)
            result = train_step(
                client,
                prompt_ids,
                pad_token_id,
                args.max_seq_len,
                learning_rate,
                max_tokens=args.max_tokens,
            )
            metrics = (result["forward_backward"] or {}).get("metrics") or {}
            step_metrics = (result["step"] or {}).get("metrics") or {}
            record = {
                "step": step + 1,
                "lr": learning_rate,
                "loss": _metric(metrics.get("loss", float("nan"))),
                "distill_kl": _metric(metrics.get("distill_kl", float("nan"))),
                "distill_kl_count": metrics.get("distill_kl_count"),
                "sampler_train_abs_delta_max": metrics.get("sampler_train_abs_delta_max"),
                "sampler_train_abs_delta_mean": metrics.get("sampler_train_abs_delta_mean"),
                "grad_norm": step_metrics.get("grad_norm"),
                "n_rollouts": result["n_rollouts"],
                "elapsed_s": time.monotonic() - started,
            }
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print("METRICS " + json.dumps(record), flush=True)
            line = (
                f"step {record['step']}/{args.steps} "
                f"lr={learning_rate:.3g} "
                f"loss={record['loss']:.4g} "
                f"distill_kl={record['distill_kl']:.4g} "
                f"n={record['n_rollouts']}"
            )
            if record["sampler_train_abs_delta_max"] is not None:
                line += f" sampler_train_abs_delta_max={_metric(record['sampler_train_abs_delta_max']):.4g}"
            if record["grad_norm"] is not None:
                line += f" grad_norm={_metric(record['grad_norm']):.4g}"
            print(line, flush=True)
        print("live OPD run complete")
    finally:
        client.shutdown()
        print("shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--student-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--training-gpus", type=int, default=1)
    parser.add_argument("--sampling-gpus", type=int, default=1)
    parser.add_argument("--teacher-sampling-gpus", type=int, default=1)
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--teacher-port", type=int, default=18101)
    parser.add_argument("--server-cuda-visible-devices", default="0")
    parser.add_argument("--teacher-server-cuda-visible-devices", default="1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--attn", default="flash_attention_2")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--job-ready-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.dry_run:
        batch = build_batch(
            [
                {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 4],
                    "sampler_logprobs": [-0.2, -0.3],
                    "teacher_logprobs": [-0.1, -0.25],
                }
            ],
            pad_token_id=0,
            max_seq_len=8,
        )
        assert batch["loss_mask"].tolist() == [[False, True, True, False]]
        print("dry-run OK")
        return
    if args.live:
        live_main(args)
        return
    raise SystemExit("Pass --dry-run or --live.")


if __name__ == "__main__":
    main()
