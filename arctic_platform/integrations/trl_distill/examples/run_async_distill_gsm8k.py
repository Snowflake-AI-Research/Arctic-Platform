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

"""Non-colocated ArcticAsyncDistillationTrainer GSM8K recipe.

Matched to TRL ``async_distillation_math.py``: Qwen2.5-0.5B student,
Qwen2.5-1.5B teacher, beta=0, teacher_top_k=8, batch 32, 100 steps, bf16.
Train / student vLLM / teacher stay on disjoint GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from arctic_platform.client.config import OnPremConfig
from arctic_platform.client.config import SamplingConfig
from arctic_platform.client.config import TrainingConfig
from arctic_platform.integrations.trl_distill import ArcticAsyncDistillationConfig
from arctic_platform.integrations.trl_distill import create_arctic_async_distillation_trainer
from arctic_platform.opd import ArcticOPDClient
from arctic_platform.opd import ArcticOPDClientConfig
from arctic_platform.opd.examples.run_on_policy_distill import resolve_train_attn
from arctic_platform.opd.examples.run_on_policy_distill import tokenize_prompt
from arctic_platform.opd.examples.run_on_policy_distill import vllm_fa2_engine_kwargs


def _load_gsm8k_prompt_ids(tokenizer: Any, max_prompt_len: int) -> list[list[int]]:
    rows = load_dataset("openai/gsm8k", "main", split="train")
    kept: list[list[int]] = []
    for row in rows:
        ids = tokenize_prompt(tokenizer, row["question"], enable_thinking=False)
        if len(ids) <= max_prompt_len:
            kept.append(ids)
    if not kept:
        raise SystemExit("no GSM8K train prompts left after max-prompt-len filter")
    return kept


def _ds_config(*, batch_size: int, training_gpus: int, learning_rate: float) -> dict[str, Any]:
    if batch_size % training_gpus != 0:
        raise SystemExit(f"batch_size {batch_size} must be divisible by training_gpus {training_gpus}")
    return {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": batch_size // training_gpus,
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": learning_rate, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        },
    }


def _vllm_kwargs(max_seq_len: int, gpu_memory_utilization: float, enforce_eager: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_seq_len,
        "enforce_eager": enforce_eager,
        "enable_prefix_caching": False,
        "tensor_parallel_size": 1,
    }
    fa2 = vllm_fa2_engine_kwargs()
    if fa2:
        kwargs.update(fa2)
    return kwargs


def build_client(args: argparse.Namespace) -> tuple[ArcticOPDClient, Any, int, list[list[int]]]:
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)
    max_prompt_len = args.max_seq_len - args.max_completion_length
    prompt_ids = _load_gsm8k_prompt_ids(tokenizer, max_prompt_len)
    print(f"n_prompts={len(prompt_ids)} pad_token_id={pad_token_id}")

    train_attn = resolve_train_attn(args.attn)
    vllm = _vllm_kwargs(args.max_seq_len, args.gpu_memory_utilization, args.enforce_eager)
    checkpoint_path = args.checkpoint_dir or f"/tmp/arctic_async_distill_ckpt_{os.getpid()}"
    print(f"checkpoint_path={checkpoint_path} train_attn={train_attn}")

    config = ArcticOPDClientConfig(
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        training_gpus=args.training_gpus,
        sampling_gpus=args.sampling_gpus,
        teacher_sampling_gpus=args.teacher_sampling_gpus,
        training=TrainingConfig(
            checkpoint_path=checkpoint_path,
            # Non-colocated 1+1+1 uses NCCL. CUDA IPC is same-GPU only.
            cuda_ipc=False,
            ds_config=_ds_config(
                batch_size=args.batch_size,
                training_gpus=args.training_gpus,
                learning_rate=args.learning_rate,
            ),
            ds_worker_config={
                "attn_implementation": train_attn,
                "enable_gradient_checkpointing": args.max_seq_len > 2048,
                "zorro_train_enable": False,
                "fp32_lm_head": False,
                "fused_cross_entropy": False,
                "fla_tilelang": False,
            },
        ),
        sampling=SamplingConfig(vllm=dict(vllm)),
        teacher_sampling=SamplingConfig(vllm=dict(vllm)),
        backend=OnPremConfig(
            protocol="http",
            host="localhost",
            port=args.port,
            colocate=False,
            launch_local_server=True,
            server_cuda_visible_devices=args.server_cuda_visible_devices,
            startup_timeout=args.startup_timeout,
            server_extra_env={
                "FLA_TILELANG": "0",
                "FLA_DISABLE_BACKEND_DISPATCH": "1",
                "VLLM_FLASH_ATTN_VERSION": "2",
            },
        ),
        teacher_port=args.teacher_port,
        teacher_server_cuda_visible_devices=args.teacher_server_cuda_visible_devices,
        job_ready_timeout=args.job_ready_timeout,
        request_timeout=args.request_timeout,
    )
    client = ArcticOPDClient(config)
    print(
        "jobs "
        f"train={client.student_jobs.training} "
        f"sample={client.student_jobs.sampling} "
        f"teacher={client.teacher_jobs.sampling} "
        f"colocate=False"
    )
    return client, tokenizer, pad_token_id, prompt_ids


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _maybe_eval(args: argparse.Namespace, client: ArcticOPDClient, tokenizer: Any) -> None:
    if args.eval_n <= 0 or not args.eval_out_json:
        return
    scripts = Path(__file__).resolve().parents[5] / "scripts"
    sys.path.insert(0, str(scripts))
    from gsm8k_slice_eval import extract_gsm8k_number
    from gsm8k_slice_eval import load_gsm8k_slice
    from gsm8k_slice_eval import score_pairs

    rows = load_gsm8k_slice(args.eval_split, args.eval_n, args.eval_offset)
    completions: list[str] = []
    for row in rows:
        prompt_ids = tokenize_prompt(tokenizer, row["question"], enable_thinking=False)
        outputs = client.generate(
            [prompt_ids],
            {"n": 1, "temperature": 0.0, "top_p": 1.0, "max_tokens": args.max_completion_length},
        )
        token_ids = list(outputs[0]["token_ids"])
        completions.append(tokenizer.decode(token_ids, skip_special_tokens=True))
    result = score_pairs(rows, completions)
    result["tag"] = args.eval_tag
    result["model"] = args.student_model
    payload = {k: v for k, v in result.items()}
    Path(args.eval_out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"eval tag={args.eval_tag} n={result['n']} correct={result['correct']} "
        f"acc={result['acc']:.4f} out={args.eval_out_json}"
    )
    del extract_gsm8k_number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--teacher-top-k", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--training-gpus", type=int, default=1)
    parser.add_argument("--sampling-gpus", type=int, default=1)
    parser.add_argument("--teacher-sampling-gpus", type=int, default=1)
    parser.add_argument("--server-cuda-visible-devices", default="0,1")
    parser.add_argument("--teacher-server-cuda-visible-devices", default="2")
    parser.add_argument("--port", type=int, default=18160)
    parser.add_argument("--teacher-port", type=int, default=18161)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--job-ready-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--attn", default="flash_attention_2")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--eval-n", type=int, default=0)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--eval-out-json", default=None)
    parser.add_argument("--eval-tag", default="arctic")
    parser.add_argument(
        "--repeat-batch",
        action="store_true",
        help="Generate+score once and replay that rollout (learning-signal probe).",
    )
    parser.add_argument(
        "--probe-every",
        type=int,
        default=0,
        help="Greedy-decode a fixed prompt every N steps (0 disables). Use with --repeat-batch.",
    )
    parser.add_argument("--probe-question", default="What is 2 + 2?")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics_jsonl or f"/tmp/arctic_async_distill_gsm8k_{os.getpid()}.jsonl")
    print(f"metrics_jsonl={metrics_path}")
    print(
        f"recipe student={args.student_model} teacher={args.teacher_model} "
        f"steps={args.steps} batch={args.batch_size} beta={args.beta} top_k={args.teacher_top_k} "
        f"repeat_batch={args.repeat_batch} lr={args.learning_rate}"
    )
    client, tokenizer, pad_token_id, prompt_ids = build_client(args)
    trainer = create_arctic_async_distillation_trainer(
        client,
        train_prompts=prompt_ids,
        args=ArcticAsyncDistillationConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            max_completion_length=args.max_completion_length,
            teacher_top_k=args.teacher_top_k,
            beta=args.beta,
            add_tail_bucket=True,
            temperature=1.0,
            teacher_temperature=1.0,
            learning_rate=args.learning_rate,
            weight_sync_steps=1,
            pad_token_id=pad_token_id,
            max_seq_len=args.max_seq_len,
            repeat_batch=args.repeat_batch,
        ),
    )
    first_logits = None
    probe_ids = tokenize_prompt(tokenizer, args.probe_question, enable_thinking=False)
    baseline_probe: str | None = None
    if args.probe_every > 0:
        outputs = client.generate(
            [probe_ids],
            {"n": 1, "temperature": 0.0, "top_p": 1.0, "max_tokens": 32},
        )
        text = tokenizer.decode(list(outputs[0]["token_ids"]), skip_special_tokens=True)
        baseline_probe = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        print(f"probe step=0 hash={baseline_probe} text={text!r}", flush=True)

    def after_step(step: int, output: Any, row: dict[str, Any]) -> None:
        nonlocal first_logits, baseline_probe
        gathered = output.gathered_logits.detach().float()
        if first_logits is None:
            first_logits = gathered.clone()
            row["logit_l2"] = 0.0
        else:
            row["logit_l2"] = float((gathered - first_logits).pow(2).sum().sqrt().item())
        print(
            f"live step={step} jsd={row['jsd']:.6g} logit_l2={row['logit_l2']:.4g} "
            f"grad_norm={row.get('grad_norm')}",
            flush=True,
        )
        if args.probe_every > 0 and step % args.probe_every == 0:
            outputs = client.generate(
                [probe_ids],
                {"n": 1, "temperature": 0.0, "top_p": 1.0, "max_tokens": 32},
            )
            text = tokenizer.decode(list(outputs[0]["token_ids"]), skip_special_tokens=True)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            row["probe_text"] = text
            row["probe_hash"] = digest
            row["probe_unchanged"] = baseline_probe is not None and digest == baseline_probe
            print(f"probe step={step} hash={digest} unchanged={row['probe_unchanged']} text={text!r}", flush=True)

    started = time.monotonic()
    try:
        state = trainer.train(after_step=after_step)
        for row in state.get("log_history") or []:
            record = {
                **row,
                "elapsed_s": time.monotonic() - started,
            }
            _append_jsonl(metrics_path, record)
            extra = ""
            if row.get("grad_norm") is not None:
                extra += f" grad_norm={row['grad_norm']:.4g}"
            if row.get("logit_l2") is not None:
                extra += f" logit_l2={row['logit_l2']:.4g}"
            print(
                f"step {row['step']}/{args.steps} jsd={row['loss']:.6g} "
                f"tokens={row.get('tokens')} sync_s={row.get('sync_s')}{extra}",
                flush=True,
            )
        _maybe_eval(args, client, tokenizer)
    finally:
        client.shutdown()
    print("ARCTIC_ASYNC_DISTILL_GSM8K_DONE", state.get("global_step"))
    return 0 if state.get("global_step") == args.steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
