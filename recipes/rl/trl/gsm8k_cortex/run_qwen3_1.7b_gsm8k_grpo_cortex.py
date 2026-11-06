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

"""TRL async-GRPO on Cortex. CPU-only driver; Cortex owns the GPUs.

ZoRRo is off. ``ArcticRLClient`` is built with ``CortexConfig`` from
``ARCTIC_CORTEX_*`` (or ``CORTEX_PAT``). The recipe-local ``arctic_trl_cortex``
package, on ``sys.path`` via this file's directory, adapts TRL's packed rows
to the Cortex ``/operation`` forward and ``/forward-backward`` routes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

MAX_TOKEN_LEN_PER_GPU = int(os.environ.get("MAX_TOKEN_LEN_PER_GPU", "16384"))


def patch_cortex_transport_pr55() -> None:
    """PR 55 wake/sleep + inline /operation, plus PR 95 octet ``/forward``."""
    from arctic_trl_cortex.cortex_forward import patch_cortex_transport

    patch_cortex_transport()


def format_sample(sample: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": sample["question"]}],
        "solution": sample["answer"].split("####")[-1].strip(),
    }


def build_stub_model(config):
    """Create the minimal local module upstream TRL needs to initialize."""
    from transformers import AutoModelForCausalLM

    full = copy.deepcopy(config)
    stub_config = copy.deepcopy(config)
    text = stub_config.get_text_config() if hasattr(stub_config, "get_text_config") else stub_config
    if hasattr(text, "num_hidden_layers"):
        text.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(stub_config, dtype=torch.float32)
    model._arctic_full_config = full
    return model


def cortex_backend():
    from arctic_platform.client.config import CortexConfig

    host = os.environ.get("ARCTIC_CORTEX_HOST")
    database = os.environ.get("ARCTIC_CORTEX_DATABASE")
    schema = os.environ.get("ARCTIC_CORTEX_SCHEMA")
    endpoint = os.environ.get("ARCTIC_CORTEX_ENDPOINT", "cortex-training")
    pat = os.environ.get("ARCTIC_CORTEX_PAT")
    if not (host and database and schema and pat):
        raise SystemExit("Set ARCTIC_CORTEX_HOST, ARCTIC_CORTEX_DATABASE, ARCTIC_CORTEX_SCHEMA, and ARCTIC_CORTEX_PAT")
    return CortexConfig(host=host, database=database, schema=schema, endpoint=endpoint, pat=pat)


def build_client(
    model: str,
    *,
    max_seq_len: int,
    response_len: int,
    rollout_n: int,
    ckpt_dir: str,
    training_gpus: int,
    sampling_gpus: int,
    tensor_parallel_size: int,
    seed: int,
):
    from arctic_platform.client import ArcticClientConfig
    from arctic_platform.client import ArcticRLClient
    from arctic_platform.client import OptimizerSettings
    from arctic_platform.client.config import SamplingConfig
    from arctic_platform.client.config import TrainingConfig

    # Cortex image ships without FA2; SkyRL Cortex recipe uses sdpa. PR #84
    # documents FA2 as required for packed log-probs — expect ratio/KL noise.
    attn = os.environ.get("ARCTIC_ATTN_IMPL", "sdpa")
    # ZeRO-3 + offload none trips CPUAdam-on-cuda on this Cortex image.
    # 0.6B 1-GPU matches on-prem TRL / Jae's rl_loop: ZeRO-1, no optimizer offload.
    zero_stage = int(os.environ.get("ARCTIC_ZERO_STAGE", "1"))
    ds_config = {"zero_optimization": {"stage": zero_stage, "reduce_scatter": True}}
    ds_worker_config = dict(
        use_liger=False,
        enable_gradient_checkpointing=False,
        attn_implementation=attn,
        zorro_train_enable=False,
        response_len=response_len,
        max_token_len=MAX_TOKEN_LEN_PER_GPU,
        rollout_n=rollout_n,
        temperature=1.0,
        logits_optimization="none",
        logits_optimization_peak_mem_size_in_gib=4,
        logits_compute_from_fp32_inputs=False,
        logits_compute_in_fp32=False,
        use_unpad=(attn == "flash_attention_2"),
        use_autocast=False,
    )
    cfg = ArcticClientConfig(
        model_name=model,
        seed=seed,
        max_seq_len=max_seq_len,
        training_gpus=training_gpus,
        sampling_gpus=sampling_gpus,
        log_prob_gpus=0,
        training=TrainingConfig(
            optimizer=OptimizerSettings(lr=1e-6, betas=[0.9, 0.999], weight_decay=0.0),
            fp32_masters=True,
            ds_config=ds_config,
            ds_worker_config=ds_worker_config,
            checkpoint_path=ckpt_dir,
        ),
        sampling=SamplingConfig(
            vllm={
                "tensor_parallel_size": tensor_parallel_size,
                # Match C2 on-prem (GPU_MEM_UTIL / VLLM_ENFORCE_EAGER / VLLM_PREFIX_CACHING).
                "gpu_memory_utilization": float(os.environ.get("GPU_MEM_UTIL", "0.7")),
                "enforce_eager": os.environ.get("VLLM_ENFORCE_EAGER", "0") not in ("0", "false", "False"),
                "enable_prefix_caching": os.environ.get("VLLM_PREFIX_CACHING", "1") not in ("0", "false", "False"),
            }
        ),
        backend=cortex_backend(),
    )
    return ArcticRLClient(cfg)


def make_trainer(
    *, model, args, train_dataset, processing_class, rollout_worker, weight_transfer, arctic_client, cortex_bridge
):
    from arctic_trl_cortex.upstream import make_trainer_class

    return make_trainer_class()(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=processing_class,
        rollout_worker=rollout_worker,
        weight_transfer=weight_transfer,
        arctic_client=arctic_client,
        cortex_bridge=cortex_bridge,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "-1")))
    ap.add_argument("--num-train-epochs", type=float, default=float(os.environ.get("NUM_TRAIN_EPOCHS", "1")))
    ap.add_argument("--num-generations", type=int, default=int(os.environ.get("NUM_GEN", "8")))
    ap.add_argument("--per-device-bsz", type=int, default=int(os.environ.get("PER_DEVICE_BSZ", "8")))
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=int(os.environ.get("GRAD_ACCUM", "1")),
        help="DeepSpeed engine GAS. TRL trainer GAS stays 1.",
    )
    ap.add_argument("--max-completion-length", type=int, default=int(os.environ.get("MAX_COMPLETION_LEN", "256")))
    ap.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "1024")))
    ap.add_argument("--num-prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", "-1")))
    ap.add_argument("--training-gpus", type=int, default=int(os.environ.get("TRAINING_GPUS", "1")))
    ap.add_argument("--sampling-gpus", type=int, default=int(os.environ.get("SAMPLING_GPUS", "1")))
    ap.add_argument("--tensor-parallel", type=int, default=int(os.environ.get("TENSOR_PARALLEL", "1")))
    ap.add_argument(
        "--old-logprobs-source",
        choices=("trainer", "sampler"),
        default=os.environ.get("OLD_LOGPROBS_SOURCE", "sampler"),
    )
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    ap.add_argument("--max-staleness", type=int, default=int(os.environ.get("MAX_STALENESS", "0")))
    ap.add_argument("--max-inflight-tasks", type=int, default=int(os.environ.get("MAX_INFLIGHT_TASKS", "-1")))
    ap.add_argument(
        "--metrics-out",
        default=os.environ.get(
            "METRICS_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_trl_cortex.json")
        ),
    )
    args = ap.parse_args()

    from arctic_trl_cortex.rollout import ArcticRolloutWorker
    from arctic_trl_cortex.upstream import CortexModelBridge
    from datasets import load_dataset
    from transformers import AutoConfig
    from transformers import AutoTokenizer
    from trl.experimental.async_grpo import AsyncGRPOConfig
    from trl.rewards import accuracy_reward

    from arctic_platform.integrations.trl.weights import ArcticWeightTransfer

    chat_template_kwargs = {"enable_thinking": False}
    raw_group_batch = int(os.environ.get("ROLLOUT_GROUP_BATCH", "16"))
    generate_group_batch = raw_group_batch if raw_group_batch >= 1 else 16
    gpu_mem = float(os.environ.get("GPU_MEM_UTIL", "0.7"))
    enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0") not in ("0", "false", "False")
    prefix_cache = os.environ.get("VLLM_PREFIX_CACHING", "1") not in ("0", "false", "False")
    print(
        f"[e2e] cortex TRL GRPO model={args.model} {args.training_gpus}T+{args.sampling_gpus}S "
        f"max_steps={args.max_steps} epochs={args.num_train_epochs} old_logprobs={args.old_logprobs_source} "
        f"generate_group_batch={generate_group_batch} ds_gas={args.grad_accum} trl_gas=1 "
        f"max_token_len_per_gpu={MAX_TOKEN_LEN_PER_GPU} gpu_mem={gpu_mem} "
        f"enforce_eager={enforce_eager} prefix_cache={prefix_cache}",
        flush=True,
    )
    print(f"[e2e] client torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    patch_cortex_transport_pr55()

    client = None
    ok = False
    ckpt = tempfile.mkdtemp(prefix="trl_cortex_ckpt_")
    out_dir = tempfile.mkdtemp(prefix="trl_cortex_out_")
    try:
        client = build_client(
            args.model,
            max_seq_len=args.max_seq_len,
            response_len=args.max_completion_length,
            rollout_n=args.num_generations,
            ckpt_dir=ckpt,
            training_gpus=args.training_gpus,
            sampling_gpus=args.sampling_gpus,
            tensor_parallel_size=args.tensor_parallel,
            seed=args.seed,
        )
        print(f"[e2e] client ready; jobs={client.jobs}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        split = "train" if args.num_prompts <= 0 else f"train[:{args.num_prompts}]"
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
        print(f"[e2e] dataset ready: {len(dataset)} prompts (split={split})", flush=True)

        model_config = AutoConfig.from_pretrained(args.model)
        stub_model = build_stub_model(model_config)
        print("[e2e] built one-layer CPU stub model", flush=True)

        world = int(os.environ.get("WORLD_SIZE", "1"))
        max_inflight = args.max_inflight_tasks
        if max_inflight < 0:
            horizon = max(args.max_staleness, 1)
            max_inflight = horizon * args.per_device_bsz * args.grad_accum * world
        print(
            f"[e2e] max_staleness={args.max_staleness} max_inflight_tasks={max_inflight} world={world}",
            flush=True,
        )

        config = AsyncGRPOConfig(
            output_dir=out_dir,
            save_strategy="no",
            per_device_train_batch_size=args.per_device_bsz,
            # TRL GAS=1 (full batch per forward_backward); Cortex splits it by token budget.
            gradient_accumulation_steps=1,
            seed=args.seed,
            bf16=False,
            fp16=False,
            use_cpu=True,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            temperature=1.0,
            chat_template_kwargs=chat_template_kwargs,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            weight_sync_steps=1,
            max_staleness=args.max_staleness,
            max_inflight_tasks=max_inflight,
            token_budget=0,
            score_centering=False,
            router_aux_loss_coef=0.0,
            logging_steps=1,
            report_to="none",
            log_completions=True,
            vllm_server_base_url="http://cortex-managed",
        )

        cortex_bridge = CortexModelBridge(
            client=client,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or 0,
            rollout_n=args.num_generations,
            max_token_len_per_gpu=MAX_TOKEN_LEN_PER_GPU,
        )
        rollout_worker = ArcticRolloutWorker(
            client,
            dataset,
            accuracy_reward,
            tokenizer,
            num_generations=args.num_generations,
            max_tokens=args.max_completion_length,
            temperature=1.0,
            queue_maxsize=args.per_device_bsz * 8,
            chat_template_kwargs=chat_template_kwargs,
            old_logprobs_source=args.old_logprobs_source,
            pad_token_id=tokenizer.pad_token_id or 0,
            max_token_len_per_gpu=MAX_TOKEN_LEN_PER_GPU,
            generate_group_batch=generate_group_batch,
        )
        weight_transfer = ArcticWeightTransfer(client)

        trainer = make_trainer(
            model=stub_model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            rollout_worker=rollout_worker,
            weight_transfer=weight_transfer,
            arctic_client=client,
            cortex_bridge=cortex_bridge,
        )

        run_desc = f"max_steps={args.max_steps}" if args.max_steps > 0 else f"{args.num_train_epochs} epoch(s)"
        print(f"[e2e] starting trainer.train() ({run_desc}) ...", flush=True)
        trainer.train()

        history = [h for h in trainer.state.log_history if "loss" in h]
        print(f"[e2e] completed {trainer.state.global_step} optimizer steps", flush=True)
        for h in history[-10:]:
            keep = {k: h[k] for k in ("loss", "reward", "ratio", "kl", "entropy", "completions/mean_length") if k in h}
            print(f"[e2e]   step {h.get('step')}: {keep}", flush=True)
        with open(args.metrics_out, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"[e2e] wrote metrics -> {args.metrics_out}", flush=True)
        target = args.max_steps if args.max_steps > 0 else 1
        if trainer.state.global_step >= target:
            ok = True
            print("[e2e] E2E PASSED: GSM8K GRPO ran on Cortex via the TRL backend", flush=True)
        else:
            print(f"[e2e] E2E INCOMPLETE: only {trainer.state.global_step} steps (target {target})", flush=True)
    except Exception:
        print("[e2e] E2E FAILED with exception:", flush=True)
        traceback.print_exc()
    finally:
        if client is not None:
            try:
                client.shutdown()
            except Exception as e:
                print(f"[e2e] client.shutdown() raised (ignored): {e}", flush=True)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
