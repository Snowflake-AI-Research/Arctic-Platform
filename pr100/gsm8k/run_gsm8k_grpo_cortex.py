#!/usr/bin/env python
"""Tunji's GSM8K GRPO recipe, with the backend swapped from on-prem to Cortex.

This is ``recipes/rl/trl/gsm8k/run_qwen3_1.7b_gsm8k_grpo_arl.py`` with three
changes and nothing else of substance:

1. ``OnPremConfig`` becomes ``CortexConfig``, so the GPU work lands on a Cortex
   job instead of a locally launched server. The recipe's client is already
   CPU-only by design -- it hides its own GPUs before importing torch -- so
   there is nothing to run locally either way.
2. The job is created up front by ``dss_client`` rather than by the transport,
   because ``forward`` needs Thong's chunking fix and pinning that image is a
   debug option the Arctic transport does not expose. The body is produced by
   ``config.to_cortex()``, the same translation the transport would have used,
   so it cannot drift from what the client expects. The client then reattaches
   through ``training_job_id`` / ``sampling_job_id``.
3. ``client_loss_encoding="grpo"`` -- the encoding under review in PR #100.

The recipe already computes its loss on the client (``--loss-placement client``
is its default). What it cannot do over Cortex is express that loss as
``weighted_logprob_sum``, because registering a loss function there means
rebuilding the image. The grpo encoding is how the same gradient gets through
using a loss the image already ships.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

# The client is CPU-only: all model compute lives on the Cortex job. Hide GPUs
# before torch initializes so accelerate keeps the stub and the loss on CPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

MAX_TOKEN_LEN_PER_GPU = 4096


def install_stub_model_loader() -> None:
    """The trainer's local model is never run; make it a 1-layer shell.

    Straight from the recipe. Arctic owns the weights, so the local module only
    needs to be a real ``*ForCausalLM`` with a config and some parameters. This
    also avoids downloading the checkpoint onto a box that would never use it.
    """
    import trl.experimental.async_grpo.async_grpo_trainer as agt
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM as _RealAuto

    class _StubLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=kwargs.get("trust_remote_code", False))
            text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
            if hasattr(text, "num_hidden_layers"):
                text.num_hidden_layers = 1
            return _RealAuto.from_config(cfg, attn_implementation="eager").to(torch.float32)

    agt.AutoModelForCausalLM = _StubLoader


def format_sample(sample: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": sample["question"]}],
        "solution": sample["answer"].split("####")[-1].strip(),
    }


def build_config(args, cortex_cfg: dict, *, training_job_id=None, sampling_job_id=None):
    from arctic_platform.client import ArcticClientConfig
    from arctic_platform.client import CortexConfig
    from arctic_platform.client import SamplingConfig
    from arctic_platform.client import TrainingConfig

    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": args.training_gpus,
        "gradient_accumulation_steps": 1,
        # No offload_optimizer key at all: spelling it as {"device": "none"} is still
        # enough for DeepSpeed to instantiate CPUAdam, which then asserts because the
        # params are on the GPU.
        "zero_optimization": {"stage": 2},
        "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "betas": [0.9, 0.999], "weight_decay": 0.0}},
        "bf16": {"enabled": True},
    }
    ds_worker_config = dict(
        use_liger=False,
        model_provider="huggingface",
        enable_gradient_checkpointing=False,
        # Flash attention is a correctness requirement, not a speed knob: the server
        # packs each forward varlen-style and only flash-attn turns the position_id
        # resets into block-diagonal cu_seqlens. Anything else attends across sequence
        # boundaries and corrupts the per-token log-probs the whole loop is built on.
        # The recipe asks for FA2; the Cortex image ships only FA3, so we use that and
        # watch the reported ratio -- corrupted log-probs blow it up away from 1.
        attn_implementation=args.attn_impl,
        response_len=args.max_completion_length,
        max_token_len=MAX_TOKEN_LEN_PER_GPU,
        rollout_n=args.num_generations,
        temperature=1.0,
        use_unpad=True,
        use_autocast=False,
    )
    host = cortex_cfg["host"]
    for prefix in ("https://", "http://"):
        host = host[len(prefix):] if host.startswith(prefix) else host

    return ArcticClientConfig(
        model_name=args.model,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        training_gpus=args.training_gpus,
        sampling_gpus=args.sampling_gpus,
        log_prob_gpus=0,
        training=TrainingConfig(ds_config=ds_config, ds_worker_config=ds_worker_config),
        sampling=SamplingConfig(
            vllm={"gpu_memory_utilization": 0.8, "enforce_eager": True, "enable_prefix_caching": False}
        ),
        backend=CortexConfig(
            host=host.rstrip("/"),
            pat=cortex_cfg["pat"],
            database=cortex_cfg["database"],
            schema=cortex_cfg.get("schema", "PUBLIC"),
            endpoint=cortex_cfg.get("endpoint", "cortex-training"),
        ),
        training_job_id=training_job_id,
        sampling_job_id=sampling_job_id,
    )


def create_job(cortex_cfg: dict, sub_job_configs: list, image_tag: str | None, client_repo: str) -> str:
    sys.path.insert(0, client_repo)
    from dss_client.neutrino_client import DEBUG_OPTIONS_ENV
    from dss_client.neutrino_client import NeutrinoClient

    host = cortex_cfg["host"]
    for prefix in ("https://", "http://"):
        host = host[len(prefix):] if host.startswith(prefix) else host
    client = NeutrinoClient.from_pat(
        host=host.rstrip("/"),
        pat=cortex_cfg["pat"],
        database=cortex_cfg["database"],
        schema=cortex_cfg.get("schema", "PUBLIC"),
        endpoint=cortex_cfg.get("endpoint", "cortex-training"),
        poll_timeout=float(cortex_cfg.get("poll_timeout", 1800.0)),
    )
    body = {"sub_job_configs": sub_job_configs}
    if image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        body["debug"] = {"job": {"image_tag": image_tag}}
        print(f"[cortex] pinning image_tag={image_tag}", flush=True)
    print(f"[cortex] creating job with {len(sub_job_configs)} sub-job(s): "
          f"{[s.get('job_type') for s in sub_job_configs]}", flush=True)
    job_id = str(client.create_job_from_body(body)["job_id"])
    print(f"[cortex] job {job_id} created; waiting for RUNNING (this takes a few minutes) ...", flush=True)
    client.wait_for_job(job_id)
    print(f"[cortex] job {job_id} RUNNING", flush=True)
    return job_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortex-config", required=True)
    ap.add_argument("--client-repo", default="/code/users/karthik/thong-client")
    ap.add_argument("--debug-image-tag")
    ap.add_argument("--job-id", help="reuse an existing Cortex job instead of creating one")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-bsz", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--num-train-epochs", type=float, default=1.0)
    ap.add_argument("--training-gpus", type=int, default=1)
    ap.add_argument("--sampling-gpus", type=int, default=1)
    ap.add_argument("--attn-impl", default="flash_attention_3")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--client-loss-encoding", default="grpo", choices=("grpo", "weighted_logprob_sum"))
    ap.add_argument("--metrics-out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_metrics.json"))
    ap.add_argument("--keep-job", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl.experimental.async_grpo import AsyncGRPOConfig
    from trl.rewards import accuracy_reward

    from arctic_platform.client import ArcticRLClient
    from arctic_platform.integrations.trl.client import ArcticTrainingClient
    from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker
    from arctic_platform.integrations.trl.weights import ArcticWeightTransfer

    cortex_cfg = json.loads(open(args.cortex_config).read())
    chat_template_kwargs = {"enable_thinking": False}

    job_id = args.job_id
    if job_id is None:
        # Same translation the transport would have applied, so the body cannot drift.
        job_id = create_job(
            cortex_cfg, build_config(args, cortex_cfg).to_cortex(), args.debug_image_tag, args.client_repo
        )
    else:
        print(f"[cortex] reusing job {job_id}", flush=True)

    cfg = build_config(
        args,
        cortex_cfg,
        training_job_id=f"{job_id}:training:0",
        sampling_job_id=f"{job_id}:sampling:0",
    )

    client = None
    ok = False
    try:
        client = ArcticRLClient(cfg)
        print(f"[cortex] client attached; jobs={client.jobs}", flush=True)
        # The TRL integration speaks the on-prem dialect; a Cortex zone does not.
        # See cortex_trl_adapter for what this bridges and what it cannot.
        from cortex_trl_adapter import CortexTRLAdapter

        engine = CortexTRLAdapter(client, temperature=1.0)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("openai/gsm8k", "main", split=f"train[:{args.num_prompts}]")
        dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
        print(f"[gsm8k] {len(dataset)} prompts", flush=True)

        install_stub_model_loader()

        config = AsyncGRPOConfig(
            output_dir=os.path.join("/tmp", f"gsm8k_cortex_{int(time.time())}"),
            save_strategy="no",
            per_device_train_batch_size=args.per_device_bsz,
            gradient_accumulation_steps=args.grad_accum,
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
            token_budget=0,
            logging_steps=1,
            report_to="none",
            log_completions=True,
            learning_rate=args.lr,
            vllm_server_base_url="http://unused",
        )

        training_client = ArcticTrainingClient(
            client=engine,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or 0,
            rollout_n=args.num_generations,
            max_token_len_per_gpu=MAX_TOKEN_LEN_PER_GPU,
            server_side_loss=False,
            client_loss_encoding=args.client_loss_encoding,
            response_len=args.max_completion_length,
        )
        print(f"[trl] client-side loss, encoding={args.client_loss_encoding}", flush=True)

        rollout_worker = ArcticRolloutWorker(
            engine, dataset, accuracy_reward, tokenizer,
            num_generations=args.num_generations,
            max_tokens=args.max_completion_length,
            temperature=1.0,
            queue_maxsize=args.per_device_bsz * 8,
            chat_template_kwargs=chat_template_kwargs,
            old_logprobs_source="trainer",
            pad_token_id=tokenizer.pad_token_id or 0,
            max_token_len_per_gpu=MAX_TOKEN_LEN_PER_GPU,
        )
        # Version skew: this trl drains `rollout_worker.metrics_queue`, while PR #84's
        # worker hangs metrics off each RolloutSample. Without a bridge the trainer
        # raises on its first log() -- and, more to the point, the reward never reaches
        # log_history, which is the only thing that makes this run worth doing. Tee the
        # per-sample metrics into a queue as they are buffered.
        import queue as _queue

        class _MetricsTee:
            def __init__(self, inner, sink):
                self._inner, self._sink = inner, sink

            def put(self, item, *a, **kw):
                metrics = getattr(item, "metrics", None)
                if metrics:
                    try:
                        self._sink.put_nowait(dict(metrics))
                    except _queue.Full:
                        pass
                return self._inner.put(item, *a, **kw)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        rollout_worker.metrics_queue = _queue.Queue()
        rollout_worker.rollout_buffer = _MetricsTee(rollout_worker.rollout_buffer, rollout_worker.metrics_queue)

        weight_transfer = ArcticWeightTransfer(client)

        from trl.experimental.async_grpo import AsyncGRPOTrainer

        from arctic_platform.integrations.trl.client import ArcticOptimizer

        class CortexAsyncGRPOTrainer(AsyncGRPOTrainer):
            def create_optimizer(self):
                if self.optimizer is None:
                    self.optimizer = ArcticOptimizer(client, self.model.parameters(), lr=self.args.learning_rate)
                return self.optimizer

        trainer = CortexAsyncGRPOTrainer(
            model=args.model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            training_client=training_client,
            rollout_worker=rollout_worker,
            weight_transfer=weight_transfer,
        )

        print("[trl] starting trainer.train() ...", flush=True)
        trainer.train()

        history = [h for h in trainer.state.log_history if "loss" in h]
        print(f"[trl] completed {trainer.state.global_step} optimizer steps", flush=True)
        for h in history:
            keep = {k: h[k] for k in ("loss", "reward", "ratio", "kl", "entropy") if k in h}
            print(f"[trl]   step {h.get('step')}: {keep}", flush=True)
        with open(args.metrics_out, "w") as fh:
            json.dump({"job_id": job_id, "args": vars(args), "log_history": trainer.state.log_history}, fh, indent=2)
        print(f"[trl] wrote {args.metrics_out}", flush=True)
        ok = trainer.state.global_step >= (args.max_steps if args.max_steps > 0 else 1)
        print("[e2e] PASSED" if ok else "[e2e] INCOMPLETE", flush=True)
    except Exception:
        print("[e2e] FAILED:", flush=True)
        traceback.print_exc()
    finally:
        if client is not None and not args.keep_job:
            try:
                client.shutdown()
                print("[cortex] job shut down", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[cortex] shutdown raised (ignored): {exc}", flush=True)
        elif args.keep_job:
            print(f"[cortex] keeping job {job_id}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
