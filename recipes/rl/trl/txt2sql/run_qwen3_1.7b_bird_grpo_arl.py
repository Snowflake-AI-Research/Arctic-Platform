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

"""BIRD GRPO on the Arctic TRL backend (disaggregated 4 train + 4 sample)."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# http: hide GPUs from this process (server is a subprocess). ray: driver must see GPUs.
_TRANSPORT = os.environ.get("ARCTIC_TRANSPORT", "http")

_SERVER_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")  # None => server inherits all GPUs
if _TRANSPORT != "ray":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

SERVER_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_server_bird_e2e.log")
MAX_TOKEN_LEN_PER_GPU = int(os.environ.get("MAX_TOKEN_LEN_PER_GPU", "8192"))


def _logits_opt_from_env() -> dict:
    """Keep worker-init and per-call TRL meta on the same ARCTIC_LOGITS_OPT knobs."""
    return {
        "logits_optimization": os.environ.get("ARCTIC_LOGITS_OPT", "none"),
        "logits_optimization_peak_mem_size_in_gib": int(os.environ.get("ARCTIC_LOGITS_OPT_PEAK_GIB", "4")),
        "logits_compute_in_fp32": os.environ.get("ARCTIC_LOGITS_COMPUTE_FP32", "0") not in ("0", "false", "False"),
    }


# --------------------------------------------------------------------------------------------------------------- #
# Inert local-model stub (README open item #1) -- identical to the GSM8K run
# --------------------------------------------------------------------------------------------------------------- #
def install_stub_model_loader() -> None:
    import trl.experimental.async_grpo.async_grpo_trainer as agt
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM as _RealAuto

    class _StubLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=kwargs.get("trust_remote_code", False))
            text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
            if hasattr(text, "num_hidden_layers"):
                text.num_hidden_layers = 1  # inert: depth doesn't matter, never forwarded
            model = _RealAuto.from_config(cfg, attn_implementation="eager")
            return model.to(torch.float32)

    agt.AutoModelForCausalLM = _StubLoader
    print("[e2e] installed inert stub-model loader (1 layer, eager attn, no checkpoint download)", flush=True)


# --------------------------------------------------------------------------------------------------------------- #
# Arctic client (disaggregated training + sampling)
# --------------------------------------------------------------------------------------------------------------- #
def build_client(
    model: str,
    host: str,
    port: int,
    *,
    max_seq_len: int,
    response_len: int,
    rollout_n: int,
    startup_timeout: float,
    ckpt_dir: str,
    training_gpus: int,
    sampling_gpus: int,
    tensor_parallel_size: int,
    comm_protocol: str = "http",
    seed: int = 42,
    colocate: bool = False,
    zorro_train_enable: bool = False,
    gpu_mem_util: float = 0.3,
    grad_accum_steps: int = 1,
):
    from arctic_platform.client import ArcticClientConfig
    from arctic_platform.client import ArcticRLClient
    from arctic_platform.client import OnPremConfig
    from arctic_platform.client import SamplingConfig
    from arctic_platform.client import TrainingConfig

    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        # train_batch_size = micro_bs * gas * world_size; worker split_dict's each DP shard.
        "train_batch_size": training_gpus * grad_accum_steps,
        "gradient_accumulation_steps": grad_accum_steps,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-6, "betas": [0.9, 0.999], "weight_decay": 0.0}},
        "bf16": {"enabled": True},
    }
    logits_opt = _logits_opt_from_env()
    ds_worker_config = dict(
        use_liger=os.environ.get("USE_LIGER", "0") not in ("0", "false", "False"),
        # FA2 required: varlen packing uses block-diagonal cu_seqlens.
        attn_implementation=os.environ.get("ARCTIC_ATTN_IMPL", "flash_attention_2"),
        zorro_train_enable=zorro_train_enable,
        response_len=response_len,
        max_token_len=MAX_TOKEN_LEN_PER_GPU,
        rollout_n=rollout_n,
        temperature=1.0,
        # none: full logits. memory: tile LM head (no-zorro via run_pipeline; zorro via patched CausalLM).
        **logits_opt,
        logits_compute_from_fp32_inputs=False,
        use_unpad=(os.environ.get("ARCTIC_ATTN_IMPL", "flash_attention_2") == "flash_attention_2"),
        use_autocast=False,
    )
    cfg = ArcticClientConfig(
        model_name=model,
        seed=seed,
        max_seq_len=max_seq_len,
        training_gpus=training_gpus,
        sampling_gpus=sampling_gpus,
        log_prob_gpus=0,
        training=TrainingConfig(ds_config=ds_config, ds_worker_config=ds_worker_config, checkpoint_path=ckpt_dir),
        sampling=SamplingConfig(
            # TP=1; leftover sampling GPUs become DP replicas.
            vllm={
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_mem_util,
                "max_num_seqs": int(os.environ.get("VLLM_MAX_NUM_SEQS", "256")),
                "max_num_batched_tokens": int(os.environ.get("VLLM_MAX_BATCHED_TOKENS", "40960")),
                "enforce_eager": os.environ.get("VLLM_ENFORCE_EAGER", "0") not in ("0", "false", "False"),
                "enable_chunked_prefill": True,
                "enable_prefix_caching": os.environ.get("VLLM_PREFIX_CACHING", "1") not in ("0", "false", "False"),
            }
        ),
        backend=OnPremConfig(
            protocol=comm_protocol,
            host=host,
            port=port,
            colocate=colocate,
            launch_local_server=False,
            startup_timeout=startup_timeout,
        ),
    )
    return ArcticRLClient(cfg)


# --------------------------------------------------------------------------------------------------------------- #
# Server lifecycle helpers
# --------------------------------------------------------------------------------------------------------------- #
def _wait_health(url: str, timeout: float, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited early with code {server.returncode} before healthy")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2)
    raise TimeoutError(f"server not healthy after {timeout}s (last: {last})")


def _tail(path: str, n: int = 100) -> str:
    try:
        with open(path) as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return "(no server log)"


# --------------------------------------------------------------------------------------------------------------- #
# Trainer subclass: build ArcticOptimizer from the trainer's own (stub) model
# --------------------------------------------------------------------------------------------------------------- #
def make_trainer(
    *, model, args, train_dataset, processing_class, training_client, rollout_worker, weight_transfer, arctic_client
):
    from trl.experimental.async_grpo import AsyncGRPOTrainer

    from arctic_platform.integrations.trl.client import ArcticOptimizer

    class ArcticAsyncGRPOTrainer(AsyncGRPOTrainer):
        def __init__(self, *a, **kw):
            self._arctic_client = arctic_client
            super().__init__(*a, **kw)

        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = ArcticOptimizer(
                    self._arctic_client, self.model.parameters(), lr=self.args.learning_rate
                )
            return self.optimizer

    return ArcticAsyncGRPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=processing_class,
        training_client=training_client,
        rollout_worker=rollout_worker,
        weight_transfer=weight_transfer,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--data-path", default=os.environ.get("BIRD_TRAIN_PARQUET", "/data/snowflakesql/txt2sql/train.parquet")
    )
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "30")))
    ap.add_argument("--num-train-epochs", type=float, default=float(os.environ.get("NUM_TRAIN_EPOCHS", "1")))
    ap.add_argument("--num-generations", type=int, default=int(os.environ.get("NUM_GEN", "8")))
    ap.add_argument("--per-device-bsz", type=int, default=int(os.environ.get("PER_DEVICE_BSZ", "8")))
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=int(os.environ.get("GRAD_ACCUM", "1")),
        help="DeepSpeed engine GAS (worker split_dict of each DP shard). TRL trainer GAS stays 1.",
    )
    ap.add_argument("--max-completion-length", type=int, default=int(os.environ.get("MAX_COMPLETION_LEN", "1024")))
    ap.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "8192")))
    ap.add_argument(
        "--val-every",
        type=int,
        default=int(os.environ.get("VAL_EVERY", "0")),
        help="Greedy val every N optimizer steps (0=off). Quality runs use 10.",
    )
    ap.add_argument("--val-parquet", default=os.environ.get("BIRD_VAL_PARQUET", ""))
    ap.add_argument("--val-max-samples", type=int, default=int(os.environ.get("VAL_MAX_SAMPLES", "0")))
    ap.add_argument("--num-prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", "128")))
    # Dedicated sampling GPUs; 0.7 matches the verl txt2sql baseline.
    ap.add_argument("--gpu-mem-util", type=float, default=float(os.environ.get("GPU_MEM_UTIL", "0.7")))
    # Disaggregated (colocated deadlocks); default 4 train + 4 sample.
    ap.add_argument("--training-gpus", type=int, default=int(os.environ.get("TRAINING_GPUS", "4")))
    ap.add_argument("--sampling-gpus", type=int, default=int(os.environ.get("SAMPLING_GPUS", "4")))
    ap.add_argument(
        "--colocate",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ARCTIC_COLOCATE", "0") not in ("0", "false", "False"),
    )
    ap.add_argument(
        "--zorro",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ARCTIC_ZORRO", "0") not in ("0", "false", "False"),
    )
    ap.add_argument(
        "--zorro-load-balancer",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ARCTIC_ZORRO_LOAD_BALANCER", "0") not in ("0", "false", "False"),
    )
    ap.add_argument("--tensor-parallel", type=int, default=int(os.environ.get("TENSOR_PARALLEL", "1")))
    ap.add_argument("--server-startup-timeout", type=float, default=1800.0)
    ap.add_argument(
        "--metrics-out",
        default=os.environ.get(
            "METRICS_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_bird_arctic.json")
        ),
    )
    ap.add_argument(
        "--old-logprobs-source",
        choices=("trainer", "sampler"),
        default=os.environ.get("OLD_LOGPROBS_SOURCE", "sampler"),
    )
    ap.add_argument("--transport", choices=("http", "ray"), default=_TRANSPORT)
    ap.add_argument(
        "--loss-placement", choices=("client", "server"), default=os.environ.get("ARCTIC_TRL_LOSS_PLACEMENT", "server")
    )
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    args = ap.parse_args()
    if args.transport != _TRANSPORT:
        raise SystemExit(
            f"--transport={args.transport} disagrees with ARCTIC_TRANSPORT={_TRANSPORT!r} (which already set GPU "
            f"visibility at import). Export ARCTIC_TRANSPORT={args.transport} before launching."
        )

    if args.zorro and args.zorro_load_balancer:
        # LB needs per_device_bsz to be a multiple of num_generations * training_gpus.
        lb_unit = args.num_generations * args.training_gpus
        if args.per_device_bsz % lb_unit != 0 or args.per_device_bsz < lb_unit:
            # Round up so each DP rank gets whole prompt groups.
            new_bsz = max(lb_unit, -(-args.per_device_bsz // lb_unit) * lb_unit)
            print(
                f"[e2e] zorro_load_balancer: per_device_bsz={args.per_device_bsz} is not a positive multiple of "
                f"num_generations*training_gpus={lb_unit}; rounding UP to {new_bsz} so reorg_global_batch can hand "
                f"whole prompt groups to each of {args.training_gpus} DP workers (else 'shouldn't reach here').",
                flush=True,
            )
            args.per_device_bsz = new_bsz
        # Deduped group is one micro-batch: max_token_len >= max_seq_len + response_len * n.
        global MAX_TOKEN_LEN_PER_GPU
        need = args.max_seq_len + args.max_completion_length * args.num_generations
        if MAX_TOKEN_LEN_PER_GPU < need:
            print(
                f"[e2e] zorro_load_balancer: raising max_token_len_per_gpu {MAX_TOKEN_LEN_PER_GPU} -> {need} to hold "
                f"a deduped group (prompt up to max_seq_len={args.max_seq_len} + "
                f"{args.num_generations}x{args.max_completion_length} responses).",
                flush=True,
            )
            MAX_TOKEN_LEN_PER_GPU = need

    import bird_task
    from transformers import AutoTokenizer
    from trl.experimental.async_grpo import AsyncGRPOConfig

    from arctic_platform.integrations.trl.client import ArcticTrainingClient
    from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker
    from arctic_platform.integrations.trl.weights import ArcticWeightTransfer

    # R1 txt2sql prompts use <think>; leave thinking on.
    chat_template_kwargs: dict = {}

    server = None
    pgid = None
    logf = None
    if args.transport == "http":
        server_cmd = [
            sys.executable,
            "-u",
            "-m",
            "arctic_platform.common.http_server",
            "--host",
            args.host,
            "--port",
            str(args.port),
            *(["--colocate"] if args.colocate else []),
            "--training-gpus",
            str(args.training_gpus),
            "--sampling-gpus",
            str(args.sampling_gpus),
        ]
        print(f"[e2e] launching server: {' '.join(server_cmd)}", flush=True)
        print(f"[e2e] server stdout/stderr -> {SERVER_LOG}", flush=True)
        server_env = dict(os.environ)
        if _SERVER_CUDA_VISIBLE_DEVICES is None:
            server_env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            server_env["CUDA_VISIBLE_DEVICES"] = _SERVER_CUDA_VISIBLE_DEVICES
        logf = open(SERVER_LOG, "w")
        server = subprocess.Popen(
            server_cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, env=server_env
        )
        pgid = os.getpgid(server.pid)
        print(f"[e2e] server pid={server.pid} pgid={pgid}", flush=True)
    else:
        print("[e2e] transport=ray: building in-process Ray server (no HTTP subprocess)", flush=True)
    print(
        f"[e2e] transport={args.transport}; client torch.cuda.is_available()={torch.cuda.is_available()}", flush=True
    )

    client = None
    ok = False
    ckpt = tempfile.mkdtemp(prefix="arl_bird_ckpt_")
    out_dir = tempfile.mkdtemp(prefix="arl_bird_out_")
    try:
        if args.transport == "http":
            _wait_health(f"http://{args.host}:{args.port}/health", args.server_startup_timeout, server)
            print("[e2e] server healthy", flush=True)

        client = build_client(
            args.model,
            args.host,
            args.port,
            max_seq_len=args.max_seq_len,
            response_len=args.max_completion_length,
            rollout_n=args.num_generations,
            startup_timeout=args.server_startup_timeout,
            ckpt_dir=ckpt,
            training_gpus=args.training_gpus,
            sampling_gpus=args.sampling_gpus,
            tensor_parallel_size=args.tensor_parallel,
            comm_protocol=args.transport,
            seed=args.seed,
            colocate=args.colocate,
            zorro_train_enable=args.zorro,
            gpu_mem_util=args.gpu_mem_util,
            grad_accum_steps=args.grad_accum,
        )
        print(f"[e2e] client ready; jobs={client.jobs}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = bird_task.load_bird_dataset(args.data_path, args.num_prompts)
        print(f"[e2e] BIRD dataset ready: {len(dataset)} prompts (from {args.data_path})", flush=True)

        import wandb_logging

        report_to, run_name = wandb_logging.configure_wandb(
            config="C3" if args.zorro else "C2",
            max_steps=args.max_steps,
            per_device_bsz=args.per_device_bsz,
            grad_accum=args.grad_accum,
            zorro=bool(args.zorro),
            seed=args.seed,
        )
        print(
            f"[e2e] report_to={report_to} run_name={run_name!r} project={os.environ.get('WANDB_PROJECT')}", flush=True
        )

        install_stub_model_loader()

        config = AsyncGRPOConfig(
            output_dir=out_dir,
            save_strategy="no",
            per_device_train_batch_size=args.per_device_bsz,
            # TRL GAS=1 (full LB batch per forward_backward); DeepSpeed GAS is args.grad_accum.
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
            token_budget=0,
            logging_steps=1,
            report_to=report_to,
            run_name=run_name,
            log_completions=False,
            vllm_server_base_url=f"http://{args.host}:{args.port}",
            heartbeat_stale_after_s=1800.0,
            request_timeout=1800,
        )

        if args.zorro and args.old_logprobs_source != "sampler":
            raise SystemExit(
                "--zorro requires --old-logprobs-source sampler (the trainer-source recompute is not "
                "zorro-shaped and crashes under the global zorro model patch)."
            )

        training_client = ArcticTrainingClient(
            client=client,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or 0,
            rollout_n=args.num_generations,
            max_token_len_per_gpu=MAX_TOKEN_LEN_PER_GPU,
            server_side_loss=(args.loss_placement == "server"),
            zorro_train_enable=args.zorro,
            response_len=args.max_completion_length,
            zorro_load_balancer=args.zorro_load_balancer,
            grad_accum_steps=args.grad_accum,
            **_logits_opt_from_env(),
        )
        prefix_cache = os.environ.get("VLLM_PREFIX_CACHING", "1") not in ("0", "false", "False")
        raw_group_batch = int(os.environ.get("ROLLOUT_GROUP_BATCH", "0"))
        max_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS", "256"))
        generate_group_batch = (
            raw_group_batch if raw_group_batch >= 1 else max(1, max_seqs // max(1, args.num_generations))
        )
        print(
            f"[e2e] loss_placement={args.loss_placement} zorro={args.zorro} "
            f"zorro_load_balancer={args.zorro_load_balancer} ds_gas={args.grad_accum} trl_gas=1 "
            f"prefix_cache={prefix_cache} generate_group_batch={generate_group_batch} "
            f"liger={os.environ.get('USE_LIGER', '0')} logits_opt={os.environ.get('ARCTIC_LOGITS_OPT', 'none')} "
            "gc=ds_worker_default(True unless enable_gradient_checkpointing set)",
            flush=True,
        )
        rollout_worker = ArcticRolloutWorker(
            client,
            dataset,
            bird_task.sql_reward,
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
            **_logits_opt_from_env(),
        )
        weight_transfer = ArcticWeightTransfer(client)

        trainer = make_trainer(
            model=args.model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            training_client=training_client,
            rollout_worker=rollout_worker,
            weight_transfer=weight_transfer,
            arctic_client=client,
        )

        import bird_val

        def _val_generate(rows):
            return bird_val.generate_arctic_greedy(
                rows,
                client=client,
                tokenizer=tokenizer,
                max_tokens=args.max_completion_length,
                chunk=int(os.environ.get("VAL_GEN_CHUNK", "32")),
                chat_template_kwargs=chat_template_kwargs,
            )

        bird_val.maybe_attach_val(
            trainer,
            tokenizer=tokenizer,
            generate_fn=_val_generate,
            max_completion_length=args.max_completion_length,
            max_model_len=args.max_seq_len,
            chat_template_kwargs=chat_template_kwargs,
            seed=args.seed,
            val_every=args.val_every,
            val_parquet=args.val_parquet or None,
            val_max_samples=args.val_max_samples,
        )

        run_desc = f"max_steps={args.max_steps}" if args.max_steps > 0 else f"{args.num_train_epochs} epoch(s)"
        print(f"[e2e] starting trainer.train() ({run_desc}) ...", flush=True)
        trainer.train()

        history = [h for h in trainer.state.log_history if "loss" in h]
        print(f"[e2e] completed {trainer.state.global_step} optimizer steps", flush=True)
        for h in history[-10:]:
            keep = {k: h[k] for k in ("loss", "reward", "ratio", "kl", "entropy", "completions/mean_length") if k in h}
            print(f"[e2e]   step {h.get('step')}: {keep}", flush=True)
        try:
            with open(args.metrics_out, "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)
            print(f"[e2e] wrote metrics -> {args.metrics_out} ({len(trainer.state.log_history)} records)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[e2e] metrics dump raised (ignored): {e}", flush=True)
        target = args.max_steps if args.max_steps > 0 else 1
        if trainer.state.global_step >= target:
            ok = True
            print("[e2e] E2E PASSED: BIRD GRPO ran end-to-end on the Arctic backend", flush=True)
        else:
            print(f"[e2e] E2E INCOMPLETE: only {trainer.state.global_step} steps (target {target})", flush=True)
    except Exception:
        print("[e2e] E2E FAILED with exception:", flush=True)
        traceback.print_exc()
    finally:
        if client is not None:
            try:
                client.shutdown()
            except Exception as e:  # noqa: BLE001
                print(f"[e2e] client.shutdown() raised (ignored): {e}", flush=True)
        if server is not None:
            print(f"[e2e] terminating server process group pgid={pgid} ...", flush=True)
            try:
                os.killpg(pgid, signal.SIGTERM)
                for _ in range(30):
                    if server.poll() is not None:
                        break
                    time.sleep(1)
                if server.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
                    server.wait(timeout=30)
            except ProcessLookupError:
                pass
            if logf is not None:
                logf.close()
            print("[e2e] ---- server log tail ----", flush=True)
            print(_tail(SERVER_LOG), flush=True)
            print("[e2e] ---- end server log tail ----", flush=True)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
