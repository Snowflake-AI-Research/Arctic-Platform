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

"""GSM8K GRPO on the Arctic TRL backend. Owns the Arctic server process group."""

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

# Transport selection is read from the environment here (before torch init) because it decides GPU visibility:
#   http (default): the Arctic server runs as a separate subprocess and this client is CPU-only.
#   ray:            this process is the Ray driver that builds the in-process server, so it must SEE the GPUs
#                   (Ray allocates them to the worker actors). The trainer still stays on CPU via use_cpu=True.
_TRANSPORT = os.environ.get("ARCTIC_TRANSPORT", "http")

# Arctic's contract: the client (this TRL trainer process) runs CPU-only -- every bit of GPU compute (model
# forward/backward, sampling, weight sync) lives on the Arctic server. For the HTTP transport, hide GPUs from this
# process *before* torch initializes CUDA, so accelerate places the inert stub model + trainer inputs on CPU and
# TRL's `loss_fn` runs on CPU. Remember the real device set so the server subprocess still receives the GPUs.
# For the Ray transport we must NOT hide them (the driver needs them to place actors); use_cpu=True still keeps
# the local stub/loss on CPU.
_SERVER_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")  # None => server inherits all GPUs
if _TRANSPORT != "ray":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

SERVER_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_server_e2e.log")
MAX_TOKEN_LEN_PER_GPU = 4096


# --------------------------------------------------------------------------------------------------------------- #
# Inert local-model stub (README open item #1)
# --------------------------------------------------------------------------------------------------------------- #
def install_stub_model_loader() -> None:
    """Replace the trainer's ``AutoModelForCausalLM`` with a loader that builds a tiny, inert stub.

    Arctic owns the weights; the trainer's local module is never run (ArcticTrainingClient ignores it) and never
    streamed (ArcticWeightTransfer ignores the iterator). So we only need a real ``*ForCausalLM`` shell with a
    ``.config`` (for MoE detection / model id), a ``.model`` + ``.lm_head`` (for ``patch_chunked_lm_head``), and
    parameters (for the optimizer + accelerate). Shrinking to 1 layer + eager attention keeps it cheap and
    avoids both the checkpoint download and the flash-attn3 kernel fetch.
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
                text.num_hidden_layers = 1  # inert: depth doesn't matter, never forwarded
            model = _RealAuto.from_config(cfg, attn_implementation="eager")
            return model.to(torch.float32)

    agt.AutoModelForCausalLM = _StubLoader
    print("[e2e] installed inert stub-model loader (1 layer, eager attn, no checkpoint download)", flush=True)


# --------------------------------------------------------------------------------------------------------------- #
# GSM8K data + reward
# --------------------------------------------------------------------------------------------------------------- #
def format_sample(sample: dict) -> dict:
    # `solution` (the gold numeric answer) is forwarded to the reward func by the rollout worker.
    return {
        "prompt": [{"role": "user", "content": sample["question"]}],
        "solution": sample["answer"].split("####")[-1].strip(),
    }


# --------------------------------------------------------------------------------------------------------------- #
# Arctic client (co-located training + sampling), mirroring the stage-2 smoke config
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
    colocate: bool = True,
    zorro_train_enable: bool = False,
):
    from arctic_platform.client import ArcticClientConfig
    from arctic_platform.client import ArcticRLClient
    from arctic_platform.client import OnPremConfig
    from arctic_platform.client import SamplingConfig
    from arctic_platform.client import TrainingConfig

    ds_config = {
        # DeepSpeed asserts train_batch_size == micro_bsz * grad_accum * world_size, and world_size is the
        # number of training GPUs. The server drives fwd/bwd with client-shaped batches, so these are just
        # DeepSpeed bookkeeping -- keep micro/grad_accum at 1 and scale the global batch by training_gpus.
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": training_gpus,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        # Optimizer folded into ds_config (the unified client drops training_config on the wire).
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-6, "betas": [0.9, 0.999], "weight_decay": 0.0}},
        "bf16": {"enabled": True},
    }
    ds_worker_config = dict(
        use_liger=False,
        enable_gradient_checkpointing=False,
        # flash_attention_2 is REQUIRED for correctness here, not just speed: the server packs each forward
        # varlen-style (multiple sequences concatenated into one [1, T] row with per-sequence position_ids resets).
        # FA2 turns those resets into block-diagonal cu_seqlens so sequences stay separated; sdpa does NOT, so the
        # packed rows attend across sequence boundaries and corrupt per-token log probs (old AND new), which blows
        # up the importance ratio. (flash-attn is built against this env's torch in setup; keep it installed.)
        attn_implementation=os.environ.get("ARCTIC_ATTN_IMPL", "flash_attention_2"),
        zorro_train_enable=zorro_train_enable,
        response_len=response_len,
        max_token_len=MAX_TOKEN_LEN_PER_GPU,
        rollout_n=rollout_n,
        temperature=1.0,
        logits_optimization=os.environ.get("ARCTIC_LOGITS_OPT", "none"),
        logits_optimization_peak_mem_size_in_gib=int(os.environ.get("ARCTIC_LOGITS_OPT_PEAK_GIB", "4")),
        logits_compute_from_fp32_inputs=False,
        logits_compute_in_fp32=os.environ.get("ARCTIC_LOGITS_COMPUTE_FP32", "0") not in ("0", "false", "False"),
        # use_unpad packs sequences varlen-style, which relies on flash-attn's cu_seqlens masking. Keep it enabled
        # only on flash_attention_2 (the correct path); any non-FA2 backend would attend across sequence boundaries.
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
            vllm={
                # TP=1 => `sampling_gpus` data-parallel replicas; the ReplicaPool spreads generate() across them.
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": 0.3,
                "enforce_eager": True,
                "enable_prefix_caching": False,
            }
        ),
        backend=OnPremConfig(
            # "ray" builds the server in-process (no HTTP, no serialization); "http" connects to a subprocess
            # server we launch separately. Either way launch_local_server=False -- the caller owns lifecycle.
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
# Server lifecycle (own process group; hardened teardown)
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
            # Called by transformers at train start, once the (stub) model exists. ArcticOptimizer.step()
            # drives client.step(); the params are only a handle for accelerate + the scheduler.
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
    # max_steps <= 0 => epoch-driven stop (num_train_epochs full passes over the prompt dataset). The async
    # trainer then treats max_steps only as a safety ceiling it computes itself.
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "-1")))
    ap.add_argument("--num-train-epochs", type=float, default=float(os.environ.get("NUM_TRAIN_EPOCHS", "1")))
    ap.add_argument("--num-generations", type=int, default=int(os.environ.get("NUM_GEN", "8")))
    ap.add_argument("--per-device-bsz", type=int, default=int(os.environ.get("PER_DEVICE_BSZ", "8")))
    ap.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "1")))
    ap.add_argument("--max-completion-length", type=int, default=int(os.environ.get("MAX_COMPLETION_LEN", "256")))
    ap.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "1024")))
    # num_prompts <= 0 => full GSM8K train split (7473 prompts).
    ap.add_argument("--num-prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", "-1")))
    ap.add_argument("--training-gpus", type=int, default=int(os.environ.get("TRAINING_GPUS", "8")))
    ap.add_argument("--sampling-gpus", type=int, default=int(os.environ.get("SAMPLING_GPUS", "8")))
    # colocate=True (default) shares the same GPUs between training and sampling with a sleep/wake handoff;
    # --no-colocate runs disaggregated (training_gpus and sampling_gpus land on disjoint devices, so they must
    # sum to <= total GPUs) which avoids the colocated wake_inference collective contention.
    ap.add_argument(
        "--colocate",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ARCTIC_COLOCATE", "1") not in ("0", "false", "False"),
    )
    # ZoRRo Train: dedup shared prompt prefixes across the rollout group in the training forward
    # (Arctic activation/logits optimization). Off by default; enables the model patch (server) and
    # the per-request meta flag (client) so run_pipeline takes the zorro path.
    ap.add_argument(
        "--zorro",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ARCTIC_ZORRO", "0") not in ("0", "false", "False"),
    )
    # Zorro-group load balancer (server reorgs same-prompt rollouts across DP workers for better dedup;
    # undone by restore_batch_order before the response). EXPERIMENTAL, off by default: reorg's bin-packer
    # assumes a verl-style global batch (world_size-divisible with fittable prompt groups) that TRL's
    # per-forward_backward microbatch does not guarantee. Only used when --zorro.
    ap.add_argument(
        "--zorro-load-balancer",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ARCTIC_ZORRO_LOAD_BALANCER", "0") not in ("0", "false", "False"),
    )
    ap.add_argument("--tensor-parallel", type=int, default=int(os.environ.get("TENSOR_PARALLEL", "1")))
    ap.add_argument("--server-startup-timeout", type=float, default=1200.0)
    ap.add_argument(
        "--metrics-out",
        default=os.environ.get(
            "METRICS_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_arctic.json")
        ),
    )
    # Where old_log_probs come from: "trainer" recomputes on the Arctic engine (verl-style); "sampler" uses
    # vLLM's generation logprobs. With the corrected shift both should hold ratio ~ 1 on-policy.
    ap.add_argument(
        "--old-logprobs-source",
        choices=("trainer", "sampler"),
        default=os.environ.get("OLD_LOGPROBS_SOURCE", "trainer"),
    )
    # Transport: "http" (subprocess server + REST) or "ray" (in-process Ray server, no HTTP/serialization).
    # NOTE: GPU visibility is decided at import time from $ARCTIC_TRANSPORT, so to use ray you must export
    # ARCTIC_TRANSPORT=ray (this flag defaults to it and must agree with it).
    ap.add_argument("--transport", choices=("http", "ray"), default=_TRANSPORT)
    # Where GRPO loss runs: "client" (default) evaluates TRL's loss on CPU then ships a
    # surrogate backward; "server" ships GRPO ingredients and runs forward+loss+backward
    # in one fused fwd_bwd (perf + parity with verl/SkyRL).
    ap.add_argument(
        "--loss-placement", choices=("client", "server"), default=os.environ.get("ARCTIC_TRL_LOSS_PLACEMENT", "client")
    )
    # Single knob for variance control: seeds the Arctic sampler (ArcticClientConfig) AND the trainer
    # (AsyncGRPOConfig) so matched-seed baseline/arctic runs are comparable.
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    args = ap.parse_args()
    if args.transport != _TRANSPORT:
        raise SystemExit(
            f"--transport={args.transport} disagrees with ARCTIC_TRANSPORT={_TRANSPORT!r} (which already set GPU "
            f"visibility at import). Export ARCTIC_TRANSPORT={args.transport} before launching."
        )

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl.experimental.async_grpo import AsyncGRPOConfig
    from trl.rewards import accuracy_reward

    from arctic_platform.integrations.trl.client import ArcticTrainingClient
    from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker
    from arctic_platform.integrations.trl.weights import ArcticWeightTransfer

    chat_template_kwargs = {"enable_thinking": False}

    # HTTP transport launches the co-located server as its own subprocess/process group; the Ray transport builds
    # the server in-process inside build_client (this process is the Ray driver), so there is nothing to launch.
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
        # The client hid its GPUs (CUDA_VISIBLE_DEVICES=""); give them back to the server subprocess.
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
    ckpt = tempfile.mkdtemp(prefix="arl_e2e_ckpt_")
    out_dir = tempfile.mkdtemp(prefix="arl_e2e_out_")
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
        )
        print(f"[e2e] client ready; jobs={client.jobs}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        split = "train" if args.num_prompts <= 0 else f"train[:{args.num_prompts}]"
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
        print(f"[e2e] dataset ready: {len(dataset)} prompts (split={split})", flush=True)

        install_stub_model_loader()

        config = AsyncGRPOConfig(
            output_dir=out_dir,
            save_strategy="no",
            per_device_train_batch_size=args.per_device_bsz,
            gradient_accumulation_steps=args.grad_accum,
            seed=args.seed,
            # Client is CPU-only; the local stub never computes, and bf16/fp16 defaults would trip CPU autocast
            # checks. Real training precision lives in the server's ds_config (bf16).
            bf16=False,
            fp16=False,
            use_cpu=True,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            temperature=1.0,
            chat_template_kwargs=chat_template_kwargs,
            # max_steps<=0 => epoch-driven stop; the trainer derives its own safety ceiling from num_train_epochs.
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            weight_sync_steps=1,
            token_budget=0,  # FixedCountBatcher: avoids needing a real vLLM URL for max_model_len
            logging_steps=1,
            report_to="none",
            log_completions=True,
            # Dummy: rollout + weight sync are overridden, and token_budget=0 skips the only other use.
            vllm_server_base_url=f"http://{args.host}:{args.port}",
        )

        if args.zorro and args.old_logprobs_source != "sampler":
            # 5a launch invariant: the zorro model patch is a process-lifetime monkeypatch, so the
            # trainer-source old-logprobs recompute (a non-zorro engine forward_no_grad) crashes under it.
            # Zorro must use sampler logprobs; it then only affects the training fwd/bwd.
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
            logits_optimization=os.environ.get("ARCTIC_LOGITS_OPT", "none"),
            logits_optimization_peak_mem_size_in_gib=int(os.environ.get("ARCTIC_LOGITS_OPT_PEAK_GIB", "4")),
            logits_compute_in_fp32=os.environ.get("ARCTIC_LOGITS_COMPUTE_FP32", "0") not in ("0", "false", "False"),
        )
        print(
            f"[e2e] loss_placement={args.loss_placement} zorro={args.zorro} "
            f"zorro_load_balancer={args.zorro_load_balancer}",
            flush=True,
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
            # verl-style: recompute old_log_probs on the training engine (not vLLM) so the ratio/KL match the
            # trainer's new log-probs. Must mirror the training client's pad/token-budget values.
            old_logprobs_source=args.old_logprobs_source,
            pad_token_id=tokenizer.pad_token_id or 0,
            max_token_len_per_gpu=MAX_TOKEN_LEN_PER_GPU,
            logits_optimization=os.environ.get("ARCTIC_LOGITS_OPT", "none"),
            logits_optimization_peak_mem_size_in_gib=int(os.environ.get("ARCTIC_LOGITS_OPT_PEAK_GIB", "4")),
            logits_compute_in_fp32=os.environ.get("ARCTIC_LOGITS_COMPUTE_FP32", "0") not in ("0", "false", "False"),
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
        # Epoch-driven runs (max_steps<=0) succeed once training completes any optimizer step; explicit-step runs
        # must reach the requested count.
        target = args.max_steps if args.max_steps > 0 else 1
        if trainer.state.global_step >= target:
            ok = True
            print("[e2e] E2E PASSED: GSM8K GRPO ran end-to-end on the Arctic backend", flush=True)
        else:
            print(f"[e2e] E2E INCOMPLETE: only {trainer.state.global_step} steps (target {target})", flush=True)
    except Exception:
        print("[e2e] E2E FAILED with exception:", flush=True)
        traceback.print_exc()
    finally:
        if client is not None:
            try:
                client.shutdown()  # ray: tears down the in-process server actors; http: destroys the jobs
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
