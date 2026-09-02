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

"""Native TRL GSM8K baseline: trainer GPU + stock ``vllm serve`` on a second GPU."""

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

# Pin this (trainer) process to a single GPU *before* torch initializes CUDA; the vLLM server gets its own GPU via the
# child process env below. Mirrors TRL's canonical layout (server and trainer on separate CUDA_VISIBLE_DEVICES).
_TRAINER_GPU = os.environ.get("BASELINE_TRAINER_GPU", "0")
_SERVER_GPU = os.environ.get("BASELINE_SERVER_GPU", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = _TRAINER_GPU

import torch  # noqa: E402

SERVER_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_server_baseline.log")


# --------------------------------------------------------------------------------------------------------------- #
# GSM8K data + reward (identical to the Arctic run)
# --------------------------------------------------------------------------------------------------------------- #
def format_sample(sample: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": sample["question"]}],
        "solution": sample["answer"].split("####")[-1].strip(),
    }


# --------------------------------------------------------------------------------------------------------------- #
# vLLM server lifecycle (own process group; hardened teardown) -- mirrors the Arctic script
# --------------------------------------------------------------------------------------------------------------- #
def _wait_health(url: str, timeout: float, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vllm server exited early with code {server.returncode} before healthy")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2)
    raise TimeoutError(f"vllm server not healthy after {timeout}s (last: {last})")


def _tail(path: str, n: int = 120) -> str:
    try:
        with open(path) as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return "(no server log)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "10")))
    ap.add_argument("--num-generations", type=int, default=int(os.environ.get("NUM_GEN", "8")))
    ap.add_argument("--per-device-bsz", type=int, default=int(os.environ.get("PER_DEVICE_BSZ", "8")))
    ap.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "1")))
    ap.add_argument("--max-completion-length", type=int, default=int(os.environ.get("MAX_COMPLETION_LEN", "256")))
    ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "1024")))
    ap.add_argument("--num-prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", "64")))
    ap.add_argument("--learning-rate", type=float, default=float(os.environ.get("LR", "1e-6")))
    ap.add_argument("--server-startup-timeout", type=float, default=float(os.environ.get("SERVER_TIMEOUT", "1200")))
    ap.add_argument(
        "--metrics-out",
        default=os.environ.get(
            "METRICS_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_baseline.json")
        ),
    )
    # Matched-seed variance control: seeds both `vllm serve` (generation) and the trainer (AsyncGRPOConfig).
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl.experimental.async_grpo import AsyncGRPOConfig
    from trl.experimental.async_grpo import AsyncGRPOTrainer
    from trl.rewards import accuracy_reward

    chat_template_kwargs = {"enable_thinking": False}

    base_url = f"http://{args.host}:{args.port}"

    # ----- launch stock `vllm serve` on the server GPU as its own process group -----
    server_cmd = [
        "vllm",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--seed",
        str(args.seed),
        # `processed_logprobs`: the sampler returns post-temperature/top-k/top-p logprobs, matching how the trainer's
        # patched model computes its (temperature-scaled) logprobs -> ratio ~ 1 on-policy.
        "--logprobs-mode",
        "processed_logprobs",
        # NCCL weight transfer endpoints (/init_weight_transfer_engine, /update_weights, ...) used by WeightTransferClient.
        "--weight-transfer-config",
        '{"backend":"nccl"}',
    ]
    server_env = dict(os.environ)
    server_env["CUDA_VISIBLE_DEVICES"] = _SERVER_GPU
    server_env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes /pause,/resume,/*_weight_update (dev-only endpoints)
    print(f"[base] launching vllm serve on GPU {_SERVER_GPU}: {' '.join(server_cmd)}", flush=True)
    print(f"[base] trainer on GPU {_TRAINER_GPU}; torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)
    print(f"[base] server stdout/stderr -> {SERVER_LOG}", flush=True)
    logf = open(SERVER_LOG, "w")
    server = subprocess.Popen(
        server_cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, env=server_env
    )
    pgid = os.getpgid(server.pid)
    print(f"[base] vllm server pid={server.pid} pgid={pgid}", flush=True)

    ok = False
    out_dir = tempfile.mkdtemp(prefix="trl_base_out_")
    trainer = None
    try:
        _wait_health(f"{base_url}/health", args.server_startup_timeout, server)
        print("[base] vllm server healthy", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        split = "train" if args.num_prompts <= 0 else f"train[:{args.num_prompts}]"
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
        print(f"[base] dataset ready: {len(dataset)} prompts (split={split})", flush=True)

        config = AsyncGRPOConfig(
            output_dir=out_dir,
            save_strategy="no",
            seed=args.seed,
            per_device_train_batch_size=args.per_device_bsz,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            bf16=True,  # real GPU training; the Arctic server also trains bf16
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            temperature=1.0,
            chat_template_kwargs=chat_template_kwargs,
            max_steps=args.max_steps,
            weight_sync_steps=1,
            token_budget=0,  # FixedCountBatcher: per_device_train_batch_size samples/step (mirrors the Arctic run)
            logging_steps=1,
            report_to="none",
            log_completions=True,
            vllm_server_base_url=base_url,
        )

        # Native backends: pass nothing -> LocalTrainingClient + AsyncRolloutWorker + WeightTransferClient.
        trainer = AsyncGRPOTrainer(
            model=args.model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=accuracy_reward,
        )

        print(f"[base] starting trainer.train() (max_steps={args.max_steps}) ...", flush=True)
        trainer.train()

        history = [h for h in trainer.state.log_history if "loss" in h]
        print(f"[base] completed {trainer.state.global_step} optimizer steps", flush=True)
        for h in history[-10:]:
            keep = {k: h[k] for k in ("loss", "reward", "ratio", "kl", "entropy", "completions/mean_length") if k in h}
            print(f"[base]   step {h.get('step')}: {keep}", flush=True)

        if trainer.state.global_step >= args.max_steps:
            ok = True
            print("[base] BASELINE PASSED: native TRL async-GRPO ran end-to-end", flush=True)
        else:
            print(f"[base] BASELINE INCOMPLETE: only {trainer.state.global_step}/{args.max_steps} steps", flush=True)
    except Exception:
        print("[base] BASELINE FAILED with exception:", flush=True)
        traceback.print_exc()
    finally:
        # Always dump whatever metrics we have, so a partial run is still comparable.
        try:
            if trainer is not None:
                with open(args.metrics_out, "w") as f:
                    json.dump(trainer.state.log_history, f, indent=2)
                print(
                    f"[base] wrote metrics -> {args.metrics_out} ({len(trainer.state.log_history)} records)",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            print(f"[base] metrics dump raised (ignored): {e}", flush=True)
        print(f"[base] terminating vllm server process group pgid={pgid} ...", flush=True)
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
        logf.close()
        print("[base] ---- vllm server log tail ----", flush=True)
        print(_tail(SERVER_LOG), flush=True)
        print("[base] ---- end vllm server log tail ----", flush=True)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
