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
"""Unified SFT training example across the on-prem and remote backends.

    python arctic_platform/client/examples/sft_example.py --backend onprem-http
    python arctic_platform/client/examples/sft_example.py --backend onprem-ray
    CORTEX_PAT=... python arctic_platform/client/examples/sft_example.py --backend remote-cortex

Every backend follows the *same* pathway: build config -> ArcticRLClient ->
loop(fwd_bwd + step) -> shutdown, with a single unified fwd_bwd/step/report.
The client + transports hide all wire/protocol differences; only the config and
the fwd_bwd batch shape differ per backend (remote Cortex tokenizes an RPC-style
{"args", "kwargs"} body; on-prem sends a pre-tokenized verl-GRPO payload).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

ARCTIC = Path(__file__).resolve().parents[3]  # Arctic-Platform/
# Run-from-anywhere: the package root + the RL test harness on the path.
sys.path.insert(0, str(ARCTIC))
sys.path.insert(0, str(ARCTIC / "tests" / "rl"))

from arctic_platform.client import ArcticRLClientConfig  # noqa: E402
from arctic_platform.client import CortexConfig  # noqa: E402
from arctic_platform.client import OnPremConfig  # noqa: E402
from arctic_platform.client import SyncArcticRLClient  # noqa: E402
from arctic_platform.client import TrainingConfig  # noqa: E402

STEPS = 20
SEED = 42
N_GPUS = 8

MODEL = "Qwen/Qwen3-0.6B"
LR = 1e-5
PROMPT = "who trained you?\nMichael Wyatt at Snowflake"
PROMPT_LEN = 8
RESPONSE_LEN = 8
SEQ_LEN = PROMPT_LEN + RESPONSE_LEN
ROLLOUT_N = 1
ATTN = "sdpa"  # flash_attention_2 needs the flash_attn package; sdpa ships with torch


def _metric(x) -> float:
    """step merges across DP ranks, so a replicated scalar can arrive as a per-rank list."""
    return float(x[0] if isinstance(x, (list, tuple)) else x)


# ── per-backend: config ──────────────────────────────────────────────────────
def _onprem_config(comm_protocol: str, launch_local_server: bool) -> Callable:
    def build(stack: contextlib.ExitStack) -> ArcticRLClientConfig:
        ckpt = stack.enter_context(tempfile.TemporaryDirectory(prefix="arl_onprem_ckpt_"))
        return ArcticRLClientConfig(
            model_name=MODEL,
            seed=SEED,
            max_seq_len=SEQ_LEN,
            training_gpus=N_GPUS,
            job_ready_timeout=600.0,
            backend_config=OnPremConfig(
                comm_protocol=comm_protocol,
                launch_local_server=launch_local_server,
            ),
            training=TrainingConfig(
                checkpoint_path=ckpt,  # server requires this for training jobs
                ds_worker_config={"attn_implementation": ATTN},
                ds_config={
                    "train_micro_batch_size_per_gpu": 1,
                    "train_batch_size": N_GPUS,
                    "gradient_accumulation_steps": 1,
                    "gradient_clipping": 1.0,
                    "optimizer": {
                        "type": "AdamW",
                        "params": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
                    },
                    "zero_optimization": {
                        "stage": 3,
                        "offload_optimizer": {"device": "none"},
                        "offload_param": {"device": "none"},
                    },
                },
            ),
        )

    return build


# ── Cortex (SnowAPI) connection info — non-secret; PAT comes from CORTEX_PAT ───
CORTEX_HOST = "dsa-test.qa6.us-west-2.aws.snowflakecomputing.com"
CORTEX_DATABASE = "NEUTRINO_DB"
CORTEX_SCHEMA = "PUBLIC"
CORTEX_ENDPOINT = "cortex-training"


def _cortex_config() -> Callable:
    def build(stack: contextlib.ExitStack) -> ArcticRLClientConfig:
        return ArcticRLClientConfig(
            model_name=MODEL,
            seed=SEED,
            max_seq_len=SEQ_LEN,
            training_gpus=N_GPUS,
            job_ready_timeout=3600.0,
            backend_config=CortexConfig(
                host=CORTEX_HOST,
                database=CORTEX_DATABASE,
                schema=CORTEX_SCHEMA,
                endpoint=CORTEX_ENDPOINT,
                pat_env_var="CORTEX_PAT",
            ),
            training=TrainingConfig(
                ds_worker_config={"attn_implementation": ATTN, "model_provider": "huggingface"},
                ds_config={
                    "train_batch_size": N_GPUS,
                    "gradient_clipping": 1.0,
                    "optimizer": {
                        "type": "AdamW",
                        "params": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
                    },
                },
            ),
        )

    return build


# ── shared data, per-backend packaging ───────────────────────────────────────
def _tokens() -> dict:
    """Tokenize the shared PROMPT with the shared MODEL — the payload the backend sends."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        [PROMPT] * N_GPUS,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].contiguous()
    attention_mask = encoded["attention_mask"].contiguous()
    # `prompts` = the prompt-length slice, matching the verl batch key set.
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": input_ids[:, :PROMPT_LEN].contiguous(),
    }


def _onprem_batch(tokens: dict) -> Any:
    from rl_harness import build_update_actor_payload

    # `processing` stays inside the payload: client.fwd_bwd folds a `processing=`
    # kwarg into the body anyway, so leaving it in yields the identical wire body
    # and lets the backend share the same `client.fwd_bwd(batch)` call.
    return build_update_actor_payload(tokens, False, ROLLOUT_N, PROMPT_LEN, RESPONSE_LEN)


def _cortex_batch(tokens: dict) -> Any:
    """Cortex fwd_bwd wants an RPC-style {"args", "kwargs"} body of client-side tensors.

    Labels use the next-token strategy (shift left, pad -> -100), matching the
    Neutrino client's ``build_forward_backward_kwargs``; the server does the causal shift.
    """
    import torch

    input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    shifted_mask = torch.roll(attention_mask, shifts=-1, dims=1)
    shifted_mask[:, -1] = 0
    labels = labels.masked_fill(shifted_mask == 0, -100)
    kwargs = {
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "labels": labels.contiguous(),
    }
    return {"args": [], "kwargs": kwargs}


# ── unified across all backends ──────────────────────────────────────────────
def _report(step: int, out: dict, step_out: dict) -> str:
    """One report for every backend, via graceful key lookups on the responses."""
    loss = _metric(out.get("avg_loss", out.get("loss")))
    line = f"step {step + 1}/{STEPS} loss={loss:.4g}"
    grad_norm = (step_out or {}).get("metrics", {}).get("grad_norm")
    if grad_norm is not None:
        line += f" grad_norm={_metric(grad_norm):.4g}"
    return line


@dataclass
class Profile:
    config: Callable[[contextlib.ExitStack], ArcticRLClientConfig]
    batch: Callable[[dict], Any]  # packages the shared tokens into the backend's wire shape


BACKENDS: dict[str, Profile] = {
    "onprem-http": Profile(_onprem_config("http", True), _onprem_batch),
    "onprem-ray": Profile(_onprem_config("ray", False), _onprem_batch),
    "remote-cortex": Profile(_cortex_config(), _cortex_batch),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=list(BACKENDS), default="onprem-http")
    profile = BACKENDS[ap.parse_args().backend]

    with contextlib.ExitStack() as stack:
        config = profile.config(stack)
        batch = profile.batch(_tokens())  # same tokens, backend-specific wire shape

        client = SyncArcticRLClient(config)
        print(f"training job: {client.jobs.training}")
        try:
            for step in range(STEPS):
                # RL note: a full loop would create a sampling job, generate() a
                # rollout, score it into advantages, then feed that here.
                out = client.fwd_bwd(batch)
                step_out = client.step()
                print(_report(step, out, step_out))
        finally:
            client.shutdown()
            print("shutdown complete")


if __name__ == "__main__":
    main()
