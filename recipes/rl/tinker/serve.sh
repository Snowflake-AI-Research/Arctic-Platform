#!/usr/bin/env bash
# Provision an Arctic RL server behind the Tinker HTTP surface.
#
# Split of concerns:
#   • The Arctic server owns *all* provisioning (ZoRRo, ZeRO stage, offload,
#     vLLM knobs) via ``POST /initialize``.
#   • ``POST /tinker/bind`` is a pure adapter — it wires two existing job
#     ids into the Tinker HTTP verbs on the same server. No sizing, no
#     optimization defaults, no reinvention.
#
# Boot the server first (native flags), then run this script.
#
# Example (colocated 8B on 8 H200s with ZoRRo + ZeRO-3):
#
#   python -m arctic_platform.rl.http_server \
#       --host 0.0.0.0 --port 7000 \
#       --training-gpus 8 --sampling-gpus 8 --colocate
#
#   MODEL=Qwen/Qwen3-8B ROLLOUT_N=16 ZERO_STAGE=3 ZORRO_ENABLE=1 \
#     recipes/rl/tinker/serve.sh

set -euo pipefail

: "${URL:=http://localhost:7000}"
: "${MODEL:=Qwen/Qwen3-1.7B}"
: "${MAX_PROMPT:=1024}"
: "${MAX_RESPONSE:=512}"
: "${ZERO_STAGE:=3}"
: "${LR:=1e-6}"
: "${ROLLOUT_N:=}"
: "${TEMPERATURE:=1.0}"
: "${ZORRO_ENABLE:=0}"
: "${USE_LIGER:=0}"
: "${GPU_MEMORY_UTIL:=0.6}"
: "${CKPT_DIR:=/tmp/arctic_tinker_ckpt}"

export URL MODEL MAX_PROMPT MAX_RESPONSE ZERO_STAGE LR ROLLOUT_N \
       TEMPERATURE ZORRO_ENABLE USE_LIGER GPU_MEMORY_UTIL CKPT_DIR

/usr/bin/env python3 - <<'PY'
import json
import os
import urllib.request

URL = os.environ["URL"]
MODEL = os.environ["MODEL"]
MAX_PROMPT = int(os.environ["MAX_PROMPT"])
MAX_RESPONSE = int(os.environ["MAX_RESPONSE"])
ZERO_STAGE = int(os.environ["ZERO_STAGE"])
LR = float(os.environ["LR"])
TEMPERATURE = float(os.environ["TEMPERATURE"])
ZORRO_ENABLE = bool(int(os.environ["ZORRO_ENABLE"]))
USE_LIGER = bool(int(os.environ["USE_LIGER"]))
GPU_MEMORY_UTIL = float(os.environ["GPU_MEMORY_UTIL"])
CKPT_DIR = os.environ["CKPT_DIR"]


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{URL}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{URL}{path}") as resp:
        return json.loads(resp.read().decode())


training_gpus = int(get("/status")["training_gpus"])

ds_worker_config = dict(
    use_liger=USE_LIGER,
    enable_gradient_checkpointing=True,
    attn_implementation="flash_attention_2",
)
if ZORRO_ENABLE:
    rn = os.environ.get("ROLLOUT_N") or ""
    assert rn.isdigit() and int(rn) > 0, "ROLLOUT_N required when ZORRO_ENABLE=1"
    ds_worker_config.update(
        zorro_train_enable=True,
        rollout_n=int(rn),
        temperature=TEMPERATURE,
        response_len=MAX_RESPONSE,
        max_token_len=MAX_PROMPT + MAX_RESPONSE,
        use_unpad=True,
        logits_optimization="memory",
    )

train = post("/initialize", {
    "job_type": "training",
    "model_name": MODEL,
    "ds_config": {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": training_gpus,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": ZERO_STAGE},
        "bf16": {"enabled": True},
        "data_types": {"grad_accum_dtype": "bf16"},
    },
    "ds_worker_config": ds_worker_config,
    "training_config": {
        "optimizer": {"lr": LR, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        "lr_scheduler": {"warmup_ratio": 0.0},
        "training_horizon": 1_000_000,
    },
    "checkpoint_path": CKPT_DIR,
})
print(f"[serve] training job={train['job_id']} zorro={ZORRO_ENABLE} zero={ZERO_STAGE}", flush=True)

sample = post("/initialize", {
    "job_type": "sampling",
    "model_name": MODEL,
    "vllm_config": {
        "max_model_len": MAX_PROMPT + MAX_RESPONSE,
        "gpu_memory_utilization": GPU_MEMORY_UTIL,
    },
})
print(f"[serve] sampling job={sample['job_id']}", flush=True)

bind = post("/tinker/bind", {
    "training_job_id": train["job_id"],
    "sampling_job_id": sample["job_id"],
    "base_model": MODEL,
    "max_prompt_length": MAX_PROMPT,
    "max_response_length": MAX_RESPONSE,
})
print(f"[serve] bound: {json.dumps(bind)}", flush=True)
print(f"[serve] tinker ready at {URL}", flush=True)
PY
