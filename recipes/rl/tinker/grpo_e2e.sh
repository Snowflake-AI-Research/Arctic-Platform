#!/usr/bin/env bash
# End-to-end GRPO run against Arctic via the Tinker HTTP layer.
#
# This is the recipe SkyRL-tx runs in its own GPU CI
# (SkyRL/tests/train/gpu_e2e_test/gsm8k_tinker.sh): boot a Tinker-compatible
# server, then drive it with Thinking Machines' upstream
# ``tinker_cookbook.recipes.math_rl.train`` — group-relative advantages
# (GRPO), no critic. Swapping SkyRL-tx's server for Arctic keeps the client
# unchanged; convergence == Tinker-wire parity.
#
# Prereqs:
#   1. Arctic server already booted on $URL (default http://localhost:7000):
#        python -m arctic_platform.rl.http_server \
#          --host 0.0.0.0 --port 7000 \
#          --training-gpus 1 --sampling-gpus 1 --colocate
#   2. tinker-cookbook checked out at $COOKBOOK_DIR (or set the path).
#
# Env vars (with defaults):
#   URL             Arctic base URL                             http://localhost:7000
#   MODEL           HF id                                       Qwen/Qwen3-0.6B
#   ENV             math_rl env: arithmetic | gsm8k | math      arithmetic
#   GROUP_SIZE      rollouts per prompt                         4
#   GROUPS_PER_BATCH prompts per step                           8
#   MAX_TOKENS      max response tokens                         128
#   MAX_STEPS       train steps                                 20
#   LR              learning rate                               1e-5
#   LOG_DIR         where math_rl.train writes ml_log/wandb     /tmp/arctic_tinker_grpo
#   COOKBOOK_DIR   tinker-cookbook path                        /data-fast/karthik/tinker-work/tinker-cookbook
#   WANDB_PROJECT  optional (skips wandb if unset)

set -euo pipefail

: "${URL:=http://localhost:7000}"
: "${MODEL:=Qwen/Qwen3-0.6B}"
: "${ENV:=arithmetic}"
: "${GROUP_SIZE:=4}"
: "${GROUPS_PER_BATCH:=8}"
: "${MAX_TOKENS:=128}"
: "${MAX_STEPS:=20}"
: "${LR:=1e-5}"
: "${LOG_DIR:=/tmp/arctic_tinker_grpo}"
: "${COOKBOOK_DIR:=/data-fast/karthik/tinker-work/tinker-cookbook}"

if [[ ! -d "$COOKBOOK_DIR" ]]; then
  echo "COOKBOOK_DIR=$COOKBOOK_DIR not found. Clone tinker-cookbook first:" >&2
  echo "  git clone https://github.com/thinking-machines-lab/tinker-cookbook.git $COOKBOOK_DIR" >&2
  exit 1
fi

# Bind the Tinker layer onto the running server if it isn't already.
if ! curl -sSf "$URL/api/v1/healthz" | grep -q '"bound":true'; then
  echo "[grpo_e2e] Tinker layer not bound; provisioning + binding via serve.sh..." >&2
  URL="$URL" MODEL="$MODEL" MAX_RESPONSE="$(( MAX_TOKENS + 64 ))" ZORRO_ENABLE=0 \
    "$(dirname "$0")/serve.sh"
fi

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

WANDB_ARGS=""
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS="wandb_project=$WANDB_PROJECT wandb_name=arctic_tinker_grpo_$(date +%H%M)"
fi

export PYTHONPATH="$COOKBOOK_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"
export TINKER_BASE_URL="$URL"

echo "[grpo_e2e] launching math_rl.train env=$ENV model=$MODEL steps=$MAX_STEPS" >&2

# math_rl.train uses chz for config; args are key=value.
exec python -u -m tinker_cookbook.recipes.math_rl.train \
  base_url="$URL" \
  model_name="$MODEL" \
  lora_rank=0 \
  env="$ENV" \
  group_size="$GROUP_SIZE" \
  groups_per_batch="$GROUPS_PER_BATCH" \
  max_tokens="$MAX_TOKENS" \
  max_steps="$MAX_STEPS" \
  learning_rate="$LR" \
  log_path="$LOG_DIR" \
  behavior_if_log_dir_exists=delete \
  $WANDB_ARGS
