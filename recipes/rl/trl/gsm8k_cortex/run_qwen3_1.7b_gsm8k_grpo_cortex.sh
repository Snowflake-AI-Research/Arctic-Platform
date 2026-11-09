#!/usr/bin/env bash
# CPU-only TRL GSM8K GRPO on Cortex. ZoRRo is off. Cortex owns the GPUs.
# Defaults are the 250-step trainer-π_old recipe. Stop the driver with
# kill -INT, not SIGTERM: the client cancels its Cortex job on the way out.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

if [[ -z "${ARCTIC_CORTEX_PAT:-}" && -n "${CORTEX_PAT:-}" ]]; then
  export ARCTIC_CORTEX_PAT="${CORTEX_PAT}"
fi
missing=()
for var in ARCTIC_CORTEX_HOST ARCTIC_CORTEX_DATABASE ARCTIC_CORTEX_SCHEMA ARCTIC_CORTEX_PAT; do
  if [[ -z "${!var:-}" ]]; then
    missing+=("${var}")
  fi
done
if ((${#missing[@]})); then
  echo "Set ${missing[*]} before launching. Do not print the PAT." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
export ARCTIC_ATTN_IMPL="${ARCTIC_ATTN_IMPL:-sdpa}"
export ARCTIC_ZERO_STAGE="${ARCTIC_ZERO_STAGE:-1}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
export VLLM_PREFIX_CACHING="${VLLM_PREFIX_CACHING:-1}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export ROLLOUT_GROUP_BATCH="${ROLLOUT_GROUP_BATCH:-16}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-16384}"

echo "== GSM8K TRL GRPO on Cortex (ZoRRo off) =="
hostname
date -Iseconds
echo "attn=${ARCTIC_ATTN_IMPL} zero=${ARCTIC_ZERO_STAGE} steps=${MAX_STEPS:-250} old_logprobs=${OLD_LOGPROBS_SOURCE:-trainer}"

exec python -u run_qwen3_1.7b_gsm8k_grpo_cortex.py \
  --max-steps "${MAX_STEPS:-250}" \
  --num-train-epochs 1 \
  --num-prompts -1 \
  --training-gpus "${TRAINING_GPUS:-4}" \
  --sampling-gpus "${SAMPLING_GPUS:-4}" \
  --num-generations "${NUM_GEN:-8}" \
  --per-device-bsz "${PER_DEVICE_BSZ:-32}" \
  --max-completion-length "${MAX_COMPLETION_LEN:-256}" \
  --max-seq-len "${MAX_SEQ_LEN:-1024}" \
  --seed "${SEED:-42}" \
  --max-staleness "${MAX_STALENESS:-0}" \
  --old-logprobs-source "${OLD_LOGPROBS_SOURCE:-trainer}" \
  "$@"
