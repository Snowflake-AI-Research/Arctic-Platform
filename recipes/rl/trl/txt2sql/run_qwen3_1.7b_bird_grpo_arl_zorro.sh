#!/usr/bin/env bash
# Arctic TRL BIRD recipe with ZoRRo train + load balancer (C3). 4 train + 4 sample H200s.
# Same generate path as C2; train GAS defaults to 16.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-arctic-trl-bird}"
unset WANDB_MODE
unset WANDB_SILENT

export TOKEN_BUDGET="${TOKEN_BUDGET:-0}"
export NUM_GEN="${NUM_GEN:-16}"
export PER_DEVICE_BSZ="${PER_DEVICE_BSZ:-256}"
export GRAD_ACCUM="${GRAD_ACCUM:-16}"
export MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-4096}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-36864}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-40960}"
export TRAINING_GPUS="${TRAINING_GPUS:-4}"
export SAMPLING_GPUS="${SAMPLING_GPUS:-4}"
export ARCTIC_COLOCATE="${ARCTIC_COLOCATE:-0}"
export ARCTIC_ZORRO=1
export ARCTIC_ZORRO_LOAD_BALANCER="${ARCTIC_ZORRO_LOAD_BALANCER:-1}"
export ARCTIC_LOGITS_OPT="${ARCTIC_LOGITS_OPT:-memory}"
export USE_LIGER="${USE_LIGER:-1}"
export VLLM_PREFIX_CACHING="${VLLM_PREFIX_CACHING:-1}"
export ROLLOUT_GROUP_BATCH="${ROLLOUT_GROUP_BATCH:-16}"
export VAL_EVERY="${VAL_EVERY:-0}"

echo "== BIRD Arctic C3 (ZoRRo+LB): train=${TRAINING_GPUS} sample=${SAMPLING_GPUS} n=${NUM_GEN} bsz=${PER_DEVICE_BSZ} gas=${GRAD_ACCUM} =="
hostname; date -Iseconds

exec python -u run_qwen3_1.7b_bird_grpo_arl.py "$@"
