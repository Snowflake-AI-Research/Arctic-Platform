#!/usr/bin/env bash
# Arctic TRL GSM8K starter. Defaults match run_qwen3_1.7b_gsm8k_grpo_arl.py
# (8-GPU colocate). Override with ARCTIC_COLOCATE=0 TRAINING_GPUS=4 SAMPLING_GPUS=4
# for the disaggregated layout used by the BIRD recipe.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

echo "== GSM8K Arctic GRPO =="
hostname; date -Iseconds

exec python -u run_qwen3_1.7b_gsm8k_grpo_arl.py "$@"
