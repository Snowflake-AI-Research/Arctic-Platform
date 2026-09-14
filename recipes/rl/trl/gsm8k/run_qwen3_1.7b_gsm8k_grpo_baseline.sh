#!/usr/bin/env bash
# Native TRL GSM8K baseline (trainer GPU 0 + vllm serve on GPU 1).
# Override with BASELINE_TRAINER_GPU / BASELINE_SERVER_GPU.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

echo "== GSM8K native-TRL baseline =="
hostname; date -Iseconds

exec python -u run_qwen3_1.7b_gsm8k_grpo_baseline.py "$@"
