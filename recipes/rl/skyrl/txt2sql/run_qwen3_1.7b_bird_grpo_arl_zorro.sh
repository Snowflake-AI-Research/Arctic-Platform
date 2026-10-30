#!/bin/bash
# SkyRL C3: the 1.7B BIRD recipe with ZoRRo on. Same step as C2.
# TRL twin: recipes/rl/trl/txt2sql/run_qwen3_1.7b_bird_grpo_arl_zorro.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export USE_ZORRO=1
exec "${HERE}/run_qwen3_1.7b_bird_grpo_arl.sh" "$@"
