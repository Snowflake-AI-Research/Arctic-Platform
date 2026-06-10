#!/bin/bash
# KL-enabled equivalent of run_qwen3_32b_bird_grpo_arl_zorro_yes.sh
#
# Diff vs non-KL bird arctic script:
#   - actor.use_kl_loss=True (ref model enabled for low_var_kl penalty)
#   - sampling_gpus / log_prob_gpus each NGPU_PER_JOB/2 (ref log-prob pool for KL)
#   - experiment_name suffix _kl
#
# Matches long-context KL arctic topology (arctic_claude run_qwen3_32b_longcontext_grpo_arl_zorro_yes_kl.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOSTFILE="/data-fast/hostfile"
if [[ -f ${HOSTFILE} ]]; then
    NNODES=$(wc -l < "${HOSTFILE}")
else
    NNODES=1
fi
NGPU_PER_NODE=8
NGPU_PER_JOB=$((NGPU_PER_NODE * NNODES))
NGPU_FOR_SAMPLING=$((NGPU_PER_JOB / 2))
NGPU_FOR_LOG_PROBS=$((NGPU_PER_JOB / 2))

exec bash "${SCRIPT_DIR}/run_qwen3_32b_bird_grpo_arl_zorro_yes.sh" \
    actor_rollout_ref.actor.use_kl_loss=True \
    trainer.experiment_name=qwen3_32b_bird_grpo_arl_zorro_yes_kl \
    arctic_rl.sampling_gpus="${NGPU_FOR_SAMPLING}" \
    arctic_rl.log_prob_gpus="${NGPU_FOR_LOG_PROBS}" \
    "$@"
