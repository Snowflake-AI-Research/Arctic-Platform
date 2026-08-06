#!/bin/bash
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

# Cortex sibling of run_gsm8k_grpo_arl.sh. ARCTIC_BACKEND=cortex routes the
# adapter through arctic_platform/rl/_cortex_dispatch.py; nothing else in
# verl or arctic_platform/integrations/verl/ has to change.
#
# See README-cortex.md for prereqs and credentials.

set -x
set -euo pipefail

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rayk}
mkdir -p "$HF_HOME" "$TMPDIR"

: "${CORTEX_PAT:?CORTEX_PAT missing; export it from your Snowflake programmatic access token}"
: "${ARCTIC_CORTEX_HOST:?ARCTIC_CORTEX_HOST missing (e.g. <account>.<region>.snowflakecomputing.com)}"

export ARCTIC_BACKEND=cortex
export ARCTIC_CORTEX_DATABASE="${ARCTIC_CORTEX_DATABASE:-NEUTRINO_DB}"
export ARCTIC_CORTEX_SCHEMA="${ARCTIC_CORTEX_SCHEMA:-PUBLIC}"
export ARCTIC_CORTEX_ENDPOINT="${ARCTIC_CORTEX_ENDPOINT:-cortex-training}"
export ARCTIC_CORTEX_PAT_ENV_VAR=CORTEX_PAT

export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCTIC_VERL_CONFIG_DIR="${SCRIPT_DIR}/../config"

NGPU_TRAIN="${NGPU_TRAIN:-4}"
NGPU_SAMPLE="${NGPU_SAMPLE:-4}"

DATA_DIR="${DATA_DIR:-${HOME}/data/gsm8k}"
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/test.parquet" ]]; then
    echo "ERROR: GSM8K parquets not found at ${DATA_DIR}/{train,test}.parquet"
    echo "       Prepare them with the sibling arl launcher's download_data.py first."
    exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LR="${LR:-1e-6}"
TRAIN_BSZ="${TRAIN_BSZ:-16}"
MINI_BSZ="${MINI_BSZ:-16}"
ROLLOUT_N="${ROLLOUT_N:-5}"
TOTAL_STEPS="${TOTAL_STEPS:-40}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gsm8k_grpo_qwen3_0p6b_cortex}"

echo "[cortex-verl] host=${ARCTIC_CORTEX_HOST} db=${ARCTIC_CORTEX_DATABASE} endpoint=${ARCTIC_CORTEX_ENDPOINT}"
echo "[cortex-verl] training_gpus=${NGPU_TRAIN} sampling_gpus=${NGPU_SAMPLE} model=${MODEL}"

python3 -m verl.trainer.main_ppo \
    hydra.searchpath="[file://${ARCTIC_VERL_CONFIG_DIR}]" \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=${TRAIN_BSZ} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    +data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    reward.num_workers=1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=arctic \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.remote_backend=arctic \
    remote_backend=arctic \
    remote_backend.colocate=False \
    "remote_backend.training_gpus=${NGPU_TRAIN}" \
    "remote_backend.sampling_gpus=${NGPU_SAMPLE}" \
    remote_backend.log_prob_gpus=0 \
    "remote_backend.sampling_tp_size=${NGPU_SAMPLE}" \
    remote_backend.train.deepspeed.zero_optimization.stage=2 \
    remote_backend.train.deepspeed.zero_optimization.offload_optimizer.device=none \
    remote_backend.train.deepspeed.zero_optimization.offload_param.device=none \
    remote_backend.train.zorro_train.enable=False \
    remote_backend.weight_sync.cuda_ipc=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    trainer.project_name=arctic_verl_cortex_gsm8k \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "trainer.total_training_steps=${TOTAL_STEPS}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
    "$@" 2>&1
