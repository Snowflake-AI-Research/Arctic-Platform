#!/bin/bash
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

# Canonical GSM8K example for the Arctic RemoteBackend on verl.
#
# Demonstrates the generic ``verl.remote_backend`` abstraction with the
# Arctic adapter (``trainer.remote_backend=arctic``). Single-GPU, GRPO,
# Qwen3-0.6B; intended as a quick convergence sanity check and as the
# reference launcher for Golden Run 1 in Arctic-Platform#35.
#
# Requirements:
#   pip install "arctic_platform[verl]"
#   pip install verl==<version pinned in your launcher>
#
# The verl side is loaded via the VERL_USE_EXTERNAL_MODULES plugin hook
# below; no verl source-tree modification is required.

set -x
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export USE_ARCTIC_TRAINING_CLIENT=1
export VLLM_BATCH_INVARIANT=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Plug the Arctic backend into verl without patching the verl source tree.
export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register

# Add the plugin's config dir to Hydra's search path so `remote_backend=arctic`
# resolves to `arctic_platform/integrations/verl/config/arctic.yaml`. Hydra
# accepts either `file://` URIs or plain absolute paths in `hydra.searchpath`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCTIC_VERL_CONFIG_DIR="${SCRIPT_DIR}/../config"

gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 2>/dev/null || echo "")
if   [[ $gpu_name == *"H200"* ]]; then flash_attention_v=flash_attention_3
elif [[ $gpu_name == *"B200"* || $gpu_name == *"B300"* ]]; then flash_attention_v=flash_attention_2
else flash_attention_v=flash_attention_2
fi

DATA_DIR="${DATA_DIR:-/code/shared/gsm8k}"

python3 -m verl.trainer.main_ppo \
    hydra.searchpath="[file://${ARCTIC_VERL_CONFIG_DIR}]" \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    +data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    reward.num_workers=1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$flash_attention_v \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=arctic \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.remote_backend=arctic \
    remote_backend=arctic \
    remote_backend.colocate=False \
    remote_backend.training_gpus=1 \
    remote_backend.sampling_gpus=1 \
    remote_backend.log_prob_gpus=0 \
    remote_backend.train.deepspeed.zero_optimization.stage=2 \
    remote_backend.train.deepspeed.zero_optimization.offload_optimizer.device=none \
    remote_backend.train.deepspeed.zero_optimization.offload_param.device=none \
    remote_backend.train.zorro_train.enable=True \
    remote_backend.weight_sync.cuda_ipc=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.experiment_name=gsm8k_grpo_qwen3_0p6b_ngpu1_gbs16_rolln5_zorroTrue \
    trainer.project_name=arctic_rl_gsm8k_arctic_platform \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=80 \
    trainer.total_epochs=15 \
    "$@" 2>&1
