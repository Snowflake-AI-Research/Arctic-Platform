#!/bin/bash
# verl × Arctic-Platform × Cortex-training on Qwen3-0.6B / GSM8K.
# Cortex sub-jobs own training + sampling; the verl driver is CPU-only.
#
# Pre-reqs (see README-cortex.md):
#   1. pip install arctic-platform[cortex]  (client-only; skips DeepSpeed/vLLM)
#   2. Data: python .../recipes/rl/verl/simple/download_data.py -> $DATA_DIR/{train,test}.parquet
#   3. ARCTIC_BACKEND=cortex + ARCTIC_CORTEX_* env vars set.

set -x

if [[ "${ARCTIC_BACKEND:-}" != "cortex" ]]; then
    echo "ERROR: ARCTIC_BACKEND=cortex is required. See README-cortex.md."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# Plug the Arctic RemoteBackend into verl.
export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register
ARCTIC_VERL_CONFIG_DIR="${REPO_ROOT}/arctic_platform/integrations/verl/config"

# Cortex owns the GPUs; verl workers run against remote sub-jobs, not local.
USE_LEGACY_WORKER_IMPL=disable
ROLLOUT_NAME=arctic
COLOCATE=False               # Cortex has no colocation lifecycle
NGPU_PER_JOB=1               # target Cortex sub-job GPU count
NGPU_FOR_LOG_PROBS=0         # no /forward on Cortex; zero-fill via _zero_logprob_response
TP_SIZE=1

BSZ=32
PPO_MINI_BSZ=32
UBS=8
ROLL_N=8
PROMPT_LEN=1024
RESPONSE_LEN=1024
MAX_TOKENS_PER_GPU=16384
ROLLOUT_MAX_BATCHED=16384
LR=1e-6
CLIP_RATIO=0.2
USE_KL_LOSS=False            # required: Cortex has no /forward for ref log-probs
KL_LOSS_COEF=0.001
TOTAL_EPOCHS=1
SAVE_FREQ=-1
TEST_FREQ=10

LOGGER="['console']"

MODEL_SHORT="${MODEL_SHORT:-Qwen3-0.6B}"
MODEL="${MODEL:-Qwen/${MODEL_SHORT}}"
experiment_name="gsm8k_grpo_${MODEL_SHORT}_cortex"

# Cortex training image ships without FA2; sdpa is the only attn impl the
# training sub-job can construct today.
flash_attention_v=sdpa

DATA_DIR="${DATA_DIR:-${HOME}/data/gsm8k}"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/test.parquet"

CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/gsm8k-rl-cortex}"

python3 -m verl.trainer.main_ppo \
    hydra.searchpath="[file://${ARCTIC_VERL_CONFIG_DIR}]" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=$PROMPT_LEN \
    data.max_response_length=$RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    data.truncation=left \
    data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$flash_attention_v \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLL_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED \
    actor_rollout_ref.nccl_timeout=1800 \
    trainer.use_legacy_worker_impl=$USE_LEGACY_WORKER_IMPL \
    trainer.remote_backend=arctic \
    remote_backend=arctic \
    trainer.balance_batch=False \
    trainer.default_local_dir=$CKPT_DIR/$experiment_name \
    trainer.logger=$LOGGER \
    trainer.project_name=arctic_rl_gsm8k_cortex \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=False \
    remote_backend.colocate=$COLOCATE \
    remote_backend.log_prob_gpus=$NGPU_FOR_LOG_PROBS \
    remote_backend.sampling_gpus=$NGPU_PER_JOB \
    remote_backend.sampling_tp_size=$TP_SIZE \
    remote_backend.train.deepspeed.zero_optimization.stage=2 \
    remote_backend.training_gpus=$NGPU_PER_JOB \
    "$@" 2>&1 | tee $experiment_name.log
