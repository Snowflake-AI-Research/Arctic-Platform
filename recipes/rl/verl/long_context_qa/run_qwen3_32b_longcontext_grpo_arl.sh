#!/bin/bash
# GRPO training for Qwen3-32B on long-context QA (LoongRL-style) with ArcticRL + Zorro
# Non-KL variant — pure GRPO, no frozen reference model (use_kl_loss=False).
#
# Adapted from:
#   - verl_opensource/examples/long_context/run_qwen2_7b_longcontext_grpo.sh (training recipe)
#   - examples/arctic_rl/run_mega_bird_grpo_arl_zorro_yes.sh   (arctic_rl / Zorro template for 32B)
#
# 4 nodes, 8 GPUs each (32 H200 GPUs total), colocate=True
#
# Prerequisites (see README.md):
#   1. Download data: python download_data.py
#      Resulting parquets must live at $DATA_DIR/merged/{train,test}.parquet
#   2. Packages installed on every node (README "Install packages": requirements.txt + overrides.txt, plus the
#      Snowflake verl fork installed editable)
#   3. Multi-node ray cluster started across the participating nodes (bash restart_multi_ray.sh)

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

HOSTFILE="${JOB_HOSTFILE:-/data-fast/hostfile}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Standard env (mirrors big_bird / mega_bird zorro perf profile)
# Do NOT set expandable_segments:True -- vLLM colocate sleep mode (cumem allocator) rejects it.
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export HF_HOME=/checkpoint/huggingface
export TORCH_COMPILE_DISABLE=1
# Select the Arctic training client for verl's remote_backend=arctic path.
export USE_ARCTIC_TRAINING_CLIENT=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT=/modeling-checkpoints/vllm
export VLLM_LOGGING_LEVEL=INFO

# Pre-launch /dev/shm cleanup: NCCL / vllm / sem files accumulate across runs and can fill 100GiB tmpfs after a few
# iterations, killing raylets (SIGBUS / OOM). Cleanup is per-user so it won't disturb other workloads. Best-effort.
DS_SSH_FLAGS=()
if [[ -n "${HOSTFILE:-}" ]]; then
    DS_SSH_FLAGS=(-f "${HOSTFILE}")
fi
if command -v ds_ssh >/dev/null 2>&1 && [[ -f "${HOSTFILE}" ]]; then
    ds_ssh "${DS_SSH_FLAGS[@]}" "find /dev/shm -maxdepth 1 -user \$USER \
        \( -name 'nccl-*' -o -name 'cuda.shm.*' -o -name 'arctic_ws_*' \
           -o -name 'torch_*' -o -name 'sem.obj*' -o -name 'sem.hdr*' \
           -o -name 'sem.loky-*' -o -name 'psm_*' -o -name 'plasma*' \) \
        -delete 2>/dev/null; \
        echo \"\$(hostname): /dev/shm \$(df -h /dev/shm | tail -1 | awk '{print \$3\"/\"\$2}')\"" \
        2>&1 | tail -10
fi

if [[ -f ${HOSTFILE} ]]; then
    NNODES=$(wc -l < ${HOSTFILE})
else
    NNODES=1
fi

# override here if you want a subset of available nodes
# NNODES=2 # to match ARL w/ colocation where 8 gpus per job are used
NGPU_PER_NODE=8

# ----- Arctic/Zorro topology -----
USE_LEGACY_WORKER_IMPL=disable
ROLLOUT_NAME=arctic

USE_ARCTIC_RL=True
USE_ARCTIC_ZORRO=True
COLOCATE=True
ARCTIC_ZERO_STAGE=3

# Total GPUs derived from NGPU_PER_NODE * NNODES (matches verl_opensource recipe: nnodes=4, n_gpus_per_node=8 -> 32
# GPUs). colocate=True bundle layout: training and sampling span all NGPU_PER_JOB bundles. log_prob is disabled
# (NGPU_FOR_LOG_PROBS=0): without KL there is no frozen ref model, and under Zorro log-probs are recomputed through
# the training engine itself.
NGPU_PER_JOB=$((NGPU_PER_NODE*NNODES))
NGPU_FOR_LOG_PROBS=0
TP_SIZE=2                # sampling TP (matches verl_opensource rollout.tensor_model_parallel_size=2)

# ----- Training hyperparams (match verl_opensource long-context recipe) -----
BSZ=256                  # data.train_batch_size
PPO_MINI_BSZ=64          # actor.ppo_mini_batch_size
UBS=8                    # actor / rollout / ref micro_batch_size_per_gpu
ROLL_N=8                 # actor_rollout_ref.rollout.n
PROMPT_LEN=16384
RESPONSE_LEN=4096
MAX_TOKENS_PER_GPU=49152 # actor.ppo_max_token_len_per_gpu (>= prompt_len + ROLL_N * response_len for Zorro tiles)
ROLLOUT_MAX_BATCHED=32768
LR=1e-6
CLIP_RATIO=0.2
USE_KL_LOSS=False        # pure GRPO, no frozen-ref KL anchoring
KL_LOSS_COEF=0.001       # unused when USE_KL_LOSS=False
TOTAL_EPOCHS=20
SAVE_FREQ=-1 # 10             # match verl baseline (save checkpoint every 10 steps)
TEST_FREQ=10             # match verl baseline (run validation every 10 steps)

LOGGER="['console']"
# if you want to use wandb, uncomment the following line and set the WANDB_API_KEY in your environment
# additionally edit below trainer.project_name and trainer.experiment_name entries to match your wandb project and
# experiment name
# LOGGER="['console','wandb']"

MODEL_SHORT=Qwen3-32B
MODEL=Qwen/${MODEL_SHORT}

experiment_name="longcontext_grpo_${MODEL_SHORT}_ngpu${NGPU_PER_JOB}_gbs${BSZ}_mbs${UBS}_rolln${ROLL_N}_arl_z${ARCTIC_ZERO_STAGE}"

# feel free to change below to hardcoded a particular attention implementation - the following logic tries to pick
# the best impelementation based on the gpu name
gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0)
if [[ $gpu_name == *"H100"* ]] || [[ $gpu_name == *"H200"* ]] ; then
    echo "Running on Hopper"
    flash_attention_v=flash_attention_3
elif [[ $gpu_name == *"B200"* ]] || [[ $gpu_name == *"B300"* ]] ; then
    echo "Running on Blackwell"
    flash_attention_v=flash_attention_2
else
    echo "Running on unknown: $gpu_name; don't know which FA version to use"
fi

# Data: LoongRL-Train-Data merged across HotpotQA + MuSiQue + 2WikiMQA, 16K context
DATA_DIR="/data/snowflakesql/long-context"
TRAIN_FILES="${DATA_DIR}/merged/train.parquet"
VAL_FILES="${DATA_DIR}/merged/test.parquet"

# In Zorro, log-probs are recomputed through the training engine, so we keep this off
LOG_PROBS=False
FREE_CACHE_ENGINE=True

python3 -m verl.trainer.main_ppo \
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
    data.filter_overlong_prompts_workers=8 \
    data.truncation=left \
    data.seed=42 \
    custom_reward_function.path=${SCRIPT_DIR}/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$flash_attention_v \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.n=$ROLL_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=$LOG_PROBS \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.nccl_timeout=1800 \
    trainer.use_legacy_worker_impl=$USE_LEGACY_WORKER_IMPL \
    trainer.remote_backend=arctic \
    remote_backend=arctic \
    trainer.balance_batch=False \
    trainer.default_local_dir=/checkpoint/long-context-rl/$experiment_name \
    trainer.logger=$LOGGER \
    trainer.project_name=arctic_rl_long_context \
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
    remote_backend.train.deepspeed.zero_optimization.offload_optimizer.device=cpu \
    remote_backend.train.deepspeed.zero_optimization.offload_param.device=none \
    remote_backend.train.deepspeed.zero_optimization.stage=$ARCTIC_ZERO_STAGE \
    remote_backend.train.logits.optimization=memory \
    remote_backend.train.zorro_train.enable=$USE_ARCTIC_ZORRO \
    remote_backend.train.zorro_train.max_rollouts=$ROLL_N \
    remote_backend.training_gpus=$NGPU_PER_JOB \
    remote_backend.weight_sync.cuda_ipc=False \
    "$@" 2>&1 | tee $experiment_name.log
