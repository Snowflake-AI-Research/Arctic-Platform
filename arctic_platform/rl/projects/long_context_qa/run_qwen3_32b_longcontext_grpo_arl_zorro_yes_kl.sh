#!/bin/bash
# GRPO training for Qwen3-32B on long-context QA (LoongRL-style) with ArcticRL + Zorro
# KL-enabled variant — matches ref verl uglrinrq (use_kl_loss=True, kl_loss_coef=0.001).
#
# Diff vs run_qwen3_32b_longcontext_grpo_arl_zorro_yes.sh:
#   - actor.use_kl_loss=True (ref model enabled for low_var_kl penalty)
#   - sampling_gpus / log_prob_gpus each NGPU_PER_JOB/2 (ref log-prob pool for KL)
#   - experiment_name suffix _kl
#
# Adapted from:
#   - verl_opensource/examples/long_context/run_qwen2_7b_longcontext_grpo.sh (training recipe)
#   - examples/arctic_rl/run_mega_bird_grpo_arl_zorro_yes.sh   (arctic_rl / Zorro template for 32B)
#
# 4 nodes, 8 GPUs each (32 H200 GPUs total), colocate=True
#
# Run Arctic then verl baseline sequentially via:
#   bash _examples/arctic_rl/launch_sequential_32b_kl_smokes.sh
#
# Prerequisites:
#   1. Download data: python examples/long_context/download_data.py  (in verl_opensource)
#      Resulting parquets must live at $DATA_DIR/merged/{train,test}.parquet
#   2. Multi-node ray cluster started across the H200 nodes (see /job/hostfile)
#   3. Arctic packages + arctic-verl installed (use the install-*.sh scripts under work_dir)

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Standard env (mirrors big_bird / mega_bird zorro perf profile)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export HF_HOME=/checkpoint/huggingface
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT=/modeling-checkpoints/vllm
export VLLM_LOGGING_LEVEL=INFO

# Pre-launch /dev/shm cleanup: NCCL / vllm / sem files accumulate across runs and
# can fill 100GiB tmpfs after a few iterations, killing raylets (SIGBUS / OOM).
# Cleanup is per-user so it won't disturb other workloads. Best-effort.
DS_SSH_HOSTFILE="${JOB_HOSTFILE:-/job/hostfile}"
DS_SSH_FLAGS=()
if [[ -n "${JOB_HOSTFILE:-}" ]]; then
    DS_SSH_FLAGS=(-f "${JOB_HOSTFILE}")
fi
if command -v ds_ssh >/dev/null 2>&1 && [[ -f "${DS_SSH_HOSTFILE}" ]]; then
    ds_ssh "${DS_SSH_FLAGS[@]}" "find /dev/shm -maxdepth 1 -user \$USER \
        \( -name 'nccl-*' -o -name 'cuda.shm.*' -o -name 'arctic_ws_*' \
           -o -name 'torch_*' -o -name 'sem.obj*' -o -name 'sem.hdr*' \
           -o -name 'sem.loky-*' -o -name 'psm_*' -o -name 'plasma*' \) \
        -delete 2>/dev/null; \
        echo \"\$(hostname): /dev/shm \$(df -h /dev/shm | tail -1 | awk '{print \$3\"/\"\$2}')\"" \
        2>&1 | tail -10
fi

HOSTFILE="/data-fast/hostfile"
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
ARCTIC_AUTOCAST=False    # match bird arctic intent (autocast off; bf16 weights used directly)

# Total GPUs derived from NGPU_PER_NODE * NNODES (matches verl_opensource recipe:
# nnodes=4, n_gpus_per_node=8 -> 32 GPUs).
# KL on: split rollout vs ref log-prob pools 50/50 (matches verl uglrinrq topology).
NGPU_PER_JOB=$((NGPU_PER_NODE*NNODES))
NGPU_FOR_SAMPLING=$((NGPU_PER_JOB/2))
NGPU_FOR_LOG_PROBS=$((NGPU_PER_JOB/2))
TP_SIZE=2                # sampling TP (matches verl_opensource rollout.tensor_model_parallel_size=2)

# ----- Training hyperparams (match verl_opensource long-context recipe) -----
BSZ=256                  # data.train_batch_size
PPO_MINI_BSZ=64          # actor.ppo_mini_batch_size
UBS=8                    # actor / rollout / ref micro_batch_size_per_gpu
ROLL_N=8                 # actor_rollout_ref.rollout.n
PROMPT_LEN=16384
RESPONSE_LEN=4096
MAX_TOKENS_PER_GPU=49152 # actor.ppo_max_token_len_per_gpu (>= prompt_len + ROLL_N * response_len for Zorro tiled compute)
ROLLOUT_MAX_BATCHED=32768
LR=1e-6
CLIP_RATIO=0.2
USE_KL_LOSS=True         # match ref verl uglrinrq; anchors policy vs frozen ref
KL_LOSS_COEF=0.001
TOTAL_EPOCHS=20
SAVE_FREQ=-1 # 10             # match verl baseline (save checkpoint every 10 steps)
TEST_FREQ=10             # match verl baseline (run validation every 10 steps)

LOGGER="['console']"
# LOGGER="['console','wandb']"

MODEL_SHORT=Qwen3-32B
MODEL=Qwen/${MODEL_SHORT}

experiment_name="longcontext_grpo_${MODEL_SHORT}_ngpu${NGPU_PER_JOB}_gbs${BSZ}_mbs${UBS}_rolln${ROLL_N}_arl_zorro_yes_z${ARCTIC_ZERO_STAGE}_kl"

gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0)
if [[ $gpu_name == *"H200"* ]]; then
    echo "Running on Hopper"
    flash_attention_v=flash_attention_3
elif [[ $gpu_name == *"B200"* ]] || [[ $gpu_name == *"B300"* ]] ; then
    echo "Running on Blackwell"
    flash_attention_v=flash_attention_2
else
    echo "Running on unknown: $gpu_name; don't know which FA version to use"
fi

# Data: LoongRL-Train-Data merged across HotpotQA + MuSiQue + 2WikiMQA, 16K context
DATA_DIR="/data/snowflakesql/xyu/long-context"
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
    trainer.use_arctic_rl=$USE_ARCTIC_RL \
    trainer.balance_batch=False \
    trainer.default_local_dir=/checkpoint/xyu/long-context-rl/$experiment_name \
    trainer.logger=$LOGGER \
    trainer.project_name=arctic_rl_long_context_$REAL_USER \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=False \
    arctic_rl.colocate=$COLOCATE \
    arctic_rl.logits_optimization=memory \
    arctic_rl.sampling_tp_size=$TP_SIZE \
    arctic_rl.training_gpus=$NGPU_PER_JOB \
    arctic_rl.sampling_gpus=$NGPU_FOR_SAMPLING \
    arctic_rl.log_prob_gpus=$NGPU_FOR_LOG_PROBS \
    arctic_rl.zorro_train.enable=$USE_ARCTIC_ZORRO \
    arctic_rl.zorro_train.max_rollouts=$ROLL_N \
    arctic_rl.use_autocast=$ARCTIC_AUTOCAST \
    arctic_rl.cuda_ipc_weight_sync=True \
    arctic_rl.zero_optimization.stage=$ARCTIC_ZERO_STAGE \
    arctic_rl.zero_optimization.offload_optimizer.device=cpu \
    arctic_rl.zero_optimization.offload_param.device=none \
    "$@" 2>&1 | tee $experiment_name.log
