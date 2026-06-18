#!/bin/bash
# GRPO training for Qwen3-32B on BIRD SQL dataset, using ArcticRL + Zorro.
#
# Mirrors the training hyperparameters of
#   verl_opensource/examples/bird_sql/run_qwen3_32b_bird_grpo.sh
# so that wall-clock speed can be compared apples-to-apples against the
# stock-verl baseline. Only the rollout backend and parallelism wiring
# differ (ArcticRL colocate + Zorro instead of standalone vllm + FSDP2).
#
# Topology (same effective resources as the verl baseline):
#   4 nodes x 8 GPUs = 32 H200 GPUs, COLOCATE=True
#
# Prerequisites:
#   1. Preprocess data:  python examples/bird_sql/preprocess_bird.py
#                        (parquets at $DATA_DIR/{train,val}.parquet)
#   2. pip install func_timeout
#   3. Multi-node ray cluster started across the H200 nodes (see /data-fast/hostfile)
#   4. Arctic packages + arctic-verl installed (see install-*.sh scripts under work_dir)

set -x

experiment_name='qwen3_32b_bird_grpo_arl_zorro_yes'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${experiment_name}_$(date +%Y%m%d_%H%M%S).log"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export HF_HOME="${HF_HOME:-/checkpoint/huggingface}"

# Standard ArcticRL env (mirrors long-context / big_bird zorro perf profile)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT=/modeling-checkpoints/vllm
export VLLM_LOGGING_LEVEL=INFO

# Pre-launch /dev/shm cleanup: NCCL / vllm / sem files accumulate across runs and
# can fill 100GiB tmpfs after a few iterations, killing raylets (SIGBUS / OOM).
# Cleanup is per-user so it won't disturb other workloads. Best-effort.
if command -v ds_ssh >/dev/null 2>&1 && [[ -f /job/hostfile ]]; then
    ds_ssh "find /dev/shm -maxdepth 1 -user \$USER \
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
# NNODES=4
NGPU_PER_NODE=8

# ----- Arctic/Zorro topology -----
USE_LEGACY_WORKER_IMPL=disable
ROLLOUT_NAME=arctic

USE_ARCTIC_RL=True
USE_ARCTIC_ZORRO=True
COLOCATE=True
ARCTIC_ZERO_STAGE=3
ARCTIC_AUTOCAST=False    # autocast off; bf16 weights used directly (match long-context arctic)

# Total GPUs: NGPU_PER_NODE * NNODES (matches long-context + verl bird baseline)
# colocate=True: training spans all NGPU_PER_JOB bundles.
# log_prob is disabled (NGPU_FOR_LOG_PROBS=0): under Zorro, log-probs are recomputed
# through the training engine, so all rollout GPUs go to sampling (mirrors long-context).
NGPU_PER_JOB=$((NGPU_PER_NODE*NNODES))
NGPU_FOR_SAMPLING=$NGPU_PER_JOB
NGPU_FOR_LOG_PROBS=0
TP_SIZE=2                # sampling TP (matches verl baseline rollout.tensor_model_parallel_size=2)
                         # NOTE: verl's rollout.tensor_model_parallel_size is hard-coded to 1
                         # below; the actual vllm TP is driven by arctic_rl.sampling_tp_size.

# ----- Training hyperparams (per-GPU values; absolute BSZ/PPO_MINI_BSZ derived) -----
# Default matches bird verl baseline at 32 GPUs: BSZ=128, PPO_MINI_BSZ=128 (4 per GPU).
BSZ_PER_GPU=4                    # samples per GPU per training step
PPO_MINI_BSZ_PER_GPU=4           # samples per GPU per PPO update step
BSZ=$((BSZ_PER_GPU*NGPU_PER_JOB))                    # data.train_batch_size
PPO_MINI_BSZ=$((PPO_MINI_BSZ_PER_GPU*NGPU_PER_JOB))  # actor.ppo_mini_batch_size
UBS=8                            # actor / rollout / ref micro_batch_size_per_gpu
ROLL_N=16                        # actor_rollout_ref.rollout.n
PROMPT_LEN=32768                 # data.max_prompt_length
RESPONSE_LEN=4096                # data.max_response_length
MAX_TOKENS_PER_GPU=98304         # actor.ppo_max_token_len_per_gpu
                                 # Zorro tiled compute requires this to be >=
                                 # prompt_len + response_len * n (= 32768 + 4096*16
                                 # = 98304) so every GRPO group fits in one tile.
                                 # Baseline used 40960, which is fine for FSDP2 + ulysses
                                 # but not for Zorro's group-level tiling.
ROLLOUT_MAX_BATCHED=40960        # rollout.max_num_batched_tokens
ULYSSES_SP=2                     # actor.ulysses_sequence_parallel_size
LR=2e-6
CLIP_RATIO=0.2
KL_COEF=0.001
KL_LOSS_COEF=0.001
FILTER_OVERLONG_PROMPTS_WORKERS=8
DATA_SEED=42
TOTAL_EPOCHS=10
SAVE_FREQ=-1                     # disable checkpointing (match long-context perf runs)
TEST_FREQ=10
UPDATE_WEIGHTS_BUCKET_MB=4096    # Match verl baseline (used for non-IPC fallback paths).

# Weight sync: CUDA IPC (zero-copy GPU handles). Avoids slow file/shm bucket writes that
# dominated timing_s/update_weights (~180s/step) when cuda_ipc_weight_sync=False on the
# reference run (wandb 0rwn1yrz). Requires training weights on GPU (param_offload=False).
CUDA_IPC_WEIGHT_SYNC=True

LOGGER="['console']"
# LOGGER="['console','wandb']"

MODEL_SHORT=Qwen3-32B
MODEL=Qwen/${MODEL_SHORT}

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

# Same dataset paths as the verl baseline.
DATA_DIR="/data/snowflakesql/xyu/open-source-text2sql"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/val.parquet"

# In Zorro, log-probs are recomputed through the training engine, so this stays False
# (the verl baseline sets calculate_log_probs=True since vllm rollout owns log-probs there).
LOG_PROBS=False
FREE_CACHE_ENGINE=True
ROLLOUT_GPU_MEM_UTIL=0.7
ROLLOUT_ENFORCE_EAGER=False
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-1}"

# Fresh run: no auto-resume from global_step_* under the experiment checkpoint root.
RESUME_MODE=disable
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CHECKPOINT_DIR="/checkpoint/xyu/sql-rl/${experiment_name}/runs/${RUN_ID}"
mkdir -p "${CHECKPOINT_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR} RESUME_MODE=${RESUME_MODE}"

# verl trainer.nnodes/n_gpus_per_node are placeholders under ArcticRL; real topology is
# arctic_rl.training_gpus=$NGPU_PER_JOB (${NGPU_PER_NODE} GPUs x ${NNODES} nodes).
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=$PROMPT_LEN \
    data.max_response_length=$RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=$FILTER_OVERLONG_PROMPTS_WORKERS \
    data.truncation=left \
    data.seed=$DATA_SEED \
    actor_rollout_ref.actor.data_loader_seed=$DATA_SEED \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$flash_attention_v \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$UBS \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SP \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.use_kl_loss=False \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.n=$ROLL_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=$LOG_PROBS \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=$UPDATE_WEIGHTS_BUCKET_MB \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_NUM_WORKERS \
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
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.logger=$LOGGER \
    trainer.project_name=arctic_rl_bird_sql \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=False \
    custom_reward_function.path="${SCRIPT_DIR}/bird_reward.py" \
    custom_reward_function.name=compute_score \
    arctic_rl.colocate=$COLOCATE \
    arctic_rl.train.logits.optimization=memory \
    arctic_rl.sampling_tp_size=$TP_SIZE \
    arctic_rl.training_gpus=$NGPU_PER_JOB \
    arctic_rl.sampling_gpus=$NGPU_FOR_SAMPLING \
    arctic_rl.log_prob_gpus=$NGPU_FOR_LOG_PROBS \
    arctic_rl.train.zorro_train.enable=$USE_ARCTIC_ZORRO \
    arctic_rl.train.zorro_train.max_rollouts=$ROLL_N \
    arctic_rl.train.deepspeed.torch_autocast.enabled=$ARCTIC_AUTOCAST \
    arctic_rl.cuda_ipc_weight_sync=$CUDA_IPC_WEIGHT_SYNC \
    arctic_rl.train.deepspeed.zero_optimization.stage=$ARCTIC_ZERO_STAGE \
    arctic_rl.train.deepspeed.zero_optimization.offload_optimizer.device=cpu \
    arctic_rl.train.deepspeed.zero_optimization.offload_param.device=none \
    "$@" 2>&1 | tee "${LOG_FILE}"
