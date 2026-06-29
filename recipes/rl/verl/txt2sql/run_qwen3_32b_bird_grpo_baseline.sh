#!/bin/bash
# Qwen3-32B GRPO on BIRD SQL -- stock verl/vLLM baseline (vLLM rollout + FSDP2, no Arctic/Zorro).
# Baseline twin of run_qwen3_32b_bird_grpo_arl_zorro_yes.sh; learning hyperparameters match it.
# 32 GPUs (4 x 8) from the hostfile. See README.md for data prep + ray cluster setup.

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${ARCTIC_VERL_ROOT:-}" ]]; then
    export PYTHONPATH="${ARCTIC_VERL_ROOT}:${PYTHONPATH:-}"
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO

HOSTFILE="${JOB_HOSTFILE:-/data-fast/hostfile}"

# Best-effort /dev/shm cleanup: stale NCCL/vllm/sem files can fill the tmpfs and kill raylets.
if command -v ds_ssh >/dev/null 2>&1 && [[ -f "${HOSTFILE}" ]]; then
    ds_ssh -f "${HOSTFILE}" "find /dev/shm -maxdepth 1 -user \$USER \
        \( -name 'nccl-*' -o -name 'cuda.shm.*' -o -name 'arctic_ws_*' \
           -o -name 'torch_*' -o -name 'sem.obj*' -o -name 'sem.hdr*' \
           -o -name 'sem.loky-*' -o -name 'psm_*' -o -name 'plasma*' \) \
        -delete 2>/dev/null; \
        echo \"\$(hostname): /dev/shm \$(df -h /dev/shm | tail -1 | awk '{print \$3\"/\"\$2}')\"" \
        2>&1 | tail -10
fi

# NNODES from hostfile (one line per node); falls back to 1.
if [[ -f ${HOSTFILE} ]]; then
    NNODES=$(wc -l < ${HOSTFILE})
else
    NNODES=1
fi
NGPU_PER_NODE=8
NGPU_PER_JOB=$((NGPU_PER_NODE*NNODES))

# ----- verl / vLLM topology -----
USE_LEGACY_WORKER_IMPL=disable
ROLLOUT_NAME=vllm        # Arctic sibling uses 'arctic'
TP_SIZE=2                # rollout tensor-parallel
ULYSSES_SP=2             # actor sequence-parallel
ACTOR_PARAM_OFFLOAD=True       # 32B needs CPU offload to fit the FSDP hybrid engine
ACTOR_OPTIMIZER_OFFLOAD=True
ROLLOUT_GPU_MEM_UTIL=0.7

# ----- Training hyperparams (match the Arctic BIRD recipe) -----
BSZ=128
PPO_MINI_BSZ=128
ROLL_N=16
PROMPT_LEN=32768
RESPONSE_LEN=4096
MAX_TOKENS_PER_GPU=40960 # dynamic-batch token cap (vs Arctic Zorro 98304)
ROLLOUT_MAX_BATCHED=40960
LR=2e-6
CLIP_RATIO=0.2
USE_KL_LOSS=False        # pure GRPO
KL_LOSS_COEF=0.001       # unused when USE_KL_LOSS=False
TOTAL_EPOCHS=10
SAVE_FREQ=-1             # no periodic checkpoints by default (matches Arctic sibling)
TEST_FREQ=10

LOGGER="['console']"
# For wandb: uncomment below, set WANDB_API_KEY, and adjust trainer.project_name/experiment_name.
# LOGGER="['console','wandb']"

MODEL_SHORT=Qwen3-32B
MODEL=Qwen/${MODEL_SHORT}

experiment_name="${EXPERIMENT_NAME:-bird_grpo_${MODEL_SHORT}_ngpu${NGPU_PER_JOB}_gbs${BSZ}_rolln${ROLL_N}_baseline}"

LOG_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${experiment_name}_$(date +%Y%m%d_%H%M%S).log"

# Pick the attention impl by GPU; hardcode if preferred.
gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 2>/dev/null || true)
if [[ $gpu_name == *"H100"* ]] || [[ $gpu_name == *"H200"* ]]; then
    flash_attention_v=flash_attention_3
elif [[ $gpu_name == *"B200"* ]] || [[ $gpu_name == *"B300"* ]]; then
    flash_attention_v=flash_attention_2
else
    flash_attention_v=flash_attention_2
fi

DATA_DIR="${DATA_DIR:-/data/snowflakesql/txt2sql}"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/val.parquet"

# Rollout log-probs off; PPO old_log_prob is recomputed by the FSDP actor.
LOG_PROBS=False
FREE_CACHE_ENGINE=True

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_DIR}/outputs/checkpoints/${experiment_name}/${RUN_ID}}"
mkdir -p "${CHECKPOINT_DIR}"

echo "NNODES=${NNODES} NGPU_PER_JOB=${NGPU_PER_JOB} CHECKPOINT_DIR=${CHECKPOINT_DIR}"

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
    actor_rollout_ref.model.use_liger=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$flash_attention_v \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SP \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.n=$ROLL_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=$LOG_PROBS \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.nccl_timeout=1800 \
    trainer.use_legacy_worker_impl=$USE_LEGACY_WORKER_IMPL \
    trainer.balance_batch=False \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.resume_mode=disable \
    trainer.logger=$LOGGER \
    trainer.project_name=arctic_rl_bird_sql \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$NGPU_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=False \
    custom_reward_function.path=${SCRIPT_DIR}/bird_reward.py \
    custom_reward_function.name=compute_score \
    "$@" 2>&1 | tee "${LOG_FILE}"
exit ${PIPESTATUS[0]}
