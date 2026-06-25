#!/bin/bash
# GRPO training for Qwen3-32B on BIRD SQL using ArcticRL + Zorro.
#
# Arctic counterpart of the stock verl/vLLM BIRD baseline: same GRPO hyperparameters
# (batch size, LR, prompt/response lengths, rollout.n) with ArcticRL colocate + Zorro
# for rollout and weight sync.
#
# Topology: 4 nodes x 8 GPUs = 32 GPUs, COLOCATE=True
#   Pass Hydra overrides via "$@" to change training settings.
#
# Prerequisites:
#   1. Preprocess data into ${DATA_DIR:-./data/bird_sql}/{train,val}.parquet
#   2. pip install func_timeout
#   3. Multi-node Ray cluster across your GPU nodes
#   4. ArcticInference + arctic-verl installed in the active Python env
#   5. Qwen/Qwen3-32B accessible via HuggingFace (set HF_HOME if using a local cache)

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/qwen3_32b_bird_grpo_arl_zorro_yes_$(date +%Y%m%d_%H%M%S).log"

if [[ -n "${ARCTIC_VERL_ROOT:-}" ]]; then
    export PYTHONPATH="${ARCTIC_VERL_ROOT}:${PYTHONPATH:-}"
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"


# Do NOT set expandable_segments:True -- vLLM colocate sleep mode rejects it.
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO

NNODES=4
NGPU_PER_NODE=8
NGPU_PER_JOB=32

gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 2>/dev/null || true)
if [[ $gpu_name == *"H200"* ]]; then
    flash_attention_v=flash_attention_3
elif [[ $gpu_name == *"B200"* ]] || [[ $gpu_name == *"B300"* ]]; then
    flash_attention_v=flash_attention_2
else
    flash_attention_v=flash_attention_2
fi

DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data/bird_sql}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_DIR}/outputs/checkpoints/qwen3_32b_bird_grpo_arl_zorro_yes/${RUN_ID}}"
mkdir -p "${CHECKPOINT_DIR}"

echo "NNODES=${NNODES} NGPU_PER_JOB=${NGPU_PER_JOB} CHECKPOINT_DIR=${CHECKPOINT_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size=128 \
    data.max_prompt_length=32768 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation=left \
    data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.model.path=Qwen/Qwen3-32B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    +actor_rollout_ref.model.override_config.attn_implementation=${flash_attention_v} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=98304 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=arctic \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.nccl_timeout=1800 \
    trainer.use_legacy_worker_impl=disable \
    trainer.use_arctic_rl=True \
    trainer.balance_batch=False \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.resume_mode=disable \
    trainer.logger="['console']" \
    trainer.project_name=arctic_rl_bird_sql \
    trainer.experiment_name=qwen3_32b_bird_grpo_arl_zorro_yes \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.val_before_train=False \
    custom_reward_function.path="${SCRIPT_DIR}/bird_reward.py" \
    custom_reward_function.name=compute_score \
    arctic_rl.colocate=True \
    arctic_rl.sampling_tp_size=2 \
    arctic_rl.training_gpus=32 \
    arctic_rl.sampling_gpus=32 \
    arctic_rl.log_prob_gpus=0 \
    arctic_rl.weight_sync.cuda_ipc=True \
    arctic_rl.weight_sync.low_memory=False \
    arctic_rl.train.logits.optimization=memory \
    arctic_rl.train.zorro_train.enable=True \
    arctic_rl.train.zorro_train.max_rollouts=16 \
    arctic_rl.train.deepspeed.zero_optimization.stage=3 \
    arctic_rl.train.deepspeed.zero_optimization.offload_optimizer.device=cpu \
    arctic_rl.train.deepspeed.zero_optimization.offload_param.device=none \
    "$@" 2>&1 | tee "${LOG_FILE}"
exit ${PIPESTATUS[0]}
