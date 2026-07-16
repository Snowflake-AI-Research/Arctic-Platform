#!/bin/bash
# Single-GPU GRPO training for Qwen3-1.7B on GSM8K with ArcticRL + ZoRRo.
# Pure GRPO, no frozen reference model (use_kl_loss=False).
#
# This is the "simple" entry-point recipe: 1 GPU, 1 node, no Ray cluster setup and no hostfile. verl starts a local
# Ray instance automatically.
#
# Adapted from:
#   - examples/arctic_rl/run_bird_grpo_arl_zorro_yes.sh   (single-GPU arctic_rl / ZoRRo template)
#   - recipes/rl/verl/long_context_qa                     (remote_backend.* schema + structure)
#
# GSM8K is scored by verl's built-in reward (data_source="openai/gsm8k"), so no custom reward function is needed.
#
# The Arctic backend is loaded into verl as a plugin via the
# `VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register`
# hook exported below; verl core carries no Arctic-specific files.
#
# Prerequisites (see README.md):
#   1. Download data: python download_data.py   (writes $DATA_DIR/{train,test}.parquet)
#   2. Packages installed (README "Install packages": requirements.txt + overrides.txt, plus the Snowflake verl fork
#      installed editable)

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Do NOT set expandable_segments:True -- vLLM colocate sleep mode (cumem allocator) rejects it.
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
# Leave HF online by default so the model + dataset can be fetched on first run.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TORCH_COMPILE_DISABLE=1
# Select the Arctic training client for verl's remote_backend=arctic path.
export USE_ARCTIC_TRAINING_CLIENT=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO

# Plug the Arctic RemoteBackend into verl. verl reads this on `import verl`
# and imports the referenced module for its registration side effects; no
# verl source-tree modification is needed.
export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register

# Make the plugin's config directory visible to Hydra so `remote_backend=arctic`
# resolves to the yaml shipped inside arctic_platform.
ARCTIC_VERL_CONFIG_DIR="${REPO_ROOT}/arctic_platform/integrations/verl/config"

# Preflight: torch/system CUDA mismatch. DeepSpeed JIT-builds CUDA extensions
# and refuses if `nvcc --version` doesn't match torch's build CUDA. In
# multi-CUDA container images (e.g. `/usr/local/cuda -> cuda-13` while torch
# is cu129), auto-point CUDA_HOME at the matching toolkit if available.
_torch_cuda=$(python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true)
if [[ -n "${_torch_cuda}" ]]; then
    _sys_nvcc=$(nvcc --version 2>/dev/null | awk -F'release ' '/release/ {split($2,a,","); print a[1]}')
    if [[ -n "${_sys_nvcc}" && "${_sys_nvcc}" != "${_torch_cuda}" ]]; then
        _match_dir="/usr/local/cuda-${_torch_cuda}"
        if [[ -x "${_match_dir}/bin/nvcc" ]]; then
            export CUDA_HOME="${_match_dir}"
            export PATH="${CUDA_HOME}/bin:${PATH}"
            export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
            echo "[preflight] system nvcc=${_sys_nvcc} != torch.version.cuda=${_torch_cuda}; auto-pointing CUDA_HOME=${CUDA_HOME}"
        else
            echo "[preflight] ERROR: system nvcc=${_sys_nvcc} != torch.version.cuda=${_torch_cuda}" >&2
            echo "[preflight]        DeepSpeed CUDA extension JIT will fail with CUDAMismatchException." >&2
            echo "[preflight]        Install cuda-${_torch_cuda} and export CUDA_HOME=/path/to/cuda-${_torch_cuda}." >&2
            exit 3
        fi
    fi
fi

# ----- Single-GPU Arctic/ZoRRo topology -----
USE_LEGACY_WORKER_IMPL=disable
ROLLOUT_NAME=arctic

USE_ARCTIC_ZORRO=True
COLOCATE=True
ARCTIC_ZERO_STAGE=2      # 1.7B fits comfortably on one GPU; no offload needed

NGPU_PER_JOB=1           # single GPU
NGPU_FOR_LOG_PROBS=0     # non-KL: no frozen ref model; ZoRRo recomputes log-probs on the training engine
TP_SIZE=1                # single-GPU sampling

# ----- Training hyperparams (small-scale single-GPU defaults) -----
BSZ=32                   # data.train_batch_size (prompts per step)
PPO_MINI_BSZ=32          # actor.ppo_mini_batch_size
UBS=8                    # actor / rollout / ref micro_batch_size_per_gpu
ROLL_N=8                 # actor_rollout_ref.rollout.n (GRPO group size)
PROMPT_LEN=1024          # GSM8K prompts are short
RESPONSE_LEN=1024
MAX_TOKENS_PER_GPU=16384 # actor.ppo_max_token_len_per_gpu (>= prompt_len + ROLL_N * response_len for ZoRRo tiles)
ROLLOUT_MAX_BATCHED=16384
LR=1e-6
CLIP_RATIO=0.2
USE_KL_LOSS=False        # pure GRPO, no frozen-ref KL anchoring
KL_LOSS_COEF=0.001       # unused when USE_KL_LOSS=False
TOTAL_EPOCHS=15
SAVE_FREQ=-1             # no checkpoint saving for this demo
TEST_FREQ=10             # run validation every 10 steps

LOGGER="['console']"
# if you want to use wandb, uncomment the following line and set the WANDB_API_KEY in your environment
# additionally edit below trainer.project_name and trainer.experiment_name entries to match your wandb project and
# experiment name
# LOGGER="['console','wandb']"

MODEL_SHORT=Qwen3-1.7B
MODEL=Qwen/${MODEL_SHORT}

experiment_name="gsm8k_grpo_${MODEL_SHORT}_ngpu${NGPU_PER_JOB}_gbs${BSZ}_mbs${UBS}_rolln${ROLL_N}_arl_z${ARCTIC_ZERO_STAGE}"

flash_attention_v=flash_attention_2

# To auto-select the attention implementation based on the GPU type instead,
# comment out the line above and uncomment the block below.
# gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0)
# if [[ $gpu_name == *"H100"* ]] || [[ $gpu_name == *"H200"* ]] ; then
#     echo "Running on Hopper"
#     flash_attention_v=flash_attention_3
# elif [[ $gpu_name == *"B200"* ]] || [[ $gpu_name == *"B300"* ]] ; then
#     echo "Running on Blackwell"
#     flash_attention_v=flash_attention_2
# else
#     echo "Running on unknown: $gpu_name; defaulting to flash_attention_2"
#     flash_attention_v=flash_attention_2
# fi

# Data: GSM8K parquets produced by download_data.py
DATA_DIR="${DATA_DIR:-${HOME}/data/gsm8k}"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/test.parquet"

# Where checkpoints would go (no-op while SAVE_FREQ=-1)
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/gsm8k-rl}"

# In ZoRRo, log-probs are recomputed through the training engine, so we keep this off
LOG_PROBS=False
FREE_CACHE_ENGINE=True

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
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
    trainer.default_local_dir=$CKPT_DIR/$experiment_name \
    trainer.logger=$LOGGER \
    trainer.project_name=arctic_rl_gsm8k \
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
    remote_backend.train.deepspeed.zero_optimization.offload_optimizer.device=none \
    remote_backend.train.deepspeed.zero_optimization.offload_param.device=none \
    remote_backend.train.deepspeed.zero_optimization.stage=$ARCTIC_ZERO_STAGE \
    remote_backend.train.logits.optimization=memory \
    remote_backend.train.zorro_train.enable=$USE_ARCTIC_ZORRO \
    remote_backend.train.zorro_train.max_rollouts=$ROLL_N \
    remote_backend.training_gpus=$NGPU_PER_JOB \
    remote_backend.weight_sync.cuda_ipc=True \
    "$@" 2>&1 | tee $experiment_name.log
