#!/bin/bash
# Arctic RL + ZoRRo: Qwen3-32B LoongRL long-context GRPO — 4 nodes / 32 H200s.
#
# Hyperparameters mirror the [verl long-context recipe][verl-lc] so the
# wall-clock comparison against `run_qwen3_32b_loongrl_grpo_fsdp_4node.sh` is
# apples-to-apples: only the training backend differs.
#
# Prereqs (see README.md):
#   1. Matching conda env with pinned deps on EVERY node (Ray requires it).
#   2. 4-node Ray cluster:
#         head:   ray start --head --port=6379 --num-gpus=8
#         worker: ray start --address=<head_ip>:6379 --num-gpus=8   # x3
#   3. SkyRL cloned at the pinned commit; `export SKYRL_HOME=<path>`.
#   4. LoongRL parquets on a shared FS visible to all 4 nodes.
#
# [verl-lc]: ../../verl/long_context_qa/run_qwen3_32b_longcontext_grpo_arl.sh

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
    echo "       Clone SkyRL at the pinned commit (see ../README.md) and"
    echo "       'export SKYRL_HOME=<path to clone>' before running this script."
    exit 1
fi
# ${SCRIPT_DIR} carries the recipe-local ``arctic_rl/`` shim + ``sitecustomize.py``
# that register ``long_context_qa`` — see README.md.
export PYTHONPATH="${SKYRL_HOME}:${SCRIPT_DIR}:${PYTHONPATH:-}"

# Matches upstream integrations/arctic_rl/examples/run_bird_grpo_32b_32gpu.sh:
# bare python from a caller-activated env. See ../README.md.
PYBIN="${PYBIN:-python}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export ARCTIC_CUDA_IPC_LOW_MEM=0
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0
# 32B + optimizer offload churns the allocator; expandable segments tame it.
# Multi-node placement doesn't hit pytorch/pytorch#147851 that bites single-node TP>1.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl_long_context}"
export WANDB_DISABLE_CODE=True

# ----- 4-node 32-GPU Arctic / ZoRRo topology -----
NUM_NODES=4
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))   # = 32

# TP=4 -> 8 engine replicas. Smaller TP OOMs the ZeRO-3 param gather during
# the optimizer step at this model size / mini-batch.
TP_SIZE="${TP_SIZE:-4}"
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

# FA3 on Hopper; set flash_attention_2 for A100/L40S.
ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
ARCTIC_ZERO_STAGE=3
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-true}"

# 256 prompts x 8 samples = 2048 trajectories/step, 4 PPO mini-batches of 64
# prompts each (Snowflake-AI-Research/verl#8: multiple PPO mini-batches are
# required at 16K prompts — a single-mini-batch update doesn't fit). Constraints:
# train_batch_size * n_samples_per_prompt divisible by num_gpus, and
# train_batch_size divisible by policy_mini_batch_size.
TRAIN_BSZ="${TRAIN_BSZ:-256}"
MINI_BSZ="${MINI_BSZ:-64}"
N_SAMPLES="${N_SAMPLES:-8}"
PROMPT_LEN="${PROMPT_LEN:-16384}"
RESPONSE_LEN="${RESPONSE_LEN:-4096}"

# Fits one full (prompt + response) at 16K/4K plus a couple decode-heavy seqs.
VLLM_MAX_BATCHED="${VLLM_MAX_BATCHED:-32768}"

LR="${LR:-1e-6}"                       # matches verl LC recipe (BIRD 32B is 2e-6)
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-false}"

LOGGER="${LOGGER:-wandb}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
MODEL_SHORT="$(basename "${MODEL}")"

DATA_DIR="${DATA_DIR:-${HOME}/data/loongrl}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/merged/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/merged/test.parquet}"
if [[ ! -f "${TRAIN_PARQUET}" || ! -f "${VAL_PARQUET}" ]]; then
    echo "ERROR: LoongRL parquets not found at ${TRAIN_PARQUET} / ${VAL_PARQUET}"
    echo "       Run 'python download_data.py --output_dir ${DATA_DIR}' first."
    echo "       NOTE: parquets must be on a shared FS visible to all 4 nodes."
    exit 1
fi

# Reward matcher — see arctic_rl/envs/long_context_qa_reward.py.
export REWARD_CALC_TYPE="${REWARD_CALC_TYPE:-pure_exact_match}"

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME="longcontext_grpo_${MODEL_SHORT}_arl_z${ARCTIC_ZERO_STAGE}_${NUM_NODES}node_${RUN_TS}"
# CKPT_DIR must be on a shared FS: head writes the CUDA-IPC weight-sync tensor,
# all workers mmap-read it.
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

SPEC_MODEL="${SPEC_MODEL:-}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
SPEC_OVERRIDE=()
if [[ -n "${SPEC_MODEL}" ]]; then
    SPEC_OVERRIDE+=("trainer.arctic_rl.speculative_model=${SPEC_MODEL}")
    SPEC_OVERRIDE+=("trainer.arctic_rl.num_speculative_tokens=${NUM_SPEC_TOKENS}")
fi

# Run from ${SKYRL_HOME} so ``integrations/`` imports resolve.
cd "${SKYRL_HOME}"

"${PYBIN}" -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=arctic_rl.entrypoint \
    trainer.arctic_rl.colocate=true \
    trainer.arctic_rl.zero_stage=${ARCTIC_ZERO_STAGE} \
    trainer.arctic_rl.offload_optimizer=${OFFLOAD_OPTIMIZER} \
    trainer.arctic_rl.attn_implementation=${ATTN_IMPL} \
    trainer.arctic_rl.cuda_ipc_weight_sync=true \
    trainer.arctic_rl.low_memory_weight_sync=true \
    trainer.arctic_rl.lr_warmup_ratio=0.05 \
    'trainer.arctic_rl.optimizer_betas=[0.9,0.95]' \
    trainer.arctic_rl.vllm_enforce_eager=false \
    trainer.arctic_rl.vllm_max_num_batched_tokens=${VLLM_MAX_BATCHED} \
    trainer.arctic_rl.server_logs=true \
    trainer.arctic_rl.startup_timeout=1800 \
    "${SPEC_OVERRIDE[@]}" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path="${MODEL}" \
    data.train_data="['${TRAIN_PARQUET}']" \
    data.val_data="['${VAL_PARQUET}']" \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_nodes=${NUM_NODES} \
    trainer.placement.policy_num_gpus_per_node=${GPUS_PER_NODE} \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.num_engines=${NUM_ENGINES} \
    generator.inference_engine.tensor_parallel_size=${TP_SIZE} \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.gpu_memory_utilization=0.5 \
    generator.batched=true \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    environment.env_class=long_context_qa \
    trainer.epochs=${TOTAL_EPOCHS} \
    trainer.train_batch_size=${TRAIN_BSZ} \
    trainer.policy_mini_batch_size=${MINI_BSZ} \
    trainer.max_prompt_length=${PROMPT_LEN} \
    generator.sampling_params.max_generate_length=${RESPONSE_LEN} \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.eval_sampling_params.max_generate_length=${RESPONSE_LEN} \
    generator.eval_sampling_params.temperature=0.0 \
    generator.eval_sampling_params.top_p=1.0 \
    generator.eval_sampling_params.top_k=-1 \
    generator.eval_n_samples_per_prompt=1 \
    trainer.eval_batch_size=16 \
    trainer.eval_before_train=${EVAL_BEFORE_TRAIN} \
    trainer.eval_interval=${EVAL_INTERVAL} \
    trainer.update_epochs_per_batch=1 \
    trainer.policy.optimizer_config.lr=${LR} \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CKPT_DIR}/logs" \
    trainer.ckpt_path="${CKPT_DIR}/ckpt" \
    trainer.ckpt_interval=-1 \
    "$@" 2>&1 | tee "${CKPT_DIR}/${EXPERIMENT_NAME}.log"
