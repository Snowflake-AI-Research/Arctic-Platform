#!/bin/bash
# Multi-node 4x8 GPU GRPO training for Qwen3-32B on LoongRL (long-context multi-hop QA)
# with Arctic RL + ZoRRo. Sibling of the txt2sql 32B 4-node launcher — same Arctic RL
# stack (FCA, CUDA-IPC weight sync, ZoRRo, Liger, FA2 trainer / FLASH_ATTN inference),
# same 4-node placement — only the env, the dataset, and the sequence-length
# defaults change vs BIRD-SQL.
#
# Hyperparameters mirror the [verl long-context recipe][verl-lc] so the wall-clock
# comparison with the FSDP-native sibling `run_qwen3_32b_loongrl_grpo_fsdp_4node.sh`
# is apples-to-apples: same global batch, same sequence lengths, same LR, same TP,
# same FA impl — only the training backend differs.
#
# Prerequisites (see README.md):
#   1. Conda env with pinned deps installed on EVERY node (Ray requires matching
#      Python + exact dep versions across the cluster).
#   2. 4-node Ray cluster up:
#         head:   ray start --head --port=6379 --num-gpus=8
#         worker: ray start --address=<head_ip>:6379 --num-gpus=8   # x3
#   3. SkyRL cloned at the pinned commit; `export SKYRL_HOME=<path>` on the driver.
#   4. LoongRL parquets staged at $DATA_DIR on a shared filesystem visible to all
#      4 nodes: `python download_data.py --output_dir $DATA_DIR`.
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
# ${SCRIPT_DIR} contains the recipe-local ``arctic_rl/`` shim + ``sitecustomize.py``
# that register the ``long_context_qa`` env — see this recipe's README for why.
export PYTHONPATH="${SKYRL_HOME}:${SCRIPT_DIR}:${PYTHONPATH:-}"

# Driver: bare python, matching upstream
# integrations/arctic_rl/examples/run_bird_grpo_32b_32gpu.sh on the pinned
# ``arctic-rl-public`` merge (7636101a). The caller is expected to activate a
# compatible env (e.g. the ``skyrl_v2`` conda env) beforehand — see this
# recipe's README.
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
# 32B + optimizer offload churns the allocator; expandable segments tame it. The
# 4-node placement layout is different enough from single-node TP>1 that this
# doesn't hit pytorch/pytorch#147851.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl_long_context}"
export WANDB_DISABLE_CODE=True

# ----- 4-node 32-GPU Arctic / ZoRRo topology -----
NUM_NODES=4
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))   # = 32

# vLLM sampling TP=4. The verl long-context recipe uses TP=2, but the SkyRL +
# Arctic RL ZeRO-3 CPU-offload optim step needs more per-GPU headroom during the
# param gather + grad-norm allreduce; TP=2 OOMs the gather on H200s here even
# with the smaller 64-prompt PPO mini-batch. TP=4 is the smallest split that
# reliably fits with our sleep-mode-2 vLLM + DeepSpeed layout. 32 GPUs / TP=4
# -> 8 engine replicas.
TP_SIZE="${TP_SIZE:-4}"
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

# FA3 on Hopper (matches the BIRD 32B recipe); flip to flash_attention_2 for
# A100/L40S.
ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
ARCTIC_ZERO_STAGE=3
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-true}"   # 32B optimizer state offload

# Matches the verl long-context ARL recipe exactly: 256 prompts x 8 samples =
# 2048 trajectories/step, chunked into 4 PPO mini-batches of 64 prompts x 8
# samples = 512 trajectories each. Per DP rank per mini-batch: 512 / 32 = 16
# trajectories. Multiple PPO mini-batches are *required* at 16K prompts — a
# single-mini-batch update over the whole rollout doesn't fit in memory (see
# Snowflake-AI-Research/verl#8 "Enable PPO mini batch training - need for long
# context recipe"). ppo_epochs defaults to 1 (single pass per mini-batch).
# NB: train_batch_size * n_samples_per_prompt must be divisible by num_gpus,
#     and train_batch_size must be divisible by policy_mini_batch_size.
TRAIN_BSZ="${TRAIN_BSZ:-256}"
MINI_BSZ="${MINI_BSZ:-64}"
N_SAMPLES="${N_SAMPLES:-8}"
PROMPT_LEN="${PROMPT_LEN:-16384}"
RESPONSE_LEN="${RESPONSE_LEN:-4096}"

# vLLM max_num_batched_tokens: comfortably fit one full (prompt + response) at
# the 16K/4K shape while leaving room for a couple of decode-heavy sequences.
VLLM_MAX_BATCHED="${VLLM_MAX_BATCHED:-32768}"

LR="${LR:-1e-6}"                       # matches verl recipe (BIRD 32B is 2e-6)
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

# Reward matcher (see envs/long_context_qa_reward.py); "pure_exact_match" is the
# blog-run default. Override via env var without touching the launcher.
export REWARD_CALC_TYPE="${REWARD_CALC_TYPE:-pure_exact_match}"

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME="longcontext_grpo_${MODEL_SHORT}_arl_z${ARCTIC_ZERO_STAGE}_${NUM_NODES}node_${RUN_TS}"
# CKPT_DIR must be on a shared FS — head writes weight-sync tensor, all nodes mmap-read.
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

SPEC_MODEL="${SPEC_MODEL:-}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
SPEC_OVERRIDE=()
if [[ -n "${SPEC_MODEL}" ]]; then
    SPEC_OVERRIDE+=("trainer.arctic_rl.speculative_model=${SPEC_MODEL}")
    SPEC_OVERRIDE+=("trainer.arctic_rl.num_speculative_tokens=${NUM_SPEC_TOKENS}")
fi

# Match upstream arl launcher: run from ${SKYRL_HOME} so integrations/ imports
# resolve relative to the checkout.
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
