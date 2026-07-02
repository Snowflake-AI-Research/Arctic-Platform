#!/bin/bash
# Arctic RL + ZoRRo: Qwen3-8B LoongRL long-context GRPO — single node / 8 H200s.
# Pure GRPO, no frozen reference model (use_kl_loss=false).
#
# Defaults to PROMPT_LEN=16384 (LoongRL's native length). With Qwen3-8B + ZeRO-3
# this fits in 8xH200 HBM; on A100-80G drop PROMPT_LEN to 8192.
#
# Prereqs (see README.md):
#   1. Activated conda env with pinned deps; `export SKYRL_HOME=<clone>`.
#   2. `python download_data.py --output_dir $DATA_DIR`.

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SkyRL is required as a checkout: integrations/arctic_rl/ is not in the
# pip-installed package. Pin: see ../README.md.
if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
    echo "       Clone SkyRL at the pinned commit (see ../README.md) and"
    echo "       'export SKYRL_HOME=<path to clone>' before running this script."
    exit 1
fi
# ${SCRIPT_DIR} carries the recipe-local ``arctic_rl/`` shim + ``sitecustomize.py``
# that register ``long_context_qa``. Ray workers pick these up via the shim's
# entrypoint.py; PPE reward-scorer children pick them up via sitecustomize.
export PYTHONPATH="${SKYRL_HOME}:${SCRIPT_DIR}:${PYTHONPATH:-}"

# Matches upstream integrations/arctic_rl/examples/run_bird_grpo_8b_32gpu.sh:
# bare python from a caller-activated env. See ../README.md.
PYBIN="${PYBIN:-python}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export ARCTIC_CUDA_IPC_LOW_MEM=0
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on single-node TP>1:
# it trips pytorch/pytorch#147851 inside vLLM's Ray-executor TP workers.

# Reward matcher — see arctic_rl/envs/long_context_qa_reward.py.
export REWARD_CALC_TYPE="${REWARD_CALC_TYPE:-pure_exact_match}"

# ----- Single-node 8-GPU Arctic / ZoRRo topology -----
NUM_NODES=1
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# TP=4 -> 2 vLLM engines. At 16K prompts KV cache is the tightest budget, so we
# shard sampling more aggressively than txt2sql (TP=2).
TP_SIZE="${TP_SIZE:-4}"
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
ARCTIC_ZERO_STAGE=3
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-false}"   # 8B fits in 8xH200 HBM

# Sizing: (MINI_BSZ * N_SAMPLES) / training_gpus must be divisible by the
# trainer's per-GPU micro-batch (default 2). Defaults below satisfy that
# ((8 * 4) / 8 = 4); if you override, keep the ratio an even multiple.
TRAIN_BSZ="${TRAIN_BSZ:-16}"
MINI_BSZ="${MINI_BSZ:-8}"
N_SAMPLES="${N_SAMPLES:-4}"
PROMPT_LEN="${PROMPT_LEN:-16384}"
RESPONSE_LEN="${RESPONSE_LEN:-2048}"

# Fits one full (prompt + response) per step at 16K/2K.
VLLM_MAX_BATCHED="${VLLM_MAX_BATCHED:-24576}"

LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-false}"

LOGGER="${LOGGER:-console}"
WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_SHORT="$(basename "${MODEL}")"

DATA_DIR="${DATA_DIR:-${HOME}/data/loongrl}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/merged/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/merged/test.parquet}"
if [[ ! -f "${TRAIN_PARQUET}" || ! -f "${VAL_PARQUET}" ]]; then
    echo "ERROR: LoongRL parquets not found at ${TRAIN_PARQUET} / ${VAL_PARQUET}"
    echo "       Run 'python download_data.py --output_dir ${DATA_DIR}' first."
    exit 1
fi

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME="longcontext_grpo_${MODEL_SHORT}_arl_z${ARCTIC_ZERO_STAGE}_${RUN_TS}"
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

# Optional Arctic speculative decoding. Off by default — the 32B head is tied
# to Qwen3-32B's hidden size and won't load on 8B; supply an 8B-sized head via
# SPEC_MODEL=<path>.
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
