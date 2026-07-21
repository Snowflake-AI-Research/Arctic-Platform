#!/bin/bash
# SkyRL FSDP-native backend: Qwen3-8B BIRD-SQL GRPO — single node / 8 H200s.
#
# Wall-clock A/B baseline for run_qwen3_8b_bird_grpo_arl.sh. Same model, data,
# batch geometry, sequence lengths, and placement as the ARL sibling; only the
# training backend differs (fsdp2 + native SkyRL colocation + plain vLLM, no
# ArcticInference / ZoRRo). Single-node counterpart of
# run_qwen3_32b_bird_grpo_fsdp_4node.sh.
#
# Prereqs (see README.md):
#   1. Activated conda env with pinned deps; `export SKYRL_HOME=<clone>`.
#   2. `python download_data.py --bird_dir <raw> --output_dir $DATA_DIR`.

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
    echo "       Clone SkyRL at the pinned commit (see ../README.md) and"
    echo "       'export SKYRL_HOME=<path to clone>' before running this script."
    exit 1
fi
# ${SCRIPT_DIR} carries the recipe-local ``arctic_rl.envs`` package + the
# ``sitecustomize.py`` hook that register ``bird`` / ``bird_sql``.
export PYTHONPATH="${SKYRL_HOME}:${SCRIPT_DIR}:${PYTHONPATH:-}"

PYBIN="${PYBIN:-python}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on single-node TP>1
# (this recipe runs TP=4 vLLM engines): it trips pytorch/pytorch#147851 inside
# vLLM's Ray-executor TP workers, same as the ARL sibling.
# Liger off: Qwen3 Liger kernel hits a Triton illegal-mem-access on packed-seq
# inputs under FSDP. Matches upstream fsdp bird recipe.
export SKYRL_USE_LIGER=0

export WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl}"
export WANDB_DISABLE_CODE=True

# ----- Single-node 8-GPU topology (matches the ARL 8B sibling) -----
NUM_NODES=1
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))

TP_SIZE="${TP_SIZE:-4}"
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

# Same batch geometry / sequence lengths as run_qwen3_8b_bird_grpo_arl.sh.
TRAIN_BSZ="${TRAIN_BSZ:-32}"
MINI_BSZ="${MINI_BSZ:-16}"
N_SAMPLES="${N_SAMPLES:-8}"
PROMPT_LEN="${PROMPT_LEN:-16384}"
RESPONSE_LEN="${RESPONSE_LEN:-2048}"
LR="${LR:-2e-6}"

LOGGER="${LOGGER:-console}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_SHORT="$(basename "${MODEL}")"

DATA_DIR="${DATA_DIR:-${HOME}/data/bird}"
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
    echo "ERROR: BIRD-SQL parquets not found at ${DATA_DIR}/{train,val}.parquet"
    echo "       Run 'python download_data.py --bird_dir <raw> --output_dir ${DATA_DIR}' first."
    exit 1
fi

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME="bird_grpo_${MODEL_SHORT}_fsdp_${NUM_NODES}node_${RUN_TS}"
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

FSDP_ENTRY="${SCRIPT_DIR}/fsdp_bird_entry.py"

# Run from ${SKYRL_HOME} so ``integrations/`` imports resolve.
cd "${SKYRL_HOME}"

"${PYBIN}" "${FSDP_ENTRY}" \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/val.parquet']" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path="${MODEL}" \
    trainer.strategy=fsdp2 \
    trainer.placement.colocate_all=true \
    trainer.placement.policy_num_gpus_per_node=${GPUS_PER_NODE} \
    trainer.placement.policy_num_nodes=${NUM_NODES} \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.policy.fsdp_config.reshard_after_forward=true \
    trainer.policy.optimizer_config.offload_after_step=true \
    trainer.policy.sequence_parallel_size=1 \
    trainer.flash_attn=true \
    trainer.micro_train_batch_size_per_gpu=${MICRO_TRAIN:-2} \
    trainer.micro_forward_batch_size_per_gpu=${MICRO_FWD:-2} \
    trainer.use_sample_packing=true \
    generator.inference_engine.num_engines=${NUM_ENGINES} \
    generator.inference_engine.tensor_parallel_size=${TP_SIZE} \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.gpu_memory_utilization=0.4 \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.max_num_batched_tokens=40960 \
    generator.inference_engine.enforce_eager=true \
    generator.batched=true \
    trainer.epochs=1 \
    trainer.eval_batch_size=32 \
    trainer.eval_before_train=false \
    trainer.eval_interval=50 \
    trainer.update_epochs_per_batch=1 \
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
    trainer.policy.optimizer_config.lr=${LR} \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false \
    environment.env_class=bird \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CKPT_DIR}/logs" \
    trainer.ckpt_path="${CKPT_DIR}/ckpt" \
    trainer.ckpt_interval=-1 \
    "$@" 2>&1 | tee "${CKPT_DIR}/${EXPERIMENT_NAME}.log"
