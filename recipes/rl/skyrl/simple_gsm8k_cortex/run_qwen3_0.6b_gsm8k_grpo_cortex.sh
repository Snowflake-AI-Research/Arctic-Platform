#!/bin/bash
# GRPO training for Qwen3-0.6B on GSM8K with the Arctic RL Cortex backend.
# Cortex sub-jobs own training + sampling; the SkyRL driver is CPU-only.
#
# Pre-reqs (see README.md):
#   1. pip install arctic-platform[cortex]
#   2. SkyRL cloned at the pinned commit with SKYRL_HOME exported.
#   3. ARCTIC_BACKEND=cortex + ARCTIC_CORTEX_* env vars set.
#   4. Data: `python download_data.py` -> $DATA_DIR/{train,validation}.parquet.

set -euo pipefail

if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
    exit 1
fi
if [[ "${ARCTIC_BACKEND:-}" != "cortex" ]]; then
    echo "ERROR: ARCTIC_BACKEND=cortex is required. See README.md."
    exit 1
fi

export PYTHONPATH="${SKYRL_HOME}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# Cortex owns the GPUs; the driver has none. SkyRL's colocated placement
# groups would deadlock on a CPU driver.
NGPU_PER_NODE=1  # target Cortex GPU count for training
NUM_NODES=1
TP_SIZE=1
NUM_ENGINES=1

TRAIN_BSZ=32
MINI_BSZ=4
N_SAMPLES=4
PROMPT_LEN=512
RESPONSE_LEN=1024
LR=1e-6
TOTAL_EPOCHS=1
EVAL_INTERVAL=10

LOGGER="${LOGGER:-console}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_SHORT="$(basename "${MODEL}")"
EXPERIMENT_NAME="gsm8k_grpo_${MODEL_SHORT}_cortex"

DATA_DIR="${DATA_DIR:-${HOME}/data/gsm8k}"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/validation.parquet"

CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

# Launch via arctic_platform.integrations.skyrl so the driver-side
# peer_access_supported shim is installed before SkyRL's Ray probe.
python -m arctic_platform.integrations.skyrl \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
    trainer.arctic_rl.colocate=false \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path="${MODEL}" \
    data.train_data="['${TRAIN_FILES}']" \
    data.val_data="['${VAL_FILES}']" \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_nodes=${NUM_NODES} \
    trainer.placement.policy_num_gpus_per_node=${NGPU_PER_NODE} \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.num_engines=${NUM_ENGINES} \
    generator.inference_engine.tensor_parallel_size=${TP_SIZE} \
    generator.inference_engine.run_engines_locally=false \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.batched=true \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    environment.env_class=gsm8k \
    trainer.epochs=${TOTAL_EPOCHS} \
    trainer.train_batch_size=${TRAIN_BSZ} \
    trainer.policy_mini_batch_size=${MINI_BSZ} \
    trainer.max_prompt_length=${PROMPT_LEN} \
    generator.sampling_params.max_generate_length=${RESPONSE_LEN} \
    trainer.eval_batch_size=256 \
    trainer.eval_before_train=false \
    trainer.eval_interval=${EVAL_INTERVAL} \
    trainer.update_epochs_per_batch=1 \
    trainer.policy.optimizer_config.lr=${LR} \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false \
    trainer.logger="${LOGGER}" \
    trainer.project_name=arctic_rl_gsm8k_cortex \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CKPT_DIR}/logs" \
    trainer.ckpt_path="${CKPT_DIR}/ckpt" \
    "$@" 2>&1 | tee "${CKPT_DIR}/${EXPERIMENT_NAME}.log"
