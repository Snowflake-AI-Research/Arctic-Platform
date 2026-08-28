#!/bin/bash
# GRPO training for Qwen3-0.6B on GSM8K with the Arctic RL Cortex backend.
# Cortex sub-jobs own training + sampling; the SkyRL driver is CPU-only.
#
# Pre-reqs (see README.md):
#   1. pip install arctic-platform[cortex]
#   2. SkyRL cloned at the pinned commit with SKYRL_HOME exported.
#   3. ARCTIC_BACKEND=cortex + ARCTIC_CORTEX_* env vars set.
#   4. Data: `python download_data.py` -> $DATA_DIR/{train,validation}.parquet.
#
# Cortex-specific overrides (Hydra):
#   trainer.arctic_rl.attn_implementation=sdpa
#     Cortex image ships without FA2.
#   generator.inference_engine.remote_urls=[http://cortex-managed, ...]
#   generator.sampling_params.logprobs=null
#     Required by SkyRL's validate_generator_cfg under run_engines_locally=false,
#     which asserts one URL per engine. The URLs are placeholders -- generation
#     is served by the shim, which never dials them.

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
NGPU_PER_NODE="${NGPU_PER_NODE:-4}"  # target Cortex training-sub-job GPU count
NUM_NODES="${NUM_NODES:-1}"
TP_SIZE="${TP_SIZE:-1}"              # driver sees Cortex sampling as TP=1
# sampling_gpus = num_engines x tp_size, so 4 engines is 4 sampling GPUs.
# 4 training + 4 sampling sits exactly at the 8-GPU per-account Cortex cap.
NUM_ENGINES="${NUM_ENGINES:-4}"

# Hyperparameters are SkyRL's canonical GSM8K GRPO recipe
# (examples/train/gsm8k/run_gsm8k.sh) unchanged, except for use_kl_loss below.
# Keep them that way: an earlier version ran TRAIN_BSZ=32 / MINI_BSZ=4, i.e. 16
# sequences per optimizer update against the canonical 1280. verl's canonical
# recipe independently lands on the same operating point, so treat this as the
# supported configuration rather than a starting guess.
TRAIN_BSZ=1024
MINI_BSZ=256
N_SAMPLES=5
MICRO_BSZ=64
PROMPT_LEN=512
RESPONSE_LEN=1024
LR=1e-6
TOTAL_EPOCHS=20
EVAL_INTERVAL=5

# SkyRL's validate_generator_cfg asserts num_engines == len(remote_urls). The
# URL itself is a placeholder -- the real endpoint lives inside the Cortex shim
# -- but the count has to track NUM_ENGINES or the driver dies before launch.
ENGINE_URLS="$(printf 'http://cortex-managed,%.0s' $(seq "${NUM_ENGINES}"))"
ENGINE_URLS="[${ENGINE_URLS%,}]"

LOGGER="${LOGGER:-console}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_SHORT="$(basename "${MODEL}")"
EXPERIMENT_NAME="gsm8k_grpo_${MODEL_SHORT}_cortex"

# SkyRL and verl use different parquet schemas for the same GSM8K dataset:
# SkyRL's GSM8kEnv reads reward_spec + env_class; verl's rl_dataset reads
# reward_model. Sharing one directory silently trains with reward=0 (schema
# hit, wrong field name). Keep this default distinct from the verl recipe.
DATA_DIR="${DATA_DIR:-${HOME}/data/gsm8k-skyrl}"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/validation.parquet"

if [[ ! -f "${TRAIN_FILES}" || ! -f "${VAL_FILES}" ]]; then
    echo "ERROR: SkyRL parquets not found under ${DATA_DIR}."
    echo "       Run: python ../simple_gsm8k/download_data.py --output_dir ${DATA_DIR}"
    exit 1
fi

# Pre-flight: refuse to launch on a verl-shaped parquet (missing reward_spec
# or env_class) — SkyRL's env would silently score every rollout 0.0 with the
# wrong field names.
python - <<PY || exit 1
import sys, pandas as pd
cols = set(pd.read_parquet("${TRAIN_FILES}").columns)
missing = {"reward_spec", "env_class"} - cols
if missing:
    print(f"ERROR: ${TRAIN_FILES} is missing SkyRL schema fields {sorted(missing)}.")
    print("       Looks like a verl-shaped parquet. Rebuild with recipes/rl/skyrl/simple_gsm8k/download_data.py.")
    sys.exit(1)
PY

CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

# Launch via arctic_platform.integrations.skyrl so the driver-side
# peer_access_supported shim is installed before SkyRL's Ray probe.
python -m arctic_platform.integrations.skyrl \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
    trainer.arctic_rl.colocate=false \
    trainer.arctic_rl.attn_implementation=sdpa \
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
    "generator.inference_engine.remote_urls=${ENGINE_URLS}" \
    "generator.inference_engine.external_server_urls=${ENGINE_URLS}" \
    "generator.sampling_params.logprobs=null" \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.batched=true \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    environment.env_class=gsm8k \
    trainer.epochs=${TOTAL_EPOCHS} \
    trainer.train_batch_size=${TRAIN_BSZ} \
    trainer.policy_mini_batch_size=${MINI_BSZ} \
    trainer.micro_train_batch_size_per_gpu=${MICRO_BSZ} \
    trainer.micro_forward_batch_size_per_gpu=${MICRO_BSZ} \
    trainer.max_prompt_length=${PROMPT_LEN} \
    generator.sampling_params.max_generate_length=${RESPONSE_LEN} \
    trainer.eval_batch_size=256 \
    trainer.eval_before_train=true \
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
