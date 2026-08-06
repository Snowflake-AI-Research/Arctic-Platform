#!/bin/bash
# SkyRL GSM8K GRPO -> Cortex-training QA6.
#
# Cortex sibling of ``simple_gsm8k/run_qwen3_0.6b_gsm8k_grpo_arl.sh``. The
# driver is CPU-only; every GPU op (training fwd/bwd/step, sampling generate,
# weight-sync) is dispatched to Cortex-training via the unified ArcticRLClient
# + Cortex dispatch shim (arctic_platform/rl/_cortex_dispatch.py).
#
# Same conda env, same pinned SkyRL commit, same Hydra config surface as the
# on-prem recipe — the only functional differences are ``ARCTIC_BACKEND=cortex``
# and the placement knobs (Cortex splits training and sampling into distinct
# sub-jobs, so ``arctic_rl.colocate=true`` doesn't apply).
#
# Prerequisites (see README.md):
#   1. Conda env ``skyrl_arl`` from the sibling ``simple_gsm8k`` recipe.
#   2. SkyRL cloned at the pinned commit (see ../README.md) and SKYRL_HOME
#      pointing at it.
#   3. Cortex PAT: ``export CORTEX_PAT=<snowflake programmatic access token>``.
#   4. Cortex endpoint config via ARCTIC_CORTEX_HOST / DATABASE / SCHEMA /
#      ENDPOINT env vars (see README.md).
#   5. Data prepared: ``python download_data.py`` -> $DATA_DIR/{train,validation}.parquet.

set -euo pipefail

if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
    echo "       Clone SkyRL at the pinned commit (see ../README.md) and"
    echo "       'export SKYRL_HOME=<path to clone>' before running this script."
    exit 1
fi
: "${CORTEX_PAT:?CORTEX_PAT missing; export it from your Snowflake programmatic access token}"
: "${ARCTIC_CORTEX_HOST:?ARCTIC_CORTEX_HOST missing (e.g. <account>.<region>.snowflakecomputing.com)}"
export PYTHONPATH="${SKYRL_HOME}:${PYTHONPATH:-}"

# Cortex routing. ARCTIC_BACKEND=cortex overrides the ArcticRL adapter's
# hard-coded ``backend="local"`` so no SkyRL-side patch is needed.
export ARCTIC_BACKEND=cortex
export ARCTIC_CORTEX_DATABASE="${ARCTIC_CORTEX_DATABASE:-NEUTRINO_DB}"
export ARCTIC_CORTEX_SCHEMA="${ARCTIC_CORTEX_SCHEMA:-PUBLIC}"
export ARCTIC_CORTEX_ENDPOINT="${ARCTIC_CORTEX_ENDPOINT:-cortex-training}"
export ARCTIC_CORTEX_PAT_ENV_VAR=CORTEX_PAT

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# Ray creates AF_UNIX sockets under TMPDIR (107-byte SUN_PATH limit); pin
# under /tmp so long ``$HOME``/checkpoint paths don't blow the budget.
export TMPDIR="${TMPDIR:-/tmp/rayk}"
mkdir -p "$HF_HOME" "$TMPDIR"

# ----- Cortex topology -----
# Cortex splits training and sampling into separate sub-jobs; 4 + 4 fits in
# a single QA6 slot. 8 + 8 is possible but subject to capacity (503s common).
NGPU_TRAIN="${NGPU_TRAIN:-4}"
NGPU_SAMPLE="${NGPU_SAMPLE:-4}"

# ----- Training hyperparams -----
TRAIN_BSZ="${TRAIN_BSZ:-64}"        # prompts per step
MINI_BSZ="${MINI_BSZ:-8}"           # actor mini-batch per DP rank
N_SAMPLES="${N_SAMPLES:-4}"         # GRPO group size
PROMPT_LEN="${PROMPT_LEN:-512}"
RESPONSE_LEN="${RESPONSE_LEN:-1024}"
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

LOGGER="${LOGGER:-console}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_SHORT="$(basename "${MODEL}")"

EXPERIMENT_NAME="gsm8k_grpo_${MODEL_SHORT}_cortex"

DATA_DIR="${DATA_DIR:-${HOME}/data/gsm8k}"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/validation.parquet"

CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

echo "[cortex] host=${ARCTIC_CORTEX_HOST} db=${ARCTIC_CORTEX_DATABASE} endpoint=${ARCTIC_CORTEX_ENDPOINT}"
echo "[cortex] training_gpus=${NGPU_TRAIN} sampling_gpus=${NGPU_SAMPLE} model=${MODEL}"

python -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
    trainer.arctic_rl.zero_stage=2 \
    trainer.arctic_rl.attn_implementation=sdpa \
    trainer.arctic_rl.use_zorro=false \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false \
    trainer.policy.model.path="${MODEL}" \
    data.train_data="['${TRAIN_FILES}']" \
    data.val_data="['${VAL_FILES}']" \
    trainer.placement.colocate_all=false \
    "trainer.placement.policy_num_gpus_per_node=${NGPU_TRAIN}" \
    trainer.placement.policy_num_nodes=1 \
    "trainer.placement.ref_num_gpus_per_node=${NGPU_TRAIN}" \
    trainer.placement.ref_num_nodes=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.num_engines=1 \
    "generator.inference_engine.tensor_parallel_size=${NGPU_SAMPLE}" \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    "generator.inference_engine.gpu_memory_utilization=${GPU_MEM_UTIL}" \
    generator.batched=true \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    environment.env_class=gsm8k \
    trainer.epochs=${TOTAL_EPOCHS} \
    trainer.train_batch_size=${TRAIN_BSZ} \
    trainer.policy_mini_batch_size=${MINI_BSZ} \
    trainer.max_prompt_length=${PROMPT_LEN} \
    generator.sampling_params.max_generate_length=${RESPONSE_LEN} \
    trainer.eval_batch_size=32 \
    trainer.eval_before_train=false \
    trainer.eval_interval=999999 \
    trainer.update_epochs_per_batch=1 \
    trainer.policy.optimizer_config.lr=${LR} \
    trainer.logger="${LOGGER}" \
    trainer.project_name=arctic_rl_gsm8k_cortex \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CKPT_DIR}/logs" \
    trainer.ckpt_path="${CKPT_DIR}/ckpt" \
    "$@" 2>&1 | tee "${CKPT_DIR}/${EXPERIMENT_NAME}.log"
