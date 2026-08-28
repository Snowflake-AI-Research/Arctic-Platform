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

# Defaults are SkyRL's canonical GSM8K GRPO recipe
# (examples/train/gsm8k/run_gsm8k.sh), with two forced deviations: N_SAMPLES
# below, and use_kl_loss further down. Everything else is canonical.
TRAIN_BSZ="${TRAIN_BSZ:-1024}"
MINI_BSZ="${MINI_BSZ:-256}"
# Canonical is 5. Cortex caps one fwd_bwd response at 128 MiB and 1024 x 5
# exceeds it by 4.2%; see the preflight below for the arithmetic.
N_SAMPLES="${N_SAMPLES:-4}"
MICRO_BSZ="${MICRO_BSZ:-64}"
PROMPT_LEN="${PROMPT_LEN:-512}"
RESPONSE_LEN="${RESPONSE_LEN:-1024}"
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"

# Cortex returns a logprob and an entropy per token (~18 B/token) against a
# 128 MiB response cap, and reports an overrun only after provisioning 8 GPUs
# and a full step -- as a 429 naming neither the cap nor the knob. Check here.
_CAP_BYTES=134217728
_SEQ_LEN=$(( PROMPT_LEN + RESPONSE_LEN ))
_SEQS=$(( TRAIN_BSZ * N_SAMPLES ))
_EST=$(( _SEQS * _SEQ_LEN * 18 ))
if (( _EST > _CAP_BYTES )); then
    _MAX_SEQS=$(( _CAP_BYTES / (_SEQ_LEN * 18) ))
    _FIT_N=$(( _MAX_SEQS / TRAIN_BSZ ))
    echo "ERROR: this configuration overruns Cortex's per-response cap." >&2
    echo >&2
    echo "  ${TRAIN_BSZ} TRAIN_BSZ x ${N_SAMPLES} N_SAMPLES = ${_SEQS} sequences/step" >&2
    echo "  x ${_SEQ_LEN} tokens x ~18 B/token = ~$(( _EST / 1048576 )) MiB" >&2
    echo "  cap = $(( _CAP_BYTES / 1048576 )) MiB  ->  at most ${_MAX_SEQS} sequences/step" >&2
    echo >&2
    if (( _FIT_N >= 1 )); then
        echo "Fix: N_SAMPLES=${_FIT_N} (or lower TRAIN_BSZ / RESPONSE_LEN)." >&2
    else
        echo "Fix: TRAIN_BSZ=${_MAX_SEQS} with N_SAMPLES=1, or shorten RESPONSE_LEN." >&2
    fi
    exit 1
fi

# pi_old is re-derived per fwd_bwd, never snapshotted, so the ratio is 1 and
# eps_clip cannot bind. Silent, and it changes the objective -- so say it.
if (( MINI_BSZ < TRAIN_BSZ )); then
    echo "NOTE: ${TRAIN_BSZ}/${MINI_BSZ} = $(( TRAIN_BSZ / MINI_BSZ )) minibatches per step, and PPO" >&2
    echo "      clipping does not engage on Cortex (pi_old is re-derived, so the ratio is 1)." >&2
    echo "      These are unclipped updates, not clipped GRPO. approx_kl=0.0 is expected." >&2
    echo "      Set MINI_BSZ=${TRAIN_BSZ} for one update per step if that matters." >&2
fi

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
