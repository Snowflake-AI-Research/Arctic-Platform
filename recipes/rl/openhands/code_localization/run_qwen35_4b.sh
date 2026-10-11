#!/bin/bash
# GRPO for an OpenHands code-localization agent on Cortex.
# Cortex owns training and sampling. This process is a CPU-only driver.
#
# The harness is arctic_platform.integrations.openhands. SkyRL drives the loop.
# OpenHands SDK rev 85ecfd93 is installed by this launcher, not vendored.
#
# Knobs match the Qwen3.5-4B localization run: 4 training GPUs, 4 sampling
# GPUs, 8 prompts x 8 rollouts, 10 turns, 40960 context, GSPO, lr 1e-6.
# use_liger is false because the fused kernel rejects this sampler's outputs.
# use_zorro is false because a multi-turn response is not one fixed length.
#
# Pass extra Hydra overrides after the script. The recorded comparison used
# trainer.max_training_steps=100.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AP_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ARCTIC_PLATFORM_SPEC="${ARCTIC_PLATFORM_SPEC:-${AP_ROOT}[rl,cortex,openhands]}"
OPENHANDS_REV="${OPENHANDS_REV:-85ecfd9333d2d2cc4404dd460fd38868d9b978e2}"
OPENHANDS_SPEC="openhands-sdk @ git+https://github.com/OpenHands/software-agent-sdk.git@${OPENHANDS_REV}#subdirectory=openhands-sdk"
OPENHANDS_TOOLS_SPEC="openhands-tools @ git+https://github.com/OpenHands/software-agent-sdk.git@${OPENHANDS_REV}#subdirectory=openhands-tools"
OPENHANDS_WORKSPACE_SPEC="openhands-workspace @ git+https://github.com/OpenHands/software-agent-sdk.git@${OPENHANDS_REV}#subdirectory=openhands-workspace"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. Install it with:" >&2
    echo "         curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or does not contain integrations/arctic_rl/." >&2
    echo "       Clone https://github.com/NovaSky-AI/SkyRL at skyrl-v0.3.0 and export SKYRL_HOME." >&2
    exit 1
fi

DATA_DIR="${DATA_DIR:-${HOME}/data/swe-smith-localization}"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/validation.parquet"
if [[ ! -f "${TRAIN_FILES}" || ! -f "${VAL_FILES}" ]]; then
    echo "ERROR: localization parquets not found under ${DATA_DIR}." >&2
    echo "       The rows are the SWE-smith split shipped with CodeScout as data/swe_smith/." >&2
    echo "       Copy train.parquet and validation.parquet there, or set DATA_DIR." >&2
    exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"
NUM_SAMPLING_ENGINES="${NUM_SAMPLING_ENGINES:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
MAX_TURNS="${MAX_TURNS:-10}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-8192}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-40960}"
LR="${LR:-1.0e-6}"
LOGGER="${LOGGER:-console}"
RUN_NAME="${RUN_NAME:-qwen35-4b-openhands-localization}"
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${RUN_NAME}}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
mkdir -p "${CKPT_DIR}"

# Cortex returns about 18 bytes of logprob and entropy per token, against a
# 128 MiB response cap. 8 x 8 sequences at 40960 tokens is about 47 MiB.
_CAP_BYTES=134217728
_SEQS=$(( BATCH_SIZE * N_ROLLOUTS ))
_EST=$(( _SEQS * MAX_CONTEXT_LENGTH * 18 ))
if (( _EST > _CAP_BYTES )); then
    echo "ERROR: ${_SEQS} sequences x ${MAX_CONTEXT_LENGTH} tokens can exceed Cortex's 128 MiB response cap." >&2
    exit 1
fi

if ! _CORTEX_TARGET="$(uv run --isolated --no-project --with "${AP_ROOT}[cortex]" -- python - <<'PY'
import sys

from pydantic import ValidationError

from arctic_platform.client import CortexConfig

try:
    cfg = CortexConfig()
except ValidationError:
    sys.exit(
        "ERROR: no Cortex connection. Either run\n"
        "         cortex-training login <config.json>\n"
        "       or export ARCTIC_CORTEX_HOST, _DATABASE, _SCHEMA, and _PAT."
    )
print(cfg.base_url or f"{cfg.host} {cfg.database}.{cfg.schema_}")
PY
)"; then
    exit 1
fi
echo "Cortex target: ${_CORTEX_TARGET}"

export PYTHONPATH="${SKYRL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export OPENHANDS_WORKSPACE="${OPENHANDS_WORKSPACE:-/tmp/testbed}"
export ARCTIC_CORTEX_JOB_ID_FILE="${CKPT_DIR}/cortex_job_id"
export CORTEX_JOB_COMMENT="${CORTEX_JOB_COMMENT:-openhands code localization ${MODEL} ${NUM_TRAIN_GPUS}+${NUM_SAMPLING_ENGINES}}"

ENGINE_URLS="$(printf 'http://cortex-managed,%.0s' $(seq "${NUM_SAMPLING_ENGINES}"))"
ENGINE_URLS="[${ENGINE_URLS%,}]"

release_job() {
    if [[ -f "${ARCTIC_CORTEX_JOB_ID_FILE}" ]]; then
        uv run --isolated --no-project --with "${AP_ROOT}[cortex]" \
            -- python -m arctic_platform.integrations.openhands.release \
            "$(cat "${ARCTIC_CORTEX_JOB_ID_FILE}")" || true
    fi
}

stop_driver() {
    if [[ -n "${DRIVER_PID:-}" ]]; then
        kill -TERM "${DRIVER_PID}" 2>/dev/null || true
        wait "${DRIVER_PID}" 2>/dev/null || true
    fi
    exit 130
}

on_exit() {
    local code=$?
    trap - EXIT
    release_job
    exit "${code}"
}

trap stop_driver INT TERM
trap on_exit EXIT

cd "${SKYRL_HOME}"
uv run --isolated --extra skyrl-train \
    --with "${ARCTIC_PLATFORM_SPEC}" \
    --with "${OPENHANDS_SPEC}" \
    --with "${OPENHANDS_TOOLS_SPEC}" \
    --with "${OPENHANDS_WORKSPACE_SPEC}" \
    -- python -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=arctic_platform.integrations.openhands.entrypoint \
    trainer.arctic_rl.colocate=false \
    trainer.arctic_rl.use_zorro=false \
    trainer.arctic_rl.use_liger=false \
    trainer.arctic_rl.zero_stage=2 \
    trainer.arctic_rl.attn_implementation=flash_attention_3 \
    trainer.arctic_rl.vllm_max_model_len="${MAX_CONTEXT_LENGTH}" \
    trainer.arctic_rl.startup_timeout="${STARTUP_TIMEOUT}" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.grpo_norm_by_std=false \
    trainer.algorithm.policy_loss_type=gspo \
    trainer.algorithm.loss_reduction=sequence_mean \
    trainer.algorithm.eps_clip_low=0.0003 \
    trainer.algorithm.eps_clip_high=0.0004 \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false \
    trainer.algorithm.use_entropy_loss=false \
    trainer.policy.model.path="${MODEL}" \
    data.train_data="['${TRAIN_FILES}']" \
    data.val_data="['${VAL_FILES}']" \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_nodes=1 \
    trainer.placement.policy_num_gpus_per_node="${NUM_TRAIN_GPUS}" \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.num_engines="${NUM_SAMPLING_ENGINES}" \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.run_engines_locally=false \
    "generator.inference_engine.external_server_urls=${ENGINE_URLS}" \
    generator.sampling_params.logprobs=null \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.enable_ray_prometheus_stats=false \
    generator.batched=false \
    generator.n_samples_per_prompt="${N_ROLLOUTS}" \
    generator.max_turns="${MAX_TURNS}" \
    generator.max_input_length="${MAX_CONTEXT_LENGTH}" \
    generator.sampling_params.max_generate_length="${MAX_GENERATE_LENGTH}" \
    generator.sampling_params.temperature=1.0 \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size="${BATCH_SIZE}" \
    trainer.policy_mini_batch_size="${BATCH_SIZE}" \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length="${MAX_CONTEXT_LENGTH}" \
    trainer.eval_before_train=false \
    trainer.eval_interval=-1 \
    trainer.ckpt_interval=10 \
    trainer.hf_save_interval=50 \
    trainer.resume_mode=none \
    trainer.policy.optimizer_config.lr="${LR}" \
    trainer.logger="${LOGGER}" \
    trainer.project_name=openhands_localization \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_path="${CKPT_DIR}" \
    trainer.export_path="${CKPT_DIR}/exported_model" \
    "$@" > >(tee "${CKPT_DIR}/${RUN_NAME}.log") 2>&1 &

DRIVER_PID=$!
set +e
wait "${DRIVER_PID}"
status=$?
set -e
exit "${status}"
