#!/bin/bash
# VeRL + Arctic RL on Cortex: Qwen3-32B GRPO on BIRD-SQL, ZoRRo on.
# Cortex plumbing is run_qwen3_8b_bird_grpo_cortex.sh. The learning config and
# the disaggregated layout are run_qwen3_32b_bird_grpo_cortex.sh (SkyRL):
#   128 prompts x 16, prompt 32768, response 4096, lr 2e-6, warmup 0.05,
#   betas 0.9/0.95, ZeRO-2, optimizer offload, FA3, Liger, pack off, balance on.
# 16 training GPUs and 16 sampling GPUs (4 engines x TP 4). One epoch, eval
# every 100 steps. Cortex owns the GPUs; this driver is CPU-only. SQL rewards
# run on the driver, so the SQLite files named in each row's extra_info.db_path
# must be readable here.
#
# Flags that differ from on-prem, all plumbing for a remote backend:
#   remote_backend.comms.protocol=cortex   selects the Cortex backend (http is an alias).
#   deepspeed zero stage 2                 Cortex rejected stage 3.
#   use_liger=True                         image dev_20261028_230845_c4bf620c1de
#                                          (build 1019, cortex c4bf620c1de). Its unpacked no-ZoRRo
#                                          worker leaves labels on the logprob post and off the
#                                          Liger call. ZoRRo replaces that forward, is not wrapped,
#                                          and splits a prompt group at the DeepSpeed micro-batch.
#   attn flash_attention_3                 the image ships FA3, not FA2.
#   colocate=False, log_prob_gpus=0, cuda_ipc=False
#                                          separate sub-jobs, NCCL weight sync, and no
#                                          DeepSpeed log-prob job on Cortex.
#
# The pinned image's forward returns log-probs and no entropy. entropy_coeff stays 0.
#
# Arm knobs (env):
#   USE_ZORRO       0|1. Defaults 1. remote_backend.train.zorro_train.enable.
#   MICRO_BSZ       DeepSpeed train_micro_batch_size_per_gpu. Defaults 16 with
#                   ZoRRo (must be a multiple of N_SAMPLES) and 1 without.
#   BALANCE_BATCH   true|false. Arctic rank assignment; false keeps arrival order.
#   PACK            true|false. Sent as ds_worker_config.pack. false (default) runs
#                   the unpacked verl_grpo path with the fwd_no_grad old log-probs.
#                   true asks Cortex for a packed job, where the request is rewritten
#                   to grpo and pi_old is the live policy (ratio 1). ZoRRo never packs.
#
# Required env:
#   ARCTIC_CORTEX_IMAGE_TAG   exactly dev_20261028_230845_c4bf620c1de. The client's
#                             job_create_payload sends it as the job's image tag.
#   ARCTIC_CORTEX_HOST, ARCTIC_CORTEX_PAT (or CORTEX_PAT), ARCTIC_CORTEX_DATABASE,
#   ARCTIC_CORTEX_SCHEMA. Source them with `set +x`; this script never prints the PAT.
#
# Other env: MODEL, DATA_DIR, TRAIN_BSZ, MINI_BSZ, N_SAMPLES, PROMPT_LEN,
# RESPONSE_LEN, MAX_STEPS (unset = one epoch), LR, TRAINING_GPUS, SAMPLING_GPUS,
# SAMPLING_TP, EVAL_INTERVAL, LOGITS_OPT,
# MAX_TOKEN_LEN, VERL_HOME, CKPT_DIR, LOGGER. PYBIN names an interpreter that
# already has the driver deps; unset, they resolve through `uv run --isolated`
# (UV_NO_CACHE=1). arctic_platform and verl always come from AP_ROOT and
# VERL_HOME on PYTHONPATH: an installed arctic_platform would make the VeRL
# plugin demand its [verl] extra, which builds arctic-inference from source.
# PREFLIGHT_ONLY=1 stops after the driver checks, before any Cortex job exists.
# Extra Hydra overrides pass through "$@".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AP_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
VERL_HOME="${VERL_HOME:-/code/users/truwase/projects/cortex_training/verl-verl-parity}"
if [[ ! -d "${VERL_HOME}/verl/remote_backend" ]]; then
    echo "ERROR: VERL_HOME=${VERL_HOME} has no verl/remote_backend (needs the plugin-hook verl, 5a6491cf or later)." >&2
    exit 1
fi
export AP_ROOT VERL_HOME

EXPECTED_IMAGE_TAG="dev_20261028_230845_c4bf620c1de"
if [[ -z "${ARCTIC_CORTEX_IMAGE_TAG:-}" ]]; then
    echo "ERROR: ARCTIC_CORTEX_IMAGE_TAG is unset. Export ARCTIC_CORTEX_IMAGE_TAG=${EXPECTED_IMAGE_TAG};" >&2
    echo "       without it the job runs the release image." >&2
    exit 2
fi
if [[ "${ARCTIC_CORTEX_IMAGE_TAG}" != "${EXPECTED_IMAGE_TAG}" ]]; then
    echo "ERROR: ARCTIC_CORTEX_IMAGE_TAG=${ARCTIC_CORTEX_IMAGE_TAG}; this recipe is pinned to ${EXPECTED_IMAGE_TAG}." >&2
    exit 2
fi
export EXPECTED_IMAGE_TAG

if [[ -z "${ARCTIC_CORTEX_PAT:-}" && -n "${CORTEX_PAT:-}" ]]; then
    export ARCTIC_CORTEX_PAT="${CORTEX_PAT}"
fi
if [[ -z "${ARCTIC_CORTEX_PAT:-}" || -z "${ARCTIC_CORTEX_HOST:-}" ]]; then
    echo "ERROR: ARCTIC_CORTEX_HOST and ARCTIC_CORTEX_PAT must be set (source the Cortex env file under set +x)." >&2
    exit 2
fi

export PYTHONPATH="${AP_ROOT}:${VERL_HOME}:${PYTHONPATH:-}"
export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register
ARCTIC_VERL_CONFIG_DIR="${AP_ROOT}/arctic_platform/integrations/verl/config"

# Cortex owns the GPUs; the driver has none.
export CUDA_VISIBLE_DEVICES=""
export UV_NO_CACHE=1
if [[ -z "${UV_PYTHON:-}" && -x /home/yak/miniconda3/envs/dev/bin/python3.12 ]]; then
    export UV_PYTHON=/home/yak/miniconda3/envs/dev/bin/python3.12
fi
export CORTEX_ALLOW_LARGE_PAYLOAD="${CORTEX_ALLOW_LARGE_PAYLOAD:-1}"
export ARCTIC_CORTEX_JOB_READY_TIMEOUT="${ARCTIC_CORTEX_JOB_READY_TIMEOUT:-3600}"
# Ray otherwise replays the driver's `uv run` line on every worker, which breaks
# under --isolated. The only Ray actors are driver-local and share its interpreter.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# A fresh Ray per run: no attaching to another run's cluster. Short path for the
# Unix socket length limit.
unset RAY_ADDRESS
RAY_TMPDIR="$(mktemp -d /tmp/rvc-XXXXXX)"
export RAY_TMPDIR
unset PYTORCH_CUDA_ALLOC_CONF
unset ARCTIC_INFERENCE_ENABLED
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

_bool() {
    case "${1,,}" in
        1|true|yes) echo True ;;
        0|false|no) echo False ;;
        *) echo "ERROR: expected a boolean, got '$1'" >&2; return 1 ;;
    esac
}

USE_ZORRO="$(_bool "${USE_ZORRO:-1}")"
BALANCE_BATCH="$(_bool "${BALANCE_BATCH:-true}")"
PACK="$(_bool "${PACK:-false}")"

TRAINING_GPUS="${TRAINING_GPUS:-16}"
SAMPLING_GPUS="${SAMPLING_GPUS:-16}"
SAMPLING_TP="${SAMPLING_TP:-4}"
TRAIN_BSZ="${TRAIN_BSZ:-128}"
MINI_BSZ="${MINI_BSZ:-128}"
N_SAMPLES="${N_SAMPLES:-16}"
PROMPT_LEN="${PROMPT_LEN:-32768}"
RESPONSE_LEN="${RESPONSE_LEN:-4096}"
LR="${LR:-2e-6}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOGGER="${LOGGER:-console}"
if (( SAMPLING_TP < 1 || SAMPLING_GPUS % SAMPLING_TP != 0 )); then
    echo "ERROR: SAMPLING_GPUS=${SAMPLING_GPUS} must be a positive multiple of SAMPLING_TP=${SAMPLING_TP}." >&2
    exit 1
fi

if [[ "${USE_ZORRO}" == "True" ]]; then
    ARM="zorro"
    MICRO_BSZ="${MICRO_BSZ:-16}"
    MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-$(( PROMPT_LEN + N_SAMPLES * RESPONSE_LEN ))}"
    LOGITS_OPT="${LOGITS_OPT:-memory}"
    if (( MICRO_BSZ % N_SAMPLES != 0 )); then
        echo "ERROR: ZoRRo MICRO_BSZ=${MICRO_BSZ} must be a multiple of N_SAMPLES=${N_SAMPLES}." >&2
        exit 1
    fi
else
    ARM="nozorro"
    MICRO_BSZ="${MICRO_BSZ:-1}"
    MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-$(( PROMPT_LEN + RESPONSE_LEN ))}"
    LOGITS_OPT="${LOGITS_OPT:-none}"
fi
_STEP_ROWS=$(( MINI_BSZ * N_SAMPLES ))
if (( _STEP_ROWS % (MICRO_BSZ * TRAINING_GPUS) != 0 )); then
    echo "ERROR: ${_STEP_ROWS} rows/step not divisible by MICRO_BSZ=${MICRO_BSZ} x TRAINING_GPUS=${TRAINING_GPUS}." >&2
    exit 1
fi

# Cortex returns ~18 B/token against a 128 MiB response cap and reports an
# overrun only after provisioning 8 GPUs and a full step.
_CAP_BYTES=134217728
_EST=$(( TRAIN_BSZ * N_SAMPLES * (PROMPT_LEN + RESPONSE_LEN) * 18 ))
if (( _EST > _CAP_BYTES )); then
    if [[ "${CORTEX_ALLOW_LARGE_PAYLOAD}" != "1" ]]; then
        echo "ERROR: ~$(( _EST / 1048576 )) MiB/step overruns the $(( _CAP_BYTES / 1048576 )) MiB Cortex response cap;" >&2
        echo "       lower TRAIN_BSZ, N_SAMPLES or the lengths, or set CORTEX_ALLOW_LARGE_PAYLOAD=1." >&2
        exit 1
    fi
    echo "WARN: payload estimate ~$(( _EST / 1048576 )) MiB is over the $(( _CAP_BYTES / 1048576 )) MiB check." >&2
    echo "      CORTEX_ALLOW_LARGE_PAYLOAD=1, so this run continues. fwd-bwd is sent in 60 MiB chunks." >&2
fi

MODEL="${MODEL:-Qwen/Qwen3-32B}"
MODEL_SHORT="$(basename "${MODEL}")"
DATA_DIR="${DATA_DIR:-/code/users/truwase/data/open-source-text2sql}"
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
    echo "ERROR: BIRD-SQL parquets not found at ${DATA_DIR}/{train,val}.parquet" >&2
    exit 1
fi

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
EXPERIMENT_NAME="verl_bird_${MODEL_SHORT}_cortex_${ARM}_mb${MICRO_BSZ}_bal${BALANCE_BATCH}_pack${PACK}_${RUN_TS}"
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"
LOG_FILE="${CKPT_DIR}/${EXPERIMENT_NAME}.log"

# Driver deps without the arctic_platform package itself (see the header):
# arctic_platform base + [sft] + [cortex], and verl + [vllm].
DRIVER_DEPS=(
    "pydantic>=2.10" "pydantic-settings>=2.0" typing-extensions
    "deepspeed>=0.19.2" "liger-kernel>=0.8.5" nvidia-ml-py "peft>=0.21.2,<0.22" psutil safetensors torch
    "transformers>=5.17,<5.18"
    aiohttp requests tenacity urllib3
    "vllm>=0.8.5,<=0.12.0" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "numpy<2.0.0"
    hydra-core omegaconf codetiming torchdata datasets pylatexenc accelerate dill pandas "pyarrow>=19"
    "ray[default]>=2.41.0" wandb "packaging>=20.0" tensorboard pybind11
)
if [[ -n "${PYBIN:-}" ]]; then
    export DRIVER_PY="${PYBIN}"
    LAUNCH=()
else
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv not found; install it or set PYBIN." >&2
        exit 1
    fi
    export DRIVER_PY=python
    LAUNCH=(uv run --isolated --no-project)
    for dep in "${DRIVER_DEPS[@]}"; do
        LAUNCH+=(--with "${dep}")
    done
    LAUNCH+=(--)
fi

# `list`/`cancel` this run's Cortex jobs, for when the driver dies without its
# own shutdown (VeRL's destroy() cancels the job on a normal exit).
cortex_jobs() {
    local mode="$1"; shift
    local runner=("${PYBIN:-}")
    if [[ -z "${PYBIN:-}" ]]; then
        runner=(uv run --isolated --no-project --with "pydantic-settings>=2.0" --with aiohttp --with tenacity
                --with safetensors --with requests --with urllib3 --with numpy --with torch -- python)
    fi
    CORTEX_JOBS_MODEL="${MODEL}" "${runner[@]}" - "${mode}" "$@" <<'PY'
import os
import sys

from arctic_platform.client.config import CortexConfig
from arctic_platform.client.requests import cancel_job_request
from arctic_platform.client.requests import list_jobs_request
from arctic_platform.client.transports.cortex import CortexTransport

TERMINAL = {"failed", "done", "cancelled", "canceled", "succeeded", "terminated"}
mode, wanted = sys.argv[1], set(sys.argv[2:])
model = os.environ["CORTEX_JOBS_MODEL"]


def is_this_model(job: dict) -> bool:
    subs = job.get("sub_jobs") or []
    return bool(subs) and all(s.get("model_name") == model for s in subs)


transport = CortexTransport(CortexConfig())
live = [
    j.get("job_id") or j.get("id") or j.get("name")
    for j in transport.call(list_jobs_request())
    if str(j.get("status") or j.get("state") or "?").lower() not in TERMINAL and is_this_model(j)
]
if mode == "list":
    print("\n".join(str(j) for j in live))
else:
    for job_id in (j for j in live if j in wanted):
        transport.call(cancel_job_request(job_id))
        print(f"released {job_id}", file=sys.stderr)
transport.shutdown()
PY
}

_SNAPSHOT_OK=0
_JOBS_BEFORE=""
_RELEASED=0
if [[ "${PREFLIGHT_ONLY:-0}" != "1" ]]; then
    if _JOBS_BEFORE="$(cortex_jobs list 2>/dev/null)"; then
        _SNAPSHOT_OK=1
    else
        _JOBS_BEFORE=""
    fi
fi

release_our_jobs() {
    (( _RELEASED )) && return 0
    _RELEASED=1
    [[ "${PREFLIGHT_ONLY:-0}" == "1" ]] && return 0
    local after ours
    if (( ! _SNAPSHOT_OK )); then
        echo "cannot tell which Cortex jobs are this run's; check for leftovers before launching again." >&2
        return 0
    fi
    # Only jobs that appeared while we ran: the cap is per-account.
    after="$(cortex_jobs list 2>/dev/null || true)"
    ours="$(comm -13 <(sort <<<"${_JOBS_BEFORE}") <(sort <<<"${after}") | tr '\n' ' ')"
    [[ -z "${ours// /}" ]] && return 0
    echo "releasing Cortex job(s) this run started: ${ours}" >&2
    cortex_jobs cancel ${ours} >&2 || true
}

_descendants() {
    local child
    for child in $(pgrep -P "$1" 2>/dev/null); do
        _descendants "${child}"
        echo "${child}"
    done
}

stop_driver() {
    echo "" >&2
    echo "stopping: killing this run's driver process tree." >&2
    if [[ -n "${DRIVER_PID:-}" ]]; then
        kill -KILL $(_descendants "${DRIVER_PID}") "${DRIVER_PID}" 2>/dev/null || true
    fi
    release_our_jobs
    exit 130
}

trap stop_driver INT TERM
trap release_our_jobs EXIT

echo "== VeRL BIRD Cortex 32B ${ARM}: zorro=${USE_ZORRO} micro=${MICRO_BSZ} balance=${BALANCE_BATCH} pack=${PACK} train=${TRAINING_GPUS} sample=${SAMPLING_GPUS} tp=${SAMPLING_TP} prompts=${TRAIN_BSZ} n=${N_SAMPLES} prompt=${PROMPT_LEN} response=${RESPONSE_LEN} lr=${LR} zero=2 offload=cpu logits=${LOGITS_OPT} token_cap=${MAX_TOKEN_LEN} image=${ARCTIC_CORTEX_IMAGE_TAG} =="
echo "ray temp dir ${RAY_TMPDIR}"

OVERRIDES=(
    hydra.searchpath="[file://${ARCTIC_VERL_CONFIG_DIR}]"
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    data.train_files="${DATA_DIR}/train.parquet"
    data.val_files="${DATA_DIR}/val.parquet"
    data.train_batch_size="${TRAIN_BSZ}"
    data.max_prompt_length="${PROMPT_LEN}"
    data.max_response_length="${RESPONSE_LEN}"
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=8
    data.truncation=left
    data.seed=42
    actor_rollout_ref.actor.data_loader_seed=42
    actor_rollout_ref.model.path="${MODEL}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_liger=True
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_3
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BSZ}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BSZ}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BSZ}"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN}"
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.optim.lr="${LR}"
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
    actor_rollout_ref.rollout.name=arctic
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.max_num_batched_tokens=40960
    actor_rollout_ref.rollout.n="${N_SAMPLES}"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.max_num_seqs=256
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.nccl_timeout=1800
    trainer.use_legacy_worker_impl=disable
    trainer.remote_backend=arctic
    remote_backend=arctic
    trainer.balance_batch=False
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.total_epochs=1
    trainer.save_freq=-1
    trainer.test_freq="${EVAL_INTERVAL}"
    trainer.val_before_train=False
    trainer.resume_mode=disable
    trainer.logger="['${LOGGER}']"
    trainer.project_name=verl_arctic_rl_bird_cortex
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${CKPT_DIR}/ckpt"
    custom_reward_function.path="${SCRIPT_DIR}/bird_reward.py"
    custom_reward_function.name=compute_score
    remote_backend.comms.protocol=cortex
    remote_backend.colocate=False
    remote_backend.training_gpus="${TRAINING_GPUS}"
    remote_backend.sampling_gpus="${SAMPLING_GPUS}"
    remote_backend.sampling_tp_size="${SAMPLING_TP}"
    remote_backend.log_prob_gpus=0
    remote_backend.weight_sync.cuda_ipc=False
    remote_backend.weight_sync.low_memory=False
    remote_backend.train.balance_batch="${BALANCE_BATCH}"
    remote_backend.train.pack="${PACK}"
    remote_backend.train.zorro_train.enable="${USE_ZORRO}"
    remote_backend.train.zorro_train.max_rollouts="${N_SAMPLES}"
    remote_backend.train.logits.optimization="${LOGITS_OPT}"
    remote_backend.train.deepspeed.zero_optimization.stage=2
    remote_backend.train.deepspeed.zero_optimization.offload_optimizer.device=cpu
    remote_backend.train.deepspeed.zero_optimization.offload_param.device=none
    remote_backend.rollout.zorro_inference.enable=False
    remote_backend.rollout.speculative_decoding.model=''
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    OVERRIDES+=(trainer.total_training_steps="${MAX_STEPS}")
fi

# One interpreter for the preflight and the trainer, so `uv run --isolated`
# resolves once. The preflight runs before any Cortex job exists.
"${LAUNCH[@]}" bash -c '
"${DRIVER_PY}" "$0" || exit $?
[[ "${PREFLIGHT_ONLY:-0}" == "1" ]] && exit 0
exec "${DRIVER_PY}" -m verl.trainer.main_ppo "$@"
' "${SCRIPT_DIR}/cortex_preflight.py" "${OVERRIDES[@]}" "$@" > >(tee -a "${LOG_FILE}") 2>&1 &

# Process substitution rather than a pipe so this is the driver's own pid and
# not tee's -- stop_driver has to kill the process tree that holds the jobs.
DRIVER_PID=$!
wait "${DRIVER_PID}" || DRIVER_STATUS=$?
exit "${DRIVER_STATUS:-0}"
