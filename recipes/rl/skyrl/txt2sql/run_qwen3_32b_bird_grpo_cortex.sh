#!/bin/bash
# GRPO training for Qwen3-32B on BIRD-SQL with the Arctic RL Cortex backend.
# Cortex sub-jobs own training and sampling; the SkyRL driver is CPU-only.
#
# This is the Qwen3-8B Cortex ZoRRo launcher
# (run_qwen3_0.6b_bird_grpo_cortex.sh plus the overrides from job
# 20261026T145558Z) with the published txt2sql 32B learning config
# (run_qwen3_32b_bird_grpo_arl_4node.sh on Arctic-Platform main):
#   128 prompts x 16, prompt 32768, response 4096, lr 2e-6, FA3,
#   optimizer offload, warmup 0.05, betas 0.9/0.95, gradient checkpointing.
#
# Disaggregated, not the colocated 4x8 on-prem layout:
#   16 training GPUs  (NUM_NODES=2, NGPU_PER_NODE=8)
#   16 sampling GPUs  (NUM_ENGINES=4, TP_SIZE=4)
# ZeRO-2, pack off, and no CUDA-IPC stay as in the 8B Cortex run. The
# published on-prem script is ZeRO-3 and colocated. Micro-batch 16 on 16
# training GPUs is grad accumulation 8 for the 2048-sequence step.
#
# Image dev_20261028_230845_c4bf620c1de (build 1019, cortex c4bf620c1de).
# The worker splits a ZoRRo prompt group at the DeepSpeed micro-batch.
# This batch is over the 128 MiB response check, so CORTEX_ALLOW_LARGE_PAYLOAD
# defaults on. fwd-bwd is sent in 60 MiB chunks.
#
# Pre-reqs:
#   1. SkyRL cloned with SKYRL_HOME exported.
#   2. ARCTIC_CORTEX_* env vars set (see ../simple_gsm8k_cortex/README.md).
#   3. Data: `python download_data.py --bird_dir <raw> --output_dir $DATA_DIR`
#      -> $DATA_DIR/{train,val}.parquet. SQL rewards run on the driver, so
#      the SQLite files named in extra_info.db_path must be readable here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This launcher ships inside the Arctic Platform checkout, so it can locate the
# repo root rather than being told. Point ARCTIC_PLATFORM_SPEC at a released
# version instead once one ships arctic_platform/integrations/ (0.1.3 does not).
# [sft] carries the RL client's DeepSpeed/liger imports; [rl] would add
# arctic-inference, which builds from source and is dead weight on a CPU driver.
AP_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ARCTIC_PLATFORM_SPEC="${ARCTIC_PLATFORM_SPEC:-${AP_ROOT}[sft,cortex]}"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. Install it with:"
    echo "         curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
    exit 1
fi
# ${SCRIPT_DIR} carries sitecustomize.py, which registers bird / bird_sql in the
# driver, its Ray workers, and the reward-scoring subprocesses.
export PYTHONPATH="${SKYRL_HOME}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
# Ray otherwise replays the driver's `uv run` line on every worker to rebuild
# its environment. Under `--isolated` that replay spawns a broken command
# (`Failed to spawn: --python`), and it buys nothing here: Cortex owns the GPUs,
# so the only Ray actors are driver-local and can share the driver interpreter.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export ARCTIC_CORTEX_IMAGE_TAG="${ARCTIC_CORTEX_IMAGE_TAG:-dev_20261028_230845_c4bf620c1de}"
export CORTEX_ALLOW_LARGE_PAYLOAD="${CORTEX_ALLOW_LARGE_PAYLOAD:-1}"

# Cortex owns the GPUs; the driver has none. SkyRL's colocated placement
# groups would deadlock on a CPU driver.
# 2 nodes x 8 GPUs = 16 training GPUs. 4 engines x TP 4 = 16 sampling GPUs.
NUM_NODES="${NUM_NODES:-2}"
NGPU_PER_NODE="${NGPU_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-4}"
NUM_ENGINES="${NUM_ENGINES:-4}"

# Batch, length, and LR are the published 32B txt2sql recipe's. MICRO_BSZ is
# the 8B Cortex ZoRRo value: SkyRL's default of 1 cannot satisfy DeepSpeed's
# micro x accum x n_gpus == train_batch * n_samples on the training sub-job.
TRAIN_BSZ="${TRAIN_BSZ:-128}"
MINI_BSZ="${MINI_BSZ:-128}"
N_SAMPLES="${N_SAMPLES:-16}"
MICRO_BSZ="${MICRO_BSZ:-16}"
PROMPT_LEN="${PROMPT_LEN:-32768}"
RESPONSE_LEN="${RESPONSE_LEN:-4096}"
LR="${LR:-2e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-false}"
ARCTIC_ZERO_STAGE="${ARCTIC_ZERO_STAGE:-2}"
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-true}"
ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-false}"

# Cortex returns a logprob and an entropy per token (~18 B/token) against a
# 128 MiB response cap, and reports an overrun only after provisioning GPUs
# and a full step -- as a 429 naming neither the cap nor the knob. Check here.
_CAP_BYTES=134217728
_SEQ_LEN=$(( PROMPT_LEN + RESPONSE_LEN ))
_SEQS=$(( TRAIN_BSZ * N_SAMPLES ))
_EST=$(( _SEQS * _SEQ_LEN * 18 ))
if (( _EST > _CAP_BYTES )) && [[ "${CORTEX_ALLOW_LARGE_PAYLOAD}" != "1" ]]; then
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
if (( _EST > _CAP_BYTES )); then
    echo "WARN: payload estimate ~$(( _EST / 1048576 )) MiB is over the $(( _CAP_BYTES / 1048576 )) MiB check." >&2
    echo "      CORTEX_ALLOW_LARGE_PAYLOAD=1, so this run continues. fwd-bwd is sent in 60 MiB chunks." >&2
fi

# run_engines_locally=false requires external_server_urls to be non-empty, so
# pass placeholders: the real endpoint lives inside the Cortex shim, which never
# dials these. Keep the count tracking NUM_ENGINES: the router is built over
# this list.
ENGINE_URLS="$(printf 'http://cortex-managed,%.0s' $(seq "${NUM_ENGINES}"))"
ENGINE_URLS="[${ENGINE_URLS%,}]"

LOGGER="${LOGGER:-console}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
MODEL_SHORT="$(basename "${MODEL}")"
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME="bird_grpo_${MODEL_SHORT}_cortex_16t16s_${RUN_TS}"

# BirdEnv reads reward_model.ground_truth and extra_info.db_path, which is the
# verl-shaped parquet download_data.py writes. environment.env_class=bird below
# picks the env, so the rows need no env_class column.
DATA_DIR="${DATA_DIR:-${HOME}/data/bird}"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/val.parquet"

if [[ ! -f "${TRAIN_FILES}" || ! -f "${VAL_FILES}" ]]; then
    echo "ERROR: BIRD-SQL parquets not found at ${DATA_DIR}/{train,val}.parquet"
    echo "       Run 'python download_data.py --bird_dir <raw> --output_dir ${DATA_DIR}' first."
    exit 1
fi

# Pre-flight: BirdEnv raises on a row without the gold SQL or its database,
# and a database the driver cannot open scores every rollout 0.0. Check the
# columns and the first row's SQLite file before paying for Cortex GPUs.
#
# Runs under uv for the same reason the training invocation does: there is no
# environment to activate, so the ambient shell is not expected to import
# pandas. pyarrow is the parquet engine and is not implied by pandas.
uv run --isolated --no-project --with pandas --with pyarrow python - <<PY || exit 1
import os
import sys

import pandas as pd

for path in ("${TRAIN_FILES}", "${VAL_FILES}"):
    frame = pd.read_parquet(path)
    missing = {"prompt", "reward_model", "extra_info"} - set(frame.columns)
    if missing:
        print(f"ERROR: {path} is missing BIRD fields {sorted(missing)}.")
        sys.exit(1)
    db_path = (frame["extra_info"].iloc[0] or {}).get("db_path")
    if not db_path or not os.path.isfile(db_path):
        print(f"ERROR: {path} row 0 names db_path={db_path!r}, which this host cannot read.")
        print("       SQL rewards run on the driver; mount the BIRD databases here.")
        sys.exit(1)
PY

CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

# ----- Stopping this run, and getting the GPUs back -----
# Nothing releases the GPUs on its own. SkyRL never calls the client's shutdown
# when training ends, so finishing the last step holds the GPUs exactly as
# firmly as being killed does, and the next launch then dies on the
# per-account cap with a 429 that never mentions your own previous run.
#
# The driver also ignores SIGINT and SIGTERM -- Ray installs handlers -- so
# there is no graceful stop to ask for: Ctrl-C alone leaves it training. Hence
# the shape below. The driver runs in the background so a signal reaches this
# shell immediately (bash defers a trap until the running foreground command
# returns), and stopping means SIGKILL.
#
# Kill the driver's descendants before the driver itself, while the tree is
# still intact. Its dataloader workers survive a bare kill of the parent,
# reparent to init, and sit on ~0.5 GB each; enough interrupted runs and the
# node OOMs a later, innocent one.

# `list`/`cancel` against the transport. Inline rather than a recipe-local CLI:
# `cortex-client/dss_neutrino_cli.py` already does this properly, and porting it
# onto the unified client is issue #101 -- once that lands, call it here instead.
cortex_jobs() {
    local mode="$1"; shift
    # Deps come from uv, not the ambient interpreter: this runs before the
    # training invocation below, so there is no environment to inherit. Only
    # the client is needed, hence --no-project and the [cortex] extra alone.
    CORTEX_JOBS_MODEL="${MODEL}" uv run --isolated --no-project --with "${AP_ROOT}[cortex]" \
        -- python - "${mode}" "$@" <<'PY'
import os
import sys

from arctic_platform.client.config import CortexConfig
from arctic_platform.client.requests import cancel_job_request
from arctic_platform.client.requests import list_jobs_request
from arctic_platform.client.transports.cortex import CortexTransport

# Anything not in here still holds its GPUs.
TERMINAL = {"failed", "done", "cancelled", "canceled", "succeeded", "terminated"}
mode, wanted = sys.argv[1], set(sys.argv[2:])
model = os.environ["CORTEX_JOBS_MODEL"]


# The before/after diff alone also catches a teammate's job launched mid-run on
# the same account; requiring every sub-job to be this run's model rules that out
# unless they are training the same model.
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

# An empty snapshot and a failed one both look like "" but mean opposite
# things: with the first, every job seen later is ours; with the second, none
# of them provably is. Only release when the baseline is real.
if _JOBS_BEFORE="$(cortex_jobs list 2>/dev/null)"; then
    _SNAPSHOT_OK=1
else
    _SNAPSHOT_OK=0
    _JOBS_BEFORE=""
fi
_RELEASED=0

release_our_jobs() {
    # INT fires this trap and then EXIT fires it again; without the guard the
    # second pass reports failing to cancel an already-cancelled job.
    (( _RELEASED )) && return 0
    _RELEASED=1
    local after ours
    if (( ! _SNAPSHOT_OK )); then
        echo "cannot tell which Cortex jobs are this run's; check for leftovers" >&2
        echo "with the Cortex job CLI before launching again." >&2
        return 0
    fi
    # Only jobs that appeared while we ran, and only ones still live: the cap
    # is per-account, so a blanket cancel could take out a teammate's run.
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
    echo "stopping: the driver ignores SIGTERM, so killing its process tree." >&2
    if [[ -n "${DRIVER_PID:-}" ]]; then
        kill -KILL $(_descendants "${DRIVER_PID}") "${DRIVER_PID}" 2>/dev/null || true
    fi
    release_our_jobs
    exit 130
}

trap stop_driver INT TERM
trap release_our_jobs EXIT

echo "cortex 32b zorro: train ${NUM_NODES}x${NGPU_PER_NODE}=$((NUM_NODES * NGPU_PER_NODE)) sample engines ${NUM_ENGINES} tp ${TP_SIZE}=$((NUM_ENGINES * TP_SIZE)) micro ${MICRO_BSZ} n ${N_SAMPLES} zero ${ARCTIC_ZERO_STAGE} activation_offload ${ACTIVATION_OFFLOAD}"
echo "image ${ARCTIC_CORTEX_IMAGE_TAG}"

# Upstream's own entrypoint is `integrations.arctic_rl.entrypoint`; naming ours
# instead is what selects Cortex, and it installs the driver-side
# peer_access_supported shim before SkyRL's Ray probe runs.
# Ray's uv runtime-env hook requires the uv project to live inside the working
# directory, so run from $SKYRL_HOME instead of pointing --project at it. Every
# path passed below is absolute, so the cd is safe.
cd "${SKYRL_HOME}"
uv run --isolated --extra skyrl-train \
    --with "${ARCTIC_PLATFORM_SPEC}" \
    -- python -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=arctic_platform.integrations.skyrl.entrypoint \
    trainer.arctic_rl.colocate=false \
    trainer.arctic_rl.attn_implementation=flash_attention_3 \
    trainer.arctic_rl.use_liger=true \
    trainer.arctic_rl.use_zorro=true \
    trainer.arctic_rl.zero_stage=${ARCTIC_ZERO_STAGE} \
    trainer.arctic_rl.offload_optimizer=${OFFLOAD_OPTIMIZER} \
    trainer.arctic_rl.offload_param=false \
    trainer.arctic_rl.pack=false \
    trainer.arctic_rl.balance_batch=true \
    trainer.arctic_rl.cuda_ipc_weight_sync=false \
    trainer.arctic_rl.use_arctic_inference=false \
    trainer.arctic_rl.logits_optimization=memory \
    trainer.arctic_rl.enable_gradient_checkpointing=true \
    trainer.arctic_rl.enable_activation_offload=${ACTIVATION_OFFLOAD} \
    trainer.arctic_rl.ulysses_sequence_parallel_size=1 \
    trainer.arctic_rl.lr_warmup_ratio=0.05 \
    'trainer.arctic_rl.optimizer_betas=[0.9,0.95]' \
    trainer.arctic_rl.vllm_enforce_eager=false \
    trainer.arctic_rl.vllm_enable_prefix_caching=true \
    trainer.arctic_rl.vllm_max_num_batched_tokens=40960 \
    trainer.arctic_rl.vllm_max_num_seqs=256 \
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
    "generator.inference_engine.external_server_urls=${ENGINE_URLS}" \
    "generator.sampling_params.logprobs=null" \
    generator.inference_engine.enable_ray_prometheus_stats=false \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.gpu_memory_utilization=0.7 \
    generator.batched=true \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    environment.env_class=bird \
    trainer.epochs=${TOTAL_EPOCHS} \
    trainer.train_batch_size=${TRAIN_BSZ} \
    trainer.policy_mini_batch_size=${MINI_BSZ} \
    trainer.micro_train_batch_size_per_gpu=${MICRO_BSZ} \
    trainer.micro_forward_batch_size_per_gpu=${MICRO_BSZ} \
    trainer.max_prompt_length=${PROMPT_LEN} \
    generator.sampling_params.max_generate_length=${RESPONSE_LEN} \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.eval_sampling_params.max_generate_length=${RESPONSE_LEN} \
    generator.eval_sampling_params.temperature=0.0 \
    generator.eval_sampling_params.top_p=1.0 \
    generator.eval_sampling_params.top_k=-1 \
    generator.eval_n_samples_per_prompt=1 \
    trainer.eval_batch_size=32 \
    trainer.eval_before_train=${EVAL_BEFORE_TRAIN} \
    trainer.eval_interval=${EVAL_INTERVAL} \
    trainer.update_epochs_per_batch=1 \
    trainer.policy.optimizer_config.lr=${LR} \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false \
    trainer.logger="${LOGGER}" \
    trainer.project_name=arctic_rl_bird_cortex \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CKPT_DIR}/logs" \
    trainer.ckpt_path="${CKPT_DIR}/ckpt" \
    trainer.ckpt_interval=-1 \
    "$@" > >(tee "${CKPT_DIR}/${EXPERIMENT_NAME}.log") 2>&1 &

# Process substitution rather than a pipe so this is the driver's own pid and
# not tee's -- stop_driver has to kill the process that holds the GPUs.
DRIVER_PID=$!

# `wait` is interruptible, so INT/TERM run their trap here immediately. Guard
# it: under `set -e` the 128+signal status would otherwise exit the script
# before the trap gets to release anything.
wait "${DRIVER_PID}" || DRIVER_STATUS=$?
exit "${DRIVER_STATUS:-0}"
