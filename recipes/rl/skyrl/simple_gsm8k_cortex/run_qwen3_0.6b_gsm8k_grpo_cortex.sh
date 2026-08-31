#!/bin/bash
# GRPO training for Qwen3-0.6B on GSM8K with the Arctic RL Cortex backend.
# Cortex sub-jobs own training + sampling; the SkyRL driver is CPU-only.
#
# Pre-reqs (see README.md):
#   1. pip install arctic-platform[cortex]
#   2. SkyRL cloned at the pinned commit with SKYRL_HOME exported.
#   3. ARCTIC_CORTEX_* env vars set (see README.md step 2).
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
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

# Hyperparameters are the on-prem sibling's, verbatim
# (../simple_gsm8k/run_qwen3_0.6b_gsm8k_grpo_arl.sh), so the two recipes differ
# only in where the GPUs live. MICRO_BSZ is the exception and has to: SkyRL
# defaults it to 1, which cannot satisfy DeepSpeed's
# micro x accum x n_gpus == train_batch at NGPU_PER_NODE=4.
#
# SkyRL's larger published GSM8K scale (train_batch_size=1024) is not the
# default here and does not currently work; see README.md section 6.
TRAIN_BSZ="${TRAIN_BSZ:-32}"
MINI_BSZ="${MINI_BSZ:-4}"
N_SAMPLES="${N_SAMPLES:-4}"
MICRO_BSZ="${MICRO_BSZ:-4}"
PROMPT_LEN="${PROMPT_LEN:-512}"
RESPONSE_LEN="${RESPONSE_LEN:-1024}"
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"

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

# SkyRL's validate_generator_cfg asserts num_engines == len(remote_urls). The
# URL itself is a placeholder -- the real endpoint lives inside the Cortex shim
# -- but the count has to track NUM_ENGINES or the driver dies before launch.
ENGINE_URLS="$(printf 'http://cortex-managed,%.0s' $(seq "${NUM_ENGINES}"))"
ENGINE_URLS="[${ENGINE_URLS%,}]"

LOGGER="${LOGGER:-console}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_SHORT="$(basename "${MODEL}")"
EXPERIMENT_NAME="gsm8k_grpo_${MODEL_SHORT}_cortex"

# SkyRL's GSM8kEnv reads reward_spec + env_class; verl's rl_dataset reads
# reward_model. A verl-shaped parquet therefore scores every rollout 0.0 here
# rather than erroring, so keep this path distinct from any ~/data/gsm8k built
# for verl.
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
import sys

try:
    import pandas as pd
except ModuleNotFoundError:
    # Bare \`python\` is also what launches the run below, so a shell that can't
    # import pandas can't train either. Say that instead of raising here.
    print("ERROR: this shell's \`python\` cannot import pandas, so it is not the")
    print("       environment arctic-platform was installed into. Activate it;")
    print("       see step 1 in README.md.")
    sys.exit(1)

cols = set(pd.read_parquet("${TRAIN_FILES}").columns)
missing = {"reward_spec", "env_class"} - cols
if missing:
    print(f"ERROR: ${TRAIN_FILES} is missing SkyRL schema fields {sorted(missing)}.")
    print("       Looks like a verl-shaped parquet. Rebuild with recipes/rl/skyrl/simple_gsm8k/download_data.py.")
    sys.exit(1)
PY

CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

# ----- Stopping this run, and getting the GPUs back -----
# Nothing releases the GPUs on its own. SkyRL never calls the client's shutdown
# when training ends, so finishing the last step holds all 8 exactly as firmly
# as being killed does, and the next launch then dies on the per-account cap
# with a 429 that never mentions your own previous run.
#
# The driver also ignores SIGINT and SIGTERM -- Ray installs handlers -- so
# there is no graceful stop to ask for: Ctrl-C alone leaves it training. Hence
# the shape below. The driver runs in the background so a signal reaches this
# shell immediately (bash defers a trap until the running foreground command
# returns, which for a 2-hour run means never), and stopping means SIGKILL.
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
    python - "${mode}" "$@" <<'PY'
import sys

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client.config import CortexConfig
from arctic_platform.client.transports.cortex import CortexTransport

# Anything not in here still holds its GPUs.
TERMINAL = {"failed", "done", "cancelled", "canceled", "succeeded", "terminated"}
mode, wanted = sys.argv[1], set(sys.argv[2:])

# model_name is required by the config but unused for job queries: this never
# provisions anything, it only reads and cancels.
transport = CortexTransport(ArcticClientConfig(model_name="unused", backend=CortexConfig()))
live = [
    j.get("job_id") or j.get("id") or j.get("name")
    for j in transport.list_jobs()
    if str(j.get("status") or j.get("state") or "?").lower() not in TERMINAL
]

if mode == "list":
    print("\n".join(str(j) for j in live))
else:
    for job_id in (j for j in live if j in wanted):
        transport.cancel_job(job_id)
        print(f"released {job_id}", file=sys.stderr)
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

# Upstream's own entrypoint is `integrations.arctic_rl.entrypoint`; naming ours
# instead is what selects Cortex, and it installs the driver-side
# peer_access_supported shim before SkyRL's Ray probe runs.
python -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=arctic_platform.integrations.skyrl.entrypoint \
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
    "$@" > >(tee "${CKPT_DIR}/${EXPERIMENT_NAME}.log") 2>&1 &

# Process substitution rather than a pipe so this is the driver's own pid and
# not tee's -- stop_driver has to kill the process that holds the GPUs.
DRIVER_PID=$!

# `wait` is interruptible, so INT/TERM run their trap here immediately. Guard
# it: under `set -e` the 128+signal status would otherwise exit the script
# before the trap gets to release anything.
wait "${DRIVER_PID}" || DRIVER_STATUS=$?
exit "${DRIVER_STATUS:-0}"
