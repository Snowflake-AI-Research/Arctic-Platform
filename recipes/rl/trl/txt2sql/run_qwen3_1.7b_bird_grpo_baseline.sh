#!/usr/bin/env bash
# Native TRL async-GRPO BIRD baseline launcher: external `vllm serve` + accelerate-DDP trainer.
#
# TRL+Arctic must run disaggregated, so the baseline is disaggregated too: trainer DDP on
# GPUs 0-3 and vLLM on GPUs 4-7. Default engine is verl txt2sql knobs at TP=1 (DP fills remaining
# sampling GPUs). This script owns the vllm-serve lifecycle.
#
# Env overrides: MODEL, BIRD_TRAIN_PARQUET, MAX_STEPS, NUM_PROMPTS, NUM_GEN, PER_DEVICE_BSZ, GRAD_ACCUM,
# MAX_COMPLETION_LEN, MAX_MODEL_LEN, LR, SEED, METRICS_OUT, PORT, TRAINER_GPUS, SERVER_GPUS,
# VAL_EVERY (0=off), BIRD_VAL_PARQUET, VAL_MAX_SAMPLES,
# VLLM_TP (default 1), GPU_MEM_UTIL, VLLM_MAX_NUM_SEQS, VLLM_MAX_BATCHED_TOKENS,
# VLLM_EXTRA_ARGS (space-separated extra `vllm serve` flags; empty default).
# W&B: REPORT_TO (default wandb; set none to disable), WANDB_PROJECT, WANDB_RUN_NAME, WANDB_RUN_GROUP.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-arctic-trl-bird}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-0}"
unset WANDB_MODE
unset WANDB_SILENT

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"
HOST="127.0.0.1"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"
SEED="${SEED:-42}"
TRAINER_GPUS="${TRAINER_GPUS:-0,1,2,3}"
SERVER_GPUS="${SERVER_GPUS:-4,5,6,7}"
NUM_TRAINERS="$(awk -F, '{print NF}' <<<"${TRAINER_GPUS}")"
N_SERVER="$(awk -F, '{print NF}' <<<"${SERVER_GPUS}")"
# verl txt2sql uses TP=2 (baseline) / sampling_tp=4 (Arctic). C1 matches those engine knobs at TP=1;
# leftover sampling GPUs become vLLM data-parallel replicas (4 GPUs × TP=1 → DP=4).
SERVER_TP="${VLLM_TP:-1}"
SERVER_DP=$((N_SERVER / SERVER_TP))
if [ $((SERVER_TP * SERVER_DP)) -ne "${N_SERVER}" ]; then
  echo "VLLM_TP=${SERVER_TP} does not divide ${N_SERVER} sampling GPUs"; exit 2
fi
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-65536}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
METRICS_OUT="${METRICS_OUT:-${HERE}/metrics_bird_baseline.json}"
SERVER_LOG="${HERE}/_server_bird_baseline.log"
BASE_URL="http://${HOST}:${PORT}"

# Spawned rollout child (AsyncRolloutWorker) pickles reward_funcs by module path -> bird_task must be importable.
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"

echo "== BIRD baseline: vllm serve (TP=${SERVER_TP} DP=${SERVER_DP}) on GPUs ${SERVER_GPUS} + accelerate DDP (${NUM_TRAINERS}) on GPUs ${TRAINER_GPUS} =="
echo "[base] GPU_MEM_UTIL=${GPU_MEM_UTIL} max_num_seqs=${VLLM_MAX_NUM_SEQS} max_batched_tokens=${VLLM_MAX_BATCHED_TOKENS} extra_args=${VLLM_EXTRA_ARGS:-<none>}"
echo "[base] wandb REPORT_TO=${REPORT_TO} project=${WANDB_PROJECT} run_name=${WANDB_RUN_NAME:-<auto>}"
hostname; date -Iseconds

# Kill ONLY the given pid's subtree (never a process group): the autorun runner is this script's process-group
# leader, so a `kill -- -<pgid>` here would take the runner down with us. Recurse over child pids instead.
kill_tree() {
  local pid="$1" sig="$2" c
  for c in $(pgrep -P "${pid}" 2>/dev/null); do kill_tree "${c}" "${sig}"; done
  kill "-${sig}" "${pid}" 2>/dev/null || true
}

# ---- launch external vllm serve on the server GPUs (plain background child; kill by pid subtree on exit) ----
# verl txt2sql: gpu_memory_utilization=0.7, max_num_seqs=256, max_num_batched_tokens=40960, enforce_eager=False.
# --generation-config vllm: do not inherit Qwen3's generation_config.json (temp 0.6 / top_p 0.95).
CUDA_VISIBLE_DEVICES="${SERVER_GPUS}" VLLM_SERVER_DEV_MODE=1 \
  vllm serve "${MODEL}" \
    --host "${HOST}" --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${SERVER_TP}" \
    --data-parallel-size "${SERVER_DP}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${VLLM_MAX_BATCHED_TOKENS}" \
    --enable-prefix-caching \
    --generation-config vllm \
    --seed "${SEED}" \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}' \
    ${VLLM_EXTRA_ARGS} \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[base] vllm serve pid=${SERVER_PID}; log -> ${SERVER_LOG}"

cleanup() {
  echo "[base] terminating vllm serve pid=${SERVER_PID} (subtree) ..."
  kill_tree "${SERVER_PID}" TERM
  for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
  kill_tree "${SERVER_PID}" KILL
  echo "[base] ---- vllm serve log tail ----"; tail -40 "${SERVER_LOG}" 2>/dev/null; echo "[base] ---- end ----"
}
trap cleanup EXIT

# ---- wait for server health ----
echo "[base] waiting for ${BASE_URL}/health ..."
HEALTHY=0
for _ in $(seq 1 900); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then echo "[base] vllm serve exited early"; exit 1; fi
  if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  sleep 2
done
[ "${HEALTHY}" = "1" ] || { echo "[base] server never healthy"; exit 1; }
echo "[base] vllm server healthy"; date -Iseconds

# ---- accelerate-DDP trainer on the trainer GPUs (connects to the external server) ----
# Run from this recipe dir: torchrun puts the launch cwd on sys.path[0]. A checkout that also
# contains a top-level `trl/` source tree would shadow the editable-installed `trl` package
# (ImportError: __version__). This folder has no `trl/`, so `import trl` resolves to the
# installed package; bird_task stays importable via cwd.
cd "${HERE}"
CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}" accelerate launch \
  --num_processes "${NUM_TRAINERS}" --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no \
  run_qwen3_1.7b_bird_grpo_baseline.py \
    --model "${MODEL}" \
    --vllm-base-url "${BASE_URL}" \
    --max-steps "${MAX_STEPS:-30}" \
    --num-generations "${NUM_GEN:-16}" \
    --per-device-bsz "${PER_DEVICE_BSZ:-2}" \
    --grad-accum "${GRAD_ACCUM:-32}" \
    --max-completion-length "${MAX_COMPLETION_LEN:-4096}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --val-every "${VAL_EVERY:-0}" \
    --val-parquet "${BIRD_VAL_PARQUET:-}" \
    --val-max-samples "${VAL_MAX_SAMPLES:-0}" \
    --num-prompts "${NUM_PROMPTS:-128}" \
    --learning-rate "${LR:-1e-6}" \
    --seed "${SEED}" \
    --metrics-out "${METRICS_OUT}"
rc=$?
echo "[base] accelerate launch rc=${rc}"; date -Iseconds
exit ${rc}
