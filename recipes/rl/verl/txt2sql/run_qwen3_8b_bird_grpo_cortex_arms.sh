#!/usr/bin/env bash
# Cortex VeRL Qwen3-8B BIRD arms, run in sequence (8-GPU per-account cap) with
# run_qwen3_8b_bird_grpo_cortex.sh. Shape matches the SkyRL 8B Cortex A/B:
# 128 x 16, prompt 36864, response 4096, 10 steps, 4 train + 4 sample, ZeRO-2,
# Liger on, FA3, image dev_20261026_083954_a6ba7da36f6 (build 994). CPU-only driver.
#
#   arm                            USE_ZORRO  MICRO_BSZ  BALANCE_BATCH  PACK
#   verl-8b-cortex-nozorro-bal     0          1          true           false
#   verl-8b-cortex-nozorro-arrival 0          1          false          false
#   verl-8b-cortex-zorro-bal       1          16         true           false
#   verl-8b-cortex-zorro-arrival   1          16         false          false
#   verl-8b-cortex-nozorro-packed  0          1          true           true    (opt-in)
#
# The packed arm is not verl_grpo: a packed Cortex job rewrites the request to
# grpo with pi_old = the live policy, so it is only in ARMS when asked for.
# NOZORRO_MICRO_BSZ / ZORRO_MICRO_BSZ override the micro-batch columns.
# Writes logs under OUT and never removes anything.
set -uo pipefail

NEUTRINO="/code/users/truwase/projects/neutrino"
cd "${NEUTRINO}"

export PATH="/home/yak/miniconda3/envs/dev/bin:/home/yak/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH=""
export WANDB_MODE=disabled
export ARCTIC_CORTEX_IMAGE_TAG="dev_20261026_083954_a6ba7da36f6"
export UV_PYTHON="/home/yak/miniconda3/envs/dev/bin/python3.12"
export UV_CACHE_DIR=/data-fast/neutrino/uv-cache
export UV_NO_CACHE=1
export CORTEX_ALLOW_LARGE_PAYLOAD=1
SCRATCH="/data-fast/neutrino/rt"
mkdir -p "${SCRATCH}/tmp" /data-fast/neutrino/uv-cache
export TMPDIR="${SCRATCH}/tmp"

set +x
set -a
# shellcheck disable=SC1091
source "${NEUTRINO}/.cortex.env"
set +a
if [[ -z "${ARCTIC_CORTEX_PAT:-}" ]]; then
  echo "ARCTIC_CORTEX_PAT missing"
  exit 2
fi
echo "ARCTIC_CORTEX_PAT set"
echo "host ${ARCTIC_CORTEX_HOST:-missing} db ${ARCTIC_CORTEX_DATABASE:-missing} schema ${ARCTIC_CORTEX_SCHEMA:-missing}"

export VERL_HOME="${VERL_HOME:-/code/users/truwase/projects/cortex_training/verl-verl-parity}"
export DATA_DIR="${DATA_DIR:-/code/users/truwase/data/open-source-text2sql}"
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export LOGGER=console
export MAX_STEPS="${MAX_STEPS:-10}"
export TRAIN_BSZ=128
export MINI_BSZ=128
export N_SAMPLES=16
export PROMPT_LEN=36864
export RESPONSE_LEN=4096
export LR=1e-6
export TRAINING_GPUS=4
export SAMPLING_GPUS=4

RECIPE="${NEUTRINO}/ap-161-on-279/recipes/rl/verl/txt2sql/run_qwen3_8b_bird_grpo_cortex.sh"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/data-fast/neutrino/cortex/smoke-logs/verl-${STAMP}}"
mkdir -p "${OUT}"
ARMS="${ARMS:-verl-8b-cortex-nozorro-bal verl-8b-cortex-nozorro-arrival verl-8b-cortex-zorro-bal verl-8b-cortex-zorro-arrival}"
NOZORRO_MICRO_BSZ="${NOZORRO_MICRO_BSZ:-1}"
ZORRO_MICRO_BSZ="${ZORRO_MICRO_BSZ:-16}"

echo "== cortex verl bird 8b 128x16 ${STAMP} image ${ARCTIC_CORTEX_IMAGE_TAG} =="
echo "logs ${OUT}"
echo "arms ${ARMS}"
echo "ap $(git -C "${NEUTRINO}/ap-161-on-279" rev-parse HEAD) verl $(git -C "${VERL_HOME}" rev-parse HEAD)"
date -Iseconds

fail=0
run_arm() {
  local name="$1" zorro="$2" micro="$3" balance="$4" pack="$5"
  case " ${ARMS} " in *" ${name} "*) ;; *) echo "== SKIP ${name} =="; return 0 ;; esac
  echo "== START ${name} zorro=${zorro} micro=${micro} balance=${balance} pack=${pack} =="
  date -Iseconds
  env CKPT_DIR="${OUT}/${name}" USE_ZORRO="${zorro}" MICRO_BSZ="${micro}" \
    BALANCE_BATCH="${balance}" PACK="${pack}" \
    bash "${RECIPE}"
  local rc=$?
  echo "== END ${name} rc=${rc} =="
  date -Iseconds
  if [[ "${rc}" -ne 0 ]]; then
    fail=1
  fi
}

run_arm verl-8b-cortex-nozorro-bal     0 "${NOZORRO_MICRO_BSZ}" true  false
run_arm verl-8b-cortex-nozorro-arrival 0 "${NOZORRO_MICRO_BSZ}" false false
run_arm verl-8b-cortex-zorro-bal       1 "${ZORRO_MICRO_BSZ}"   true  false
run_arm verl-8b-cortex-zorro-arrival   1 "${ZORRO_MICRO_BSZ}"   false false
run_arm verl-8b-cortex-nozorro-packed  0 "${NOZORRO_MICRO_BSZ}" true  true

echo "== cortex verl bird 8b 128x16 done fail=${fail} =="
date -Iseconds
exit "${fail}"
