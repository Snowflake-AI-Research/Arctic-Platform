#!/usr/bin/env bash
# On-prem VeRL Qwen3-8B BIRD arms, run in sequence on one 8-GPU node with
# run_qwen3_8b_bird_grpo_arl.sh. Shape matches the SkyRL 8B A/B: 128 x 16,
# prompt 36864, response 4096, 10 steps, 4 train + 4 sample, ZeRO-2, Liger, FA2.
#
#   arm                     USE_ZORRO  MICRO_BSZ  BALANCE_BATCH  PACK
#   verl-8b-nozorro-bal     0          1          true           false
#   verl-8b-nozorro-arrival 0          1          false          false
#   verl-8b-zorro-bal       1          16         true           false
#   verl-8b-zorro-arrival   1          16         false          false
#
# No-ZoRRo micro-batch 4 ran out of memory on SkyRL at this shape; 1 gives
# 1 x 512 accumulation x 4 GPUs = 2048. NOZORRO_MICRO_BSZ / ZORRO_MICRO_BSZ
# override the two columns.
#
# PACK stays false: on-prem always runs unpacked, so packed vs unpacked is not
# an on-prem axis. ARMS selects a subset by name (space separated).
# Each arm gets its own RAY_TMPDIR (short, under /tmp, for the Unix socket path
# limit) with RAY_ADDRESS unset, so it starts a fresh Ray and cannot attach to
# another run's cluster. Nothing here stops Ray processes it did not start.
# Writes logs under OUT and never removes anything.
set -uo pipefail

NEUTRINO="/code/users/truwase/projects/neutrino"
cd "${NEUTRINO}"

CONDA_BASE="${CONDA_BASE:-/home/yak/miniconda3}"
export LD_LIBRARY_PATH=""
export PATH="${CONDA_BASE}/envs/verl_arl/bin:${CONDA_BASE}/bin:${PATH}"
export WANDB_MODE=disabled
hash -r

export PYBIN="${CONDA_BASE}/envs/verl_arl/bin/python"
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
export ZERO_STAGE=2
export USE_LIGER=true
export ATTN_IMPL=flash_attention_2

RECIPE="${NEUTRINO}/ap-161-on-279/recipes/rl/verl/txt2sql/run_qwen3_8b_bird_grpo_arl.sh"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/data-fast/neutrino/onprem/smoke-logs/verl-${STAMP}}"
mkdir -p "${OUT}"
ARMS="${ARMS:-verl-8b-nozorro-bal verl-8b-nozorro-arrival verl-8b-zorro-bal verl-8b-zorro-arrival}"
NOZORRO_MICRO_BSZ="${NOZORRO_MICRO_BSZ:-1}"
ZORRO_MICRO_BSZ="${ZORRO_MICRO_BSZ:-16}"
unset RAY_ADDRESS

echo "== onprem verl bird 8b 128x16 ${STAMP} on $(hostname) =="
echo "logs ${OUT}"
echo "arms ${ARMS}"
echo "ap $(git -C "${NEUTRINO}/ap-161-on-279" rev-parse HEAD) verl $(git -C "${VERL_HOME}" rev-parse HEAD)"
date -Iseconds
nvidia-smi -L || true
"${PYBIN}" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)'

fail=0
run_arm() {
  local name="$1" zorro="$2" micro="$3" balance="$4" pack="$5"
  case " ${ARMS} " in *" ${name} "*) ;; *) echo "== SKIP ${name} =="; return 0 ;; esac
  echo "== START ${name} zorro=${zorro} micro=${micro} balance=${balance} pack=${pack} =="
  date -Iseconds
  local ray_dir
  ray_dir="$(mktemp -d "/tmp/rv-${STAMP}-XXXX")"
  echo "ray temp dir ${ray_dir}"
  env CKPT_DIR="${OUT}/${name}" USE_ZORRO="${zorro}" MICRO_BSZ="${micro}" \
    BALANCE_BATCH="${balance}" PACK="${pack}" RAY_TMPDIR="${ray_dir}" \
    bash "${RECIPE}"
  local rc=$?
  echo "== END ${name} rc=${rc} =="
  date -Iseconds
  if [[ "${rc}" -ne 0 ]]; then
    fail=1
  fi
}

run_arm verl-8b-nozorro-bal     0 "${NOZORRO_MICRO_BSZ}" true  false
run_arm verl-8b-nozorro-arrival 0 "${NOZORRO_MICRO_BSZ}" false false
run_arm verl-8b-zorro-bal       1 "${ZORRO_MICRO_BSZ}"   true  false
run_arm verl-8b-zorro-arrival   1 "${ZORRO_MICRO_BSZ}"   false false

echo "== onprem verl bird 8b 128x16 done fail=${fail} =="
date -Iseconds
exit "${fail}"
