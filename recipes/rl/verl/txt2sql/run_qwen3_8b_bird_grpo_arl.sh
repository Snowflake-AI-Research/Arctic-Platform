#!/bin/bash
# VeRL + Arctic RL on-prem: Qwen3-8B GRPO on BIRD-SQL, one arm per invocation.
# The VeRL counterpart of recipes/rl/skyrl/txt2sql/run_qwen3_1.7b_bird_grpo_arl.sh,
# with the same shape as the SkyRL 8B A/B:
#
#   Qwen3-8B, 128 prompts x 16 samples (2048 sequences/step), prompt 36864,
#   response 4096, 10 steps, 4 training + 4 sampling GPUs (disaggregated,
#   colocate=False), ZeRO-2, no offload, Liger, FA2, temperature 1, no KL,
#   LR 1e-6, betas (0.9, 0.999), no warmup, grad clip 1.0, loss verl_grpo.
#
# Not matched: SkyRL also sets vLLM max_num_batched_tokens=40960 and prefix
# caching. The VeRL adapter does not forward either to the Arctic vLLM
# engines, so sampling runs on vLLM defaults.
#
# Both arms use the Arctic backend; only remote_backend.train.zorro_train.enable
# changes. run_qwen3_32b_bird_grpo_baseline.sh is stock verl (FSDP2 + vLLM),
# not a no-ZoRRo Arctic arm.
#
# Arm knobs (env):
#   USE_ZORRO       0|1. remote_backend.train.zorro_train.enable.
#   MICRO_BSZ       actor.ppo_micro_batch_size_per_gpu, which the adapter sends as
#                   DeepSpeed train_micro_batch_size_per_gpu. Defaults 1 without
#                   ZoRRo (4 ran out of memory on SkyRL at this shape) and 16
#                   with it. ZoRRo needs a multiple of N_SAMPLES.
#   BALANCE_BATCH   true|false. remote_backend.train.balance_batch: Arctic rank
#                   assignment (sample length, or whole prompt groups under
#                   ZoRRo). false keeps arrival order. verl's own
#                   trainer.balance_batch stays False either way: with the single
#                   forwarder it only sorts the batch on the driver.
#   PACK            true|false. remote_backend.train.pack, sent as
#                   ds_worker_config.pack. On-prem always runs unpacked, so this
#                   only changes the job config.
#
# Other env: MODEL, DATA_DIR, TRAIN_BSZ, MINI_BSZ, N_SAMPLES, PROMPT_LEN,
# RESPONSE_LEN, MAX_STEPS, LR, TRAINING_GPUS, SAMPLING_GPUS, TP_SIZE, ZERO_STAGE,
# USE_LIGER, ATTN_IMPL, LOGITS_OPT, MAX_TOKEN_LEN, VERL_HOME, PYBIN, CKPT_DIR,
# LOGGER. Extra Hydra overrides pass through "$@".
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AP_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
VERL_HOME="${VERL_HOME:-/code/users/truwase/projects/cortex_training/verl-verl-parity}"
if [[ ! -d "${VERL_HOME}/verl/remote_backend" ]]; then
    echo "ERROR: VERL_HOME=${VERL_HOME} has no verl/remote_backend (needs the plugin-hook verl, 5a6491cf or later)."
    exit 1
fi
export PYTHONPATH="${AP_ROOT}:${VERL_HOME}:${PYTHONPATH:-}"
PYBIN="${PYBIN:-python}"

export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register
ARCTIC_VERL_CONFIG_DIR="${AP_ROOT}/arctic_platform/integrations/verl/config"

unset PYTORCH_CUDA_ALLOC_CONF
unset ARCTIC_INFERENCE_ENABLED
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO

_bool() {
    case "${1,,}" in
        1|true|yes) echo True ;;
        0|false|no) echo False ;;
        *) echo "ERROR: expected a boolean, got '$1'" >&2; return 1 ;;
    esac
}

USE_ZORRO="$(_bool "${USE_ZORRO:-0}")"
BALANCE_BATCH="$(_bool "${BALANCE_BATCH:-true}")"
PACK="$(_bool "${PACK:-false}")"
USE_LIGER="$(_bool "${USE_LIGER:-true}")"

TRAINING_GPUS="${TRAINING_GPUS:-4}"
SAMPLING_GPUS="${SAMPLING_GPUS:-4}"
TP_SIZE="${TP_SIZE:-1}"
ZERO_STAGE="${ZERO_STAGE:-2}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

TRAIN_BSZ="${TRAIN_BSZ:-128}"
MINI_BSZ="${MINI_BSZ:-128}"
N_SAMPLES="${N_SAMPLES:-16}"
PROMPT_LEN="${PROMPT_LEN:-36864}"
RESPONSE_LEN="${RESPONSE_LEN:-4096}"
MAX_STEPS="${MAX_STEPS:-10}"
LR="${LR:-1e-6}"
LOGGER="${LOGGER:-console}"

# ZoRRo needs every rank's micro-batch to hold whole prompt groups, and a
# per-GPU token cap of prompt + n * response for one group. logits=memory
# is ZoRRo-only; the non-ZoRRo pipeline rejects it.
if [[ "${USE_ZORRO}" == "True" ]]; then
    ARM="zorro"
    MICRO_BSZ="${MICRO_BSZ:-16}"
    MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-$(( PROMPT_LEN + N_SAMPLES * RESPONSE_LEN ))}"
    LOGITS_OPT="${LOGITS_OPT:-memory}"
    if (( MICRO_BSZ % N_SAMPLES != 0 )); then
        echo "ERROR: ZoRRo MICRO_BSZ=${MICRO_BSZ} must be a multiple of N_SAMPLES=${N_SAMPLES}."
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
    echo "ERROR: ${_STEP_ROWS} rows/step not divisible by MICRO_BSZ=${MICRO_BSZ} x TRAINING_GPUS=${TRAINING_GPUS}."
    exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_SHORT="$(basename "${MODEL}")"
DATA_DIR="${DATA_DIR:-/code/users/truwase/data/open-source-text2sql}"
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
    echo "ERROR: BIRD-SQL parquets not found at ${DATA_DIR}/{train,val}.parquet"
    exit 1
fi

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
EXPERIMENT_NAME="verl_bird_${MODEL_SHORT}_${ARM}_mb${MICRO_BSZ}_bal${BALANCE_BATCH}_pack${PACK}_${RUN_TS}"
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"
LOG_FILE="${CKPT_DIR}/${EXPERIMENT_NAME}.log"

# The plugin must come from AP_ROOT (verl_arl was built against another
# Arctic Platform tree) and must carry the micro-batch / pack / balance checks.
"${PYBIN}" - "${AP_ROOT}" "${VERL_HOME}" <<'PY'
import os
import sys

import arctic_platform
import verl
from arctic_platform.integrations.verl.adapter import ArcticRLClientWrapper

ap_root, verl_home = (os.path.realpath(p) for p in sys.argv[1:3])
ap_file = os.path.realpath(arctic_platform.__file__)
verl_file = os.path.realpath(verl.__file__)
print("arctic_platform", ap_file)
print("verl", verl_file)
if not ap_file.startswith(ap_root + os.sep):
    raise SystemExit(f"arctic_platform resolves outside {ap_root}")
if not verl_file.startswith(verl_home + os.sep):
    raise SystemExit(f"verl resolves outside {verl_home}")
if not hasattr(ArcticRLClientWrapper, "_check_train_options"):
    raise SystemExit("verl adapter predates the micro-batch / pack / balance checks")
PY

echo "== VeRL BIRD ${ARM}: zorro=${USE_ZORRO} micro=${MICRO_BSZ} balance=${BALANCE_BATCH} pack=${PACK} train=${TRAINING_GPUS} sample=${SAMPLING_GPUS} prompts=${TRAIN_BSZ} n=${N_SAMPLES} steps=${MAX_STEPS} zero=${ZERO_STAGE} logits=${LOGITS_OPT} token_cap=${MAX_TOKEN_LEN} =="

"${PYBIN}" -m verl.trainer.main_ppo \
    hydra.searchpath="[file://${ARCTIC_VERL_CONFIG_DIR}]" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size="${TRAIN_BSZ}" \
    data.max_prompt_length="${PROMPT_LEN}" \
    data.max_response_length="${RESPONSE_LEN}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation=left \
    data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.model.path="${MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger="${USE_LIGER}" \
    +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPL}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BSZ}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BSZ}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BSZ}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]' \
    actor_rollout_ref.rollout.name=arctic \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n="${N_SAMPLES}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.nccl_timeout=1800 \
    trainer.use_legacy_worker_impl=disable \
    trainer.remote_backend=arctic \
    remote_backend=arctic \
    trainer.balance_batch=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${MAX_STEPS}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.logger="['${LOGGER}']" \
    trainer.project_name=verl_arctic_rl_bird \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}/ckpt" \
    custom_reward_function.path="${SCRIPT_DIR}/bird_reward.py" \
    custom_reward_function.name=compute_score \
    remote_backend.colocate=False \
    remote_backend.training_gpus="${TRAINING_GPUS}" \
    remote_backend.sampling_gpus="${SAMPLING_GPUS}" \
    remote_backend.sampling_tp_size="${TP_SIZE}" \
    remote_backend.log_prob_gpus=0 \
    remote_backend.weight_sync.cuda_ipc=False \
    remote_backend.weight_sync.low_memory=False \
    remote_backend.train.balance_batch="${BALANCE_BATCH}" \
    remote_backend.train.pack="${PACK}" \
    remote_backend.train.zorro_train.enable="${USE_ZORRO}" \
    remote_backend.train.zorro_train.max_rollouts="${N_SAMPLES}" \
    remote_backend.train.logits.optimization="${LOGITS_OPT}" \
    remote_backend.train.deepspeed.zero_optimization.stage="${ZERO_STAGE}" \
    remote_backend.train.deepspeed.zero_optimization.offload_optimizer.device=none \
    remote_backend.train.deepspeed.zero_optimization.offload_param.device=none \
    remote_backend.rollout.zorro_inference.enable=False \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
