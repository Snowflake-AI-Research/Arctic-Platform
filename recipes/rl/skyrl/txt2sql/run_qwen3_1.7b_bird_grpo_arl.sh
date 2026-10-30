#!/bin/bash
# SkyRL port of the TRL 1.7B BIRD pair, for a ZoRRo step-time comparison.
#
#   C2  this file, USE_ZORRO=0
#       TRL: recipes/rl/trl/txt2sql/run_qwen3_1.7b_bird_grpo_arl.sh
#   C3  run_qwen3_1.7b_bird_grpo_arl_zorro.sh (sets USE_ZORRO=1)
#       TRL: run_qwen3_1.7b_bird_grpo_arl_zorro.sh
#
# Kept from TRL, both arms: Qwen3-1.7B, 4 train + 4 sample, disaggregated,
# 16 samples/prompt, prompt 36864, response 4096, 8 prompts/step (128
# sequences), ZeRO-2, no offload, Liger, FA2, logits_optimization=memory,
# prefix caching, temperature 1, no KL, 30 steps.
# Both arms are unpacked (pack=false; ZoRRo is unpacked either way) and both
# balance ranks. No ZoRRo balances by sample length. ZoRRo balances by
# prompt group.
# DeepSpeed uses trainer.micro_train_batch_size_per_gpu. The Qwen3-8B
# figure is micro-batch 4 without ZoRRo and 16 with ZoRRo. MICRO_BSZ
# overrides that. ZoRRo still requires a multiple of N_SAMPLES.
#
# ZoRRo's token cap is prompt + n_samples * response = 102400, the value
# TRL raises to at runtime when ZoRRo balancing is on. The non-ZoRRo
# micro-batch is one sequence, whose cap is prompt + response = 40960.
#
# Env overrides: MODEL, DATA_DIR, TRAIN_BSZ, MINI_BSZ, N_SAMPLES, MICRO_BSZ,
# PROMPT_LEN, RESPONSE_LEN, MAX_STEPS, LR, TRAINING_GPUS, SAMPLING_GPUS,
# USE_ZORRO, LOGGER.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
    exit 1
fi
export PYTHONPATH="${SKYRL_HOME}:${SCRIPT_DIR}:${PYTHONPATH:-}"

PYBIN="${PYBIN:-python}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export ARCTIC_CUDA_IPC_LOW_MEM=0
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0

USE_ZORRO="${USE_ZORRO:-0}"
if [[ "${USE_ZORRO}" == "1" || "${USE_ZORRO}" == "true" ]]; then
    USE_ZORRO_FLAG=true
    ARM="c3"
else
    USE_ZORRO_FLAG=false
    ARM="c2"
fi

NUM_NODES=1
TRAINING_GPUS="${TRAINING_GPUS:-4}"
SAMPLING_GPUS="${SAMPLING_GPUS:-4}"
TP_SIZE="${TP_SIZE:-1}"
NUM_ENGINES="${NUM_ENGINES:-${SAMPLING_GPUS}}"

ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

TRAIN_BSZ="${TRAIN_BSZ:-8}"
MINI_BSZ="${MINI_BSZ:-8}"
N_SAMPLES="${N_SAMPLES:-16}"
# Figure: micro-batch 4 without ZoRRo, 16 with ZoRRo.
if [[ "${USE_ZORRO_FLAG}" == "true" ]]; then
    MICRO_BSZ="${MICRO_BSZ:-16}"
else
    MICRO_BSZ="${MICRO_BSZ:-4}"
fi
PROMPT_LEN="${PROMPT_LEN:-36864}"
RESPONSE_LEN="${RESPONSE_LEN:-4096}"
MAX_STEPS="${MAX_STEPS:-30}"

LR="${LR:-1e-6}"
LOGGER="${LOGGER:-console}"
WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl}"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MODEL_SHORT="$(basename "${MODEL}")"

DATA_DIR="${DATA_DIR:-${HOME}/data/bird}"
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
    echo "ERROR: BIRD-SQL parquets not found at ${DATA_DIR}/{train,val}.parquet"
    echo "       Run 'python download_data.py --bird_dir <raw> --output_dir ${DATA_DIR}' first."
    exit 1
fi

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME="bird_grpo_${MODEL_SHORT}_${ARM}_${RUN_TS}"
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

echo "== BIRD SkyRL ${ARM}: zorro=${USE_ZORRO_FLAG} train=${TRAINING_GPUS} sample=${SAMPLING_GPUS} prompts=${TRAIN_BSZ} n=${N_SAMPLES} micro=${MICRO_BSZ} steps=${MAX_STEPS} =="

cd "${SKYRL_HOME}"

"${PYBIN}" -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=arctic_rl.entrypoint \
    trainer.arctic_rl.colocate=false \
    trainer.arctic_rl.zero_stage=2 \
    trainer.arctic_rl.offload_optimizer=false \
    trainer.arctic_rl.offload_param=false \
    trainer.arctic_rl.log_prob_gpus=0 \
    trainer.arctic_rl.use_zorro=${USE_ZORRO_FLAG} \
    trainer.arctic_rl.pack=false \
    trainer.arctic_rl.balance_batch=true \
    trainer.arctic_rl.use_liger=true \
    trainer.arctic_rl.attn_implementation=${ATTN_IMPL} \
    trainer.arctic_rl.enable_gradient_checkpointing=true \
    trainer.arctic_rl.ulysses_sequence_parallel_size=1 \
    trainer.arctic_rl.logits_optimization=memory \
    trainer.arctic_rl.cuda_ipc_weight_sync=false \
    trainer.arctic_rl.low_memory_weight_sync=false \
    trainer.arctic_rl.lr_warmup_ratio=0.0 \
    'trainer.arctic_rl.optimizer_betas=[0.9,0.999]' \
    trainer.arctic_rl.vllm_enforce_eager=false \
    trainer.arctic_rl.vllm_enable_prefix_caching=true \
    trainer.arctic_rl.vllm_max_num_batched_tokens=40960 \
    trainer.arctic_rl.vllm_max_num_seqs=256 \
    trainer.arctic_rl.use_arctic_inference=false \
    trainer.arctic_rl.server_logs=true \
    trainer.arctic_rl.startup_timeout=1800 \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path="${MODEL}" \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/val.parquet']" \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_nodes=${NUM_NODES} \
    trainer.placement.policy_num_gpus_per_node=${TRAINING_GPUS} \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.num_engines=${NUM_ENGINES} \
    generator.inference_engine.tensor_parallel_size=${TP_SIZE} \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.gpu_memory_utilization=0.7 \
    generator.batched=true \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    environment.env_class=bird \
    trainer.epochs=1 \
    trainer.max_training_steps=${MAX_STEPS} \
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
    trainer.eval_before_train=false \
    trainer.eval_interval=1000 \
    trainer.update_epochs_per_batch=1 \
    trainer.policy.optimizer_config.lr=${LR} \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CKPT_DIR}/logs" \
    trainer.ckpt_path="${CKPT_DIR}/ckpt" \
    trainer.ckpt_interval=-1 \
    "$@" 2>&1 | tee "${CKPT_DIR}/${EXPERIMENT_NAME}.log"
