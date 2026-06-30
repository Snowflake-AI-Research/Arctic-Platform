#!/bin/bash
# Single-node 8-GPU GRPO training for Qwen3-8B on BIRD-SQL with Arctic RL + ZoRRo.
# Pure GRPO, no frozen reference model (use_kl_loss=False).
#
# This is the single-node iteration target for the 4-node Qwen3-32B run that
# produced the 2x speedup behind the Arctic RL launch blog. Same Arctic RL
# stack (FCA, CUDA-IPC weight sync, ZoRRo, Liger, FA3 trainer / FLASH_ATTN
# inference), just scaled down to one node so it can run on a standalone host.
#
# Prerequisites (see README.md):
#   1. Conda env with pinned deps (`uv pip install -r requirements.txt
#      --override overrides.txt`). No SkyRL checkout needed.
#   2. Raw BIRD-SQL staged at $BIRD_RAW (see README), then
#      `python download_data.py --bird_dir $BIRD_RAW --output_dir $DATA_DIR`
#      to produce $DATA_DIR/{train,val}.parquet.

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_LIB_DIR="$(cd "${SCRIPT_DIR}/../_lib" && pwd)"
export PYTHONPATH="${SKYRL_LIB_DIR}:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export ARCTIC_CUDA_IPC_LOW_MEM=0
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on single-node TP>1:
# it triggers torch's "Expandable segments are not compatible with memory pool"
# assertion in vLLM's Ray-executor TP workers (pytorch/pytorch#147851). The
# multi-node 32B launcher gets away with it because its placement group layout
# is different — but for this single-node setup we leave allocator defaults.

# ----- Single-node 8-GPU Arctic / ZoRRo topology -----
# `trainer.arctic_rl.colocate=true` keeps Arctic RL's training + sampling on the
# same 8 GPUs. `trainer.placement.colocate_all=false` is required so SkyRL does
# not also try to claim a placement group for its own inference engines (Arctic
# RL already owns the GPUs).
NUM_NODES=1
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# TP=2 -> 4 inference engines, same multi-rank FlashInfer code path the
# multi-node 8B / 32B runs exercise.
TP_SIZE=2
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

# FA3 is the default on Hopper; flip to flash_attention_2 for A100/L40S.
ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
ARCTIC_ZERO_STAGE=3
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-false}"   # 8B easily fits in 8xH200 HBM

# Smaller batch / context than the multi-node 8B recipe so a single node has
# room to breathe — easy to override on the command line for stress runs.
TRAIN_BSZ="${TRAIN_BSZ:-32}"
MINI_BSZ="${MINI_BSZ:-16}"
N_SAMPLES="${N_SAMPLES:-8}"
PROMPT_LEN="${PROMPT_LEN:-8192}"
RESPONSE_LEN="${RESPONSE_LEN:-2048}"

LR="${LR:-2e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-false}"

LOGGER="${LOGGER:-console}"
WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_SHORT="$(basename "${MODEL}")"

DATA_DIR="${DATA_DIR:-${HOME}/data/bird}"
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
    echo "ERROR: BIRD-SQL parquets not found at ${DATA_DIR}/{train,val}.parquet"
    echo "       Run 'python download_data.py --bird_dir <raw> --output_dir ${DATA_DIR}' first."
    exit 1
fi

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME="bird_grpo_${MODEL_SHORT}_arl_z${ARCTIC_ZERO_STAGE}_${RUN_TS}"
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

# Optional: arctic speculative decoding (drop in a Qwen3-8B-trained 3-head
# checkpoint via SPEC_MODEL=<path>). Off by default — the 32B blog-run head is
# tied to Qwen3-32B's hidden size and won't load on 8B.
SPEC_MODEL="${SPEC_MODEL:-}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
SPEC_OVERRIDE=()
if [[ -n "${SPEC_MODEL}" ]]; then
    SPEC_OVERRIDE+=("trainer.arctic_rl.speculative_model=${SPEC_MODEL}")
    SPEC_OVERRIDE+=("trainer.arctic_rl.num_speculative_tokens=${NUM_SPEC_TOKENS}")
fi

python -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=arctic_rl.entrypoint \
    trainer.arctic_rl.colocate=true \
    trainer.arctic_rl.zero_stage=${ARCTIC_ZERO_STAGE} \
    trainer.arctic_rl.offload_optimizer=${OFFLOAD_OPTIMIZER} \
    trainer.arctic_rl.attn_implementation=${ATTN_IMPL} \
    trainer.arctic_rl.cuda_ipc_weight_sync=true \
    trainer.arctic_rl.low_memory_weight_sync=true \
    trainer.arctic_rl.lr_warmup_ratio=0.05 \
    'trainer.arctic_rl.optimizer_betas=[0.9,0.95]' \
    trainer.arctic_rl.vllm_enforce_eager=false \
    trainer.arctic_rl.vllm_max_num_batched_tokens=20480 \
    trainer.arctic_rl.server_logs=true \
    trainer.arctic_rl.startup_timeout=1800 \
    "${SPEC_OVERRIDE[@]}" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path="${MODEL}" \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/val.parquet']" \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_nodes=${NUM_NODES} \
    trainer.placement.policy_num_gpus_per_node=${GPUS_PER_NODE} \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.num_engines=${NUM_ENGINES} \
    generator.inference_engine.tensor_parallel_size=${TP_SIZE} \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.gpu_memory_utilization=0.5 \
    generator.batched=true \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    environment.env_class=bird \
    trainer.epochs=${TOTAL_EPOCHS} \
    trainer.train_batch_size=${TRAIN_BSZ} \
    trainer.policy_mini_batch_size=${MINI_BSZ} \
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
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CKPT_DIR}/logs" \
    trainer.ckpt_path="${CKPT_DIR}/ckpt" \
    trainer.ckpt_interval=-1 \
    "$@" 2>&1 | tee "${CKPT_DIR}/${EXPERIMENT_NAME}.log"
