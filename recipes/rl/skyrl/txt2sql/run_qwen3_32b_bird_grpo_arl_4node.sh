#!/bin/bash
# Multi-node 4x8 GPU GRPO training for Qwen3-32B on BIRD-SQL with Arctic RL + ZoRRo.
#
# This is the run behind the ~2x speedup vs SkyRL FSDP-native baseline that ships
# in the Arctic RL launch blog. The single-node 8B sibling
# (`run_qwen3_8b_bird_grpo_arl.sh`) is the iteration target — same Arctic RL
# stack (FCA, CUDA-IPC weight sync, ZoRRo, Liger, FA3 trainer / FLASH_ATTN
# inference), scaled up to 32 H200s.
#
# Prerequisites (see README.md):
#   1. Conda env with pinned deps (`uv pip install -r requirements.txt
#      --override overrides.txt`) on EVERY node (Ray requires matching Python
#      and exact dep versions across the cluster).
#   2. 4-node Ray cluster up:
#         head:   ray start --head --port=6379 --num-gpus=8
#         worker: ray start --address=<head_ip>:6379 --num-gpus=8   # x3
#      Verify with `ray status` -> "4 active node(s)" and 32 GPUs total.
#   3. SkyRL cloned at the pinned commit and `export SKYRL_HOME=<path>` on the
#      driver (workers pick it up from the runtime_env injected by entrypoint.py).
#   4. Raw BIRD-SQL staged at $BIRD_RAW, then
#      `python download_data.py --bird_dir $BIRD_RAW --output_dir $DATA_DIR`
#      to produce $DATA_DIR/{train,val}.parquet. $DATA_DIR MUST live on a shared
#      filesystem visible to all 4 nodes.

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_LIB_DIR="$(cd "${SCRIPT_DIR}/../_lib" && pwd)"

if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
    echo "       Clone SkyRL at the pinned commit (see ../README.md) and"
    echo "       'export SKYRL_HOME=<path to clone>' before running this script."
    exit 1
fi
export PYTHONPATH="${SKYRL_HOME}:${SKYRL_LIB_DIR}:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
# Also disable torch.inductor's on-disk cache — a stale compiled graph from a
# prior run with fuse_allreduce_rms=true can otherwise fail warm-up with
#   AssertionError: Flashinfer workspace must be initialized when using flashinfer
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export ARCTIC_CUDA_IPC_LOW_MEM=0
# Qwen3-32B's tie_word_embeddings is False; the strict-name bypass is a no-op
# today and a safety net if upstream Qwen3 adds new tied buffers.
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0
# 32B optimizer-state CPU offload churns the allocator; expandable segments
# tame it. Unlike the single-node TP>1 sibling, the multi-node placement
# group layout doesn't hit pytorch/pytorch#147851 here.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# WandB — set WANDB_API_KEY to enable; LOGGER=console to skip wandb entirely.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl}"
export WANDB_DISABLE_CODE=True

# ----- 4-node 32-GPU Arctic / ZoRRo topology -----
NUM_NODES=4
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))   # = 32

# vLLM sampling TP=4 (Qwen3-32B doesn't fit per-GPU at bf16 + 0.5 mem_util
# headroom on H200). 32 GPUs / TP=4 -> 8 engine replicas.
TP_SIZE=4
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

# FA3 is the default on Hopper; flip to flash_attention_2 for A100/L40S.
ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
ARCTIC_ZERO_STAGE=3
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-true}"   # 32B optimizer state offload

# Global batch: 128 prompts x 16 samples = 2048 trajectories/step. With DP=32:
# per-DP mini=64, ZoRRo micro=16 (n_samples), grad_accum=4.
# NB: train_batch_size * n_samples_per_prompt must be divisible by num_gpus.
TRAIN_BSZ="${TRAIN_BSZ:-128}"
MINI_BSZ="${MINI_BSZ:-128}"
N_SAMPLES="${N_SAMPLES:-16}"
PROMPT_LEN="${PROMPT_LEN:-32768}"
RESPONSE_LEN="${RESPONSE_LEN:-4096}"

LR="${LR:-2e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-false}"

LOGGER="${LOGGER:-wandb}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
MODEL_SHORT="$(basename "${MODEL}")"

DATA_DIR="${DATA_DIR:-${HOME}/data/bird}"
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
    echo "ERROR: BIRD-SQL parquets not found at ${DATA_DIR}/{train,val}.parquet"
    echo "       Run 'python download_data.py --bird_dir <raw> --output_dir ${DATA_DIR}' first."
    echo "       NOTE: \$DATA_DIR must be on a shared filesystem visible to all 4 nodes."
    exit 1
fi

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME="bird_grpo_${MODEL_SHORT}_arl_z${ARCTIC_ZERO_STAGE}_${NUM_NODES}node_${RUN_TS}"
# CKPT_DIR must be on a shared FS — head writes weight-sync tensor, all nodes mmap-read.
CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

# Optional Arctic speculative decoding (drop in a Qwen3-32B-trained N-head ckpt
# via SPEC_MODEL=<path>; the blog-run head is tied to Qwen3-32B's hidden size).
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
    trainer.arctic_rl.vllm_max_num_batched_tokens=40960 \
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
