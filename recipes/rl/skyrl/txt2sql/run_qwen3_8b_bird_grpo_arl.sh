#!/bin/bash
# Arctic RL + ZoRRo: Qwen3-8B BIRD-SQL GRPO — single node / 8 H200s.
# Pure GRPO, no frozen reference model (use_kl_loss=false).
#
# Iteration target for the 4-node Qwen3-32B run (`run_qwen3_32b_bird_grpo_arl_4node.sh`).
#
# Prereqs (see README.md):
#   1. Activated conda env with pinned deps; `export SKYRL_HOME=<clone>`.
#   2. `python download_data.py --bird_dir <raw> --output_dir $DATA_DIR`.

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SkyRL is required as a checkout: integrations/arctic_rl/ is not in the
# pip-installed package. Pin: see ../README.md.
if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
    echo "       Clone SkyRL at the pinned commit (see ../README.md) and"
    echo "       'export SKYRL_HOME=<path to clone>' before running this script."
    exit 1
fi
# ${SCRIPT_DIR} carries the recipe-local ``arctic_rl/`` shim + ``sitecustomize.py``
# that register ``bird`` / ``bird_sql`` — see README.md.
export PYTHONPATH="${SKYRL_HOME}:${SCRIPT_DIR}:${PYTHONPATH:-}"

# Bare python from a caller-activated env. See ../README.md.
PYBIN="${PYBIN:-python}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# NOTE: do NOT force-disable the inductor/vLLM compile caches. vLLM 0.18.0's
# CUDA-graph precompile path asserts `Cannot precompile with
# torch._inductor.config.force_disable_caches=True; caching is required`, so
# TORCHINDUCTOR_FORCE_DISABLE_CACHES / TORCH_COMPILE_DISABLE crash the vLLM
# engine at startup. Upstream's arctic_rl launchers set none of these.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export ARCTIC_CUDA_IPC_LOW_MEM=0
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on single-node TP>1:
# it trips pytorch/pytorch#147851 inside vLLM's Ray-executor TP workers.

# ----- Single-node 8-GPU Arctic / ZoRRo topology -----
NUM_NODES=1
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# TP=4 -> 2 vLLM engines. Matches the 4-node 32B recipe's multi-rank code
# path so TP>1-only bugs surface on the smaller iteration target first.
TP_SIZE="${TP_SIZE:-4}"
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

# FA2 + OFFLOAD=true match upstream's ARL 8B example. FA3 on this
# single-node checkout empirically re-triggers the FlashInfer-workspace
# assertion even with the AI cfg workaround — followup to isolate.
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
ARCTIC_ZERO_STAGE=3
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-true}"

TRAIN_BSZ="${TRAIN_BSZ:-32}"
MINI_BSZ="${MINI_BSZ:-16}"
N_SAMPLES="${N_SAMPLES:-8}"
PROMPT_LEN="${PROMPT_LEN:-16384}"
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

# Inference knobs forwarded to vLLM via trainer.arctic_rl.arctic_inference_config
# (raw passthrough). Same rationale as upstream run_bird_grpo_{8b,32b}_32gpu.sh:
# optimization_level=1 hard-codes fuse_allreduce_rms=false in vLLM's compile
# pipeline, avoiding the `Flashinfer workspace must be initialized` assertion
# on TP>1 + Hopper. Unconditional here (upstream gates it behind USE_FCA=True).
USE_FCA="${USE_FCA:-False}"     # requires arctic_inference wheel built with FCA support
SPEC_MODEL="${SPEC_MODEL:-}"    # 32B-tied; supply 8B-sized 3-head via SPEC_MODEL=<path>
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"

AI_CFG_PARTS=('optimization_level: 1'
              'compilation_config: {cudagraph_mode: PIECEWISE, pass_config: {fuse_allreduce_rms: false}}')
if [[ "${USE_FCA}" == "True" ]]; then
    AI_CFG_PARTS+=('forest_cascade_attn_configs: "{}"')
fi
if [[ -n "${SPEC_MODEL}" && -d "${SPEC_MODEL}" ]]; then
    AI_CFG_PARTS+=("speculative_config: {method: arctic, model: ${SPEC_MODEL}, num_speculative_tokens: ${NUM_SPEC_TOKENS}}")
fi
IFS=, AI_CFG_BODY="${AI_CFG_PARTS[*]}" ; unset IFS
AI_CFG_OVERRIDE=("trainer.arctic_rl.arctic_inference_config={${AI_CFG_BODY}}")

# Run from ${SKYRL_HOME} so ``integrations/`` imports resolve.
cd "${SKYRL_HOME}"

"${PYBIN}" -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=arctic_rl.entrypoint \
    trainer.arctic_rl.colocate=true \
    trainer.arctic_rl.zero_stage=${ARCTIC_ZERO_STAGE} \
    trainer.arctic_rl.offload_optimizer=${OFFLOAD_OPTIMIZER} \
    trainer.arctic_rl.offload_param=false \
    trainer.arctic_rl.log_prob_gpus=0 \
    trainer.arctic_rl.use_zorro=true \
    trainer.arctic_rl.use_liger=true \
    trainer.arctic_rl.attn_implementation=${ATTN_IMPL} \
    trainer.arctic_rl.enable_gradient_checkpointing=true \
    trainer.arctic_rl.ulysses_sequence_parallel_size=1 \
    trainer.arctic_rl.logits_optimization=memory \
    trainer.arctic_rl.cuda_ipc_weight_sync=true \
    trainer.arctic_rl.low_memory_weight_sync=true \
    trainer.arctic_rl.lr_warmup_ratio=0.05 \
    'trainer.arctic_rl.optimizer_betas=[0.9,0.95]' \
    trainer.arctic_rl.vllm_enforce_eager=false \
    trainer.arctic_rl.vllm_enable_prefix_caching=true \
    trainer.arctic_rl.vllm_max_num_batched_tokens=40960 \
    trainer.arctic_rl.vllm_max_num_seqs=256 \
    trainer.arctic_rl.use_arctic_inference=true \
    trainer.arctic_rl.server_logs=true \
    trainer.arctic_rl.startup_timeout=1800 \
    "${AI_CFG_OVERRIDE[@]}" \
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
