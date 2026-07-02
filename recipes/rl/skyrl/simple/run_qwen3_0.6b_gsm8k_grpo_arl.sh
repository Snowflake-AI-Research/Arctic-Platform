#!/bin/bash
# Single-GPU GRPO training for Qwen3-0.6B on GSM8K with Arctic RL + ZoRRo.
# Pure GRPO, no frozen reference model (use_kl_loss=False).
#
# This is the "simple" entry-point recipe: 1 GPU, 1 node, no Ray cluster setup
# and no hostfile. SkyRL starts a local Ray instance automatically.
#
# Prerequisites (see README.md):
#   1. Conda env with the recipe's pinned deps installed
#      (`uv pip install -r requirements.txt --override overrides.txt`).
#   2. SkyRL cloned at the pinned commit (see ../README.md) and SKYRL_HOME
#      pointing at it — the Arctic RL × SkyRL integration code lives at
#      $SKYRL_HOME/integrations/arctic_rl/ (NOT in the pip-installed package).
#   3. Data prepared: `python download_data.py` -> $DATA_DIR/{train,validation}.parquet.

set -euo pipefail

# SkyRL is required as a checkout (the Arctic RL × SkyRL integration code lives
# at integrations/arctic_rl/ which is NOT inside the pip-installed package).
# Pin: see ../README.md.
if [[ -z "${SKYRL_HOME:-}" || ! -d "${SKYRL_HOME}/integrations/arctic_rl" ]]; then
    echo "ERROR: SKYRL_HOME is unset or doesn't contain integrations/arctic_rl/."
    echo "       Clone SkyRL at the pinned commit (see ../README.md) and"
    echo "       'export SKYRL_HOME=<path to clone>' before running this script."
    exit 1
fi
# $SKYRL_HOME provides integrations.arctic_rl.* (config/trainer/generator/BirdEnv).
# The GSM8K env is registered by upstream skyrl_gym, so this recipe doesn't ship
# any Python — it dispatches straight to upstream's Ray entrypoint below.
export PYTHONPATH="${SKYRL_HOME}:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
# Belt-and-suspenders for TP>1 CLI overrides — see the BIRD-8B launcher note on
# the FlashInfer-workspace assertion. Harmless at TP=1.
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO

# ----- Single-GPU Arctic/ZoRRo topology -----
# `trainer.arctic_rl.colocate=true` keeps Arctic RL's training + sampling jobs
# on the same GPU. `trainer.placement.colocate_all=false` is required so SkyRL
# does NOT also try to grab a placement group for its own inference engines —
# Arctic RL already owns the GPU, so a SkyRL PG would deadlock.
ARCTIC_ZERO_STAGE=2       # 0.6B fits comfortably on one GPU; no offload needed
NGPU_PER_NODE=1
NUM_NODES=1
TP_SIZE=1
NUM_ENGINES=1
GPU_MEM_UTIL=0.3          # leave headroom for training in colocated mode

# ----- Training hyperparams (small-scale single-GPU defaults) -----
TRAIN_BSZ=32              # prompts per step
MINI_BSZ=4                # actor mini-batch (per DP rank)
N_SAMPLES=4               # GRPO group size
PROMPT_LEN=512            # GSM8K prompts are short
RESPONSE_LEN=1024
LR=1e-6
TOTAL_EPOCHS=1
EVAL_INTERVAL=10          # validate every 10 steps

# Defaulted to console so the recipe runs without WANDB_API_KEY. Set
# LOGGER=wandb to enable wandb (export WANDB_API_KEY first).
LOGGER="${LOGGER:-console}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_SHORT="$(basename "${MODEL}")"

EXPERIMENT_NAME="gsm8k_grpo_${MODEL_SHORT}_arl_z${ARCTIC_ZERO_STAGE}"

# Data: GSM8K parquets produced by download_data.py
DATA_DIR="${DATA_DIR:-${HOME}/data/gsm8k}"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/validation.parquet"

CKPT_DIR="${CKPT_DIR:-${HOME}/checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}"

# Arctic-Inference config. At the default TP=1 the fused_allreduce_rms pass is
# a no-op, but keeping the same override shape as the multi-GPU recipes means
# users who override TP>1 (e.g. bumping to 8B on 8 GPUs) inherit the escape
# hatch for free.
AI_CFG_PARTS=('optimization_level: 1'
              'compilation_config: {cudagraph_mode: PIECEWISE, pass_config: {fuse_allreduce_rms: false}}')
IFS=, AI_CFG_BODY="${AI_CFG_PARTS[*]}" ; unset IFS
AI_CFG_OVERRIDE=("trainer.arctic_rl.arctic_inference_config={${AI_CFG_BODY}}")

python -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
    trainer.arctic_rl.colocate=true \
    trainer.arctic_rl.zero_stage=${ARCTIC_ZERO_STAGE} \
    trainer.arctic_rl.use_zorro=true \
    trainer.arctic_rl.use_liger=true \
    trainer.arctic_rl.logits_optimization=memory \
    trainer.arctic_rl.enable_gradient_checkpointing=true \
    trainer.arctic_rl.use_arctic_inference=true \
    trainer.arctic_rl.vllm_enforce_eager=false \
    trainer.arctic_rl.vllm_enable_prefix_caching=true \
    trainer.arctic_rl.vllm_max_num_seqs=256 \
    "${AI_CFG_OVERRIDE[@]}" \
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
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.gpu_memory_utilization=${GPU_MEM_UTIL} \
    generator.batched=true \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    environment.env_class=gsm8k \
    trainer.epochs=${TOTAL_EPOCHS} \
    trainer.train_batch_size=${TRAIN_BSZ} \
    trainer.policy_mini_batch_size=${MINI_BSZ} \
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
    trainer.project_name=arctic_rl_gsm8k \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CKPT_DIR}/logs" \
    trainer.ckpt_path="${CKPT_DIR}/ckpt" \
    "$@" 2>&1 | tee "${CKPT_DIR}/${EXPERIMENT_NAME}.log"
