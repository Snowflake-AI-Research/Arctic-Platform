#!/usr/bin/env bash
# Overfit 8 prompts to convergence.
#
# Sized from the throughput measurement across four prior runs: generate holds at
# ~0.7-0.8 req/s no matter the concurrency (24 -> 96) or the sampling GPU count
# (16 -> 32). Two consequences drive every number below.
#
#   Collection wall clock is total_generate_calls x ~1.33s, so the only lever on
#   step time is calls per step. 8 prompts x 8 rollouts = 64 rollouts at ~62 turns
#   is ~3,968 calls ~= 1.5h, against 2.9h for the 16-prompt shape.
#
#   Concurrency buys no throughput, it only multiplies per-rollout latency
#   (33s/turn at 24, 113s/turn at 64). Three earlier runs died on the rollout
#   timeout for exactly this reason, so run at 24 and the timeout stops binding.
#
# 6 sampling GPUs rather than 32: they were 1-3% utilised and throughput was flat
# across 16 and 32, so the extra 26 bought nothing. This also matches the
# reference topology (2 train + 6 infer).
#
# Tasks are the 8 fastest of the 11 that showed group spread in overfit16d; the 5
# that went 0/8 and would carry no gradient are dropped.
#
# Cross-entropy materialises a [tokens x 151936] logit tensor, which asks for
# ~67 GiB in one allocation at these sequence lengths and OOMs an H200. Lowering
# micro-batch cannot help: the backend raises it to cover the training GPUs, so
# 2 GPUs means at least 2. --liger is the actual fix -- Liger's fused linear
# cross-entropy chunks the lm head, which is how the reference trains the same
# lengths on the same two GPUs (it uses fused_lm_head_token_chunk_size = 16384).
#
# betas2 0.95 and --no-zero-on-violation also follow the reference's overfit
# config, which detects format violations but does not zero reward on them.
set -uo pipefail

RUN="${1:-/modeling-code/karthik/abstract-remote-exps/runs/conv8-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN"

export PRIME_RL_ROOT=/modeling-code/boyiliu/prime-rl
export R2E_DATASET=/data/fshu/important/swe_data/r2e_family/R2E-Gym-Subset_validgold_unique_baseline/train.jsonl
set -a; . /data-fast/cortex.env; set +a
export KUBECONFIG=/data-fast/k3s/kubeconfig.yaml
# A previous 8-GPU job was cancelled by an external client three minutes in. Every
# job reports submitted_by=ADMIN, so the comment is the only way anyone sweeping
# the schema can tell this one is in use.
export CORTEX_JOB_COMMENT="karthik r2e-gym overfit8 convergence run -- in use, please ask before cancelling"
PY=/data-fast/ap-venv/bin/python
K="/data-fast/k3s/bin/k3s kubectl"

# Start clean: leftover pods and CRs from a previous run hold node memory.
$K get pods -n default --no-headers 2>/dev/null | awk '/^r2e-/{print $1}' \
  | xargs -r -n20 $K delete pod -n default --grace-period=0 --force >/dev/null 2>&1
$PY /modeling-code/karthik/abstract-remote-exps/poc/reap_sandboxes.py --max-age 0 >/dev/null 2>&1

# The controller recreates pods while a CR lives, so without this the run wedges
# on node memory after a few steps.
nohup $PY /modeling-code/karthik/abstract-remote-exps/poc/reap_sandboxes.py \
  --loop 300 --max-age 7200 > "$RUN/reaper.log" 2>&1 &
echo "reaper pid $!" | tee "$RUN/reaper.pid"

TASKS=coveragepy@16997254,numpy@c8a09822,orange3@78213643,orange3@a2c7ac74,coveragepy@c4fc3833,pyramid@24c63558,numpy@a5322429,aiohttp@274c54e4

cd /modeling-code/karthik/abstract-remote-exps
exec $PY poc/r2e_driver.py \
  --steps 20 --group 8 --concurrency 24 --seed 42 --lr 1e-6 \
  --mops 4 --send-logprobs \
  --rollout-timeout 6000 --job-ready-timeout 7200 \
  --task-ids "$TASKS" \
  --allow-content --no-std-norm --no-length-penalty --adam-beta2 0.95 \
  --liger --attn-impl flash_attention_2 --no-zero-on-violation \
  --train-gpus 2 --sample-gpus 6 \
  --micro-batch 2 --gpu-mem-util 0.85 --max-num-seqs 56 --tensor-parallel 1 \
  --keepalive-interval 2400 \
  --out "$RUN" 2>&1 | tee -a "$RUN/run.log"
