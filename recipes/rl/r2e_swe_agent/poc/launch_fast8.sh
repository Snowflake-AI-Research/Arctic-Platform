#!/usr/bin/env bash
# Overfit the same 8 prompts, but ~7x more optimizer updates per hour.
#
# WHY THE PREVIOUS RUN FAILED. runs/overfit8-20261004-042900 ran 9 complete
# steps over 12 hours at concurrency 24 / mops 1 / lr 5e-6 and the reward never
# moved:
#   0.297 0.422 0.281 0.344 0.375 0.391 0.359 0.422 0.281
# Linear trend +0.0023/step, t=+0.30 -- indistinguishable from flat. Meanwhile
# entropy fell 0.497 -> 0.383, slope -0.0141/step at t=-3.56, the only
# statistically significant trend in the run. A policy sharpening without
# improving. Per-task it specialised on what it could already do (coveragepy
# 0.62->0.88) and abandoned what it could not (tornado 0.25->0.00).
#
# This was NOT instability: informative groups held at 6-8 of 8 every step, so
# the 4-prompt failure mode never recurred, and grad_norm had no trend
# (t=-0.63). It was simply too few updates. Nine steps bought NINE updates.
#
# WHERE THE TIME WENT -- measured, not guessed. Per-turn cost is ~40s while the
# model generates only 229 tokens, so the obvious suspects were checked first
# and all three were cleared:
#   gateway chat-template + tokenize : 0.074s/req, 135 req/s at conc 24
#   prefix cache                     : idle 11k-token prefill is 0.40-0.54s
#                                      whether cold, warm or grown -- caching
#                                      is not the lever, prefill is already
#                                      cheap (poc/probe_prefix_cache.py)
#   request serialization            : none. 12.7x overlap at conc 16
#                                      (poc/probe_concurrency.py)
# What it actually is: the sampler is underfed. poc/probe_sampler_scaling.py
# pushed 11k-token prompts at it *while 24 agents were live* and aggregate
# throughput still scaled near-linearly --
#   conc  1 ->    233 tok/s    conc  8 ->  1,449 tok/s  (6.2x)
#   conc  2 ->    412 tok/s    conc 16 ->  2,725 tok/s  (11.7x)
# -- for only a 1.4x latency penalty (47s -> 65s). Six H200s serving a 4B model
# had headroom we were not using. This also retires the older note that
# throughput was flat from concurrency 24 to 96; that is not what the sampler
# does when measured directly.
#
# SO: CONCURRENCY 64, ONE WAVE. 64 rollouts, 64 slots, no wave quantisation.
# Fitting the measured scaling (exponent 0.88) puts collection at ~2,000s
# against 4,729s, i.e. ~2.4x.
#
# AND MOPS 8, WHICH IS THE BIGGER WIN. Collection is 90% of a step, so mops 1
# throws the expensive half away: one update per 1.4h. Updates per hour is what
# drives convergence, and it is maximised by reusing the batch we already paid
# for:
#   conc 24 mops  1 -> 0.70 upd/h   (what just failed)
#   conc 64 mops  8 -> 5.15 upd/h   (7.4x)
# mops 8 is not a guess. Updates per informative group was 1.07 on the stable
# 15-task run (mops 16) and 4.0 on the 4-task run that collapsed (mops 16). At
# 8 tasks with ~7 informative groups, mops 8 gives 1.14 -- next to the stable
# configuration, far from the one that broke.
#
# LR BACK TO 2e-6. 5e-6 was justified only by mops 1 having no drift to
# compound; with 8 epochs that argument is gone. It also doubles the entropy
# budget: at the measured -0.0141/update per 5e-6, entropy 0.497 is exhausted
# after ~35 updates, against ~88 at 2e-6. 20 steps x 8 = 160 updates, so
# entropy, not step count, is what will end this run.
#
# THE NUMBER TO WATCH is epoch-0 grad_norm per step (the gradient on fresh
# data) together with entropy. Both flat means keep going. grad_norm climbing
# step over step while entropy falls is the 4-prompt failure and wants mops 4.
#
# Inherited and measured elsewhere: turn cap 40 (cutting it to 20 would push
# the four 1/8 tasks to 0/8 and starve the gradient), the 41,000-token pack
# guard fitted from four OOMs, no --attn-impl, no reward zeroing.
set -uo pipefail

RUN="${1:-/modeling-code/karthik/abstract-remote-exps/runs/fast8-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN"

# The pod and sandbox cleanup below is indiscriminate, so starting a second run
# while one is live silently kills the first one's rollouts. Refuse instead.
/data-fast/ap-venv/bin/python \
  /modeling-code/karthik/abstract-remote-exps/poc/run_lock.py acquire "$RUN" || exit 1

export PRIME_RL_ROOT=/modeling-code/boyiliu/prime-rl
export R2E_DATASET=/data/fshu/important/swe_data/r2e_family/R2E-Gym-Subset_validgold_unique_baseline/train.jsonl
set -a; . /data-fast/cortex.env; set +a
export KUBECONFIG=/data-fast/k3s/kubeconfig.yaml
export CORTEX_JOB_COMMENT="karthik r2e-gym fast8 convergence run -- in use, please ask before cancelling"
PY=/data-fast/ap-venv/bin/python
K="/data-fast/k3s/bin/k3s kubectl"

$K get pods -n default --no-headers 2>/dev/null | awk '/^r2e-/{print $1}' \
  | xargs -r -n20 $K delete pod -n default --grace-period=0 --force >/dev/null 2>&1
$PY /modeling-code/karthik/abstract-remote-exps/poc/reap_sandboxes.py --max-age 0 >/dev/null 2>&1

nohup $PY /modeling-code/karthik/abstract-remote-exps/poc/reap_sandboxes.py \
  --loop 300 --max-age 7200 > "$RUN/reaper.log" 2>&1 &
echo "reaper pid $!" | tee "$RUN/reaper.pid"

TASKS=numpy@a5322429,scrapy@b51b52ff,coveragepy@16997254,coveragepy@c4fc3833,numpy@c8a09822,orange3@a2c7ac74,aiohttp@274c54e4,tornado@37081d79

cd /modeling-code/karthik/abstract-remote-exps
exec $PY poc/r2e_driver.py \
  --steps 20 --group 8 --concurrency 64 --seed 42 --lr 2e-6 \
  --mops 8 --send-logprobs \
  --max-turns 40 --max-pack-tokens 41000 \
  --rollout-timeout 6000 --job-ready-timeout 7200 \
  --task-ids "$TASKS" \
  --allow-content --no-std-norm --no-length-penalty --adam-beta2 0.95 \
  --no-zero-on-violation \
  --train-gpus 2 --sample-gpus 6 \
  --micro-batch 2 --gpu-mem-util 0.85 --max-num-seqs 56 --tensor-parallel 1 \
  --keepalive-interval 2400 \
  --out "$RUN" 2>&1 | tee -a "$RUN/run.log"
