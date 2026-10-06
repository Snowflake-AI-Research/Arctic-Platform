#!/usr/bin/env bash
# Same recipe as launch_fast8.sh. Only the eight prompts change.
#
# WHY FAST8 FAILED, AND IT WAS NOT THE HYPERPARAMETERS. fast8 fixed throughput
# (7.5x more optimizer updates per hour, confirmed: 4,738s/step for 8 updates
# against 4,729s for 1) and then produced a worse curve than the run it
# replaced. Six steps, 48 updates:
#   reward  0.359 0.266 0.359 0.234 0.219 0.234   slope -0.025/step, t=-2.21
#   entropy 0.452 0.633 0.611 0.753 0.792 0.787   slope +0.065/step, t=+4.88
# Entropy *rising* is the tell. The previous run sharpened without improving;
# this one diffused. Same data path, same tasks, more updates -- so the updates
# were injecting noise, and more of them made it worse. That rules out step
# count, which is what fast8 was built to fix.
#
# It was not behavioural collapse either. The failure mix is flat across all
# six steps (truncation 64-77%, format_invalid 11-23%, mean turns 34.7-36.8),
# so the policy kept doing the same things while getting less certain.
#
# THE PROMPTS WERE THE PROBLEM. Per task over this run (n~49 each):
#   coveragepy@16997254  0.64      scrapy@b51b52ff   0.29
#   coveragepy@c4fc3833  0.59      orange3@a2c7ac74  0.27
#   aiohttp@274c54e4     0.15      numpy@a5322429    0.10
#   numpy@c8a09822       0.08      tornado@37081d79  0.06
# Six of the eight sit at or below 0.29. At p=0.08 a group of 8 lands one
# success, GRPO hands that single rollout advantage +0.875, and the model is
# taught to imitate a trajectory that succeeded by luck. Most of the gradient
# *magnitude* in every step came from those groups. Note this is invisible in
# the diagnostic fast8 was watching: those groups are still "informative" (6-8
# of 8 every step), they are just informative about noise.
#
# Mining all 39 runs in runs/ for per-task pass rate (84 distinct tasks, each
# scored over 20-472 rollouts) says this is a property of the task pool, not of
# our eight picks: 70 of 84 tasks pass below 0.25, and none passes above 0.75.
# Exactly seven sit in the band where the gradient means something. This takes
# those seven plus the best of the remainder.
#
# Mean pass goes 0.27 -> 0.385, and every prompt here is one the model has
# demonstrably solved many times, so advantage tracks behaviour it controls.
#
# WHAT IS DELIBERATELY UNCHANGED: concurrency 64, mops 8, lr 2e-6, the
# 41,000-token pack guard, no reward zeroing. fast8 established the throughput
# and nothing about it was wrong, so the two changes here are both about the
# quality of the reward signal: which prompts, and how many turns they get. If
# reward still will not move on prompts the model can actually solve, within a
# turn budget that lets it finish them, the problem is in the loss or the
# credit assignment rather than the curriculum.
#
# TURN CAP 55, AND THIS IS THE SECOND FIX. 272 of 381 fast8 rollouts stop at
# exactly 40 turns, logged as stop_condition=truncation reason=response_length.
# That label is misleading -- it reads like a response hitting
# --max-tokens-per-turn, but the largest single response among them is 6,355
# tokens against a 32,768 limit. The real wall is the turn cap.
#
# How much that wall costs us, over the 399 clean finishes these eight tasks
# have produced across all runs in runs/: turns-to-finish is p50=38, p75=58,
# p90=76. A cap of 40 therefore keeps only 56% of the finishes the model is
# capable of, and the other 44% are scored as if they had failed. Since clean
# finishes pass at 0.76 against 0.31 for truncations, that lost 44% is both a
# large chunk of the available reward and a large chunk of the "ran out of
# turns" noise this run is trying to get away from.
#
# Note this closes off the obvious way to make steps cheaper. Cutting the cap
# to 25 would keep 18% of clean finishes and to 20 just 5%, so trading turns
# for speed would destroy the signal. Collection is ~90% of a step, so the
# honest conclusion is that this run buys signal and pays wall time for it.
#
# 55 is where the token budget runs out, not a preference. Fitted over 1,487
# band8 trajectories, tokens = 2,775 + 572 per turn, so median packed length is
# ~34,200 at cap 55 against the 41,000-token guard (itself fitted from four
# OOMs on 2 training GPUs). Cap 60 puts the p90 at ~46,600 and starts feeding
# the guard; cap 70 puts the *median* over it. Going past 55 needs more
# training GPUs, not a bigger number here.
#
# Expected: ~17 more points of rollouts convert from truncated to clean, each
# moving from 0.31 to 0.76, so mean pass should land near 0.46 -- which is also
# where GRPO's informative-group rate and gradient magnitude peak (p=0.5).
# Steps get ~1.3x longer because the 62% that truncate now run 55 turns.
#
# STEPS 12, NOT 20, because 12 x mops 8 = 96 updates and the entropy budget at
# lr 2e-6 is ~88. Twenty steps would have been 160 updates, i.e. the back half
# of the run spent past the point where entropy is exhausted.
#
# CAP 45, REVISED DOWN FROM 55 AFTER TWO MEASURED STEPS. Cap 55 worked too
# well: step 0 came in at reward 0.727 and step 1 at 0.688, against the 0.46 it
# was predicted to reach. Five of the eight prompts landed at 7/8 or 8/8, and a
# group where every rollout scores the same yields zero advantage by
# construction, so groups-with-spread fell 6/8 -> 4/8 and the training batch
# shrank 45 -> 31 rollouts. The surviving gradient came from the three prompts
# sitting at 1/8, 2/8 and 3/8 -- the same low-p noise regime that sank fast8.
#
# The lesson is that the turn cap and prompt difficulty are not independent
# knobs. The per-task pass rates this set was selected on were measured at the
# old cap, so raising the cap invalidated the very calibration used to pick the
# prompts. Raising pass rate helps GRPO only up to ~0.5; past that, groups
# saturate and the signal disappears from the top instead of the bottom.
#
# So interpolate to the optimum rather than guess: these eight prompts average
# ~0.41 at cap 40 and 0.71 at cap 55, which puts p=0.5 at cap ~45. That also
# pulls median packed length back to ~28,500 tokens (2,775 + 572/turn), well
# under the 41,000 guard that was dropping 13% of the batch at cap 55, and it
# cuts the training half of each step, which cap 55 had inflated from 1,002s to
# ~6,300s by pushing the longest sequences to 39,644 tokens.
#
# THE NUMBER TO WATCH is groups-with-spread, logged every step. It should sit at
# 7-8 of 8. Falling means the prompts are saturating (too easy) and rising
# reward is about to stop meaning anything; combined with entropy rising it is
# the fast8 pattern and wants a harder set, not a hyperparameter change.
set -uo pipefail

RUN="${1:-/modeling-code/karthik/abstract-remote-exps/runs/band8-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN"

# The pod and sandbox cleanup below is indiscriminate, so starting a second run
# while one is live silently kills the first one's rollouts. Refuse instead.
/data-fast/ap-venv/bin/python \
  /modeling-code/karthik/abstract-remote-exps/poc/run_lock.py acquire "$RUN" || exit 1

export PRIME_RL_ROOT=/modeling-code/boyiliu/prime-rl
export R2E_DATASET=/data/fshu/important/swe_data/r2e_family/R2E-Gym-Subset_validgold_unique_baseline/train.jsonl
set -a; . /data-fast/cortex.env; set +a
export KUBECONFIG=/data-fast/k3s/kubeconfig.yaml
export CORTEX_JOB_COMMENT="karthik r2e-gym band8 convergence run -- in use, please ask before cancelling"
PY=/data-fast/ap-venv/bin/python
K="/data-fast/k3s/bin/k3s kubectl"

$K get pods -n default --no-headers 2>/dev/null | awk '/^r2e-/{print $1}' \
  | xargs -r -n20 $K delete pod -n default --grace-period=0 --force >/dev/null 2>&1
$PY /modeling-code/karthik/abstract-remote-exps/poc/reap_sandboxes.py --max-age 0 >/dev/null 2>&1

nohup $PY /modeling-code/karthik/abstract-remote-exps/poc/reap_sandboxes.py \
  --loop 300 --max-age 7200 > "$RUN/reaper.log" 2>&1 &
echo "reaper pid $!" | tee "$RUN/reaper.pid"

# Pass rates from 39 runs: 0.66 0.48 0.47 0.38 0.32 0.29 0.26 0.20.
TASKS=coveragepy@16997254,coveragepy@c4fc3833,pyramid@a43abd25,coveragepy@da1b282d,scrapy@7910fa01,orange3@a2c7ac74,aiohttp@37b2604d,aiohttp@274c54e4

cd /modeling-code/karthik/abstract-remote-exps
exec $PY poc/r2e_driver.py \
  --steps 12 --group 8 --concurrency 64 --seed 42 --lr 2e-6 \
  --mops 8 --send-logprobs \
  --max-turns 45 --max-pack-tokens 41000 \
  --rollout-timeout 6000 --job-ready-timeout 7200 \
  --task-ids "$TASKS" \
  --allow-content --no-std-norm --adam-beta2 0.95 \
  --no-zero-on-violation \
  --train-gpus 2 --sample-gpus 6 \
  --micro-batch 2 --gpu-mem-util 0.85 --max-num-seqs 56 --tensor-parallel 1 \
  --keepalive-interval 2400 \
  --out "$RUN" 2>&1 | tee -a "$RUN/run.log"
