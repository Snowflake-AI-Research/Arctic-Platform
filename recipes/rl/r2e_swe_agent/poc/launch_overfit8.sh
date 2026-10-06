#!/usr/bin/env bash
# Overfit 8 prompts -- the middle ground after 15 was too slow and 4 collapsed.
#
# WHAT WENT WRONG AT 4 PROMPTS. With 4 tasks x group 8, reward never rose and
# then fell apart: 0.500, 0.469, 0.406, 0.469, 0.156. The collapse is visible in
# the training internals well before the reward, in grad_norm at the *first*
# epoch of each step, which is the honest per-step gradient on fresh data:
#   step 0: 1.179   step 1: 1.393   step 2: 2.519   step 3: 4.393   step 4: 4.523
# A gradient that grows every step is a policy being destabilised, not trained.
# Mean turns fell with it -- 36.4, 34.4, 34.8, 32.1, 24.2 -- the same march
# toward degenerate short outputs that an earlier lr 1e-5 run died of.
#   The cause is too few groups, not too little learning. GRPO's advantage is
#   computed within a group, so 4 groups means the whole update direction comes
#   from 4 noisy estimates; 16 epochs then amplifies that noise into the
#   weights. The identical mops 16 / lr 2e-6 settings were stable on 15 tasks
#   (first-epoch grad_norm 1.298 and 1.562, entropy flat), so the setting was
#   never the problem on its own -- the batch was.
#
# SO: 8 GROUPS, 1 EPOCH. Twice the groups of the run that collapsed, and no
# batch reuse at all, which removes the amplification half of the failure
# outright: with one epoch the sampled policy *is* the trained policy, the
# importance ratio is 1 by construction, and a noisy advantage estimate is
# applied once rather than sixteen times. Noise then averages across steps
# instead of compounding within one.
#   Cost: 64 rollouts/step is ~3,950s of collection plus ~120s of training, so
#   ~1.1h per step -- but only ONE optimizer step per collection, against 16
#   before. That is the price of being fully on-policy and it is the reason for
#   the lr below.
#
# LR 5e-6, raised from 2e-6, specifically because mops is 1. The risk that made
# a high lr dangerous was compounding: many updates on one stale batch drifting
# the policy off the sampler. At one epoch there is no drift to compound, so the
# ceiling of the reference range (its configs span 1e-6 to 5e-6) is the right
# end to sit at when each collection buys a single update. Drop to 2e-6 if
# first-epoch grad_norm climbs.
#
# TASK CHOICE. All eight are mid-range as measured on the 15-prompt run (step 0
# -> step 1 solves out of 8), since GRPO gradient is maximised near p=0.5 and
# vanishes at 0/8 or 8/8:
#   numpy@a5322429 6->6, scrapy@b51b52ff 5->3, coveragepy@16997254 4->5,
#   coveragepy@c4fc3833 4->4, numpy@c8a09822 3->3, orange3@a2c7ac74 2->3,
#   aiohttp@274c54e4 2->3, tornado@37081d79 1->3
# Baseline ~0.45, headroom to 1.0. Still excluded: coveragepy@5c3d0946 (0/8,
# nothing to reinforce) and pyramid@a43abd25 (7/8, already near saturation).
#
# THE NUMBER TO WATCH is first-epoch grad_norm per step. Flat or falling across
# steps means stable; climbing past ~2.5 means this is heading the same way as
# the 4-prompt run and wants lr 1e-6 or mops 4, not patience.
#
# Inherited and measured elsewhere: turn cap 40, the 41,000-token pack guard
# fitted from four OOMs, lr 2e-6, no --attn-impl, no reward zeroing.
set -uo pipefail

RUN="${1:-/modeling-code/karthik/abstract-remote-exps/runs/overfit8-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN"

# The pod and sandbox cleanup below is indiscriminate, so starting a second run
# while one is live silently kills the first one's rollouts. Refuse instead.
/data-fast/ap-venv/bin/python \
  /modeling-code/karthik/abstract-remote-exps/poc/run_lock.py acquire "$RUN" || exit 1

export PRIME_RL_ROOT=/modeling-code/boyiliu/prime-rl
export R2E_DATASET=/data/fshu/important/swe_data/r2e_family/R2E-Gym-Subset_validgold_unique_baseline/train.jsonl
set -a; . /data-fast/cortex.env; set +a
export KUBECONFIG=/data-fast/k3s/kubeconfig.yaml
export CORTEX_JOB_COMMENT="karthik r2e-gym overfit8 convergence run -- in use, please ask before cancelling"
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
  --steps 30 --group 8 --concurrency 24 --seed 42 --lr 5e-6 \
  --mops 1 --send-logprobs \
  --max-turns 40 --max-pack-tokens 41000 \
  --rollout-timeout 6000 --job-ready-timeout 7200 \
  --task-ids "$TASKS" \
  --allow-content --no-std-norm --no-length-penalty --adam-beta2 0.95 \
  --no-zero-on-violation \
  --train-gpus 2 --sample-gpus 6 \
  --micro-batch 2 --gpu-mem-util 0.85 --max-num-seqs 56 --tensor-parallel 1 \
  --keepalive-interval 2400 \
  --out "$RUN" 2>&1 | tee -a "$RUN/run.log"
