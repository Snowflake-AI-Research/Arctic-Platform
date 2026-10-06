#!/usr/bin/env bash
# Overfit 15 prompts to convergence.
#
# Every number here is measured rather than chosen. The measurements live in
# poc/analyze_pass_and_turns.py (run it against runs/overfit16d-20260929-145849).
#
# TASK SET. Of overfit16d's 16 tasks, 15 of 16 are solvable at pass@8 (earned
# 51/128 = 0.398 overall), so the task set itself was never the problem. We keep
# 15 and drop exactly one, pillow@e0b95724, because it is the only task whose
# packed sequence exceeds the memory ceiling at this turn cap (50,009 tokens
# against 37,886 for every other task). Dropping that one task is what makes a
# cap of 40 possible instead of 25.
#   Two kept tasks carry no gradient and are kept deliberately:
#   pyramid@a43abd25 went 8/8, so it is a retention control -- if it regresses we
#   are damaging the policy; coveragepy@5c3d0946 went 0/8 and is a hard negative.
#   Both are short, so neither costs anything in memory.
#
# TURN CAP 40, not the argparse default of 100. The default was the single cause
# of both failure modes: it let conversations grow to 98,443 tokens (which OOMs
# the trainer) and to ~63 model calls per rollout (which is what makes a
# collection take 2.5h, since cost tracks call count). At cap 40 the longest
# sequence across these 15 tasks is 37,886 tokens.
#   The cap costs little: 80% of rollouts end on format_invalid at a median of
#   47 turns, i.e. they stop because the agent breaks protocol, not because it
#   runs out of turns. Only 9 of 89 ended on truncation.
#
# MEMORY CEILING. Fitted from four measured OOMs (2 and 4 training GPUs, 57k to
# 98k tokens): total GiB = 87.6 + 1.083 per 1k tokens, against 139.80 GiB per
# H200. That puts the safe maximum at ~41,000 tokens. The fit predicted 127.3
# GiB for a 36,709-token batch that then trained, and 128.6 GiB for this
# configuration, which was confirmed directly: TRAIN OK in 182s at 37,886
# tokens, grad_norm 6.83.
#   Things that do NOT fix it, both measured: a smaller --micro-batch (the
#   backend raises it to cover the training GPUs, so 2 GPUs means at least 2),
#   and more training GPUs (4 GPUs at 73k tokens still OOM'd with 117.54 GiB
#   per-GPU in use -- activations dominate, not optimizer state). --liger avoids
#   the allocation entirely but Cortex's RL pipeline then has no logits to
#   compute logprobs from, so it is unusable here.
#
# NO REWARD ZEROING. Gating cost overfit16d more than half its signal: 51
# rollouts earned reward and only 27 survived, and it zeroed three whole tasks
# to 0/8 that had actually solved 2-3 of 8. The reference overfit config detects
# format violations but does not zero on them, so neither do we.
#
# TOKEN GUARD, not just a turn cap. The first full run trained steps 0 and 1
# (longest 34,625 and 36,770 tokens) and then OOM'd on step 2, whose longest
# packed trajectory was 52,543 tokens: a turn cap bounds turns, and one turn can
# return a whole file or test log. --max-pack-tokens 41000 drops those, which on
# step 2 would have cost 1 trajectory out of 102 instead of the whole step.
# A training failure is also no longer fatal, since a step costs ~2h to collect.
#
# MOPS 16, raised from 4. Each mops is a full epoch over the step's batch, and
# collection is 89% of a step's wall clock (7,600s collect vs 915s train), so
# extra epochs are nearly free while extra steps are not: mops 4 bought 1.9
# optimizer steps per hour, mops 16 buys ~5.2. Three steps at mops 4 moved the
# policy so little that reward stayed inside its own sampling noise (0.342,
# 0.333, 0.275 against a measured +-0.03 noise floor), which is a throughput
# problem, not a learning-rate one.
#   Safe because the batch is nowhere near exhausted after 4 epochs: clip_ratio
#   ended step 0 at 0.0031 and step 1 at 0.0016, and importance_weight was
#   0.9993, so the policy had barely moved off the sampler. Watch clip_ratio --
#   if it climbs past ~0.2 the epochs have gone too far off-policy.
#
# LR 2e-6, raised from 1e-6. Three steps gave 0.292, 0.333, 0.267 -- noise, no
# trend, and at 4 optimizer steps per 2h collection, 1e-6 against gradients
# clipped to norm 1.0 moves very little. The reference's own r2e/swe configs for
# this model use 1e-6 to 5e-6 with 2e-6 common, so this stays inside tested
# ground. Fall back to 1e-6 if reward collapses or grad_norm climbs.
#
# NO --attn-impl. Asking for flash_attention_2 makes Cortex fail both sub-jobs
# during startup, before any batch is sent: a 2-GPU job with 4,096-token
# sequences fails exactly like a full one, so it is the flag and not capacity or
# length. The memory ceiling above was fitted without it anyway, so the default
# attention is the configuration those numbers actually describe.
#
# TOPOLOGY 2 train + 6 sample matches the reference. Concurrency stays at 24:
# throughput is flat from 24 to 96 while per-turn latency tracks concurrency, so
# raising it only pushes rollouts into the timeout.
set -uo pipefail

RUN="${1:-/modeling-code/karthik/abstract-remote-exps/runs/conv15-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN"

# The pod and sandbox cleanup below is indiscriminate, so starting a second run
# while one is live silently kills the first one's rollouts. Refuse instead.
/data-fast/ap-venv/bin/python \
  /modeling-code/karthik/abstract-remote-exps/poc/run_lock.py acquire "$RUN" || exit 1

export PRIME_RL_ROOT=/modeling-code/boyiliu/prime-rl
export R2E_DATASET=/data/fshu/important/swe_data/r2e_family/R2E-Gym-Subset_validgold_unique_baseline/train.jsonl
set -a; . /data-fast/cortex.env; set +a
export KUBECONFIG=/data-fast/k3s/kubeconfig.yaml
# Every job in the shared schema reports submitted_by=ADMIN, so the comment is
# the only way anyone sweeping it can tell this one is in use.
export CORTEX_JOB_COMMENT="karthik r2e-gym overfit15 convergence run -- in use, please ask before cancelling"
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

# overfit16d's 16 tasks minus pillow@e0b95724. Trailing comment is each task's
# measured pass@8, which is also its expected group spread.
TASKS=pyramid@a43abd25,coveragepy@16997254,numpy@a5322429,coveragepy@c4fc3833,tornado@37081d79,scrapy@b51b52ff,pillow@bfaa0a1f,orange3@a2c7ac74,orange3@78213643,aiohttp@274c54e4,scrapy@721df895,pyramid@24c63558,numpy@c8a09822,aiohttp@fa628a21,coveragepy@5c3d0946
#     1.00            0.75                0.50            0.50                0.38             0.38            0.38            0.38             0.38             0.38             0.25             0.25              0.25            0.25            0.00

cd /modeling-code/karthik/abstract-remote-exps
exec $PY poc/r2e_driver.py \
  --steps 20 --group 8 --concurrency 24 --seed 42 --lr 2e-6 \
  --mops 16 --send-logprobs \
  --max-turns 40 --max-pack-tokens 41000 \
  --rollout-timeout 6000 --job-ready-timeout 7200 \
  --task-ids "$TASKS" \
  --allow-content --no-std-norm --no-length-penalty --adam-beta2 0.95 \
  --no-zero-on-violation \
  --train-gpus 2 --sample-gpus 6 \
  --micro-batch 2 --gpu-mem-util 0.85 --max-num-seqs 56 --tensor-parallel 1 \
  --keepalive-interval 2400 \
  --out "$RUN" 2>&1 | tee -a "$RUN/run.log"
