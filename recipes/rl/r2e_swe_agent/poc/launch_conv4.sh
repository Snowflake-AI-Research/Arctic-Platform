#!/usr/bin/env bash
# Overfit 4 prompts to convergence -- the fast version of launch_conv15.sh.
#
# WHY 4 AND NOT 15. Collection is 76% of a step and its wall clock tracks
# rollout count, not concurrency (throughput measured flat from concurrency 24
# to 96 while per-request latency rose in proportion). Dropping 15x8=120
# rollouts to 4x8=32 therefore takes a step from ~2.9h to ~0.8h.
#   Each step's reward is noisier for it -- sigma goes from ~0.025 to ~0.087 --
#   but detecting a trend improves anyway, because the standard error of a
#   fitted slope falls as n^1.5: 4x the steps against 3.5x the noise is ~2.3x
#   better trend detection per hour.
#   The real argument is that 4 prompts are memorisable. On 15 prompts the
#   honest reading after two steps was 0.325 -> 0.375 at p~0.10, i.e. an
#   argument about statistics. Four prompts can plausibly reach reward 1.0,
#   which is a curve rather than an argument.
#
# TASK CHOICE. GRPO gradient comes only from groups with reward spread and is
# maximised near p=0.5, so every task here is mid-range as measured on the
# 15-prompt run (step 0 / step 1 solves out of 8):
#   coveragepy@16997254  4/8 -> 5/8
#   coveragepy@c4fc3833  4/8 -> 4/8
#   numpy@a5322429       6/8 -> 6/8
#   numpy@c8a09822       3/8 -> 3/8
# Baseline is therefore ~0.55 with headroom to 1.0. Deliberately excluded:
# coveragepy@5c3d0946 (0/8, nothing to reinforce) and pyramid@a43abd25 (7/8,
# nearly saturated already).
#
# KNOWN RISK. With only 4 groups, one task saturating to 8/8 removes a quarter
# of the gradient at once, where 15 tasks gave redundancy. If reward stalls
# while individual tasks sit at 8/8, raise --group to 16 (still ~1.8x faster
# than the 15-prompt run) rather than adding tasks back.
#
# Everything else is inherited from launch_conv15.sh and was measured there:
# turn cap 40, the 41,000-token pack guard fitted from four OOMs, mops 16
# (clip_ratio ended step 0 at 0.015 against a 0.2 intervention threshold),
# lr 2e-6, no --attn-impl, and no reward zeroing.
set -uo pipefail

RUN="${1:-/modeling-code/karthik/abstract-remote-exps/runs/conv4-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN"

# The pod and sandbox cleanup below is indiscriminate, so starting a second run
# while one is live silently kills the first one's rollouts. Refuse instead.
/data-fast/ap-venv/bin/python \
  /modeling-code/karthik/abstract-remote-exps/poc/run_lock.py acquire "$RUN" || exit 1

export PRIME_RL_ROOT=/modeling-code/boyiliu/prime-rl
export R2E_DATASET=/data/fshu/important/swe_data/r2e_family/R2E-Gym-Subset_validgold_unique_baseline/train.jsonl
set -a; . /data-fast/cortex.env; set +a
export KUBECONFIG=/data-fast/k3s/kubeconfig.yaml
export CORTEX_JOB_COMMENT="karthik r2e-gym overfit4 convergence run -- in use, please ask before cancelling"
PY=/data-fast/ap-venv/bin/python
K="/data-fast/k3s/bin/k3s kubectl"

$K get pods -n default --no-headers 2>/dev/null | awk '/^r2e-/{print $1}' \
  | xargs -r -n20 $K delete pod -n default --grace-period=0 --force >/dev/null 2>&1
$PY /modeling-code/karthik/abstract-remote-exps/poc/reap_sandboxes.py --max-age 0 >/dev/null 2>&1

nohup $PY /modeling-code/karthik/abstract-remote-exps/poc/reap_sandboxes.py \
  --loop 300 --max-age 7200 > "$RUN/reaper.log" 2>&1 &
echo "reaper pid $!" | tee "$RUN/reaper.pid"

TASKS=coveragepy@16997254,coveragepy@c4fc3833,numpy@a5322429,numpy@c8a09822

cd /modeling-code/karthik/abstract-remote-exps
exec $PY poc/r2e_driver.py \
  --steps 40 --group 8 --concurrency 24 --seed 42 --lr 2e-6 \
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
