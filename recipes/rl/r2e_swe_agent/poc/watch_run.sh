#!/usr/bin/env bash
# Emit one line per completed step, and shout if the run stops moving.
#
# This run has died twice without anyone noticing for hours: once when the
# sampler had no room for the weight-sync buffer, once when a fire-and-forget
# launch was SIGKILLed at its hard timeout. Both cost a full collection window
# (about two hours of GPU) before the silence was investigated. The cost of
# polling a log file is nothing, so the asymmetry is obvious.
#
# usage: watch_run.sh <run_dir> [stall_seconds]
set -uo pipefail

RUN=${1:?usage: watch_run.sh <run_dir> [stall_seconds]}
LOG="$RUN/run.log"
HIST="$RUN/history.json"
# A step is two hours, and within it the log only moves when a rollout lands,
# which is minutes apart. Half an hour of total silence is well outside normal
# and still catches a wedge long before it costs a whole step.
STALL=${2:-1800}
POLL=60

last_steps=-1
last_trains=-1
while true; do
  if [ ! -f "$LOG" ]; then
    echo "WATCH-DEAD no log at $LOG"
    exit 1
  fi

  now=$(date +%s)
  mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
  quiet=$((now - mtime))

  steps=$(python3 - "$HIST" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    print(len(json.load(open(sys.argv[1]))))
except Exception:
    print(0)
PY
)

  if [ "$steps" -gt "$last_steps" ] && [ "$last_steps" -ge 0 ]; then
    line=$(python3 - "$HIST" <<'PY' 2>/dev/null
import json, sys
h = json.load(open(sys.argv[1]))[-1]
t = h.get("mean_turns")
print(f"step {h['step']} reward={h['mean_reward']:.3f} "
      f"pass={h['pass_rate']:.3f} solved={h['solved']}/{h['n']} "
      f"zeroed={h['zeroed']}" + (f" turns={t:.0f}" if t else ""))
PY
)
    echo "WATCH-STEP $line"
  fi
  [ "$steps" -gt "$last_steps" ] && last_steps=$steps

  # Optimizer metrics are a separate event from collection metrics, because
  # they happen at a different time: history.json is written when the rollouts
  # land, and the optimizer does not run until after that. Reporting them
  # together means always reporting the previous step's numbers, or none.
  #
  # Entropy is the one to watch at a ten-fold learning rate. The failure mode
  # is not a crash but a policy collapsing onto a single output and quietly
  # ceasing to explore, which reward does not reveal for several more steps.
  # ``grep -c`` prints 0 and exits 1 when nothing matches, so an ``|| echo 0``
  # appends a second zero and every later comparison fails on "0\n0".
  trains=$(grep -c "optimizer step, importance_weight" "$LOG" 2>/dev/null)
  trains=${trains:-0}
  if [ "$trains" -gt "$last_trains" ] && [ "$last_trains" -ge 0 ]; then
    # One line per off-policy inner step. The importance weight is the whole
    # point of replay: at the first inner step it should sit just off 1.0,
    # proving the sampler log-probs arrived on the right alignment, and it
    # should drift as the later steps train away from the sampling policy. A
    # weight far from 1.0 on the *first* inner step means misalignment, not
    # drift, and the run is worthless.
    m=$(grep "optimizer step, importance_weight" "$LOG" | tail -1)
    which=$(echo "$m" | grep -o "step [0-9]*\.[0-9]*" | tail -1)
    iw=$(echo "$m" | grep -o "importance_weight=[0-9.e+-]*" | cut -d= -f2)
    cr=$(echo "$m" | grep -o "clip_ratio=[0-9.e+-]*" | cut -d= -f2)
    kl=$(echo "$m" | grep -o "approx_kl=[0-9.e+-]*" | cut -d= -f2)
    gn=$(echo "$m" | grep -o "grad_norm=[0-9.e+-]*" | cut -d= -f2)
    en=$(grep "train metrics" "$LOG" | tail -1 \
         | grep -o "'entropy': [0-9.e+-]*" | cut -d' ' -f2)
    echo "WATCH-TRAIN ${which:-?} importance_weight=${iw:-?} clip_ratio=${cr:-?}" \
         "approx_kl=${kl:-?} grad_norm=${gn:-?} entropy=${en:-?}"
  fi
  [ "$trains" -gt "$last_trains" ] && last_trains=$trains

  if [ "$quiet" -gt "$STALL" ]; then
    echo "WATCH-STALL no log output for ${quiet}s (limit ${STALL}s) after $steps steps"
    exit 2
  fi

  if ! pgrep -f "r2e_driver.py" >/dev/null 2>&1; then
    # The driver runs in the sandbox pod, not here, so its absence locally
    # proves nothing. Only the log going cold is evidence, and that is the
    # check above.
    :
  fi

  sleep "$POLL"
done
