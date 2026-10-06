#!/usr/bin/env bash
# Keep launching band8 until Cortex actually places the job on GPUs.
#
# Two consecutive launches died with reason=placement_timeout: the job is
# accepted, waits for 8 H200s, never gets them, and the driver exits ~45min
# later having collected nothing. That is a capacity race, not a config error,
# and the only thing that fixes it is being in the queue when GPUs free up.
#
# So this retries. Each attempt gets a fresh run dir, because a failed attempt
# leaves a run.log containing a traceback and reusing the dir would make the
# next attempt look failed too.
#
# An attempt is judged by the first rollout line, not by job status: the whole
# point is that "job accepted" means nothing here. Once a rollout lands, GPUs
# are serving and the run is handed off to the normal monitor.
#
#   ./launch_until_placed.sh [max_attempts] [wait_minutes_per_attempt]
set -uo pipefail

ROOT=/modeling-code/karthik/abstract-remote-exps
cd "$ROOT"
ATTEMPTS=${1:-12}
WAIT_MIN=${2:-50}
PY=/data-fast/ap-venv/bin/python

for a in $(seq 1 "$ATTEMPTS"); do
  RUN="$ROOT/runs/band8c-$(date +%Y%m%d-%H%M%S)"
  echo "[retry] attempt $a/$ATTEMPTS -> $(basename "$RUN")"

  # A dead attempt leaves the lock claimed; the launcher would then refuse.
  ./rexec.sh "cd $ROOT && $PY poc/run_lock.py release" >/dev/null 2>&1

  echo "$RUN" > /tmp/band8_run_dir
  HARDKILL=172000 ./rexec.sh -n "cd $ROOT && bash poc/launch_band8.sh '$RUN'" >/dev/null

  placed=0
  for _ in $(seq 1 $((WAIT_MIN * 2))); do
    sleep 30
    [ -f "$RUN/run.log" ] || continue
    if grep -qE '^[[:space:]]+\[[0-9]+/64\]' "$RUN/run.log" 2>/dev/null; then
      placed=1; break
    fi
    # Any terminal state, not just 'failed': a job the fleet cannot place comes
    # back 'cancelled', which read as "still waiting" and parked the loop for
    # the whole window against an already-dead job.
    if grep -qE "placement_timeout|terminal state '[a-z]+'" "$RUN/run.log" 2>/dev/null; then
      echo "[retry] attempt $a: placement failed $(grep -oE "terminal state '[a-z]+'" "$RUN/run.log" | tail -n 1)"
      break
    fi
    if grep -qE 'address already in use' "$RUN/run.log" 2>/dev/null; then
      echo "[retry] attempt $a: PORT CONFLICT -- another driver is alive, stopping"
      exit 2
    fi
  done

  if [ "$placed" = 1 ]; then
    echo "[retry] PLACED on attempt $a: $RUN"
    exit 0
  fi

  # Leave nothing behind that the next attempt could trip over.
  ./rexec.sh "pkill -f r2e_driver.py; pkill -f 'reap_sandboxes.py --loop'" >/dev/null 2>&1
  sleep 60
done

echo "[retry] GAVE UP after $ATTEMPTS attempts -- no GPU capacity"
exit 1
