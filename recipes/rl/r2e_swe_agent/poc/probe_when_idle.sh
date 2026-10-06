#!/usr/bin/env bash
# Measure sampler prefill latency while no rollouts are in flight.
#
# Probing during collection measures queueing, not prefill: 24 live rollouts
# make a single request's latency swing 6-33s, which swamped the cold/warm
# comparison the prefix-cache verdict depends on. The training phase is the
# only window where the sampler is idle but still up -- collection has
# finished and the trainer has the GPUs -- so wait for it and probe there.
set -uo pipefail

RUN=${1:?usage: probe_when_idle.sh <run_dir> <gateway_url>}
URL=${2:?usage: probe_when_idle.sh <run_dir> <gateway_url>}
PY=/data-fast/ap-venv/bin/python
LOG="$RUN/run.log"
OUT="$RUN/prefix_probe.txt"

# fwd_bwd only appears once collection is done and the optimizer is running,
# but the log keeps the *previous* step's fwd_bwd lines, so matching anywhere
# in the tail fires immediately and measures collection instead -- which is
# what happened on the first attempt. Anchor to lines written from now on.
START=$(wc -l < "$LOG" 2>/dev/null || echo 0)
for _ in $(seq 1 900); do
  if tail -n "+$((START + 1))" "$LOG" 2>/dev/null | grep -q 'fwd_bwd [0-9]*/'; then
    break
  fi
  sleep 10
done

{
  echo "=== probed at $(date -u +%H:%M:%SZ), sampler idle during training ==="
  tail -n 2 "$LOG" | head -n 1
  echo "--- with routing_key (replica affinity requested) ---"
  $PY poc/probe_prefix_cache.py "$URL" 40 3 2>&1 | tail -n 10
  echo "--- without routing_key (affinity off, for contrast) ---"
  $PY poc/probe_prefix_cache.py "$URL" 40 3 none 2>&1 | tail -n 10
} >> "$OUT" 2>&1
