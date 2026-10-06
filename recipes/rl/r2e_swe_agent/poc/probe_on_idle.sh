#!/usr/bin/env bash
# Measure sampler latency during a training window, when nothing else is using it.
#
# Per-request latency under live collection is mostly queue wait, which is why
# a tiny prompt and a 40k one cost the same. The number that separates queue
# wait from an irreducible floor can only be taken while the sampler is idle,
# and the only such window is the forward/backward phase -- a few minutes,
# unannounced, hours apart. So watch for it rather than trying to catch it.
set -uo pipefail

RUN="${1:?usage: probe_on_idle.sh <run_dir> <base_url> [out]}"
URL="${2:?}"
OUT="${3:-/tmp/idle_probe.txt}"
PY=/data-fast/ap-venv/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "waiting for a training window in $RUN" > "$OUT"

# fwd_bwd 1/N is printed once collection is done and training has begun.
tail -Fn0 "$RUN/run.log" 2>/dev/null | while read -r line; do
  case "$line" in
    *"fwd_bwd 1/"*)
      {
        echo "=== idle probe at $(date -u +%H:%M:%S) ==="
        echo "$line"
        "$PY" "$HERE/probe_concurrency.py" "$URL" 8
      } >> "$OUT" 2>&1
      break
      ;;
  esac
done
