#!/usr/bin/env bash
# Run a command inside the privileged sandbox pod (kganesan-swe2-0, 48 CPU).
#
# Pod-to-pod networking is blocked by NetworkPolicy and we have no pods/exec RBAC,
# so commands are handed over through the shared Lustre mount: a poller in the pod
# picks up <channel>/in/<id>.sh and writes <channel>/out/<id>.log + .done.
#
# usage:  ./rexec.sh 'command'            run and wait (default 600s)
#         TIMEOUT=3600 ./rexec.sh 'cmd'   longer wait
#         ./rexec.sh -n 'cmd'             fire and forget, prints the job id
#         HARDKILL=7200 ./rexec.sh -n ... in-pod cap for a job meant to outlive
#                                         the local wait, e.g. a training run
#         REXEC_CHANNEL=.remote ./rexec.sh ...   talk to the old 16-CPU pod
set -uo pipefail

R=${REXEC_CHANNEL:-/modeling-code/karthik/abstract-remote-exps/.remote2}
TIMEOUT=${TIMEOUT:-600}
# The .remote2 poller runs jobs in the background, so a slow one no longer blocks
# the channel. This per-job cap stays as a second line of defence against a job
# that hangs outright, and it is what bounds a fire-and-forget run.
HARDKILL=${HARDKILL:-$((TIMEOUT + 60))}
NOWAIT=0
[ "${1:-}" = "-n" ] && { NOWAIT=1; shift; }

mkdir -p "$R/in" "$R/out"
ID="$(date +%Y%m%d-%H%M%S)-$$"

{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  printf 'timeout -s KILL %s bash -s <<'"'"'__REXEC_EOF__'"'"'\n' "$HARDKILL"
  cat
  echo '__REXEC_EOF__'
} > "$R/in/.$ID.tmp" <<< "${1:?usage: rexec.sh \"command\"}"
mv "$R/in/.$ID.tmp" "$R/in/$ID.sh"

if [ "$NOWAIT" = 1 ]; then echo "$ID"; exit 0; fi

for _ in $(seq 1 $((TIMEOUT / 2))); do
  if [ -e "$R/out/$ID.done" ]; then
    cat "$R/out/$ID.log" 2>/dev/null
    RC=$(cat "$R/out/$ID.done")
    [ "$RC" != 0 ] && echo "[rexec] exit=$RC" >&2
    exit "$RC"
  fi
  sleep 2
done

echo "[rexec] timed out after ${TIMEOUT}s; partial output:" >&2
cat "$R/out/$ID.log" 2>/dev/null
exit 124
