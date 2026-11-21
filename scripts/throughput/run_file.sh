#!/bin/bash
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Measure one catalog training file's SFT throughput on a local DSS gateway (README.md):
#
#   [RUN_HOSTFILE=<file>] [STEPS=12] run_file.sh <run name> <catalog training json> [run_sft.py arguments]
#
# Sequence length and rows per step come from the file (max_seq_len, train_batch_size), the checkpoint from
# $BAKED_ROOT/<HF id>. A file with more than 8 GPUs spans the first nodes of RUN_HOSTFILE ("<host> slots=8" per line);
# start it on the first of them. The nodes must be idle. Afterwards the gateway is stopped, and whatever it started is
# killed on every node. Output: $OUT_DIR/results/<name>.json, $OUT_DIR/logs/<name>.log and <name>.gateway.log, and a
# line in $OUT_DIR/summary.txt.
set -u
D=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=SCRIPTDIR/env.sh
source "$D/env.sh"
name=${1:?usage: run_file.sh <run name> <catalog training json> [run_sft.py arguments]}
cfg=$(readlink -f "${2:?catalog training json}")
shift 2
[ -f "$cfg" ] || die "no such file: $cfg"
read -r model seq batch gpus < <("$VENV/bin/python" -I -c '
import json, sys

d = json.load(open(sys.argv[1]))
s = next(x for x in d["sub_job_configs"] if x["job_type"] == "training")
t = s["training_config"]
print(s["model_name"], t["max_seq_len"], t.get("train_batch_size", 1), t["n_gpus"])' "$cfg")
[ -n "${gpus:-}" ] || die "cannot read the training sub-job of $cfg"

nodes=$(((gpus + 7) / 8))
if [ "$nodes" -gt 1 ]; then
  [ -f "${RUN_HOSTFILE:-}" ] || die "$name needs $gpus GPUs: set RUN_HOSTFILE ('<host> slots=8' per node)"
  hosts=$(awk 'NF { print $1 }' "$RUN_HOSTFILE" | head -n "$nodes")
  [ "$(echo "$hosts" | wc -l)" -eq "$nodes" ] || die "$RUN_HOSTFILE has fewer than $nodes nodes"
  is_local "$(echo "$hosts" | head -1)" || die "start on $RUN_HOSTFILE's first node: the gateway runs here"
else
  hosts=$(hostname -I | awk '{ print $1 }')
fi
log=$OUT_DIR/logs/$name.log glog=$OUT_DIR/logs/$name.gateway.log hostfile=$OUT_DIR/logs/$name.hostfile
for h in $hosts; do echo "$h slots=8"; done > "$hostfile"
for h in $hosts; do
  on_node "$h" node_idle || die "$h is busy (GPU processes, or a DSS gateway or zone): a run needs its nodes to itself"
  on_node "$h" bake_model "$BAKED_ROOT" "$MODEL_ROOT" "$model" || exit 1
done
if curl -s -o /dev/null "localhost:$PORT/health"; then die "port $PORT is taken: another gateway is running"; fi

echo "$(date -Is) $name: $model $seq x $batch rows on $gpus GPUs ($cfg)"
{
  echo "host $(hostname) nodes=$nodes AP=$AP DSSP=$DSSP DSP=${DSP:-venv} CORTEX=$CORTEX"
  for t in "$AP" "$DSSP" ${DSP:+"$DSP"} "$CORTEX"; do
    echo "$t $(git -C "$t" rev-parse --abbrev-ref HEAD) $(git -C "$t" rev-parse --short HEAD)" \
      "dirty=$(git -C "$t" status --short | wc -l)"
  done
} > "$log" 2>&1

# Every process the gateway starts, here and on the other nodes (DSS propagates DSS_* variables), carries the tag.
tag=$name-$(hostname)-$$
gw=
# shellcheck disable=SC2329  # the EXIT trap runs it
cleanup() {
  [ -n "$gw" ] || return 0
  kill "$gw" 2> /dev/null  # SIGTERM: the gateway shuts its zones down
  for _ in $(seq 90); do
    kill -0 "$gw" 2> /dev/null || break
    sleep 2
  done
  for h in $hosts; do
    if is_local "$h"; then kill_run "$tag" "$gw"; else on_node "$h" kill_run "$tag"; fi  # setsid: its session is $gw
  done
  for _ in $(seq 30); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0
    sleep 2
  done
  echo "$(hostname): GPUs still busy after $name" >&2
}
trap cleanup EXIT
trap 'exit 130' INT TERM

cd "$SCRATCH" || die "cannot cd to $SCRATCH"
DSS_THROUGHPUT_RUN=$tag setsid nohup "$VENV/bin/python" -m dss.ray_dss.gateway_server --host 0.0.0.0 --port "$PORT" \
  --backend gpu --placement local --hostfile "$hostfile" > "$glog" 2>&1 < /dev/null &
gw=$!
until curl -s -o /dev/null "localhost:$PORT/health"; do
  kill -0 "$gw" 2> /dev/null || die "the gateway exited: $glog"
  sleep 3
done
"$VENV/bin/python" -I "$D/run_sft.py" --gateway "http://localhost:$PORT" --cortex "$CORTEX" --job-config "$cfg" \
  --model-name "$BAKED_ROOT/$model" --mode text ${CORPUS:+--corpus "$CORPUS"} --seq-len "$seq" --batch-size "$batch" \
  --steps "$STEPS" --warmup 2 --set step_peak_memory_log=true "$@" --out "$OUT_DIR/results/$name.json" \
  >> "$log" 2>&1 < /dev/null
rc=$?
err=$(grep -a -h -o 'Tried to allocate.\{0,60\}\|Error.\{0,200\}' "$log" "$glog" | tail -1)
echo "$name: $(grep -a SUMMARY "$log" | cut -c1-220) $err" >> "$OUT_DIR/summary.txt"
grep -a "^$name: " "$OUT_DIR/summary.txt" | tail -1 | cut -c1-300
exit $rc
