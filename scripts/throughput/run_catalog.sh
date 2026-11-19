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

# Run a plan (plan.py's output) on one node set, one line after another, each once its nodes are idle:
#
#   [RUN_HOSTFILE=<file, this node first>] run_catalog.sh <plan>
#
# A line needing more nodes than RUN_HOSTFILE has (one without it) is skipped: run it on a bigger set. A run whose
# $OUT_DIR/results/<name>.json exists is not repeated.
set -u
D=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=SCRIPTDIR/env.sh
source "$D/env.sh"
plan=${1:?usage: run_catalog.sh <plan>}
if [ -n "${RUN_HOSTFILE:-}" ]; then
  hosts=$(awk 'NF { print $1 }' "$RUN_HOSTFILE")
else
  hosts=$(hostname -I | awk '{ print $1 }')
fi
have=$(echo "$hosts" | wc -l)
idle() {  # <nodes>: the first <nodes> hosts are idle
  local h
  for h in $(echo "$hosts" | head -n "$1"); do on_node "$h" node_idle || return 1; done
}
while read -r name nodes cfg rest; do
  case $name in '' | '#'*) continue ;; esac
  if [ "$nodes" -gt "$have" ]; then
    echo "$(date -Is) skip $name: needs $nodes nodes"
    continue
  fi
  if [ -f "$OUT_DIR/results/$name.json" ]; then
    echo "$(date -Is) skip $name: done"
    continue
  fi
  until idle "$nodes"; do sleep 30; done
  echo "$(date -Is) start $name"
  # shellcheck disable=SC2086  # the plan's extra arguments
  "$D/run_file.sh" "$name" "$cfg" $rest < /dev/null
done < "$plan"
