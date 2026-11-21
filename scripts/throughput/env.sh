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

# The environment of a local DSS gateway and the throughput driver, sourced by run_file.sh and run_catalog.sh.
# Variables (README.md): AP, DSSP, CORTEX required; DSP, VENV, CORPUS, SCRATCH, BAKED_ROOT, MODEL_ROOT, OUT_DIR,
# STEPS, PORT, POD_ENV optional. Remote nodes get what they need through DSS's env propagation.

die() {
  echo "$*" >&2
  exit 1
}

# The pod's own settings (NCCL / EFA, CUDA_HOME, cache paths) first, so that they win over the defaults below.
if [ -n "${POD_ENV:-}" ]; then
  # shellcheck disable=SC1090
  source "$POD_ENV" || die "POD_ENV=$POD_ENV failed"
fi

for v in AP DSSP CORTEX; do
  [ -d "${!v:-}" ] || die "set $v (README.md): '${!v:-}' is not a directory"
done
AP=$(readlink -f "$AP") DSSP=$(readlink -f "$DSSP") CORTEX=$(readlink -f "$CORTEX") DSP=${DSP:-} CORPUS=${CORPUS:-}
if [ -n "$DSP" ]; then
  [ -d "$DSP" ] || die "DSP=$DSP is not a directory"
  DSP=$(readlink -f "$DSP")
fi
VENV=${VENV:-${VIRTUAL_ENV:-}}
[ -x "$VENV/bin/python" ] || die "set VENV to the DSS venv (or activate it): '$VENV/bin/python' is missing"
if [ -n "$CORPUS" ]; then
  [ -f "$CORPUS" ] || die "CORPUS=$CORPUS is not a file"
  CORPUS=$(readlink -f "$CORPUS")
fi
export AP DSSP DSP CORTEX VENV CORPUS

# Node-local storage: the same path on every node of a job.
export SCRATCH=${SCRATCH:-$HOME/.cache/dss-throughput}
export BAKED_ROOT=${BAKED_ROOT:-$SCRATCH/baked} MODEL_ROOT=${MODEL_ROOT:-$SCRATCH/models}
OUT_DIR=$(readlink -m "${OUT_DIR:-$PWD/throughput-out}")
export OUT_DIR STEPS=${STEPS:-12} PORT=${PORT:-8003}
mkdir -p "$SCRATCH/cache" "$OUT_DIR/results" "$OUT_DIR/logs" || die "cannot create $SCRATCH or $OUT_DIR"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
# The source trees shadow the venv's installs; anything already on PYTHONPATH (e.g. the UCCL-EP overlay) follows them.
export PYTHONPATH=${DSP:+$DSP:}$AP:$DSSP${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/cache/triton}
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$SCRATCH/cache/tilelang}
# FlashInfer's JIT build files name the venv's include paths: one cache per venv, or each venv rebuilds every module.
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$SCRATCH/cache/$(basename "$VENV")}
# A DeepSpeed source tree JIT-builds its ops with its own paths: one extensions cache per tree.
if [ -n "$DSP" ] && [ -z "${TORCH_EXTENSIONS_DIR:-}" ]; then
  TORCH_EXTENSIONS_DIR=$SCRATCH/cache/torch_extensions/$(echo "$DSP" | md5sum | cut -c1-12)
  export TORCH_EXTENSIONS_DIR
fi
# The job loads its checkpoint by local path (BAKED_ROOT), not from DSS's managed model cache.
export DSS_ALLOW_REMOTE_MODELS=${DSS_ALLOW_REMOTE_MODELS:-1}
export DSS_WEIGHT_CONVERSION_CACHE_DIR=${DSS_WEIGHT_CONVERSION_CACHE_DIR:-$SCRATCH/weight-conversion-cache}
export DSS_WEIGHT_CONVERSION_CACHE_SCOPE=${DSS_WEIGHT_CONVERSION_CACHE_SCOPE:-node}
export DSS_LOCAL_CHECKPOINT_DIR=${DSS_LOCAL_CHECKPOINT_DIR:-$SCRATCH/dss-checkpoints}
# The first conversion of a large MoE checkpoint and the first step's kernel builds outlast DSS's defaults.
export DSS_TRAINING_CREATE_TIMEOUT_SECONDS=${DSS_TRAINING_CREATE_TIMEOUT_SECONDS:-7200}
export DSS_DIST_TIMEOUT_SECONDS=${DSS_DIST_TIMEOUT_SECONDS:-3600}
# Remote nodes' zones get PYTHONPATH and the DSS_ / TORCH_ / NCCL_ / HF_ variables by default; these too.
export DSS_PROPAGATE_ENV_EXTRA=${DSS_PROPAGATE_ENV_EXTRA:-'["CUDA_HOME", "PATH", "TILELANG_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "OMP_NUM_THREADS", "DS_SKIP_CUDA_CHECK"]'}
export DSS_PROPAGATE_ENV_PREFIXES_EXTRA=${DSS_PROPAGATE_ENV_PREFIXES_EXTRA:-'["MALLOC_", "ULTRA_EP_", "EP_"]'}
# torch is built for one CUDA minor version and images often ship another; DeepSpeed's op builder refuses that.
export DS_SKIP_CUDA_CHECK=${DS_SKIP_CUDA_CHECK:-1}
# Some images set this in the container's environment, which makes every FP32 cuBLAS matmul TF32; DSS forwards TORCH_*
# variables to every worker. Keep FP32 matmuls FP32.
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

ssh_node() {
  ssh -n -o BatchMode=yes -o LogLevel=ERROR "$@"
}

is_local() {
  case " $(hostname -I) $(hostname) $(hostname -f 2>/dev/null) localhost 127.0.0.1 " in
    *" $1 "*) return 0 ;;
  esac
  return 1
}

# The helpers below run on any node: on_node <host> <helper> [args].
node_idle() {  # no GPU process, no DSS gateway or zone
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] &&
    ! pgrep -f "[d]ss.ray_dss.(gateway_server|zones.server_worker)" > /dev/null
}

run_procs() {  # <tag>: this user's processes that carry DSS_THROUGHPUT_RUN=<tag> (a gateway and what it started)
  local p
  for p in $(pgrep -u "$(id -u)"); do
    grep -qxzF "DSS_THROUGHPUT_RUN=$1" "/proc/$p/environ" 2> /dev/null && echo "$p"
  done
  return 0
}

# <tag> [session id...]: the tagged processes, the rest of their sessions (Ray's agents drop the environment) and the
# given sessions; never the caller's own session.
kill_run() {
  local tag=$1 p s sids own
  shift
  p=$(run_procs "$tag")
  sids="$*"
  # shellcheck disable=SC2086
  [ -z "$p" ] || sids="$sids $(ps -o sid= -p "$(echo $p | tr ' ' ',')")"
  own=$(ps -o sid= -p $$ | tr -d ' ')
  for s in $sids; do
    [ "$s" = "$own" ] || p="$p $(pgrep -u "$(id -u)" -s "$s")"
  done
  # shellcheck disable=SC2086
  p=$(echo $p | tr ' ' '\n' | sort -u)
  if [ -n "$p" ]; then
    echo "$(hostname): killing $(echo "$p" | wc -l) leftover processes of $tag" >&2
    # shellcheck disable=SC2086
    kill -9 $p 2> /dev/null
  fi
  return 0
}

bake_model() {  # <baked root> <model root> <HF id>: link <model root>/<name> to <baked root>/<HF id> if it is missing
  local dst=$1/$3 src=$2/${3#*/}
  [ -f "$dst/config.json" ] && return 0
  [ -f "$src/config.json" ] || {
    echo "no checkpoint for $3 on $(hostname): download it to $dst or $src" >&2
    return 1
  }
  mkdir -p "$dst" && cp -rs "$(readlink -f "$src")/." "$dst/"
}

on_node() {
  local h=$1
  shift
  if is_local "$h"; then
    "$@"
  else
    ssh_node "$h" "$(declare -f node_idle run_procs kill_run bake_model); $(printf '%q ' "$@")"
  fi
}
