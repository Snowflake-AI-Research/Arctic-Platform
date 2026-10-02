#!/usr/bin/env bash
#
# Apply the rollout-replay patches to the active vLLM install.
#
# Adds a `max_tokens_n: list[int] | None` field to `SamplingParams` and
# threads it through `parallel_sampling.ParentRequest._get_child_sampling_params`
# so that, when `n > 1`, each child sample can have its own `max_tokens`
# (used to faithfully replay a recorded rollout trace).
#
# Targets vLLM 0.18.0. The patches apply against pristine v0.18 source
# files; for any other version you may need to regenerate them with
# `diff -u`.
#
# Usage:  bash patch_sampling.sh
#
# The script is idempotent: re-running it after a successful patch is a
# no-op (the second call exits 0 with "already patched" rather than
# corrupting the file).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_SAMPLING_PARAMS="$SCRIPT_DIR/sampling_params.patch"
PATCH_PARALLEL_SAMPLING="$SCRIPT_DIR/parallel_sampling.patch"

VLLM_PATH="$(pip show vllm | awk '/^Location: /{print $2}')"
if [ -z "$VLLM_PATH" ]; then
  echo "Error: could not find vLLM in current env" >&2
  exit 1
fi
VLLM_VERSION="$(pip show vllm | awk '/^Version: /{print $2}')"
echo "vLLM path:    $VLLM_PATH"
echo "vLLM version: $VLLM_VERSION  (patches were generated against 0.18.0)"

apply_one() {
  local target="$1" patch_file="$2"
  if [ ! -f "$target" ]; then
    echo "Error: target file does not exist: $target" >&2
    exit 1
  fi
  # Already-applied check.
  if patch --dry-run --reverse --silent "$target" < "$patch_file" \
      >/dev/null 2>&1; then
    echo "  [skip] already patched: $target"
    return 0
  fi
  if ! patch --dry-run --silent "$target" < "$patch_file" >/dev/null 2>&1
  then
    echo "Error: patch does not apply cleanly to $target. The vLLM" >&2
    echo "       source has likely diverged from 0.18.0; regenerate" >&2
    echo "       the patch with diff -u." >&2
    exit 1
  fi
  patch "$target" < "$patch_file"
}

apply_one "$VLLM_PATH/vllm/sampling_params.py"           "$PATCH_SAMPLING_PARAMS"
apply_one "$VLLM_PATH/vllm/v1/engine/parallel_sampling.py" "$PATCH_PARALLEL_SAMPLING"

echo "Done."
