#!/usr/bin/env bash
# End-to-end validation: SkyRL gsm8k GRPO recipe driven against `fake_cortex_gs`.
#
# What this proves (or breaks): the whole client chain — SkyRL launcher →
# `integrations/arctic_rl/config.py::build_rl_config` (hardcoded `backend=local`)
# → `create_arctic_rl_client` → `ARCTIC_RL_BACKEND=cortex` env-override rewrite
# → `_CortexClientShim` → `arctic_platform.client.ArcticRLClient` →
# `CortexTransport` → HTTP → local fake Cortex GS → back — actually runs a
# real training step. No mocks anywhere in the client / transport / adapter
# code paths; only the *server* is fake.
#
# Convergence is NOT validated — the fake GS returns random losses. This is
# a plumbing gate: prove the driver → wire round-trip works and both
# integrations reach the Cortex path with zero adapter change.
#
# Usage:
#     export SKYRL_ROOT=/path/to/SkyRL
#     bash tests/e2e/run_skyrl_against_mock.sh
#
# Optional env:
#     CACHE_ROOT — where uv/HF/tmp caches live (default: $HOME/.cache/cortex-e2e).
#                  Point at a scratch disk if $HOME is size-constrained; uv
#                  will pull ~15 GiB of wheels on first resolution.
#     STEPS      — training steps to run (default 3).
#     GPUS       — GPUs to split policy/generator across (default 4).

set -euo pipefail

STEPS=${STEPS:-3}
GPUS=${GPUS:-4}

# ---------------------------------------------------------------------------
# Env caches — uv will otherwise fill $HOME with vllm + torch wheels (~15 GiB).
# Override CACHE_ROOT to a scratch disk (e.g. /data-fast/$USER/cortex-e2e) if
# $HOME is size-constrained.
# ---------------------------------------------------------------------------
CACHE_ROOT=${CACHE_ROOT:-${HOME}/.cache/cortex-e2e}
export UV_CACHE_DIR=${CACHE_ROOT}/uv-cache
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python
export XDG_CACHE_HOME=${CACHE_ROOT}/cache
export HF_HOME=${HF_HOME:-${CACHE_ROOT}/hf-cache}
export PIP_CACHE_DIR=${CACHE_ROOT}/pip-cache
export TMPDIR=${TMPDIR:-${CACHE_ROOT}/tmp}
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" "$HF_HOME" \
         "$PIP_CACHE_DIR" "$TMPDIR"

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------
E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCTIC_PLATFORM_ROOT="$(cd "$E2E_DIR/../.." && pwd)"
DATA_DIR=${DATA_DIR:-${CACHE_ROOT}/gsm8k}

# SkyRL checkout — required, no default. Point this at your local clone of
# https://github.com/NovaSky-AI/SkyRL (any branch that has
# integrations/arctic_rl/ works).
if [[ -z "${SKYRL_ROOT:-}" ]]; then
    cat <<EOF >&2
ERR: SKYRL_ROOT is not set. Export it to your local SkyRL checkout, e.g.:
     export SKYRL_ROOT=/path/to/your/SkyRL
EOF
    exit 2
fi
if [[ ! -d "$SKYRL_ROOT" ]]; then
    echo "ERR: SkyRL checkout not found at SKYRL_ROOT=$SKYRL_ROOT" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Prep GSM8K parquets under $DATA_DIR if missing.
# ---------------------------------------------------------------------------
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/validation.parquet" ]]; then
    echo "[e2e] GSM8K parquets missing under $DATA_DIR; using SkyRL prep script..."
    cd "$SKYRL_ROOT"
    uv run --isolated examples/train/gsm8k/gsm8k_dataset.py --output_dir "$DATA_DIR"
fi

# ---------------------------------------------------------------------------
# Start the fake Cortex GS on a random port in the background. It runs from
# the local Arctic-Platform checkout — SkyRL's uv env doesn't need
# arctic-platform installed for the mock's sake (we're driving verl/SkyRL,
# not the mock, through the uv env).
# ---------------------------------------------------------------------------
MOCK_LOG="${CACHE_ROOT}/fake_cortex_gs.log"

MOCK_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
echo "[e2e] starting fake_cortex_gs on port $MOCK_PORT (log: $MOCK_LOG)"

# Use the arctic_platform checkout's own env (dev conda) to run the mock —
# it only needs fastapi + uvicorn + safetensors + torch which are already
# there. Don't share the SkyRL uv env with this.
(
    cd "$ARCTIC_PLATFORM_ROOT"
    # `tests.e2e` is intentionally not a package; drive the file directly.
    exec python "$E2E_DIR/fake_cortex_gs.py" --port "$MOCK_PORT" --host 127.0.0.1
) >"$MOCK_LOG" 2>&1 &
MOCK_PID=$!
cleanup() {
    if kill -0 "$MOCK_PID" 2>/dev/null; then
        echo "[e2e] stopping fake_cortex_gs (pid=$MOCK_PID)"
        kill "$MOCK_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for it to bind (openapi is served by FastAPI as soon as uvicorn is up).
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${MOCK_PORT}/openapi.json" >/dev/null; then
        echo "[e2e] fake_cortex_gs is up"
        break
    fi
    sleep 0.2
done
if ! curl -sf "http://127.0.0.1:${MOCK_PORT}/openapi.json" >/dev/null; then
    echo "ERR: fake_cortex_gs failed to start within 12s" >&2
    tail -50 "$MOCK_LOG" >&2 || true
    exit 3
fi

# ---------------------------------------------------------------------------
# Flip the launcher to Cortex via the env-var override.
# ---------------------------------------------------------------------------
export ARCTIC_RL_BACKEND=cortex
export CORTEX_BASE_URL="http://127.0.0.1:${MOCK_PORT}"
export CORTEX_DATABASE=e2e_db
export CORTEX_SCHEMA=e2e_sch
export CORTEX_ENDPOINT=cortex-training
export CORTEX_MAX_SEQ_LEN=1536

# Point uv at a local wheel of *this* Arctic-Platform checkout so `--with
# arctic-platform` in the SkyRL recipe resolves to our Cortex changes (0.1.3.dev0
# beats PyPI's 0.1.2). We do this via env vars — no edits to the recipe.
WHEEL_DIR=${WHEEL_DIR:-${CACHE_ROOT}/wheels}
if ! ls "${WHEEL_DIR}"/arctic_platform-*.whl >/dev/null 2>&1; then
    echo "[e2e] building local arctic-platform wheel into $WHEEL_DIR"
    ( cd "$ARCTIC_PLATFORM_ROOT" && uv build --wheel -o "$WHEEL_DIR" ) >/dev/null
fi
export UV_FIND_LINKS="file://${WHEEL_DIR}"
export UV_PRERELEASE=allow
export UV_INDEX_STRATEGY=unsafe-best-match

# ---------------------------------------------------------------------------
# Launch SkyRL. Its bundled recipe already handles isolated uv resolution
# for arctic-inference[vllm] + flash-attn.
# ---------------------------------------------------------------------------
cd "$SKYRL_ROOT"

RECIPE=integrations/arctic_rl/examples/run_gsm8k_grpo_4gpu.sh
if [[ ! -f "$RECIPE" ]]; then
    echo "ERR: expected recipe not found at $SKYRL_ROOT/$RECIPE" >&2
    exit 4
fi

RUN_LOG="${CACHE_ROOT}/skyrl_run.log"
echo "[e2e] launching SkyRL recipe (log: $RUN_LOG)"
echo "[e2e]   steps=${STEPS} gpus=${GPUS} mock=http://127.0.0.1:${MOCK_PORT}"

set +e
DATA_DIR="$DATA_DIR" \
    bash "$RECIPE" \
        trainer.epochs=1 \
        trainer.total_training_steps="$STEPS" \
        trainer.placement.policy_num_gpus_per_node=$((GPUS / 2)) \
        generator.inference_engine.num_engines=$((GPUS / 2)) \
        trainer.policy_mini_batch_size=4 \
        trainer.train_batch_size=64 \
        generator.n_samples_per_prompt=4 \
        trainer.eval_before_train=false \
        trainer.eval_interval=999999 \
        2>&1 | tee "$RUN_LOG"
STATUS=$?
set -e

echo "[e2e] SkyRL exit=$STATUS"
echo "[e2e] mock log tail:"
tail -30 "$MOCK_LOG" || true

exit "$STATUS"
