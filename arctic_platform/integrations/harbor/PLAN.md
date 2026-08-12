# Plan — any Harbor agent trains on Cortex

Status of the Harbor <-> Cortex integration and what's left to make any
Harbor `BaseAgent` — not just the reference `CortexRLAgent` this package
ships — drive a training loop.

## What ships in this PR

| Piece | File(s) | Status |
| --- | --- | --- |
| Post-training data contract | `models.py` | Done. `Rollout` carries `loss_mask` for multi-turn. |
| RFC backend impl (`connect`/`generate`/`train`/`deploy_inference`) | `backend.py` | Done. E2E on Cortex Training. |
| Harbor `BaseEnvironment` (Docker-less dev) | `env.py` | Done. |
| Reference `BaseAgent` (native reconnect-config path) | `agent.py` | Done. |
| Multi-turn rollout flatten + per-token loss mask | `adapter.py` | New. Earlier code trained only on turn 0. |
| E2E CLI | `train.py` (`harbor-cortex-train`) | Done. Native mode validated E2E; OpenAI-compat mode wired + unit-tested. |
| Multi-seed aggregator | `aggregate.py` (`harbor-cortex-aggregate`) | Done. |
| OpenAI-compat HTTP surface | `arctic_platform/openai_compat.py` + wired into `arctic_platform/rl/http_server.py` | **New.** `/v1/chat/completions`, `/v1/completions`, `/v1/models` — any OpenAI-SDK client (Terminus 2 via LiteLLM, LangChain, the openai CLI) samples from the same sub-job that `/generate` serves, no vLLM HTTP subprocess. |
| Auto-derive `--sampling-api-base` | `train.py::_derive_openai_base_url` | New. `--sampling-api-base auto` reads the client's transport (on-prem: `http://host:port/v1`). |
| vLLM token-id echo (`prompt_token_ids`, per-choice `token_ids`) | `arctic_platform/openai_compat.py` | New. Emitted on chat + text completions (non-stream and terminal SSE frame). Required for Harbor's `LiteLLM._extract_token_ids` — without them `RolloutDetail` has no token ids and the adapter drops the trial. |
| Unit tests for the router | `tests/openai_compat/test_openai_compat.py` | 23 tests: parameter mapping, `n>1`, `stop`, streaming SSE, error paths, tokenizer-without-chat-template, token-id echo (chat, text, token-id prompt, streaming). |
| Real OpenAI-SDK contract test | `tests/openai_compat/test_real_openai_sdk.py` | 4 tests: live uvicorn subprocess + `openai` client hitting chat/completions (non-stream + stream), completions, models list. |
| Harbor + LiteLLM integration test | `tests/openai_compat/test_harbor_litellm_integration.py` | Live server + real `harbor.llms.lite_llm.LiteLLM(collect_rollout_details=True)` — asserts `response.completion_token_ids` and `response.prompt_token_ids` come back non-empty. Closes the client-side loop end-to-end without GPUs. |
| Adapter tests | `tests/integrations/harbor/test_adapter_multiturn.py` | 6 tests for multi-turn flatten. |

## How the OpenAI-compat surface works

Cortex's sampling sub-job runs `AsyncLLM` in-process via
`ReplicaPool` — there's no separate vLLM HTTP server to proxy to. The
router in `arctic_platform/openai_compat.py` translates OpenAI requests
straight into `pool.generate()` calls:

* `/v1/chat/completions` renders the message list through the tokenizer's
  chat template (loaded once at `/initialize`), calls `pool.generate`, and
  formats the response.
* `/v1/completions` passes the prompt through (`str`, `list[int]`, or
  batched `list[list[int]]`) with no template.
* `/v1/models` returns the currently-loaded sampling model; empty list
  before init so LiteLLM's discovery probe doesn't give up.
* `stream=True` runs the full generation, then replays the completed
  text as SSE deltas in OpenAI's chunked shape. Client contract is
  correct; first-token latency is not. Incremental streaming needs a
  delta-yielding surface on `ReplicaPool` — the router is written so
  swapping in that surface is a one-function change.

Because we route through `ReplicaPool`, scheduling, prefix caching,
tensor-parallel fan-out, and sleep/wake continue to work exactly as on
the RL-shaped `/generate` route. Both routes share the same worker
processes.

## Blocking work outside this PR

### Cortex sub-job routing to the OpenAI URL

`arctic_platform.rl.http_server` serves both `/generate` (RL wire) and
`/v1/*` (OpenAI wire) on the same port. On-prem you already get both:

```
http://localhost:7000/generate      # RL clients
http://localhost:7000/v1/*          # OpenAI clients
```

On Cortex, clients today reach the parent job via SnowAPI
(`https://<host>/api/v2/.../cortex-training/<parent-job-id>/<op>`) which
routes to sub-jobs by op name. The public sub-job URL shape
`https://<cortex>/sub-jobs/<sampling-sub-job-id>/v1/*` — the one an
OpenAI SDK would speak against — is a Cortex control-plane routing rule,
not something this repo can add. Once that's in place, no code here
needs to change: the sub-job process already answers on `/v1/*`.

### (Non-blocking) Harbor upstream plugin ergonomics

Companion RFC in the Harbor tree: `rfcs/0002-plugin-first-extension-points.md`.
`LLMFactory` + entry-point group so third parties register LLM backends
by installing a package, and CLI short-name resolution for `--agent`,
`--env`, `--verifier`. Users can pass `module:Class` today, so this is
ergonomics.

## End-to-end demo: any Harbor agent

On-prem, this works today:

```bash
# 1. Start the RL server (one training + one sampling GPU).
python -m arctic_platform.rl.http_server \
  --training-gpus 1 --sampling-gpus 1 --port 7000 &

# 2. Drive training with Harbor's stock Terminus 2 agent over OpenAI-compat.
export CORTEX_PAT=...
harbor-cortex-train \
  --tasks-dir ./my_bench/train  --heldout-dir ./my_bench/heldout \
  --model             Qwen/Qwen3-0.6B \
  --agent             harbor.agents.terminus_2:Terminus2 \
  --sampling-api-base auto \
  --llm-backend       litellm \
  --iters 30 --prompts-per-step 8 --n-attempts 4 --lr 5e-6 \
  --out ./training-run/
```

`--sampling-api-base auto` reads the client's transport and points
Terminus 2 at `http://localhost:7000/v1`. `Cortex Training` deployments
substitute `https://<cortex>/sub-jobs/<sampling-sub-job-id>/v1` once the
Cortex-side routing above is live.

## Current E2E status

**Native mode (this package's `CortexRLAgent`):** `Qwen/Qwen3-0.6B`,
3-digit × 2-digit multiplication, 15 GRPO steps × 24 rollouts/step,
lr = 5e-6, three seeds. Mean held-out reward +10.0 pp, 95% CI [+9.0,
+11.6]. Full transcript in [RUN_LOG.md](./RUN_LOG.md).

**OpenAI-compat mode (any Harbor agent):**

* Router + token-id echo: 23 unit tests + 4 real-`openai`-SDK
  contract tests pass.
* Real Harbor pipe: `harbor.llms.lite_llm.LiteLLM(collect_rollout_details=True)`
  → LiteLLM → HTTP → our router → fake pool round-trips
  `prompt_token_ids` and `completion_token_ids`. Test asserts non-empty
  values on both, i.e. Harbor's `RolloutDetail` would carry the tokens
  the adapter needs.
* Live GPU-backed loop (real vLLM behind the router, Terminus 2 as
  agent): **not yet run** on this branch — the box is CPU-only. Run
  the on-prem demo from the E2E block above to close it out.
