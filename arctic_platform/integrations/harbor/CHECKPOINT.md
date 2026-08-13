# Checkpoint — Harbor × Arctic Platform

Handoff doc so this thread of work can be paused and resumed without
losing context. Everything below is what is *actually true* on
[PR #66](https://github.com/Snowflake-AI-Research/Arctic-Platform/pull/66)
as of the last commit. Overclaims have been stripped; anything not
listed under "Real" is code, not observation.

## Real (has numbers)

**Native mode, Cortex Training, end-to-end.**

* Config: `Qwen/Qwen3-0.6B`, 3-digit × 2-digit multiplication (`a ∈ [100, 999]`, `b ∈ [10, 99]`), 15 GRPO steps × 24 rollouts/step, lr = 5e-6.
* Three independent seeds. Held-out: 80 problems, greedy re-eval.
* Result: mean held-out reward **+10.0 pp**, 95 % CI **[+9.0, +11.6]** (bootstrap, n=3).
* `pass@1` moves +0.029, CI [-0.013, +0.050] — spans zero. A 0.6 B model doesn't learn multiplication in 15 steps, but reward-shape confirms the loop works.
* Full transcript: [RUN_LOG.md](./RUN_LOG.md).

Everything used by that run is in this PR: `agent.py`, `env.py`,
`backend.py`, `adapter.py`, `train.py`, `aggregate.py`, plus the
adjacent `arctic_platform/rl/_cortex_dispatch.py` shim (from an earlier
PR).

## Shipped as code, not exercised on real GPUs

`arctic_platform/openai_compat.py`
: `/v1/chat/completions`, `/v1/completions`, `/v1/models` mounted on
  `arctic_platform.rl.http_server`. Translates OpenAI requests to
  `ReplicaPool.generate()` in-process. Emits vLLM's OpenAI extensions
  (`prompt_token_ids`, per-choice `token_ids`) so Harbor's
  `LiteLLM._extract_token_ids` can populate `RolloutDetail`.

`train.py::_derive_openai_base_url` + `--sampling-api-base auto`
: Auto-derives the OpenAI base URL from the client's transport
  (on-prem: `http://host:port/v1`). Would let `harbor.agents.terminus_2:Terminus2`
  drive the loop without any Harbor-side change once a GPU-backed run
  proves it.

`tests/openai_compat/` — **28 tests, all pass; all against a fake `ReplicaPool` + stub tokenizer.**
: `test_openai_compat.py` — 23 unit tests (`FastAPI TestClient`,
  parameter mapping, `n>1`, `stop`, streaming SSE, error paths,
  tokenizer-without-chat-template, token-id echo).
: `test_real_openai_sdk.py` — 4 tests. Live `uvicorn` + real `openai`
  Python client + fake pool. Validates wire-shape against the OpenAI
  SDK.
: `test_harbor_litellm_integration.py` — 1 test. Live `uvicorn` + real
  `harbor.llms.lite_llm.LiteLLM(collect_rollout_details=True)` + fake
  pool. Asserts Harbor's `_extract_token_ids` finds our
  `prompt_token_ids` and per-choice `token_ids` in the response.

None of these prove real generations, real chat templates, or real
training. They prove the wire is shaped right.

## Not real, blocked

**GPU-backed E2E of the OpenAI-compat path.**
: This host is CPU-only (`nvidia-smi` absent, `torch.cuda.is_available() == False`).
: To close: on a GPU host, run
  ```bash
  python -m arctic_platform.rl.http_server --training-gpus 1 --sampling-gpus 1 --port 7000 &
  curl -sS http://localhost:7000/v1/models
  harbor-cortex-train --tasks-dir ./bench/train --heldout-dir ./bench/heldout \
    --model Qwen/Qwen3-0.6B --agent harbor.agents.terminus_2:Terminus2 \
    --sampling-api-base auto --llm-backend litellm \
    --iters 15 --prompts-per-step 8 --n-attempts 4 --lr 5e-6 \
    --out ./training-run/
  ```

**Cortex-side OpenAI-compat.**
: On Cortex, sub-jobs are only reachable via SnowAPI's op-name
  dispatch on the parent job (`POST /{prefix}/{parent_job_id}/{op}`).
  Recognized ops: `forward-backward`, `generate`, `step`, `save`,
  `operation`. `/v1/*` isn't in the set. Two ways to close, both
  external to this PR:
  1. **Cortex control-plane routing.** Expose sub-jobs directly at
     `https://<cortex>/sub-jobs/<sampling-sub-job-id>/v1/*`.
  2. **Op-envelope smuggle.** Add `operation_type: "openai-chat-completion"`
     to the existing `operation` op, forward `payload` into the same
     handler the on-prem `/v1/*` routes call, plus a client-side
     gateway that translates OpenAI SDK calls into `operation`
     envelopes.
: Detail in [PLAN.md](./PLAN.md#cortex-side-gap-for-openai-compat).

## Drafted, not filed

RFC to Harbor maintainers: [`rfcs/harbor-post-training-backend.md`](../../../../rfcs/harbor-post-training-backend.md)
(one level up from the Arctic-Platform tree, workspace-relative).

Five asks:

1. `harbor train` CLI verb + `--train <backend>` on `harbor run`.
2. `PostTrainingBackend` Protocol in `harbor.backends`.
3. Entry-point group `harbor.plugins.post_training_backends` (mirrors the existing plugin pattern).
4. `RolloutDetail.loss_mask` field.
5. Promote `collect_rollout_details` from Terminus-2/Computer-1 to `BaseAgent`.

**Not yet a PR against `harbor-framework/harbor`.** Not yet shared with
Stas or the other maintainers.

Minimum-useful subset if the full RFC is too big for a first PR:
items 4 + 5 + Protocol only, no CLI. Lets Arctic Platform's plugin
live entirely as a third-party package and users drive it via
`harbor-cortex-train` (this PR's script) until a real `harbor train`
CLI lands.

## What Arctic Platform ships if the RFC lands

The current `harbor-cortex-train` script is Arctic-Platform-as-driver
(shells out to `harbor run` as a subprocess, reads `result.json`,
loops). If the RFC lands, the script deletes. `arctic_platform.integrations.harbor.backend:ArcticCortexBackend`
becomes a plain `PostTrainingBackend` implementation, registered
under the `harbor.plugins.post_training_backends` entry point. Users
`pip install arctic-platform` and type `harbor train --backend
arctic-cortex ...`; nothing in the Arctic Platform CLI needs to
exist.

Until the RFC lands, `harbor-cortex-train` is the glue.

## Package layout (files in this PR under `arctic_platform/integrations/harbor/`)

| file | role |
| --- | --- |
| `models.py` | RFC data contract — `Rollout`, `RolloutDataset`, `PostTrainingConfig`, `TrainingRun`, `InferenceEndpoint`. Mirrors Harbor's `RolloutDetail` 1:1. |
| `backend.py` | `ArcticCortexBackend` — implements the RFC `PostTrainingBackend` Protocol over `ArcticRLClient` + Cortex transport. Owns GRPO advantage + batch construction. |
| `env.py` | `HostEnvironment(BaseEnvironment)` — host-subprocess trial dir, no container. Dev only. |
| `agent.py` | `CortexRLAgent(BaseAgent)` — reattaches via `ArcticRLClient.reconnect_config`, does one sampling call per turn, writes Harbor's `RolloutDetail`. |
| `task_gen.py` | Arithmetic task-dir + `dataset.toml` writer. Demo helper. |
| `adapter.py` | Reads `{trial_dir}/result.json`, materializes a `RolloutDataset`. Handles multi-turn flatten + per-token `loss_mask` (was training only turn 0 previously). |
| `train.py` | `harbor-cortex-train` CLI. Drives Harbor via `harbor run` subprocess. Supports `--sampling-api-base` for the OpenAI-compat mode. |
| `aggregate.py` | `harbor-cortex-aggregate` — multi-seed aggregator with bootstrap 95 % CI. |
| `README.md` | User-facing docs. Sampling-mode table is honest about native vs. OpenAI-compat status. |
| `PLAN.md` | Critical path. What's real, what's shipped-as-code, what's blocked and why. |
| `RUN_LOG.md` | Full transcript of the converged 3-seed run. |
| `CHECKPOINT.md` | This file. |

Alongside, in the top-level Arctic Platform tree:

| file | role |
| --- | --- |
| `arctic_platform/openai_compat.py` | OpenAI-compat router. Zero coupling to `arctic_platform.rl`. |
| `arctic_platform/rl/http_server.py` | Mounts the router; loads `AutoTokenizer` at `/initialize` time. |
| `tests/openai_compat/` | 28 tests. Wire-shape only (fake pool + stub tokenizer). |

## Decisions still open

* **Full RFC (5 items) vs. minimum-useful (items 4 + 5 + Protocol).** Full is drafted at `rfcs/harbor-post-training-backend.md`. Minimum would be a much easier first PR against Harbor. No decision made.
* **Cortex OpenAI-compat now (op-envelope smuggle) vs. wait (control-plane routing).** No decision made.
* **File the Harbor RFC as a PR now, or hold until we've had a GPU-backed OpenAI-compat run to include as reference data.** No decision made.

## Prior-work references

* Cortex dispatch shim: `arctic_platform/rl/_cortex_dispatch.py` (earlier PR — this is what glues SkyRL / verl / our `CortexRLAgent` to the sync `ArcticRLClient`).
* SkyRL + Cortex, verl + Cortex GSM8K recipes: `arctic_platform/integrations/verl/examples/` and `recipes/rl/skyrl/`. Both green from earlier PRs.
* Unified client PRs upstream: #52 (client), #54 (transports), #58 (async client). All merged.

## To resume

1. Read this file.
2. Skim [PLAN.md](./PLAN.md) — the "Not yet real" and "Cortex-side gap" sections.
3. Pick one of: (a) get on a GPU host and close the OpenAI-compat E2E, (b) file the RFC as a Harbor PR, (c) build the Cortex op-envelope smuggle.
