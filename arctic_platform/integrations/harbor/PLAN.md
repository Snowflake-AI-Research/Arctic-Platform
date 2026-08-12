# Plan — "any Harbor agent trains on Cortex"

Status of the Harbor <-> Cortex integration and the concrete work left to
support arbitrary Harbor agents (not just our reference `CortexRLAgent`).
This PR lands the pieces that live in Arctic-Platform. Two follow-ups are
tracked separately because they live in other repos.

## What ships in this PR

| Piece | File(s) | Status |
| --- | --- | --- |
| Post-training data contract | `models.py` | Done. `Rollout` now carries `loss_mask` for multi-turn. |
| RFC backend impl (`connect`/`generate`/`train`/`deploy_inference`) | `backend.py` | Done. E2E on Cortex QA6. |
| Harbor `BaseEnvironment` (Docker-less dev) | `env.py` | Done. |
| Reference `BaseAgent` (native reconnect-config path) | `agent.py` | Done. |
| Multi-turn rollout flattening + per-token loss mask | `adapter.py` | **New in this PR.** Every model-produced token across all turns is marked trainable; earlier code only trained on turn 0. |
| OpenAI-compat sampling mode plumbing | `train.py` (`--sampling-api-base`, `--llm-backend`) | **New in this PR.** No-op until Cortex ships `/v1/chat/completions` (see below). |
| E2E CLI | `train.py` (`harbor-cortex-train`) | Done. Native mode validated E2E; OpenAI-compat mode wired but end-to-end blocked on the Cortex-side item below. |
| Multi-seed aggregator | `aggregate.py` (`harbor-cortex-aggregate`) | Done. |
| Adapter tests | `tests/integrations/harbor/test_adapter_multiturn.py` | New in this PR — 6 tests. |

## Blocking work outside this PR

### 1. Cortex sampling sub-job: expose `/v1/chat/completions`

**Why:** Cortex's sampling sub-job already runs vLLM internally, and vLLM
already speaks the OpenAI protocol. Cortex's HTTP server proxies
RL-specific routes (`/generate`, `/fwd_bwd`, `/sync_weights`) but does
not currently proxy vLLM's OpenAI routes to the outside. Until it does,
any Harbor agent that only knows LiteLLM / OpenAI-chat (Terminus 2, most
community agents) cannot sample directly from a Cortex sub-job — hence
our reference `CortexRLAgent` that goes through `ArcticRLClient` on
Cortex's RL-shaped route.

**Change:** ~50–100 LOC in `arctic_platform/rl/http_server.py`. Register
one route per OpenAI path (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`) that forwards the request body to the sub-job's internal
vLLM HTTP server and streams the response back. Reuse the sub-job's
existing bearer auth. No change to vLLM, no change to Harbor.

**Verification once shipped:**

```bash
# Any OpenAI SDK, unmodified:
OPENAI_BASE_URL=https://<cortex>/sub-jobs/<sampling-id>/v1 \
  openai chat.completions.create -m Qwen/Qwen3-0.6B \
  -m user "2+2=?"
```

and then:

```bash
harbor-cortex-train ... \
  --sampling-api-base https://<cortex>/sub-jobs/<sampling-id>/v1 \
  --agent harbor.agents.terminus_2:Terminus2 \
  --llm-backend litellm
```

should train Terminus 2 without any Terminus-2 or Harbor changes.

### 2. Harbor upstream: plugin-first extension points

Tracked in a Harbor-side branch (`arctic/plugin-discoverable-llm-backends`)
and RFC (`harbor-framework/rfcs/0002-plugin-first-extension-points.md`).
Two small changes:

* **`LLMFactory` + `harbor.plugins.llm_backends` entry-point group.**
  Today Harbor's `LLMBackend` is a closed enum (`LITELLM`, `TINKER`).
  A factory + entry-point group lets third parties (us, others) register
  a backend by installing a package — no core Harbor change per backend.
  Refactored Terminus 2 to dispatch through the factory without behavior
  change for the built-ins.

* **CLI short-name resolution for `--agent` / `--env` / `--verifier`.**
  `--env` already resolves plugin short names; `--agent` and `--verifier`
  did not. Unified so `--verifier arctic-arithmetic` works the same as
  `--agent arctic-cortex-agent`.

Neither change is required for *this* Arctic-Platform PR to be useful —
users can always pass full `module:Class` paths today. Shipping the
Harbor upstream changes just makes the plugin ergonomics symmetric.

## What "any Harbor agent trains on Cortex" looks like when done

Steps a user runs (post-Cortex-fix):

```
1. install       :  pip install harbor arctic_platform[harbor]
2. tasks + tests :  <bench>/{train,heldout}/<task>/{instruction.md,tests/test.sh}
3. train         :  harbor-cortex-train --tasks-dir ... --heldout-dir ... \
                       --agent harbor.agents.terminus_2:Terminus2 \
                       --sampling-api-base <cortex-sub-job>/v1 \
                       --llm-backend litellm \
                       --iters ... --lr ...
4. eval          :  harbor run -p ./heldout \
                       -a harbor.agents.terminus_2:Terminus2 \
                       --model-base-url <cortex-sub-job>/v1
```

No Harbor changes. No agent changes. The `harbor-cortex-train` driver
is the only new binary; every other step is Harbor's own CLI.

## Current E2E status (native mode)

Cortex QA6, `Qwen/Qwen3-0.6B`, 3-digit × 2-digit multiplication, 15 GRPO
steps × 24 rollouts/step, lr = 5e-6, three seeds. Mean held-out reward
+10.0 pp, 95% CI [+9.0, +11.6] — statistically significant with the
0.6B model. Full transcript in [RUN_LOG.md](./RUN_LOG.md).
