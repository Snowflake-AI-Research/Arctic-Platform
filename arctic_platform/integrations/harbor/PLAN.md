# Plan — any Harbor agent trains on Cortex

Status of the Harbor <-> Cortex integration and the work left to support
arbitrary Harbor agents, not just the reference `CortexRLAgent` this
package ships.

## What ships in this PR

| Piece | File(s) | Status |
| --- | --- | --- |
| Post-training data contract | `models.py` | Done. `Rollout` carries `loss_mask` for multi-turn. |
| RFC backend impl (`connect`/`generate`/`train`/`deploy_inference`) | `backend.py` | Done. E2E on Cortex Training. |
| Harbor `BaseEnvironment` (Docker-less dev) | `env.py` | Done. |
| Reference `BaseAgent` (native reconnect-config path) | `agent.py` | Done. |
| Multi-turn rollout flatten + per-token loss mask | `adapter.py` | New. Earlier code trained only on turn 0. |
| OpenAI-compat sampling plumbing | `train.py` (`--sampling-api-base`, `--llm-backend`) | New. No-op until Cortex exposes `/v1/chat/completions` (see below). |
| E2E CLI | `train.py` (`harbor-cortex-train`) | Done. Native mode validated E2E; OpenAI-compat wired but not yet E2E. |
| Multi-seed aggregator | `aggregate.py` (`harbor-cortex-aggregate`) | Done. |
| Adapter tests | `tests/integrations/harbor/test_adapter_multiturn.py` | New — 6 tests. |

## Blocking work outside this PR

### 1. Cortex sampling sub-job: expose `/v1/chat/completions`

Cortex's sampling sub-job already runs vLLM, and vLLM already speaks the
OpenAI protocol. Cortex's HTTP server proxies RL-specific routes
(`/generate`, `/fwd_bwd`, `/sync_weights`) but does not proxy vLLM's
OpenAI routes to the outside. Until it does, any Harbor agent that only
knows LiteLLM / OpenAI-chat (Terminus 2, most community agents) cannot
sample from a Cortex sub-job — hence the reference `CortexRLAgent` that
goes through `ArcticRLClient` on the RL-shaped route.

Change: ~50–100 LOC in `arctic_platform/rl/http_server.py`. Add one
route per OpenAI path (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`) that forwards to the sub-job's internal vLLM HTTP server
and streams the response back. Reuse the sub-job's bearer auth. No
change to vLLM. No change to Harbor.

Verification once shipped:

```bash
OPENAI_BASE_URL=https://<cortex>/sub-jobs/<sampling-id>/v1 \
  openai chat.completions.create -m Qwen/Qwen3-0.6B -m user "2+2=?"
```

and then:

```bash
harbor-cortex-train ... \
  --sampling-api-base https://<cortex>/sub-jobs/<sampling-id>/v1 \
  --agent harbor.agents.terminus_2:Terminus2 \
  --llm-backend litellm
```

trains Terminus 2 with no Terminus-2 or Harbor changes.

### 2. Harbor upstream: plugin-first extension points

Tracked by a companion RFC in the Harbor tree
(`rfcs/0002-plugin-first-extension-points.md`). Two small changes:

* `LLMFactory` + `harbor.plugins.llm_backends` entry-point group.
  Today Harbor's `LLMBackend` is a closed enum (`LITELLM`, `TINKER`);
  the factory + entry point lets third parties register a backend by
  installing a package. Terminus 2 refactored to dispatch through the
  factory with no behavior change for the built-ins.
* CLI short-name resolution for `--agent` / `--env` / `--verifier`.
  `--env` already resolves plugin short names; `--agent` and
  `--verifier` did not.

Users can pass full `module:Class` paths today, so this is ergonomics,
not a blocker.

## What "any Harbor agent trains on Cortex" looks like when done

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

`harbor-cortex-train` is the only new binary; every other flag is
Harbor's own.

## Current E2E status (native mode)

`Qwen/Qwen3-0.6B`, 3-digit × 2-digit multiplication, 15 GRPO steps × 24
rollouts/step, lr = 5e-6, three seeds. Mean held-out reward +10.0 pp,
95% CI [+9.0, +11.6]. Full transcript in [RUN_LOG.md](./RUN_LOG.md).
