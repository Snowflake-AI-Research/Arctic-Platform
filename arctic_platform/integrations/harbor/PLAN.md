# Plan — any Harbor agent trains on Cortex

## What's real today

**Converged on Cortex, native mode.** `CortexRLAgent` (this package's
`BaseAgent`) driving Cortex Training via `ArcticRLClient`.
`Qwen/Qwen3-0.6B`, 3-digit × 2-digit multiplication, 15 GRPO steps × 24
rollouts/step, lr = 5e-6, three independent seeds. Mean held-out reward
+10.0 pp, 95% CI [+9.0, +11.6]. Full transcript in
[RUN_LOG.md](./RUN_LOG.md).

Everything else described below is either **shipped code without a
GPU-backed run**, or **explicitly external work**. Nothing is asserted
as "works" unless a real training loop produced numbers.

## What ships in this PR

| Piece | File(s) | State |
| --- | --- | --- |
| Post-training data contract | `models.py` | Code + adapter tests (multi-turn flatten, per-token `loss_mask`). |
| RFC backend impl (`connect` / `generate` / `train` / `deploy_inference`) | `backend.py` | Code. Native mode is what produced the converged run above. |
| Harbor `BaseEnvironment` (host subprocess, dev-only) | `env.py` | Code. Used by the converged run above. |
| Reference `BaseAgent` (native reconnect-config path) | `agent.py` | Code. Used by the converged run above. |
| Multi-turn rollout flatten + per-token loss mask | `adapter.py` | Code + 6 unit tests. |
| E2E CLI (`harbor-cortex-train`) | `train.py` | Code. Native mode used above. |
| Multi-seed aggregator (`harbor-cortex-aggregate`) | `aggregate.py` | Code. Used above. |
| OpenAI-compat router (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) | `arctic_platform/openai_compat.py`, wired into `arctic_platform/rl/http_server.py` | Code. Never exercised against a live vLLM. See "Not yet real" below. |
| `--sampling-api-base` on `harbor-cortex-train` + auto URL derivation | `train.py::_derive_openai_base_url` | Code. Never driven end-to-end. |
| vLLM token-id echo (`prompt_token_ids`, per-choice `token_ids`) | `arctic_platform/openai_compat.py` | Code. Required so Harbor's `LiteLLM._extract_token_ids` can populate `RolloutDetail`. |

## Not yet real

Everything about the OpenAI-compat "any Harbor agent" path is code, not
observation:

* No run has been done against a real vLLM behind the router. There is
  no GPU on my current host (`nvidia-smi` absent,
  `torch.cuda.is_available() == False`), so this branch has not booted
  `arctic_platform.rl.http_server` with `--sampling-gpus 1`, loaded a
  real model, and served `/v1/*` against it.
* No run has been done with Terminus 2 (or any other Harbor agent
  speaking OpenAI-chat) driving a full training loop through `/v1/*`.
* The unit tests for `openai_compat.py`, the `openai`-SDK contract
  tests, and the `LiteLLM` integration test all run against a fake
  `ReplicaPool` that returns hard-coded tokens, and a stub tokenizer.
  They prove the **wire shape** (routes, response schema, SSE frames,
  `_extract_token_ids` compatibility). They do **not** prove that a
  real Qwen tokenizer's `apply_chat_template` output aligns with what
  vLLM tokenizes internally, that real generations round-trip
  correctly, or that a multi-turn Terminus 2 loop produces trainable
  rollouts.

To upgrade this section to "real", the following has to happen on a
GPU host:

```bash
python -m arctic_platform.rl.http_server \
  --training-gpus 1 --sampling-gpus 1 --port 7000 &

curl -sS http://localhost:7000/v1/models
# expect: {"object":"list","data":[{"id":"Qwen/Qwen3-0.6B", ...}]}

curl -sS http://localhost:7000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hi"}],"max_tokens":32}' \
  | jq '.choices[0].message.content, (.prompt_token_ids | length), (.choices[0].token_ids | length)'

harbor-cortex-train \
  --tasks-dir ./my_bench/train --heldout-dir ./my_bench/heldout \
  --model Qwen/Qwen3-0.6B \
  --agent harbor.agents.terminus_2:Terminus2 \
  --sampling-api-base auto --llm-backend litellm \
  --iters 15 --prompts-per-step 8 --n-attempts 4 --lr 5e-6 \
  --out ./training-run/
```

Until that sequence completes and produces held-out numbers, the
OpenAI-compat any-agent story is not proven.

## Cortex-side gap for OpenAI-compat

Separate from the on-prem GPU run above: on Cortex, sub-jobs are only
reachable through SnowAPI's op-name dispatch on the parent job
(`POST /{prefix}/{parent_job_id}/{op}`). The recognized ops are
`forward-backward`, `generate`, `step`, `save`, `operation`. `/v1/*` is
not in that set. Two ways to close this — both external to this PR
until picked up:

1. **Cortex control-plane routing.** Expose sub-jobs directly, e.g.
   `https://<cortex>/sub-jobs/<sampling-sub-job-id>/v1/*`. Cleanest
   because clients (OpenAI SDK, LiteLLM) talk to the sub-job with no
   intermediary.
2. **Op-envelope smuggle.** The existing `operation` op accepts a
   generic `{operation_type, payload}` envelope. Add
   `operation_type: "openai-chat-completion"` that forwards `payload`
   into the same handler our on-prem `/v1/*` routes call, plus a
   client-side gateway that translates OpenAI SDK calls into `operation`
   envelopes. Small, ugly, deployable without control-plane changes.

## Follow-ups (non-blocking)

- Streaming: the router replays a completed generation as SSE deltas.
  First-token latency is not correct until `ReplicaPool` exposes a
  delta-yielding surface.
- `harbor train` subcommand in Harbor itself would let users type
  `harbor train --backend arctic-cortex ...`. Users can already do
  `harbor-cortex-train ...` today.
