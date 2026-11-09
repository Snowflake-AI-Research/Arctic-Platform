# prime-rl on Snowflake Cortex

Train and serve a [PrimeIntellect prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)
RL run on Snowflake Cortex, with **no fork of prime-rl**. This is the prime-rl analogue of
the `verl` adapter: prime-rl is user-supplied and version-pinned by the launcher, and this
package drives it from the outside.

prime-rl has no in-process plugin hook, so this is a **launcher**, not a registered backend.
`rl.py` reuses an unmodified prime-rl install — its `orchestrator` and `env-server` run as
subprocesses exactly as under prime-rl's own `rl` entrypoint — while the Cortex **worker**
(`worker.py`) stands in for both the trainer and the vLLM-shaped inference server: it serves
`/inference/v1/generate` + `/v1/chat/completions` + the admin routes to the orchestrator, and
on the training side it runs one forward-backward + optimizer step + weight sync per policy
version on the Cortex **training** sub-job, generating rollouts on the Cortex **sampling**
sub-job. Everything crosses the wire through `arctic_platform.client`.

## Install

```bash
# 1. The integration (this package) + Cortex client stack.
pip install "arctic-platform[prime-rl]"

# 2. prime-rl, user-supplied and pinned (brings its own verifiers / renderers / configs).
#    Follow prime-rl's install for its torch/vLLM closure; the pin below is what this
#    integration targets:
PRIME_RL=84e94bf05c29d73e293341b24f6eda2b15bbc900   # v0.9.1.dev234
pip install "prime-rl @ git+https://github.com/PrimeIntellect-ai/prime-rl@${PRIME_RL}"
```

`prime-rl` and its `verifiers` runtime are **not** dependencies of the `[prime-rl]` extra
(the same contract `[verl]` uses for verl): prime-rl's dependency closure — `prime-rl-configs`,
`verifiers`, `renderers`, `prime-pydantic-config`, and the torch/vLLM indexes — is wired by
prime-rl's own `[tool.uv.sources]`/workspace, which does not travel with a plain dependency
pin, so install prime-rl from its repo rather than relying on this extra to pull it.

Pins this integration targets:

| component | pin | source |
|---|---|---|
| prime-rl | `84e94bf05c` (v0.9.1.dev234) | `PrimeIntellect-ai/prime-rl` |
| verifiers | `da75527ca` (brought by prime-rl) | `PrimeIntellect-ai/verifiers` |

The reused prime-rl surface (`entrypoints.rl` config plumbing, `transports.batch.MicroBatch`,
`transports.weights.base` markers, `utils.pathing`) is internal to prime-rl; bump the pin
deliberately and re-run the dry run below after any bump.

## Run

```bash
# SWE (r2e-gym with the agentic grading policy; sandbox via PrimeIntellect's sandbox
# service — verifiers' default `prime` runtime):
python -m arctic_platform.integrations.prime_rl.rl \
  @ arctic_platform/integrations/prime_rl/examples/qwen3.5-4b/swe-grade.toml

# Resolve-only, writes cortex_job_body.json and exits:
python -m arctic_platform.integrations.prime_rl.rl @ <recipe>.toml --dry-run
```

A math recipe runs the same way (`--env-package math-env`); the one for the current
math experiment is added here once its dataset is published.

Cortex credentials come from `ARCTIC_CORTEX_*` (host, database, schema, PAT), read by
`CortexConnectionConfig`. The shipped SWE recipe serves `Qwen/Qwen3.5-4B` from the Cortex
model catalog (one 8-GPU training sub-job + one 8-GPU sampling sub-job).

## Sandboxes

The SWE recipes use PrimeIntellect's sandbox service — verifiers' default `prime` runtime —
so an external user reproduces SWE with PrimeIntellect's own sandbox. Point
`env.agent.runtime.*` at any other verifiers runtime (`docker`, `e2b`, `modal`, …) to use a
different sandbox backend.

## SWE grading policy

`swe-grade.toml` sets `env.agent.grading_policy = true` (off by default): the rollout is
terminated and its reward forced to 0 on a format-invalid turn (any non-OK renderer tool-call
parse status), a per-turn output-budget overrun (a turn finishing on `length`), or a framework
turn/token limit, and the disqualified fraction is logged as `grading_policy/disqualified`.

This is applied over the **unmodified public `verifiers`** by a startup monkey-patch shipped in
this package — `_grading_startup.py`, installed at the venv root with a `.pth` (`import … ;
apply()`) so it runs at every interpreter startup, including the EnvWorker subprocesses that
prime-rl spawns. There is **no verifiers fork** and nothing to pass at launch; the patch only
adds an `AgentConfig.grading_policy` field, a `parse_issues` carrier on each `Response`, two
response `@stop` predicates, and a disqualification branch in `Task.score`, and is a no-op unless
a recipe opts in. See `_grading_startup.py` for the exact seams.
