# `arctic_platform.integrations.harbor` — Harbor plugin for Cortex training

A Harbor plugin. Ships with the `arctic_platform` wheel, in the same
shape as the sibling `arctic_platform.integrations.verl` adapter.

Every LLM call runs inside a `harbor run` trial: Harbor's own trial
runner spawns `CortexRLAgent` (BaseAgent) under `HostEnvironment`
(BaseEnvironment) and scores it with Harbor's stock `Verifier` execing
each task's `tests/test.sh`. Between trials the driver reads Harbor's
`result.json`, runs one Cortex GRPO step, and `sync_weights` propagates
the new weights back to the same sampling sub-job — so the next
`harbor run` samples from an improved model at the same endpoint. No
Harbor code is modified; no custom `BaseVerifier` subclass.

## Install

```bash
pip install harbor
pip install 'arctic_platform[harbor]'
```

Harbor discovers the plugin via `importlib.metadata` entry points in
the `harbor.plugins` group, registered on Arctic-Platform's top-level
`pyproject.toml`:

```
$ harbor plugins list
┃ Name                ┃ Import path                                             ┃
│ arctic-cortex-agent │ arctic_platform.integrations.harbor.agent:CortexRLAgent │
│ arctic-cortex-env   │ arctic_platform.integrations.harbor.env:HostEnvironment │
```

## Train against your Harbor benchmark

You already have a directory of Harbor task subdirs, each with an
`instruction.md` and `tests/test.sh`, and optionally a `SKILL.md`.
Split them into a training pool and a held-out pool and run:

```bash
export CORTEX_PAT=...
export ARCTIC_CORTEX_HOST=...

harbor-cortex-train \
  --tasks-dir     ./my_bench/train    \
  --heldout-dir   ./my_bench/heldout  \
  --skill-md      ./SKILL.md          \
  --model         Qwen/Qwen3-0.6B     \
  --iters 30 --prompts-per-step 8 --n-attempts 4 --lr 5e-6 \
  --out ./training-run/
```

What runs end-to-end:

1. **Cortex cold-start.** Provisions a training sub-job (GRPO trainer,
   holds weights) and a sampling sub-job (vLLM). Cortex owns every GPU
   op; the driver box needs no GPUs.
2. **Baseline `harbor run`** on `--heldout-dir`, greedy, one attempt per
   task. Records `pass@1` and mean reward.
3. **N GRPO iterations.** Each step samples `--prompts-per-step` tasks
   from `--tasks-dir`, writes a temporary `dataset.toml`, invokes
   `harbor run -a arctic-cortex-agent -e arctic-cortex-env` with
   `--n-attempts` rollouts each, hands the rollouts to
   `ArcticCortexBackend.train`, then `sync_weights` to the sampling
   sub-job.
4. **Final `harbor run`** on `--heldout-dir` at the same sampling sub-job
   — whatever weights `sync_weights` pushed are what the model uses.
5. **Writes `./training-run/summary.json`** with metrics and Cortex
   sub-job ids for downstream evals or production inference.

### Eval-only

```bash
harbor run \
  -p ./my_bench/heldout \
  --agent arctic-cortex-agent \
  --env   arctic-cortex-env   \
  -m      Qwen/Qwen3-0.6B     \
  --ak    reconnect_config_path=./training-run/reconnect_config.json
```

Reattaches to the same sampling sub-job at the weights
`harbor-cortex-train` finished on.

### What each user artifact maps to

| your artifact | Harbor flag | at trial time |
| --- | --- | --- |
| `task_dir/instruction.md` | (implicit) | Shown to the agent as the user prompt. |
| `task_dir/tests/test.sh` | (implicit; Harbor's stock `Verifier`) | Uploaded into the env, execed after the agent runs; writes `reward.txt` under `/logs/verifier/`. |
| `task_dir/task.toml` | (implicit) | Timeouts, target OS, optional `[verifier].env`. |
| `SKILL.md` | `--extra-instruction-path` | Appended to every task's `instruction.md`. |
| `./skills/` dir | `--skill` | Mounted at `/harbor/skills` in the env. |
| custom `BaseAgent` | `--agent module:MyAgent` | Replaces `arctic-cortex-agent`. Needs only to call `client.generate` from the sub-job. |
| custom `BaseEnvironment` | `--env module:MyEnv` | Swap in Harbor's `DockerEnvironment` / Modal / Daytona for prod. |

## Files

| File | Role |
| --- | --- |
| `models.py` | RFC data contract — `Rollout`, `RolloutDataset`, `PostTrainingConfig`, `TrainingRun`, `InferenceEndpoint`. Mirrors Harbor's `RolloutDetail` 1:1. |
| `backend.py` | `ArcticCortexBackend` — the RFC's `PostTrainingBackend` protocol (`connect`, `generate`, `train`, `deploy_inference`, `cancel`) over `ArcticRLClient` + Cortex transport. Owns GRPO advantage + batch construction. |
| `env.py` | `HostEnvironment(BaseEnvironment)` — runs commands as host subprocesses under a per-trial root, no container. Development-only. |
| `agent.py` | `CortexRLAgent(BaseAgent)` — reattaches to the Cortex sub-job via `ArcticRLClient.reconnect_config`, does one sampling call, writes Harbor's `RolloutDetail`. |
| `task_gen.py` | Arithmetic task-dir + `dataset.toml` writer for the demo. Skip if you have your own task dirs. |
| `adapter.py` | Reads Harbor's `{trial_dir}/result.json`, materializes a `RolloutDataset` for `backend.train`. Preserves every verifier reward key on `Rollout.metadata`. |
| `train.py` | The `harbor-cortex-train` CLI. |
| `aggregate.py` | `harbor-cortex-aggregate` — multi-seed aggregator with bootstrap 95% CI on pass@1 and mean held-out reward. |

Entry points and console scripts are registered in Arctic-Platform's
top-level `pyproject.toml`; no nested `pyproject.toml` here.

## Result on Cortex QA6

`Qwen/Qwen3-0.6B`, 3-digit × 2-digit multiplication (a ∈ [100, 999],
b ∈ [10, 99]), 15 GRPO steps × 24 rollouts/step, lr = 5e-6. Held-out
80 problems, greedy re-eval. Three independent seeds:

```
                seed 0            seed 1            seed 2            aggregate (n=3)
pass@1          0.362 → 0.350     0.350 → 0.400     0.375 → 0.425     Δ +0.029  95% CI [-0.013, +0.050]
mean held-out r 0.580 → 0.696     0.600 → 0.690     0.648 → 0.741     Δ +0.100  95% CI [+0.090, +0.116]
```

`pass@1` CI spans zero — a 0.6B model doesn't learn 3-digit × 2-digit
multiplication in 15 GRPO steps. Mean held-out reward moves +10.0 pp,
95% CI [+9.0, +11.6] across three seeds: on seed 0, 19 held-out
problems moved out of the reward ≤ 0.05 bucket into "close but wrong"
or "verbose correct". See `RUN_LOG.md` for the full transcript.

## Follow-ups (not in this PR)

- **`harbor train` subcommand in Harbor itself.** A ~30-LOC Typer
  addition would let users write `harbor train --backend arctic-cortex
  ...` and dispatch through a `harbor.backends` entry point.
  `arctic_platform.integrations.harbor.train:cli` is a drop-in backend.
- **`PostTrainingBackend.stream_progress()`.** The RFC's async iterator
  for per-step metrics. `backend.train()` already returns them in a
  dict; wrapping as an iterator is small.
- **Real sandbox.** `HostEnvironment` is dev-only; users swap in
  Harbor's `DockerEnvironment` / `ModalEnvironment` / `DaytonaEnvironment`
  without our code changing.
