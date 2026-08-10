# `arctic_platform.integrations.harbor` — Harbor plugin for Cortex training

Reference implementation of the [Harbor Post-Training RFC](../../../../rfcs/harbor-post-training-backend.md).
**Harbor is the primary product; this subpackage is a plugin.** Ships
with the `arctic_platform` wheel (mirrors the layout of the sibling
`arctic_platform.integrations.verl` adapter). Harbor discovers our
agent + environment by name through `importlib.metadata` entry points
in the `harbor.plugins` group, registered on Arctic-Platform's own
top-level `pyproject.toml`.

Every LLM call runs inside a real `harbor run` trial (real `BaseAgent`,
real `BaseEnvironment`, real `RolloutDetail` on disk), scored by Harbor's
stock `Verifier` execing each task's `tests/test.sh`. Between trials, the
driver reads Harbor's `result.json`, runs one Cortex GRPO step, and
`sync_weights` propagates the new weights back to the same sampling
sub-job — so the next `harbor run` samples from an improved model at the
same endpoint. **No Harbor code is modified.**

## Install

```bash
pip install harbor
pip install 'arctic_platform[harbor]'   # pulls Arctic + registers the plugin
```

Then verify Harbor sees the plugin:

```bash
$ harbor plugins list
                                 Harbor Plugins
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name                ┃ Import path                                             ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ arctic-cortex-agent │ arctic_platform.integrations.harbor.agent:CortexRLAgent │
│ arctic-cortex-env   │ arctic_platform.integrations.harbor.env:HostEnvironment │
└─────────────────────┴─────────────────────────────────────────────────────────┘
```

## The Harbor user's flow

You already have a Harbor benchmark — a directory of task subdirs, each
with `instruction.md`, `tests/test.sh`, and optionally a `SKILL.md`. To
train a model against those same tasks on Cortex, split them into a
training pool and a held-out pool and run:

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

That's the whole invocation. What it does end-to-end:

1. **Cortex cold-start.** Provisions a training sub-job (GRPO trainer,
   holds weights) and a sampling sub-job (vLLM). Cortex owns every GPU
   op — the driver box needs no GPUs.
2. **Baseline `harbor run`** on `--heldout-dir` with the current weights,
   greedy, one attempt per task. Records `pass@1` and mean reward.
3. **N GRPO iterations.** Each step: sample `--prompts-per-step` tasks
   from `--tasks-dir`, write a temporary `dataset.toml`, invoke
   `harbor run -a arctic-cortex-agent -e arctic-cortex-env` with
   `--n-attempts` rollouts each, hand the resulting rollouts to
   `ArcticCortexBackend.train`, `sync_weights` to the sampling sub-job.
4. **Final `harbor run`** on `--heldout-dir` at the same sampling
   sub-job — whatever weights `sync_weights` pushed are what the model
   uses. Records post-training `pass@1` and mean reward.
5. **Writes `./training-run/summary.json`** with metrics + Cortex sub-job
   ids for downstream evals or production inference.

### Eval-only (no training) — same package, same CLI

```bash
harbor run \
  -p ./my_bench/heldout \
  --agent arctic-cortex-agent \
  --env   arctic-cortex-env   \
  -m      Qwen/Qwen3-0.6B     \
  --ak    reconnect_config_path=./training-run/reconnect_config.json
```

Harbor resolves `arctic-cortex-agent` / `arctic-cortex-env` through the
entry point group and hands each trial to `CortexRLAgent`, which
reattaches to the sub-job whose weights `harbor-cortex-train` just
finished updating.

### What each of your files becomes at runtime

| your artifact | Harbor flag | what it does at trial time |
| --- | --- | --- |
| `task_dir/instruction.md` | (implicit) | Shown to the agent as the user prompt. |
| `task_dir/tests/test.sh` | (implicit, Harbor's stock `Verifier`) | Uploaded into the env, execed after the agent runs; writes `reward.txt` under `/logs/verifier/`. |
| `task_dir/task.toml` | (implicit) | Timeouts, target OS, optional `[verifier].env`. |
| `SKILL.md` | `--extra-instruction-path` (Harbor's own flag) | Appended to every task's `instruction.md`. |
| `./skills/` dir | `--skill` (Harbor's own flag) | Mounted at `/harbor/skills` in the env. |
| custom `BaseAgent` | `--agent module:MyAgent` | Replaces `arctic-cortex-agent` if you have your own agent. Only requirement: `client.generate` from the sub-job. |
| custom `BaseEnvironment` | `--env module:MyEnv` or a Harbor-shipped `DockerEnvironment` for prod | We default to `HostEnvironment` (no sandbox). |

Nothing in this list is Arctic-specific plumbing — every knob is a
Harbor knob. Our plugin only contributes the two `harbor.plugins`
entries and one training-loop CLI.

## What's in this package

| File | Role |
| --- | --- |
| `models.py` | RFC data contract — `Rollout`, `RolloutDataset`, `PostTrainingConfig`, `TrainingRun`, `InferenceEndpoint`. Mirrors Harbor's `RolloutDetail` 1:1. |
| `backend.py` | `ArcticCortexBackend` — implements the RFC's `PostTrainingBackend` protocol (`connect`, `generate`, `train`, `deploy_inference`, `cancel`) over `ArcticRLClient` + Cortex transport. Owns GRPO advantage + batch construction. |
| `env.py` | `HostEnvironment(BaseEnvironment)` — runs commands as host subprocesses under a per-trial root. Used because this demo box has no Docker; **not for prod**. Real deployments use `DockerEnvironment` / Modal / Daytona. |
| `agent.py` | `CortexRLAgent(BaseAgent)` — reattaches to the running Cortex sub-job via `ArcticRLClient.reconnect_config`, does one sampling call, writes Harbor's `RolloutDetail` with `prompt_token_ids` + `completion_token_ids`. |
| `task_gen.py` | Programmatic Harbor task-dir + `dataset.toml` writer for the arithmetic demo. Real users skip this — they bring their own task dirs. |
| `adapter.py` | Reads Harbor's `{trial_dir}/result.json`, materializes `RolloutDataset` for `backend.train`. Preserves every verifier reward key on metadata so eval can report multiple metrics. |
| `train.py` | The `harbor-cortex-train` CLI — connects Cortex, runs the alternating `harbor run` → `backend.train` → `sync_weights` loop, emits `summary.json`. |
| `aggregate.py` | `harbor-cortex-aggregate` — multi-seed aggregator with bootstrap 95% CI on pass@1 and mean held-out reward. |
| _(no nested `pyproject.toml`)_ | The `harbor.plugins` entry points and the two `harbor-cortex-*` console scripts are registered in Arctic-Platform's top-level `pyproject.toml`, alongside the `[harbor]` optional dependency. Ships with the `arctic_platform` wheel. |

## Result on Cortex QA6

`Qwen/Qwen3-0.6B`, 3-digit × 2-digit multiplication (a ∈ [100, 999],
b ∈ [10, 99]), 15 GRPO steps × 24 rollouts/step, `lr = 5e-6`. Held-out
80 problems, greedy re-eval. Three independent seeds:

```
                  seed 0            seed 1            seed 2            aggregate (n=3)
pass@1            0.362 → 0.350     0.350 → 0.400     0.375 → 0.425     Δ +0.029  95%CI [-0.013, +0.050]
mean held-out r   0.580 → 0.696     0.600 → 0.690     0.648 → 0.741     Δ +0.100  95%CI [+0.090, +0.116]
```

`pass@1` CI spans zero — a 0.6B model doesn't reliably learn 3-digit ×
2-digit multiplication in 15 GRPO steps. **Mean held-out reward moves
+10.0 pp with 95% CI [+9.0, +11.6]** across all three seeds — real,
statistically clean improvement in output quality (19 held-out problems
on seed 0 moved out of the "catastrophic-loop" bucket into "close but
wrong"). See `RUN_LOG.md` for the full transcript.

## What it proves

1. **Harbor's `RolloutDetail` shape is exactly the RFC's `Rollout`
   shape.** The adapter is ~50 LOC.
2. **Training and eval endpoints are the same endpoint.** Baseline and
   post-training `harbor run` invocations use identical CLI arguments;
   the number difference is only what `sync_weights` pushed.
3. **No Harbor code was modified, no custom `BaseVerifier`.** `harbor
   run` accepts our agent + env through its plugin registry; scoring uses
   Harbor's stock `Verifier` on each task's own `tests/test.sh`. Exactly
   how a real Harbor benchmark task works.
4. **Arctic-side dependencies are contained.** This package is what a
   Harbor user installs. It depends on `harbor` and `arctic-platform`;
   users don't import Arctic modules directly.

## What's still an RFC-side follow-up (not in this package)

- **`harbor train` subcommand.** Ideal is `harbor train --tasks ...
  --backend arctic-cortex ...`, dispatching to a backend discovered via
  a `harbor.backends` entry point. That's a ~30-LOC Typer addition
  inside Harbor and would call `arctic_platform.integrations.harbor.train:cli`
  unchanged. Contributing that PR is next.
- **Real sandbox.** `--env arctic-cortex-env` is `HostEnvironment` (no
  isolation). Prod users pass Harbor's `DockerEnvironment` /
  `ModalEnvironment` / `DaytonaEnvironment` — none of our code changes.
- **`PostTrainingBackend.stream_progress()`.** RFC has an async
  iterator for streaming per-step loss/grad_norm/reward. Metrics are
  already in `backend.train()`'s return dict — wrapping as an iterator
  is small when Harbor wants to consume them.
