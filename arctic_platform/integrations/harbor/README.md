# Harbor + Arctic Cortex — post-training backend, driven by Harbor's own CLI

Reference implementation of the [Harbor Post-Training RFC](../../../../rfcs/harbor-post-training-backend.md).
Every LLM call in this demo happens inside a real `harbor run` trial (real
`BaseAgent`, real `BaseEnvironment`, real `BaseVerifier`, real `RolloutDetail`
written to `result.json`). The middle step reads Harbor's on-disk output,
runs one Cortex GRPO step through `ArcticCortexBackend.train`, and
`sync_weights` propagates the new weights to the same sampling sub-job that
Harbor's next trial samples from.

The point is: no code inside Harbor changes. A user configures Harbor with
this package's agent + env + verifier and points at Cortex Training as the
backend.

## What's in the box

| File                       | Role |
| -------------------------- | ---- |
| `models.py`                | `Rollout`, `RolloutDataset`, `PostTrainingConfig`, `TrainingRun`, `InferenceEndpoint` — the RFC's data contract. Mirrors Harbor's `RolloutDetail` 1:1. |
| `backend.py`               | `ArcticCortexBackend` — implements the RFC protocol (`connect`, `generate`, `train`, `deploy_inference`, `cancel`) over `ArcticRLClient` + Cortex transport. Owns GRPO advantage + batch construction. |
| `host_environment.py`      | `HostEnvironment(BaseEnvironment)` — Harbor's environment interface implemented against host subprocesses. Lets Harbor drive a trial where Docker/Daytona/Modal aren't available (e.g., a locked-down VM). Not for prod; there's no isolation. |
| `cortex_agent.py`          | `CortexRLAgent(BaseAgent)` — reattaches to a running Cortex sub-job via `ArcticRLClient.reconnect_config`, does one sampling call, writes Harbor's `RolloutDetail` with `prompt_token_ids` + `completion_token_ids`. |
| `arithmetic_verifier.py`   | `ArithmeticVerifier(BaseVerifier)` — in-process verifier that reads the trial's `agent/completion.txt`, extracts the last integer, scores against `task.toml [metadata].expected`. Dense partial credit (relative closeness) so GRPO always has gradient. |
| `task_gen.py`              | Programmatic Harbor task-dir + `dataset.toml` writer. One task dir per problem, so every problem is a real Harbor task. |
| `adapter.py`               | Reads Harbor's `{trial_dir}/result.json` under a job dir, extracts `agent_result.rollout_details` + verifier reward, materializes a `RolloutDataset` the backend can train on. |
| `harbor_runner.py`         | End-to-end driver: connect Cortex → real `harbor run` for baseline → per step (real `harbor run` → `backend.train` → weight sync) → real `harbor run` for post-training re-eval. |

## Result on Cortex QA6

`Qwen/Qwen3-0.6B`, 3-digit × 2-digit multiplication (a ∈ [100, 999],
b ∈ [10, 99]), 15 GRPO steps, 6 prompts/step × 4 attempts (24 rollouts/step),
`lr = 5e-6`. Every step is a fresh `harbor run` subprocess.

```
BASELINE pass@1 = 0.250   (5/20)
FINAL    pass@1 = 0.400   (8/20)   Δ +0.150

held-out flips (baseline greedy vs. post-training greedy, same problems):
  965 × 22:  21630 (wrong)  ->  21230 (right)   [actual 21230]
  376 × 13:  4908  (wrong)  ->  4888  (right)   [actual 4888]
  991 × 96:  95184 (wrong)  ->  95136 (right)   [actual 95136]
  regressions (right -> wrong): 0

training reward curve (per-step mean, partial credit):
  0.375 0.327 0.840 0.744 0.658 0.758 0.925 0.606
  0.642 0.787 0.625 0.610 0.723 0.756 0.771
```

Three held-out problems flipped wrong → right (real arithmetic changes,
not formatting) with zero regressions. Same sampling sub-job before and
after — Harbor's post-training re-eval reads whatever weights the last
`sync_weights` pushed. Real GRPO gradients throughout (`grad_norm` 6–21).

`RUN_LOG.md` (next to this README) has the full transcript: the actual
`harbor run` command lines, Harbor's own per-trial results tables,
per-step gradients, and the complete baseline / post-training completion
tables.

## What it proves

1. **Harbor's `RolloutDetail` shape is exactly the RFC's `Rollout` shape.** The
   adapter is 40 LOC and copies `prompt_token_ids` / `completion_token_ids` /
   reward straight across, no re-encoding.
2. **The training and eval endpoints are the same endpoint.** Baseline and
   post-training `harbor run` invocations use identical CLI arguments — the
   difference in numbers is entirely from the new weights `sync_weights`
   pushed to the sampling sub-job.
3. **No Harbor code was modified.** `harbor run` accepts our agent, env, and
   verifier by import path (`module:Class`). This is Harbor's built-in
   extension surface, not a fork.
4. **A user's existing `harbor run` invocation is the training loop.** Swap
   `--agent`, `--env`, `--verifier` for the RFC's registered ids and add
   `--backend arctic-cortex`; the demo runs unchanged.

## What a real integration would add

- Harbor calls `harbor train --job-dir <dir> --backend arctic-cortex`. This
  driver stands in; wiring it into Harbor is a ~30 LOC subcommand.
- `PostTrainingBackend.stream_progress()` yields real `TrainingProgress`
  events (loss, grad_norm, reward_mean) instead of the runner logging them.
- A production `BaseEnvironment` (Docker/Modal/Daytona) instead of
  `HostEnvironment` — the demo uses HostEnv only because this box has no
  container runtime.
- A stronger base model + a task with headroom (Terminal-Bench-Lite,
  Aider-Polyglot easy tier, GSM8K-style problems). The whole plumbing is
  model- and task-agnostic; only the task-dirs change.

## Run against Cortex QA6

Requires `CORTEX_PAT` + `ARCTIC_CORTEX_HOST` in env, and an account with
`NEUTRINO_ACCOUNT_TIER='internal'`.

```bash
python -m arctic_platform.integrations.harbor.harbor_runner \
  --model Qwen/Qwen3-0.6B \
  --task mul --a-low 100 --a-high 999 --b-low 10 --b-high 99 \
  --iters 15 --prompts-per-step 6 --n-attempts 4 --heldout 20 \
  --max-tokens 64 --temperature 0.8 --lr 5e-6
```

Cortex cold-start (weights + vLLM warmup on both sub-jobs) takes a few
minutes; each subsequent `harbor run` is 15–60s wall clock.

## Under the hood, per Harbor trial

1. `harbor run` spawns a trial worker.
2. Trial worker starts `HostEnvironment` (creates a per-trial root dir, sets
   up `/tests`, `/logs`, `/solution` under it).
3. `CortexRLAgent.run(instruction, env, context)` loads the reconnect config
   from `--ak reconnect_config_path=...`, opens an `ArcticRLClient` attached
   to the running sub-jobs (no fresh cold-start), calls `client.generate`,
   populates `context.rollout_details = [{prompt_token_ids, completion_token_ids}]`.
4. `ArithmeticVerifier.verify()` reads the completion the agent wrote, scores
   against `task.toml [metadata].expected`, returns
   `VerifierResult(rewards={"reward": ...})`.
5. Trial writes `result.json` with `agent_result.rollout_details` and
   `verifier_result.rewards` — Harbor's canonical output format.

The driver then reads every `result.json` under the job dir, constructs a
`RolloutDataset` grouped by task, calls `backend.train(...)`, and repeats.
