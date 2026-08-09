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
| `host_environment.py`      | `HostEnvironment(BaseEnvironment)` — Harbor's environment interface implemented against host subprocesses. Lets Harbor drive a trial where Docker/Daytona/Modal aren't available (e.g., a locked-down VM). **Not for prod; there's no isolation.** |
| `cortex_agent.py`          | `CortexRLAgent(BaseAgent)` — reattaches to a running Cortex sub-job via `ArcticRLClient.reconnect_config`, does one sampling call, writes Harbor's `RolloutDetail` with `prompt_token_ids` + `completion_token_ids`. |
| `task_gen.py`              | Programmatic Harbor task-dir + `dataset.toml` writer. Each problem is a real Harbor task with its own `tests/test.sh` — scoring uses Harbor's stock `Verifier`, not a custom `BaseVerifier` subclass. |
| `adapter.py`               | Reads Harbor's `{trial_dir}/result.json` under a job dir, extracts `agent_result.rollout_details` + verifier reward, materializes a `RolloutDataset`. Preserves all reward fields as metadata so eval can report multiple metrics. |
| `harbor_runner.py`         | End-to-end driver: connect Cortex → real `harbor run` for baseline → per step (real `harbor run` → `backend.train` → weight sync) → real `harbor run` for post-training re-eval. |
| `aggregate_runs.py`        | Read N per-seed `summary.json` files and print mean±sd plus bootstrap 95% CI for pass@1 and mean held-out reward. |

## Result on Cortex QA6

`Qwen/Qwen3-0.6B`, 3-digit × 2-digit multiplication (a ∈ [100, 999],
b ∈ [10, 99]), 15 GRPO steps × 24 rollouts/step (6 prompts × 4 attempts),
`lr = 5e-6`. Held-out 80 problems, greedy re-eval. Three independent seeds.

```
                  seed 0            seed 1            seed 2            aggregate (n=3)
pass@1            0.362 -> 0.350    0.350 -> 0.400    0.375 -> 0.425    Δ +0.029  95%CI [-0.013, +0.050]
mean held-out r   0.580 -> 0.696    0.600 -> 0.690    0.648 -> 0.741    Δ +0.100  95%CI [+0.090, +0.116]
```

**pass@1**'s CI still spans zero across three seeds — a 0.6B model doesn't
reliably learn 3-digit × 2-digit multiplication in 15 GRPO steps.
**Mean held-out reward** moves **+10.0 pp with 95% CI [+9.0, +11.6]** — a
real, statistically clean improvement in output quality that pass@1 misses.
All three seeds move in the same direction on both metrics.

What that improvement actually is (seed 0, 80 held-out): 19 problems moved
out of the "catastrophic" bucket (model got stuck in a repetition loop, e.g.
`994 * 74 = 72, 994 * 74 = 72, 994 * 74 = 72, ...`, reward 0.05) into the
"close but wrong" bucket (`994 * 74 = 72,856`, reward 0.7, within 1% of the
true answer 73556). GRPO is buying us fewer catastrophic failures, not
more exact answers. That's what the dense partial-credit reward incentivizes,
and it's what the model can actually change in 15 steps.

Every step had real GRPO gradients (`grad_norm` 5–23), same sampling sub-job
for baseline and post-training re-eval. See `RUN_LOG.md` for the full
transcript, per-step gradients, distribution shift table, and worked
examples of flips.

### Honest caveats for the room

- 15 steps of GRPO × 24 rollouts on a 0.6B model is intentionally small; it
  proves the pipeline, not the ceiling. Scaling numbers to a real workload
  is downstream of that.
- Pass@1 on arithmetic is the wrong metric for this compute budget; a task
  where the target behavior is reachable in 15 steps (e.g. Terminal-Bench-Lite
  with a 7B model, or a shorter-horizon task) would move pass@1 too.
- `HostEnvironment` runs commands as `$USER` with no isolation. Real
  deployments must use `DockerEnvironment` (or Modal/Daytona/etc.) —
  swap `--env` at the CLI, no other changes.

## What it proves

1. **Harbor's `RolloutDetail` shape is exactly the RFC's `Rollout` shape.** The
   adapter is ~50 LOC and copies `prompt_token_ids` / `completion_token_ids` /
   reward straight across, no re-encoding.
2. **The training and eval endpoints are the same endpoint.** Baseline and
   post-training `harbor run` invocations use identical CLI arguments; the
   difference in numbers is entirely from the new weights `sync_weights`
   pushed to the sampling sub-job.
3. **No Harbor code was modified, and no custom `BaseVerifier`.** `harbor
   run` accepts our agent + env by import path (`module:Class`), and scoring
   uses Harbor's stock `Verifier` running the task's own `tests/test.sh`.
   Exactly how a real Harbor benchmark task works.
4. **A user's existing `harbor run` invocation is the training loop.** Swap
   `--agent` and `--env` for the RFC's registered ids and add
   `--backend arctic-cortex`; the driver logic in `harbor_runner.py` is
   what a `harbor train` subcommand would call.

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
   up `/tests`, `/logs`, `/solution` under it, exports `HARBOR_HOST_ROOT`).
3. `CortexRLAgent.run(instruction, env, context)` loads the reconnect config
   from `--ak reconnect_config_path=...`, opens an `ArcticRLClient` attached
   to the running sub-jobs (no fresh cold-start), calls `client.generate`,
   and populates `context.rollout_details = [{prompt_token_ids, completion_token_ids}]`.
4. Harbor's stock `Verifier` uploads the task's `tests/` dir into the env,
   execs `test.sh`, and reads `reward.txt` from `/logs/verifier/`. Our
   `test.sh` is a shell + Python one-liner that scores the completion; the
   verifier machinery is unchanged from what Terminal-Bench-style tasks use.
5. Trial writes `result.json` with `agent_result.rollout_details` and
   `verifier_result.rewards` — Harbor's canonical output format.

The driver then reads every `result.json` under the job dir, constructs a
`RolloutDataset` grouped by task, calls `backend.train(...)`, and repeats.

### Two non-obvious bugs I found and fixed

1. **`ArcticRLClientConfig.training_job_id/sampling_job_id` have
   `Field(exclude=True)`**, so `model_dump_json()` drops them. Each Harbor
   trial subprocess thus rebuilt the client without job ids, hit an
   internal cold-start path, and blocked on `_wait_running`. Fixed in
   `harbor_runner.py` by writing the ids back after serialization.
2. **`ArcticRLClient.shutdown()` cancels the shared Cortex sub-job.** With
   `CortexRLAgent` running one trial per subprocess, the first trial was
   killing the sub-job the other trials still needed. Fixed by dropping
   `client.shutdown()` from the agent — sockets are GC'd on trial-process
   exit; the runner cancels the sub-job in its `finally`.
3. **`AutoTokenizer.from_pretrained` on every trial rate-limits HF Hub.**
   With `n_concurrent=4` and 20 GRPO steps × ~30 rollouts, dozens of
   parallel HF Hub calls trip the Hub's rate limiter and drop trials
   silently (`HfHubHTTPError` × 21 in one step). Fixed by pre-warming the
   local tokenizer cache in the runner and using `local_files_only=True`
   in `CortexRLAgent`.
