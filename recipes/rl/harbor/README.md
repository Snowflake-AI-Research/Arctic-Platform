# Harbor RL Post-Training on Cortex Training (`recipes/rl/harbor`)

You have iterated on your agent harness (prompts, tools, `SKILL.md`), evaluated it with `harbor run`, and scaled up a directory of Harbor tasks. **Now you want to train the model weights inside that exact harness and environment.**

With `arctic_platform.integrations.harbor` and `recipes/rl/harbor`, you do not rewrite your Harbor tasks, agent harness, or verifier into an RL-specific format. Point **`TRAIN_HARBOR_TASKS`** and **`EVAL_HARBOR_TASKS`** at your Harbor task directories and launch `recipes/rl/harbor/run_qwen35_4b.sh` from a CPU driver machine. Training and sampling run on `4 training + 4 sampling H200 GPUs` on Snowflake Cortex Training.

## How It Works

Every Harbor task already defines `instruction.md`, `environment/`, and `tests/test.sh`. Each rollout runs through `arctic_platform.integrations.harbor.adapter.HarborTrialAdapter` without any custom task glue:
1. **`Trial`**: Harbor's `Trial.create(trial_cfg)` $\to$ `trial.run()` starts the sandbox (`HARBOR_ENV`) and runs the agent (`HARBOR_AGENT`) pointed at `ChatCompletionsProxy` (`request.base_url`, `request.api_key`).
2. **Token Splicing**: `ChatCompletionsProxy` records every `/v1/chat/completions` call and splices the exact sampled token IDs across turns (`loss_mask = 1` on assistant turns, `0` on prompts and tool outputs).
3. **`Verifier`**: Harbor's `Verifier` executes the task's `tests/test.sh` and reads `/logs/verifier/reward.txt` or `reward.json`.

## Before You Start

The driver needs [`uv`](https://docs.astral.sh/uv/) and a SkyRL checkout; the launcher installs OpenHands and Harbor itself. Run the commands below from the root of this repository.

```bash
git clone --branch skyrl-v0.3.0 --depth 1 https://github.com/NovaSky-AI/SkyRL
export SKYRL_HOME=$PWD/SkyRL

# The launcher reads the Cortex connection only from these variables.
cfg=~/cortex-training-config.json
export ARCTIC_CORTEX_HOST=$(jq -r .host "$cfg") ARCTIC_CORTEX_DATABASE=$(jq -r .database "$cfg")
export ARCTIC_CORTEX_SCHEMA=$(jq -r .schema "$cfg") ARCTIC_CORTEX_PAT=$(jq -r .pat "$cfg")
```

Rollouts run in Snowflake Cortex Sandboxes. Add a [connection](https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#connecting-using-the-connections-toml-file) for your account to `~/.snowflake/connections.toml` and pass its name as `connection` below. Use one that logs in without a browser, such as a programmatic access token (`authenticator = "PROGRAMMATIC_ACCESS_TOKEN"`) or a key pair: a run logs in again every 20 minutes to keep its session valid, and an SSO (`externalbrowser`) connection may open a browser each time.

## Train on Your Harbor Tasks

### 1. Any Harbor Task Set

```bash
export TRAIN_HARBOR_TASKS=/path/to/train_harbor_tasks
export EVAL_HARBOR_TASKS=/path/to/eval_harbor_tasks
export HARBOR_ENV_KWARGS='{"base_image": "sandbox-base", "connection": "MY_CONN", "role": "MY_ROLE"}'
export HARBOR_SKILL=/path/to/SKILL.md   # optional: ":"-separated SKILL.md paths
export ARCTIC_CORTEX_JOB_COMMENT="<your name> harbor rl"

bash recipes/rl/harbor/run_qwen35_4b.sh trainer.max_training_steps=50
```

Each rollout runs in its own sandbox (`HARBOR_ENV=snowflake`, the default). `HARBOR_ENV_KWARGS` is a JSON object passed to that environment, like `harbor run --ek`:

- `connection` and `role`: the Snowflake connection name and the role the sandboxes run as.
- `base_image`: the image each task's `environment/Dockerfile` is replayed on, in place of its `FROM` image. The sandbox catalog image `sandbox-base` is Chainguard Linux with Python 3.11, pip, git, and `apk`, but no `apt-get`, so the Dockerfile's steps must work there. Replay handles one `FROM` stage with `RUN`, `COPY` and `ADD` of local files, `ENV`, `ARG`, `WORKDIR`, `USER`, and `ENTRYPOINT`. It refuses `RUN` flags, `COPY --from`, `ADD` from a URL or of an archive, and `SHELL`, and docker-compose tasks aren't supported.
- `egress_eai`: a comma-separated list of external access integrations the sandboxes may use. Without one, sandboxes reach Snowflake and public package registries but not GitHub or other hosts.

Before it provisions any GPUs, the launcher runs the first training task once with an agent that does nothing: the sandbox starts, the Dockerfile replays, a command runs, and the verifier grades the result. If any of that fails, the launcher exits with the cause. For example, a SWE-smith task without GitHub egress fails within seconds with `GitHub egress is not configured for this sandbox`. Set `HARBOR_SKIP_TASK_CHECK=1` to skip this check.

To evaluate on the same environment, pass its import path to `harbor run`. From the repository root, `uv run --extra harbor` provides the pinned `harbor` CLI with the environment importable:

```bash
uv run --extra harbor harbor run -p "$EVAL_HARBOR_TASKS" -a <your agent> -m <your model> \
  --env arctic_platform.integrations.harbor.snowflake_sandbox:SnowflakeSandboxEnvironment \
  --ek base_image=sandbox-base --ek connection=MY_CONN --ek role=MY_ROLE
```

Snowflake allows 100 live sandboxes per owner, and suspended ones count. The environment creates at most 16 sandboxes at a time and keeps at most 90 live. It renews its Snowflake session while the run lasts, retries rate-limited creates, and terminates every sandbox it started when the run ends, including when the run fails. A driver that is killed outright can't do that, and the next launch warns when sandboxes it left behind would push you past the limit. When no other run is using the connection, terminate them:

```bash
uv run --extra harbor python -c "
from snowflake.sandbox import Sandbox
for s in Sandbox.list(connection='MY_CONN'):
    if s.status in ('ready', 'pending') and (s.name or '').startswith('harbor-rl-'):
        s.terminate()"
```

#### Tasks with their own images (not tested end to end in this release)

The environment can also mirror each task's `FROM` image, or the `docker_image` in its `task.toml`, into a Snowflake image repository (`image_repo=DATABASE.SCHEMA.REPOSITORY`), or run every task on one `prebuilt_image`. Mirroring needs [`crane`](https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md) on the driver's `PATH`, WRITE on the repository for `role`, and a registry that accepts pushes from the driver's network. A `prebuilt_image` or `docker_image` that is already a sandbox image path (`/DB/SCHEMA/REPO/IMAGE:TAG`) is used as is, with no mirroring. These paths have unit tests but were not run end to end for this release. The check above shows within a minute whether they work for your tasks, before any GPUs are provisioned.

### Where Rollouts Run

The agent's `terminal` tool and the verifier both run through the selected Harbor environment, so the agent works in the same filesystem that `tests/test.sh` grades. Each command runs to completion in a fresh shell: the working directory carries over between commands, but exported variables and background jobs do not. `file_editor` is not available because it would edit the training driver's filesystem instead.

The agent itself runs on the driver, and only its commands go to the environment. `HARBOR_AGENT` must be such a driver-side agent: the default `harbor_rl.agent:OpenHandsHarborAgent` or your own Harbor `BaseAgent` subclass (`module:Class`), which receives the rollout's model endpoint and key as the `base_url` and `api_key` constructor arguments. Harbor's installed agents (`openhands-sdk`, `mini-swe-agent`, `claude-code`, ...) run inside the environment, where task code could read that key, so the launcher and the adapter refuse them.

`HARBOR_ENV=host` runs tasks as subprocesses on the driver machine with no isolation. Task `RUN` lines, verifiers, and agent commands can read anything the driver's user can, so it is only for tasks you trust. It needs `HARBOR_ALLOW_HOST_EXEC=1`. Host commands get an allowlisted environment (`PATH`, locale, `TERM`, `TMPDIR`, plus any names in `HARBOR_HOST_ENV_PASSTHROUGH`), never the driver's `ARCTIC_*`, `WANDB_*`, or other credentials. All task paths are resolved and must stay inside the trial directory or the task's `environment/`.

A rollout whose trial ends without a verifier result (the sandbox never comes up, the Dockerfile fails to replay, the agent or the verifier crashes) is an error, not a reward of 0: it is saved as `$CKPT_DIR/trajectories/step_<n>/<phase>/<instance>_<n>.error` with Harbor's traceback and is masked out of the loss. If every rollout of a step errors, the step is rerun up to 3 times, after waits of 1, 2, and 4 minutes, so a brief sandbox outage doesn't end the run. If it still errors, the run stops with the first error.

### 2. SWE-smith Code Localization via Harbor

```bash
uv run recipes/rl/harbor/prepare_data.py \
  --export-swe-smith-to ~/data/swe-smith-harbor-tasks \
  --out ~/data/harbor-rl

export TRAIN_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/train
export EVAL_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/eval
export HARBOR_ENV_KWARGS='{"base_image": "sandbox-base", "connection": "MY_CONN", "role": "MY_ROLE", "egress_eai": "MY_GITHUB_EAI"}'
export ARCTIC_CORTEX_JOB_COMMENT="<your name> harbor localization"

bash recipes/rl/harbor/run_qwen35_4b.sh trainer.max_training_steps=100
```

Each task clones its repository from GitHub inside the sandbox, so `egress_eai` must name an external access integration that allows `github.com`; your account administrator creates it. A 25-step run of this example on Snowflake sandboxes averaged a reward of `0.59` over steps 1–10, `1.81` over steps 11–20, and `2.39` over steps 21–25, with a best step of `2.54` (the maximum is `3.0`).

## Two Ways to Customize Your Training Run

### Option A: Use the `cortex-training-rl` Skill

Install the **[`cortex-training-rl`](https://github.com/OpenHands/extensions/tree/main/skills/cortex-training-rl)** skill from `OpenHands/extensions` in OpenHands and tell your coding agent what you want to train:

> *"Use `/cortex-training-rl` to post-train `Qwen/Qwen3.5-4B` on my Harbor tasks in `~/my-harbor-tasks/train` and `~/my-harbor-tasks/eval` with `HARBOR_SKILL=~/my-skill/SKILL.md` and `MAX_TURNS=15` for 50 steps."*

The skill walks the agent through the same steps as this README: it checks Cortex GPU capacity and live jobs, prepares the data, launches the run detached, reports 10-step reward means and `.error` counts from `trajectories/step_*/`, draws `reward_graph.svg`, and stops the run so its Cortex job is cancelled.

### Option B: Tweak `recipes/rl/harbor/run_qwen35_4b.sh` Directly

Every harness setting, sandbox backend, model, and RL hyperparameter in `recipes/rl/harbor/run_qwen35_4b.sh` can be overridden via environment variables or trailing Hydra CLI flags (`"$@"`):

| What you want to change | Variable / Flag | Default | Example |
| :--- | :--- | :--- | :--- |
| **Task datasets** | `TRAIN_HARBOR_TASKS`, `EVAL_HARBOR_TASKS` | — | `~/tasks/train`, `~/tasks/eval` |
| **Agent skill (`SKILL.md`)** | `HARBOR_SKILL` | — | `~/skills/debug/SKILL.md:~/skills/git/SKILL.md` |
| **Agent tools** | `OPENHANDS_TOOLS` | `terminal` (+ finish tool) | `terminal,task_tracker` |
| **Agent system prompt** | `OPENHANDS_SYSTEM_PROMPT_FILE` | built-in prompt | `~/prompts/custom_agent.j2` |
| **Harbor agent class** | `HARBOR_AGENT` | `harbor_rl.agent:OpenHandsHarborAgent` | `my_pkg:MyAgent` (a driver-side `BaseAgent`; see [Where Rollouts Run](#where-rollouts-run)) |
| **Harbor environment (`--env`)** | `HARBOR_ENV` / `HARBOR_ENV_KWARGS` | `snowflake` / — | `host` (trusted tasks only) |
| **Base model** | `MODEL` | `Qwen/Qwen3.5-4B` | — |
| **GPU allocation (train / sample)** | `NUM_TRAIN_GPUS` / `NUM_SAMPLING_ENGINES` | `4` / `4` | `8` / `8` |
| **Rollout concurrency** | `BATCH_SIZE` $\times$ `N_ROLLOUTS` | `8` $\times$ `8` (`64`/step) | `16` $\times$ `8` (`128`/step) |
| **Horizon & context length** | `MAX_TURNS` / `MAX_CONTEXT_LENGTH` | `10` / `32768` | `20` / `32768` |
| **Learning rate & RL loss** | `LR` / Hydra CLI args | `1.0e-6` (`gspo`) | `LR=5e-7 trainer.algorithm.policy_loss_type=grpo` |
| **Attention & weight sync** | `ATTN_IMPLEMENTATION` / `WEIGHT_SYNC_BACKEND` | `flash_attention_3` / `nccl` | `flash_attention_2` (non-Hopper GPUs) |

This release was run end to end with the defaults above, on Snowflake sandboxes with `base_image` and on `host`. The launcher also accepts Harbor's other environments (`docker`, `modal`, `daytona`, `e2b`), other models, and other GPU counts, but those weren't tested with this recipe.
