# Harbor RL Post-Training on Cortex Training (`recipes/rl/harbor`)

You have iterated on your agent harness (prompts, tools, `SKILL.md`), evaluated it with `harbor run`, and scaled up a directory of Harbor tasks. **Now you want to train the model weights inside that exact harness and environment.**

With `arctic_platform.integrations.harbor` and `recipes/rl/harbor`, you do not rewrite your Harbor tasks, agent harness, or verifier into an RL-specific format. Point **`TRAIN_HARBOR_TASKS`** and **`EVAL_HARBOR_TASKS`** at your Harbor task directories and launch `recipes/rl/harbor/run_qwen35_4b.sh` from a CPU driver machine to train on the model weights on `4 training + 4 sampling H200 GPUs` on Snowflake Cortex Training.

## How It Works

Every Harbor task already defines `instruction.md`, `environment/`, and `tests/test.sh`. Each rollout runs through `arctic_platform.integrations.harbor.adapter.HarborTrialAdapter` without any custom task glue:
1. **`Trial`**: Harbor's `Trial.create(trial_cfg)` $\to$ `trial.run()` starts the sandbox (`HARBOR_ENV`) and runs the agent (`HARBOR_AGENT`) pointed at `ChatCompletionsProxy` (`request.base_url`, `request.api_key`).
2. **Token Splicing**: `ChatCompletionsProxy` records every `/v1/chat/completions` call and splices the exact sampled token IDs across turns (`loss_mask = 1` on assistant turns, `0` on prompts and tool outputs).
3. **`Verifier`**: Harbor's `Verifier` executes the task's `tests/test.sh` and reads `/logs/verifier/reward.txt` or `reward.json`.

## Go from `harbor run` (eval) to RL Post-Training in 3 Steps

### 1. Any Harbor Task Set

```bash
export TRAIN_HARBOR_TASKS=/path/to/train_harbor_tasks
export EVAL_HARBOR_TASKS=/path/to/eval_harbor_tasks
export HARBOR_ENV=snowflake             # default: each rollout runs in its own Snowflake Cortex Sandbox
export HARBOR_ENV_KWARGS='{"image_repo": "DB.SCHEMA.REPO", "connection": "MY_CONN", "role": "MY_ROLE"}'
export HARBOR_SKILL=/path/to/SKILL.md   # optional: ":"-separated SKILL.md paths
export ARCTIC_CORTEX_JOB_COMMENT="<your name> harbor rl"

bash recipes/rl/harbor/run_qwen35_4b.sh trainer.max_training_steps=50
```

`HARBOR_ENV_KWARGS` is a JSON object passed to the environment, like `harbor run --ek`. For `snowflake` it takes `image_repo` (where task images are mirrored), `connection`, `role`, `prebuilt_image`, `egress_eai` (needed for tasks that reach the internet, such as `git clone` from GitHub), and `image_platform`. Mirroring task images needs [`crane`](https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md) on the driver's `PATH` (no Docker daemon), and the role needs WRITE on `image_repo`.

### Where Rollouts Run

The agent's `terminal` tool and the verifier both run through the selected Harbor environment, so the agent works in the same filesystem that `tests/test.sh` grades, on every backend. Each command runs to completion in a fresh shell: the working directory carries over between commands, but exported variables and background jobs do not. `file_editor` is not available because it would edit the training driver's filesystem instead.

`HARBOR_ENV=host` runs tasks as subprocesses on the driver machine with no isolation. Task `RUN` lines, verifiers, and agent commands can read anything the driver's user can, so it is only for tasks you trust. It needs `HARBOR_ALLOW_HOST_EXEC=1`. Host commands get an allowlisted environment (`PATH`, locale, `TERM`, `TMPDIR`, plus any names in `HARBOR_HOST_ENV_PASSTHROUGH`), never the driver's `ARCTIC_*`, `WANDB_*`, or other credentials. All task paths are resolved and must stay inside the trial directory or the task's `environment/`.

### 2. SWE-smith Code Localization via Harbor

```bash
uv run recipes/rl/harbor/prepare_data.py \
  --export-swe-smith-to ~/data/swe-smith-harbor-tasks \
  --out ~/data/harbor-rl

export TRAIN_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/train
export EVAL_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/eval
export HARBOR_ENV_KWARGS='{"image_repo": "DB.SCHEMA.REPO", "egress_eai": "GITHUB_ACCESS"}'
export ARCTIC_CORTEX_JOB_COMMENT="<your name> harbor localization"

bash recipes/rl/harbor/run_qwen35_4b.sh trainer.max_training_steps=100
```

## Two Ways to Customize Your Training Run

### Option A: Use the `cortex-training-rl` Skill (Recommended)

Install the **[`cortex-training-rl`](https://github.com/OpenHands/extensions/tree/main/skills/cortex-training-rl)** skill from `OpenHands/extensions` (or invoke `/cortex-training-rl` in OpenHands, Claude Code, or Cursor) and tell your coding agent what you want to train:

> *"Use `/cortex-training-rl` to post-train `Qwen/Qwen3-8B` on my Harbor tasks in `~/my-harbor-tasks/train` and `~/my-harbor-tasks/eval` with `HARBOR_SKILL=~/my-skill/SKILL.md`, `MAX_TURNS=15`, and `--env snowflake` for 50 steps."*

The skill configures `run_qwen35_4b.sh`, checks Cortex H200 capacity, verifies the 128 MiB sequence-budget cap, launches detached training, monitors per-step rewards and `.error` counts across `trajectories/step_*/`, generates `reward_graph.svg`, and cleans up remote Cortex GPU jobs on exit.

### Option B: Tweak `recipes/rl/harbor/run_qwen35_4b.sh` Directly

Every harness setting, sandbox backend, model, and RL hyperparameter in `recipes/rl/harbor/run_qwen35_4b.sh` can be overridden via environment variables or trailing Hydra CLI flags (`"$@"`):

| What you want to change | Variable / Flag | Default | Example |
| :--- | :--- | :--- | :--- |
| **Task datasets** | `TRAIN_HARBOR_TASKS`, `EVAL_HARBOR_TASKS` | — | `~/tasks/train`, `~/tasks/eval` |
| **Agent skill (`SKILL.md`)** | `HARBOR_SKILL` | — | `~/skills/debug/SKILL.md:~/skills/git/SKILL.md` |
| **Agent tools** | `OPENHANDS_TOOLS` | `terminal` (+ finish tool) | `terminal,task_tracker` |
| **Agent system prompt** | `OPENHANDS_SYSTEM_PROMPT_FILE` | built-in prompt | `~/prompts/custom_agent.j2` |
| **Harbor agent class** | `HARBOR_AGENT` | `harbor_rl.agent:OpenHandsHarborAgent` | `openhands-sdk`, `mini-swe-agent`, `my_pkg:MyAgent` |
| **Harbor sandbox (`--env`)** | `HARBOR_ENV` / `HARBOR_ENV_KWARGS` | `snowflake` / — | `docker`, `modal`, `daytona`, `e2b`, `host` (trusted tasks only) |
| **Base model** | `MODEL` | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3-8B`, `Qwen/Qwen2.5-Coder-7B-Instruct` |
| **GPU allocation (train / sample)** | `NUM_TRAIN_GPUS` / `NUM_SAMPLING_ENGINES` | `4` / `4` | `8` / `8` |
| **Rollout concurrency** | `BATCH_SIZE` $\times$ `N_ROLLOUTS` | `8` $\times$ `8` (`64`/step) | `16` $\times$ `8` (`128`/step) |
| **Horizon & context length** | `MAX_TURNS` / `MAX_CONTEXT_LENGTH` | `10` / `32768` | `20` / `32768` |
| **Learning rate & RL loss** | `LR` / Hydra CLI args | `1.0e-6` (`gspo`) | `LR=5e-7 trainer.algorithm.policy_loss_type=grpo` |
| **Attention & weight sync** | `ATTN_IMPLEMENTATION` / `WEIGHT_SYNC_BACKEND` | `flash_attention_3` / `nccl` | `flash_attention_2` (non-Hopper GPUs) |
