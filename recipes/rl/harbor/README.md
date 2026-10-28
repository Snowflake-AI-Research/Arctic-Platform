# Harbor RL Post-Training on Cortex Training (`recipes/rl/harbor`)

Go from any [Harbor](https://github.com/harbor-framework/harbor) evaluation tasks to RL post-training on **Snowflake Cortex Training** (`4 training + 4 sampling H200 GPUs`) in two environment variables: **`TRAIN_HARBOR_TASKS`** and **`EVAL_HARBOR_TASKS`**.

## How It Works

Every Harbor task already defines `instruction.md`, `environment/`, and `tests/test.sh`. Each rollout runs through `arctic_platform.integrations.harbor.adapter.HarborTrialAdapter` without any custom task glue:
1. **`Trial`**: Harbor's `Trial.create(trial_cfg)` $\to$ `trial.run()` starts the sandbox (`HARBOR_ENV`) and runs the agent (`HARBOR_AGENT`) pointed at `ChatCompletionsProxy` (`request.base_url`, `request.api_key`).
2. **Token Splicing**: `ChatCompletionsProxy` records every `/v1/chat/completions` call and splices the exact sampled token IDs across turns (`loss_mask = 1` on assistant turns, `0` on prompts and tool outputs).
3. **`Verifier`**: Harbor's `Verifier` executes the task's `tests/test.sh` and reads `/logs/verifier/reward.txt` or `reward.json`.

## Tweaking `recipes/rl/harbor/run_qwen35_4b.sh`

All task paths, agent/sandbox settings, and training hyperparameters in `recipes/rl/harbor/run_qwen35_4b.sh` can be tweaked via environment variables or trailing Hydra CLI overrides (`"$@"`):

| Variable | Default | Description |
| --- | --- | --- |
| `TRAIN_HARBOR_TASKS` | — | Directory of training Harbor tasks (`instruction.md`, `environment/`, `tests/test.sh`) |
| `EVAL_HARBOR_TASKS` | `TRAIN_HARBOR_TASKS` | Directory of evaluation Harbor tasks |
| `MODEL` | `Qwen/Qwen3.5-4B` | Any HuggingFace model ID (`Qwen/Qwen3-8B`, etc.) |
| `NUM_TRAIN_GPUS` / `NUM_SAMPLING_ENGINES` | `4` / `4` | Remote Cortex H200 GPUs for FSDP training and vLLM sampling |
| `BATCH_SIZE` / `N_ROLLOUTS` | `8` / `8` | Prompts per step $\times$ rollouts per prompt (`64` concurrent Harbor trials/step) |
| `MAX_TURNS` / `MAX_CONTEXT_LENGTH` | `10` / `32768` | Max agent tool turns per trial and context window |
| `LR` | `1.0e-6` | Policy learning rate |
| `HARBOR_SKILL` | — | Optional `:`-separated `SKILL.md` file(s) or skill directories loaded into OpenHands `AgentContext` |
| `OPENHANDS_TOOLS` | `terminal` (+ finish tool) | Optional comma-separated OpenHands tools (e.g. `terminal,file_editor,task_tracker`) |
| `OPENHANDS_SYSTEM_PROMPT_FILE` | built-in prompt | Optional custom system prompt file for the target OpenHands agent |
| `HARBOR_AGENT` | `harbor_rl.agent:OpenHandsHarborAgent` | Any Harbor `BaseAgent` (`openhands-sdk`, `openhands`, `module:Class`) |
| `HARBOR_ENV` | `arctic_platform.integrations.harbor.env:HostEnvironment` | Any Harbor `BaseEnvironment` (`arctic_platform.integrations.harbor.snowflake_sandbox:SnowflakeSandboxEnvironment` / `snowflake`, `docker`, `modal`, `daytona`) |

## Quickstart

### 1. Any Harbor Task Set

```bash
export TRAIN_HARBOR_TASKS=/path/to/train_harbor_tasks
export EVAL_HARBOR_TASKS=/path/to/eval_harbor_tasks
export HARBOR_SKILL=/path/to/SKILL.md   # optional
export ARCTIC_CORTEX_JOB_COMMENT="<your name> harbor rl"

bash recipes/rl/harbor/run_qwen35_4b.sh trainer.max_training_steps=50
```

### 2. SWE-smith Code Localization via Harbor

```bash
uv run recipes/rl/harbor/prepare_data.py \
  --export-swe-smith-to ~/data/swe-smith-harbor-tasks \
  --out ~/data/harbor-rl

export TRAIN_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/train
export EVAL_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/eval
export ARCTIC_CORTEX_JOB_COMMENT="<your name> harbor localization"

bash recipes/rl/harbor/run_qwen35_4b.sh trainer.max_training_steps=100
```

### 3. Drive & Tweak Runs Automatically with the `cortex-training-rl` Skill

Install the [`cortex-training-rl`](https://github.com/OpenHands/extensions/pull/771) skill from `OpenHands/extensions` (or invoke `/cortex-training-rl` in OpenHands / Claude Code / Cursor) to let your agent automatically tweak `run_qwen35_4b.sh` (model, GPU count, batch size, tools, sandbox, or `HARBOR_SKILL`), run preflight capacity checks, monitor per-step rewards, and generate `reward_graph.svg`.
