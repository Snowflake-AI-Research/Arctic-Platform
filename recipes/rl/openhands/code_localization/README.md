# OpenHands code localization on Cortex

Train Qwen3.5-4B to name the file, class, and function a bug touches. The
driver is a CPU process. Cortex runs 4 training GPUs and 4 sampling GPUs.
SkyRL runs GRPO. The agent is OpenHands: `terminal` to search, then
`localization_finish` to submit.

The reward is file F1 + module F1 + function F1 (maximum 3). A rollout that
never submits within 10 turns is left out of the loss.

Job setup goes through `arctic_platform.integrations.skyrl.entrypoint.launch_driver`.
The branch is `karthik/openhands-cortex`.

This directory is one example. The chat endpoint and the episodes live in
`arctic_platform.integrations.harness` and do not import a harness.
`HarnessGenerator` calls `submit`, `get_status`, and `collect_output`. The
request carries the dataset row, the phase, `base_url` from
`ChatCompletionsProxy.url_for(routing_key)`, and a bearer token from `issue`.
The proxy keeps each sampled response and splices it into the next prompt.
The grader scores `collect_output`. The adapter does not return tokens or a
reward. OpenHands, OpenCode, or Codex can supply those three methods.

`openhands_localization` is the OpenHands example: the sandbox, the tools,
and `grade_localization`. A different task copies this directory and replaces
those three. The launcher refuses to start when a cached wheel shadows
`launch_driver`.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone --branch skyrl-v0.3.0 --depth 1 https://github.com/NovaSky-AI/SkyRL
export SKYRL_HOME=$PWD/SkyRL

git clone --branch karthik/openhands-cortex https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform
```

`SKYRL_HOME` must contain `integrations/arctic_rl`. The launcher installs
OpenHands SDK `85ecfd93`.

Cortex credentials are `cortex-training login ~/cortex-training-config.json`,
or `ARCTIC_CORTEX_HOST`, `ARCTIC_CORTEX_DATABASE`, `ARCTIC_CORTEX_SCHEMA`, and
`ARCTIC_CORTEX_PAT`. A file that assigns those without `export` has to be
loaded with `set -a`.

You need a Cortex account with 8 H200s free, and disk on the driver for one
git checkout per in-flight rollout. That checkout is the sandbox.

## Data

Copy `train.parquet` and `validation.parquet` from CodeScout
[`data/swe_smith`](https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2)
at `abab719` to `~/data/swe-smith-localization`, or set `DATA_DIR`.

Rows need `instance_id`, `repo`, `base_commit`, `problem_statement`, `patch`,
`use_patch`, `file_changes`, and `prompt`.

## Run

```bash
export LOGGER=console
export OPENHANDS_WORKSPACE=/tmp/testbed
export ARCTIC_CORTEX_JOB_COMMENT="your name, openhands localization"

bash recipes/rl/openhands/code_localization/run_qwen35_4b.sh \
  trainer.max_training_steps=100
```

`LOGGER=console` skips Weights & Biases. `HF_HOME` and `UV_CACHE_DIR` relocate
the model cache and the wheel cache. `STARTUP_TIMEOUT=14400` extends the
placement wait past the default hour.

The script exits if `uv` is missing, `SKYRL_HOME` is wrong, a parquet is
missing, credentials do not load, or the batch would exceed Cortex's 128 MiB
response cap. Leave the process up until the run finishes. The log is
`$CKPT_DIR/$RUN_NAME.log` (default
`~/checkpoints/qwen35-4b-openhands-localization`).

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B` |
| GPUs | 4 training + 4 sampling |
| Batch | 8 prompts × 8 rollouts |
| Turns | 10 |
| Context | 32768 tokens, generation 8192 |
| Loss | GSPO, sequence mean, clip 3e-4 / 4e-4, no KL |
| Optimizer | AdamW, lr 1e-6, one update per batch, grad clip 1.0 |
| ZeRO | stage 2, FlashAttention 3, Liger off |

Liger is off because this sampler returns token ids and no logits. With Liger
off, a 40960-token sequence asks for a 37.9 GiB logit tensor, which does not
fit on an H200 next to the forward.

Each step clones the task repo, runs up to 10 turns, and scores the single
`localization_finish` call. Early `reward/avg_raw_reward` values are often
near 0. Later steps on this setup sit around 2. `policy/grad_norm` should
stay under 1.

Checkpoints are server-side every 10 steps. Ids are appended to
`$CKPT_DIR/cortex_checkpoints.jsonl`. The recipe cannot resume from them.

## Stop

Ctrl-C, `kill`, or the end of training cancels the Cortex job. A driver killed
with `kill -9` or by the OOM killer cannot. The server reclaims that job after
30 idle minutes; set `ARCTIC_CORTEX_IDLE_TIMEOUT_SECONDS` (at least 300) to
shorten that.

## Sources

| Piece | Source |
| --- | --- |
| Finish tool, F1 reward, rollout, chat proxy, Hermes parser | [codescout@abab719](https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2) |
| Qwen3.5 XML tool calls | [codescout@8184b42](https://github.com/18jeffreyma/codescout/commit/8184b42) |
| OpenHands SDK | [software-agent-sdk@85ecfd93](https://github.com/OpenHands/software-agent-sdk/commit/85ecfd9333d2d2cc4404dd460fd38868d9b978e2) |
| GRPO loop and Cortex client | SkyRL `skyrl-v0.3.0`, `arctic_platform.integrations.skyrl` |

The CodeScout prompt says `bash` and "up to 4 turns". This prompt says
`terminal`, which is the registered tool, and states the turn budget of 10.
There is no last-turn reminder.
