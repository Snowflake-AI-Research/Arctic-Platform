# Bring your own agent: OpenHands code localization

Train a model inside the agent you already run. The agent keeps its own loop,
tools, and sandbox. Training changes where it sends model calls: each rollout
gets its own OpenAI-compatible URL and API key, served by the policy being
trained. This recipe is the worked example, with an unmodified
[OpenHands](https://github.com/OpenHands/software-agent-sdk) agent.

The task is code localization: given a bug report, name the file, class, and
function to change. The agent searches with `terminal` and submits with
`localization_finish`. The reward is file F1 + module F1 + function F1
(maximum 3). Qwen3.5-4B trains with GRPO. SkyRL drives the loop from a CPU
machine, and Cortex runs 4 training and 4 sampling GPUs.

## How an agent plugs in

The agent connects in two places:

1. **The model endpoint.** `submit` receives a `base_url` and an `api_key` for
   the rollout. The agent's OpenAI client uses them. To the agent, a training
   rollout looks like ordinary inference.
2. **The rollout lifecycle.** An adapter with three methods:
   `submit(request) -> rollout_id`, `get_status(rollout_id)` returning
   `running`, `completed`, or `errored`, and `collect_output(rollout_id)`
   returning the artifacts.

A grader turns the artifacts into a reward. The agent never sees the reward.
The endpoint records every model call with the token ids the policy sampled,
and reuses them when the next call continues the conversation, so training
sees exactly what the model generated.

| Piece | This example |
| --- | --- |
| `submit` | Clones the task repository and starts an OpenHands conversation in a thread, with the rollout's `base_url` and `api_key` as its LLM settings |
| `get_status` | The thread's state |
| `collect_output` | The conversation and the `localization_finish` locations |
| Grader | `grade_localization`: F1 sum. A rollout that uses all 10 turns without submitting is left out of the loss |

The adapter and grader are in `openhands_localization/rollout.py`. The wiring
is `get_generator` in `openhands_localization/entrypoint.py`. The endpoint
(`ChatCompletionsProxy`) and the generator (`HarnessGenerator`) are in
`arctic_platform.integrations.harness` and know nothing about OpenHands.

## Use your own agent

1. Copy this directory and rename the `openhands_localization` package.
2. Write an adapter and a grader, then build them in `get_generator` in place
   of `OpenHandsAdapter` and `grade_localization`. Keep the endpoint and
   `HarnessGenerator`.
3. In the run script, point `trainer.override_entrypoint` at your package and
   replace the OpenHands `--with` packages with your agent's.
4. Write `train.parquet` and `validation.parquet`. Each row needs a `prompt`
   column of chat messages. The adapter and grader read the other columns from
   `request.env_extra`.

An adapter that runs the agent as a command:

```python
import os
import subprocess
import tempfile
import threading

from arctic_platform.integrations.harness.generator import Grade
from arctic_platform.integrations.harness.generator import RolloutRequest


class CommandAdapter:
    """Runs the agent as a subprocess pointed at the rollout's model endpoint."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.runs = {}
        self.lock = threading.Lock()

    def submit(self, request: RolloutRequest) -> str:
        workdir = tempfile.mkdtemp(prefix="rollout-")
        env = dict(os.environ, OPENAI_BASE_URL=request.base_url, OPENAI_API_KEY=request.api_key)
        task = request.env_extra["problem_statement"]
        process = subprocess.Popen([*self.command, task], cwd=workdir, env=env)
        with self.lock:
            self.runs[request.routing_key] = (process, workdir)
        return request.routing_key

    def get_status(self, rollout_id: str) -> str:
        with self.lock:
            process, _ = self.runs[rollout_id]
        if process.poll() is None:
            return "running"
        return "completed" if process.returncode == 0 else "errored"

    def collect_output(self, rollout_id: str) -> dict:
        with self.lock:
            _, workdir = self.runs.pop(rollout_id)
        answer = os.path.join(workdir, "answer.txt")
        if not os.path.exists(answer):
            return {}
        with open(answer) as f:
            return {"answer": f.read().strip()}


def grade(episodes, artifacts, request) -> Grade:
    correct = artifacts.get("answer") == request.env_extra["expected_answer"]
    return Grade(reward=float(correct), reward_detail={"correct": float(correct)})
```

- `submit` must return without waiting for the agent. Rollouts in a batch run
  concurrently, and the methods are called from worker threads.
- The endpoint takes OpenAI chat completions with `n=1` and no streaming.
  `tools` work; Hermes JSON and Qwen3.5 XML tool calls come back as
  `tool_calls`. `chat_template_kwargs` goes to the chat template.
- `Grade(reward, reward_detail, exhausted=False)`. Numeric `reward_detail` keys
  are logged as `reward/<key>`. `exhausted=True` drops the rollout from the
  loss.
- A call that does not continue the conversation, such as a compaction
  summary or a subagent, starts a new context. Set
  `generator.step_wise_trajectories=true` to train on every context.

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

The tasks come from the Hugging Face dataset
[`OpenHands/SWE-smith-py-code-search`](https://huggingface.co/datasets/OpenHands/SWE-smith-py-code-search).
From the `Arctic-Platform` checkout, download it and write the training files:

```bash
uv run recipes/rl/openhands/code_localization/prepare_data.py
```

That writes 39,187 training and 100 validation rows to
`~/data/swe-smith-localization`, which is where the run script looks. To keep
them elsewhere, pass `--out DIR` and set `DATA_DIR=DIR` when you run.

The script pins the dataset revision and reproduces CodeScout's `data/swe_smith`
split. It adds the chat `prompt` and the `target` locations, sets `use_patch`
because a SWE-smith patch introduces the bug, and holds out 100 shuffled rows
for validation.

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
missing, credentials do not load, the batch would exceed Cortex's 128 MiB
response cap, or a cached Arctic Platform wheel would hide this checkout.
Leave the process up until the run finishes. The log is
`$CKPT_DIR/$RUN_NAME.log` (default
`~/checkpoints/qwen35-4b-openhands-localization`). Each rollout's conversation,
reward, and submission are saved under `$CKPT_DIR/trajectories/step_<n>/`.

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
| Tasks | [`OpenHands/SWE-smith-py-code-search`](https://huggingface.co/datasets/OpenHands/SWE-smith-py-code-search) via CodeScout `src/build_dataset.py --use_patch` |
| Finish tool, F1 reward, rollout, chat proxy, Hermes parser | [codescout@abab719](https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2) |
| Qwen3.5 XML tool calls | [codescout@8184b42](https://github.com/18jeffreyma/codescout/commit/8184b42) |
| OpenHands SDK | [software-agent-sdk@85ecfd93](https://github.com/OpenHands/software-agent-sdk/commit/85ecfd9333d2d2cc4404dd460fd38868d9b978e2) |
| GRPO loop and Cortex client | SkyRL `skyrl-v0.3.0`, `arctic_platform.integrations.skyrl` |

The CodeScout prompt says `bash` and "up to 4 turns". This prompt says
`terminal`, which is the registered tool, and states the turn budget of 10.
There is no last-turn reminder.
