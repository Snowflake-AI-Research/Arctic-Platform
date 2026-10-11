# OpenHands code localization on Cortex

Train a Qwen model to localize a code change with an OpenHands agent. The
driver is CPU-only. Cortex runs a training sub-job and a sampling sub-job.
SkyRL runs GRPO. The harness, reward, and chat proxy are
`arctic_platform.integrations.openhands`.

The agent searches with the `terminal` tool and submits with
`localization_finish`. The reward is the sum of file, module, and function
F1, so the maximum is 3. A rollout that never submits inside the turn budget
is left out of the loss.

## Sources

| Piece | Source |
| --- | --- |
| Finish tool, F1 reward, rollout, chat proxy, Hermes parser | [codescout@abab719](https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2) |
| Qwen3.5 XML tool calls | [codescout@8184b42](https://github.com/18jeffreyma/codescout/commit/8184b42) |
| OpenHands SDK | [software-agent-sdk@85ecfd93](https://github.com/OpenHands/software-agent-sdk/commit/85ecfd9333d2d2cc4404dd460fd38868d9b978e2) |
| GRPO loop | SkyRL `skyrl-v0.3.0`, `integrations/arctic_rl` |

Two deliberate differences from the cited CodeScout prompt. The prompt there
says `bash`, and OpenHands registers the executor as `terminal`, so those
calls fail. This prompt says `terminal`. The cited prompt also says "up to 4
turns" while the cutoff in code is 10. This prompt states the configured
turn budget. There is no last-turn reminder.

`trainer.arctic_rl.use_liger` is false. The fused kernel rejects completions
that carry token ids and no logits, which is what this sampler returns.

## Hardware

One node, 8 GPUs: 4 for training and 4 for sampling. The driver needs no GPU.
It does need disk for one git checkout per in-flight rollout, under
`OPENHANDS_WORKSPACE` (default `/tmp/testbed`).

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/NovaSky-AI/SkyRL
git -C SkyRL checkout skyrl-v0.3.0
export SKYRL_HOME=$PWD/SkyRL
```

Cortex credentials are `cortex-training login ~/cortex-training-config.json`,
or `ARCTIC_CORTEX_HOST`, `ARCTIC_CORTEX_DATABASE`, `ARCTIC_CORTEX_SCHEMA`, and
`ARCTIC_CORTEX_PAT`.

## Data

`DATA_DIR` must contain `train.parquet` and `validation.parquet` with the
SWE-smith columns CodeScout trains on: `instance_id`, `repo`, `base_commit`,
`problem_statement`, `patch`, `use_patch`, `file_changes`, and a `prompt`
column SkyRL uses only to filter length. The default path is
`~/data/swe-smith-localization`. The parquets are the `data/swe_smith` split
in the CodeScout repo. They are not committed here.

## Run

```bash
bash recipes/rl/openhands/code_localization/run_qwen35_4b.sh \
  trainer.max_training_steps=100
```

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B` |
| GPUs | 4 training + 4 sampling |
| Batch | 8 prompts × 8 rollouts |
| Turns | 10 |
| Context | 40960 tokens, generation 8192 |
| Loss | GSPO, sequence mean, clip 3e-4 / 4e-4, no KL |
| Optimizer | AdamW, lr 1e-6, one update per batch |
| ZeRO | stage 2, FlashAttention 3, Liger off |

The first step is slow: wheels, then placement of the two sub-jobs, then the
rollouts. A step is on the order of five minutes once the jobs are up.
`CORTEX_JOB_COMMENT` is written onto the job. The id is written to
`$CKPT_DIR/cortex_job_id` before placement finishes, and the launcher cancels
that job on exit.

Stopping the driver does not release the GPUs by itself. The launcher's EXIT
trap calls `python -m arctic_platform.integrations.openhands.release`.

## What this does not do

Resume from a Cortex checkpoint. A second optimizer step on the same batch.
KL against a reference model. Docker sandboxes. The agent edits nothing. It
only names the locations.
