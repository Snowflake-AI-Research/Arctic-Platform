# R2E-Gym SWE agent on Cortex Training — run log

Second workload on this backend, after the reasoning-gym run in
[RUN_LOG.md](./RUN_LOG.md). The goal here was not a new backend feature
but a much harder question: does the Harbor → Cortex path hold up under
a *real* black-box SWE agent — multi-turn, native tool calling, 100-turn
trajectories, a container per rollout — rather than a single-turn
arithmetic task?

The reference is the prime-rl run of the same recipe, so every
hyper-parameter below was matched to it on purpose and the deviations
are listed explicitly.

## Setup

| | Ours | Reference run |
|---|---|---|
| Model | `Qwen/Qwen3.5-4B` | same |
| Harness | `mini-swe-agent-plus` (staged verbatim) | same |
| Tasks | R2E-Gym subset, 3606 instances | same |
| Sampling | temp 1.0, top_p 1.0, 32768 tok/turn, 100 turns | same |
| Optimiser | lr 1e-6, Adam β2 0.95, eps 1e-15, wd 0 | same |
| Group size | 8 | 8 |
| Loss | **GRPO (PPO-clip, Cortex server-side)** | **CISPO** |
| Trainer | **Cortex Training** | prime-rl on local GPUs |

The loss is the one deliberate deviation: CISPO needs a Cortex-side
registration, and the immediate question was whether the *recipe and
architecture* transfer, not whether we can match its final number.

Sandboxes are in-pod k3s (agent-sandbox), one container per rollout, on
a 48-CPU CPU-only pod. No GPU is used on our side at all — sampling and
`fwd_bwd` both go to Cortex over HTTP, and the agent reaches the sampler
through `DriverOpenAIGateway` on the cluster bridge.

## Real (has numbers)

A 3-step GRPO run, 96 rollouts total, landed end to end on Cortex:
rollouts collected in sandboxes, rewards graded by R2E's own test
command, advantages computed, `fwd_bwd` accepted, weights updated.

**The model solves these tasks at a reasonable rate.**

```
earned reward (pass rate)   40/96   41.7%
```

**But most of that signal is thrown away by protocol grading.**
prime-rl zeroes the reward of any trace that trips a terminal detector,
and we match that behaviour:

```
terminal-invalid            71/96   74.0%
  ... of which had solved the task and get zeroed:  22
effective reward after override   18/96   18.8%
```

The comparable run sits at 10–23 % terminal-invalid. At 74 % the reward
signal is nearly gone: with group size 8, most groups end up with
near-zero variance and therefore near-zero advantage, which is why this
is the blocker rather than anything in the loss or the transport.

**What actually fires** (a trace can trip several detectors):

| Detector | Count | Share |
|---|---:|---:|
| `unknown_tool` | 24 | 25.0 % |
| `nonempty_content` | 19 | 19.8 % |
| output-token limit / turn budget | 12 | 12.5 % |
| `exact_repeat_no_source_edit_since` | 9 | 9.4 % |
| `invalid_tool_arguments_schema` | 9 | 9.4 % |
| `missing_tool_call` | 7 | 7.3 % |

Two of these are ours, not the model's:

* **`unknown_tool` (24)** was a prompt bug. Our task wrapper told the
  agent to "submit", but `mini-swe-agent-plus` has no `submit` tool —
  submission is `echo MINI_SWE_AGENT_FINAL_OUTPUT` through
  `execute_bash`. The model dutifully called a tool that does not
  exist. Fixed by passing the bare problem statement, mirroring the
  taskset's own `get_instruction`; not yet re-measured.
* **The 12 "truncations"** all ended at exactly turn 100, which is our
  `max_turns`. They are the gateway's turn budget, not the model running
  past its output-token limit. Same terminal class either way, but it
  means output length is not the problem it looked like.

The remaining format failures (`nonempty_content`, `missing_tool_call`,
`invalid_tool_arguments_schema`) are still being split between genuine
model errors and artefacts of our OpenAI-compat parsing. Ending turns:
clean finishes at a median of 56 turns, format failures at 48, and only
3 of 96 failed within 2 turns — so this is not a systematic
first-exchange breakage.

## Shipped as code, not yet exercised

* **Micro-batched `fwd_bwd`** with length-sorted padding trim. A
  222-rollout batch at 32 k tokens each overflows a single Cortex
  request; this splits it and aggregates metrics across the pieces
  (`tests/test_micro_batching.py`, `tests/test_metric_merge.py`). The
  ceiling itself has not been probed.
* **Sampler log-probs as `old_log_probs_shifted`**
  (`tests/test_cortex_offpolicy.py`). Only meaningful for an off-policy
  loss; on the single-epoch GRPO path π_old ≡ π_new and this is inert.
  Sampler and trainer log-probs currently disagree (approx_kl ≈ −0.21),
  which is an open question about sequence reconstruction rather than
  about the loss.
* **Native tool calling in `openai_compat`** — Qwen3.5's XML envelope
  converted to OpenAI `tool_calls`, plus `<think>` handling for a
  generation prompt that opens the block and a completion that never
  closes it (`tests/openai_compat/test_tool_calling.py`).

## Not done

No convergence run. The reference curve moves 0.28 → 0.71 by step 91 of a planned
500; we have 3 steps, which says the loop closes, not that it learns.
Getting there needs the terminal-invalid rate down first, then batch 128
and ~30 steps.

## A note on the reference run

The run originally cited as the target turned out to use the **coco**
harness, not `mini-swe-agent-plus`, so it is the wrong baseline for this
recipe. The right comparison reaches reward 0.652, eviction-adjusted
0.835, at step 500 and 743 s/step. Two settings differ between those two
runs, and the `mini-swe-agent-plus` one is what we match:
`std_normalization` is true, and a linear length penalty is present.
Both are still outstanding on our side.
