# GSM8K GRPO with TRL on Cortex

Same task as [`../gsm8k/`](../gsm8k/), with training and sampling dispatched to
Cortex. The TRL driver is CPU-only: it builds a one-layer stub so
`AsyncGRPOTrainer` can run its loop, and Cortex owns the GPUs. ZoRRo is off.

The logged `kl` is not a loss term. The update is clipped GRPO (importance
ratio times advantage) with `entropy_coeff=0`. After each forward, TRL records
the Schulman approximation between the new log-probs and that step's π_old.
A real trainer-π_old step is about `1e-4` to `1e-3`. `kl` of 0 means the old
log-probs were missing or identical to the new ones.

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen3-1.7B` |
| Reward | TRL `accuracy_reward` (exact match on `#### <number>`) |
| Data | HuggingFace `openai/gsm8k` at launch (full train split, 7473 prompts) |
| π_old | `trainer` (an extra Cortex `/forward`). `sampler` is the other choice |
| Loss | Client-side clipped surrogate. The Cortex `/fwd_bwd` carries advantages |
| Attention | `sdpa` (`ARCTIC_ATTN_IMPL`) |
| Optimizer | AdamW `1e-6`, betas `0.9/0.999`, ZeRO-1, DeepSpeed gas 1, TRL gas 1 |
| Batch | 32 prompts × 8 samples, completion 256, sequence 1024, seed 42 |
| Schedule | 250 optimizer steps, `max_staleness=0` |
| GPUs | 4 training + 4 sampling, allocated by Cortex |

## 1. Install

The driver does not need a local GPU, vLLM, or FlashAttention. Install the
Cortex client from this checkout and TRL from the same pin as
[`../README.md`](../README.md). `math-verify` is what `accuracy_reward` imports.

```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
git clone https://github.com/huggingface/trl
git -C trl checkout 914196293543f609d0a8cb68b16154934e6b0c51

conda create -y -n trl_cortex python=3.12
conda activate trl_cortex
pip install -q uv
uv pip install -e ./Arctic-Platform[cortex]
uv pip install -e ./trl
uv pip install 'math-verify' datasets
```

`arctic_platform[cortex]` pulls the SnowAPI client (`aiohttp`, `tenacity`,
`torch`). It does not pull DeepSpeed or vLLM.

## 2. Set Cortex env

```bash
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<db>
export ARCTIC_CORTEX_SCHEMA=<schema>
export ARCTIC_CORTEX_PAT=<your PAT>
```

`CortexConfig` reads `ARCTIC_CORTEX_*`. If only `CORTEX_PAT` is set, the
launcher copies it to `ARCTIC_CORTEX_PAT`. Do not print the token.

There is no backend switch. This launcher constructs `backend=CortexConfig(...)`.
The on-prem GSM8K launcher does not.

## 3. Run

From this directory:

```bash
bash run_qwen3_1.7b_gsm8k_grpo_cortex.sh
```

That is the 250-step trainer-π_old recipe. A 2-step path check:

```bash
MAX_STEPS=2 bash run_qwen3_1.7b_gsm8k_grpo_cortex.sh
```

Sampler π_old, instead of the extra trainer forward:

```bash
OLD_LOGPROBS_SOURCE=sampler bash run_qwen3_1.7b_gsm8k_grpo_cortex.sh
```

Stop the driver with `kill -INT`, not `SIGTERM`. The client cancels the Cortex
job in a `finally` block. `kill -9` skips that. One live Cortex job at a time:
4 train + 4 sample is the account GPU cap.

`grad_norm=0` in the TRL log is expected. The backward ran on Cortex, and the
local optimizer only records the server step.

## 4. What a healthy run looks like

On current `main`, a 2-step trainer-π_old run (Qwen3-1.7B, 4+4, ZoRRo off,
seed 42) finished with `kl` `3.4e-4` then `6.8e-4`. That sits in the band of
the earlier 250-step trainer-π_old run below, whose step-1 `kl` was `2.9e-4`.

The 250-step quality pins were measured with the pre-main Cortex client, same
knobs. They are training-batch reward over the last 20 steps, not held-out
pass@1. `kl` was nonzero on every step.

| Arm | Last-20 reward | Last-20 completion length | Mean step | `kl` over 250 |
| --- | ---: | ---: | ---: | --- |
| Sampler π_old | 0.838 | 131 | 5.10 s | mean `1.2e-3`, never 0 |
| Trainer π_old | 0.734 | 184 | 6.02 s | mean `8.0e-4`, never 0 |

A 250-step run on this checkout's client is the reproducibility check. Do not
overwrite those pins; write a new metrics file (`--metrics-out`).

## Files

| File | What it is |
| --- | --- |
| `run_qwen3_1.7b_gsm8k_grpo_cortex.sh` | 250-step launcher. Reads `ARCTIC_CORTEX_*` |
| `run_qwen3_1.7b_gsm8k_grpo_cortex.py` | CPU driver: stub model, `ArcticRLClient`, TRL loop |
| `arctic_trl_cortex/` | Forward and `/fwd_bwd` adapters, rollout worker, training client |

`arctic_platform.integrations.trl` remains the on-prem adapter. This recipe
does not import it for the training step. It does use
`arctic_platform.integrations.trl.weights.ArcticWeightTransfer` to call
`client.sync_weights`.
