# GSM8K GRPO with TRL on Cortex

Same task as [`../gsm8k/`](../gsm8k/), with training and sampling dispatched to
Cortex. The TRL driver is CPU-only: it builds a one-layer stub so
`AsyncGRPOTrainer` can run its loop, and Cortex owns the GPUs. ZoRRo is off.

The update is clipped GRPO (importance ratio times advantage) with
`entropy_coeff=0`. Logged `kl` is the Schulman approximation between the new
log-probs and that step's π_old. A missing or identical π_old logs `kl` of 0.

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
git -C trl checkout 3b24f2fcd85102893a9175483da2636eb49c1a22

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

This recipe (upstream TRL `3b24f2fcd`, trainer π_old, ZoRRo off, `sdpa`, seed
42) finished 250 steps in 29:35. `train_runtime` was 1775 s, about 7.1 s/step.
Step-1 training reward was 0.359 and step-1 `kl` was `2.8e-4`. The first 20
steps averaged reward 0.426 and `kl` `5.5e-4`. Last-20 training reward was
0.809, completion length 135, and `kl` `2.9e-3`. Mean `kl` over all 250 steps
was `1.3e-3`, and `kl` was nonzero on every step. `grad_norm` stayed 0.

The reward below is the mean training-batch `accuracy_reward` over the 20
steps ending at that step, not a held-out pass@1.

| Step | Reward |
| --- | ---: |
| 20 | 0.43 |
| 40 | 0.62 |
| 60 | 0.63 |
| 80 | 0.75 |
| 100 | 0.79 |
| 120 | 0.77 |
| 140 | 0.73 |
| 160 | 0.80 |
| 180 | 0.76 |
| 200 | 0.80 |
| 220 | 0.85 |
| 240 | 0.76 |
| 250 | 0.81 |

Write that run to a new `--metrics-out`. The older pins below stay where they
are.

The 250-step quality pins were measured with the pre-main Cortex client, same
knobs. They are training-batch reward over the last 20 steps. `kl` was
nonzero on every step.

| Arm | Last-20 reward | Last-20 completion length | Mean step | `kl` over 250 |
| --- | ---: | ---: | ---: | --- |
| Sampler π_old | 0.838 | 131 | 5.10 s | mean `1.2e-3`, never 0 |
| Trainer π_old | 0.734 | 184 | 6.02 s | mean `8.0e-4`, never 0 |

The previous adapter on `main` (`c40f78bca`, trainer π_old, ZoRRo off) finished
250 steps in 27:34. A 2-step run of that adapter logged `kl` `3.4e-4` then
`6.8e-4`. Over 250 steps, step-1 `kl` was `3.0e-4` and the first 20 steps
averaged `5.9e-4`. Last-20 training reward was 0.767 with completion length
126. Last-20 `kl` was `1.8e-2`, and `kl` was nonzero on every step. Mean step
time was 6.56 s.

## Files

| File | What it is |
| --- | --- |
| `run_qwen3_1.7b_gsm8k_grpo_cortex.sh` | 250-step launcher. Reads `ARCTIC_CORTEX_*` |
| `run_qwen3_1.7b_gsm8k_grpo_cortex.py` | CPU driver: stub model, `ArcticRLClient`, TRL loop |
| `arctic_trl_cortex/` | Upstream-TRL forward/backward bridge, rollout worker, and Cortex payload adapters |

`arctic_platform.integrations.trl` remains the on-prem adapter. This recipe
does not import it for the training step. It does use
`arctic_platform.integrations.trl.weights.ArcticWeightTransfer` to call
`client.sync_weights`.
