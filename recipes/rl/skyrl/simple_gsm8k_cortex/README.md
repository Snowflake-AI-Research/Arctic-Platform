# Simple (Cortex) — GRPO on GSM8K, dispatched to Cortex-training

Cortex-backend sibling of [`simple_gsm8k/`](../simple_gsm8k/README.md). Same
SkyRL / Arctic RL adapter surface, same conda env, same pinned SkyRL commit —
but every GPU op (training fwd/bwd/step, sampling generate, weight-sync) is
dispatched to **Cortex-training** via the unified `ArcticRLClient` and the
Cortex dispatch shim at
[`arctic_platform/rl/_cortex_dispatch.py`](../../../arctic_platform/rl/_cortex_dispatch.py).

Runs from a CPU-only driver (no local GPU required).

| Knob             | Value |
| ---              | --- |
| Model            | `Qwen/Qwen3-0.6B` |
| Reward           | SkyRL built-in `gsm8k` env (exact match on `#### <number>`) |
| Trainer          | DeepSpeed ZeRO-2, no offload |
| Sampling         | vLLM (TP=`NGPU_SAMPLE`, 1 engine) |
| GPU layout       | Cortex sub-jobs: `NGPU_TRAIN` training + `NGPU_SAMPLE` sampling (defaults 4 + 4) |
| Sequence lengths | prompt 512, response 1024 |

## 1. Install

Same env as the sibling `simple_gsm8k`/`txt2sql`/`long_context_qa` recipes.
If you already have `skyrl_arl`, `conda activate skyrl_arl` and skip step 2.

```bash
git clone https://github.com/Snowflake-AI-Research/SkyRL
cd SkyRL && git checkout 7636101a71f1849b6127ee10232fb277d2f31174 && cd ..
export SKYRL_HOME=$PWD/SkyRL

conda create -y -n skyrl_arl python=3.12.13
conda activate skyrl_arl
pip install -q uv
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 -U
uv pip install -r ../simple_gsm8k/requirements.txt --override ../simple_gsm8k/overrides.txt
```

## 2. Data

```bash
python ../simple_gsm8k/download_data.py --output_dir ~/data/gsm8k
```

## 3. Cortex credentials + endpoint

Set the Snowflake Programmatic Access Token and the SnowAPI endpoint fields:

```bash
export CORTEX_PAT=<snowflake PAT>
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=NEUTRINO_DB   # default
export ARCTIC_CORTEX_SCHEMA=PUBLIC          # default
export ARCTIC_CORTEX_ENDPOINT=cortex-training  # default
```

`ARCTIC_BACKEND=cortex` is set by the launcher; it overrides the ArcticRL
adapter's hard-coded `backend="local"` so **no SkyRL-side patch is needed**.

## 4. Train

```bash
bash run_qwen3_0.6b_gsm8k_grpo_cortex.sh
```

Common overrides:

```bash
NGPU_TRAIN=4 NGPU_SAMPLE=4 \        # Cortex sub-job GPU counts (default 4+4)
LOGGER=wandb WANDB_MODE=online \    # wandb (export WANDB_API_KEY first)
DATA_DIR=~/data/gsm8k \             # default
MODEL=Qwen/Qwen3-1.7B \              # default: Qwen/Qwen3-0.6B
bash run_qwen3_0.6b_gsm8k_grpo_cortex.sh
```

Any SkyRL Hydra override passes through:

```bash
bash run_qwen3_0.6b_gsm8k_grpo_cortex.sh \
    trainer.train_batch_size=128 \
    generator.n_samples_per_prompt=8
```

## How this is wired

- `trainer.override_entrypoint=integrations.arctic_rl.entrypoint` — SkyRL
  dispatches into the Arctic RL × SkyRL glue in `$SKYRL_HOME/integrations/arctic_rl/`,
  same as `simple_gsm8k/`.
- `ARCTIC_BACKEND=cortex` — routes `create_arctic_rl_client(config)` through
  `arctic_platform.rl._cortex_dispatch.create_cortex_client`, which wraps the
  unified `arctic_platform.client.ArcticRLClient` (Cortex transport) in a
  small async shim exposing the surface the adapter expects.
- `trainer.placement.colocate_all=false` — Cortex splits training and
  sampling into separate sub-jobs, so SkyRL must not try to grab a local
  placement group.
- `trainer.algorithm.use_kl_loss=false` / `use_kl_in_reward=false` — Cortex
  has no `/forward` or `/log_probs` endpoint for a ref model; run pure GRPO.

The shim translates SkyRL/verl `fwd_bwd` payloads onto Cortex's canonical
shape (`{args, kwargs, context, processing}`) and **omits
`old_log_probs_shifted` from context**. Cortex's `grpo` loss defaults
`old_log_probs = logprobs.detach()` when the field is absent, which is the
correct π_old for the single-epoch on-policy regime (`update_epochs_per_batch=1`)
this recipe runs. This also matches the canonical Cortex loop at
`Arctic-Platform/examples/cortex-client/recipes/rl_loop.py`, which never
populates the field and never calls `fwd_no_grad`.

## Verified run (QA6)

`4 + 4` GPUs, `Qwen/Qwen3-0.6B`, 1 epoch (116 steps), ~1 h wall:

- Held-out **pass@1 on GSM8K validation: 32.9 %** (in-loop eval at step 116).
- Training `reward/avg_raw_reward`: 0.24 (steps 1–10) → 0.28 (mid) → **0.29** (last 20).
- `approx_kl ≈ 0`, `clip_ratio = 0`, `entropy ≈ 0.55`, `grad_norm ≈ 0.36` — clean single-epoch on-policy.
- `tokens/sec/GPU ≈ 6.5 k`, ~26 s/step steady, no NCCL errors.

## Known gaps

- **Model weights aren't retrievable client-side.** The Arctic RL adapter's
  `save_checkpoint` drops the local `ckpt_dir` and delegates to the client;
  our shim forwards to Cortex's `POST save` but discards the response.
  Cortex's job payload also doesn't surface a durable checkpoint URI, so
  after the training sub-job is cancelled there is no addressable artifact.
  Follow-up: capture `checkpoint_id`, add an eval-only sampling-job path.
- **8 + 8 GPUs frequently 503s** on QA6; 4 + 4 is reliable.
