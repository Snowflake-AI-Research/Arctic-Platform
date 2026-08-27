# Math RL (GRPO) — standalone loop

An end-to-end RL training loop driving the unified client
(`arctic_platform.client`) directly, with no external RL framework. Trains on
Hendrycks MATH with MATH-500 held out for eval.

The loop itself lives in `rl_loop.py`; connection, rendering, collation, and the
`fwd_bwd` + `step` pair are shared with the SFT recipe in
`recipes/recipe_utils.py`.

> **Backend support.** This recipe drives the **Cortex** backend only. The client
> API is backend-agnostic, but `fwd_bwd`'s *batch* is not yet: Cortex takes an
> RPC-style `{"args", "kwargs", "context"}` body, while on-prem takes a
> pre-tokenized verl-GRPO `{"batch", "meta", "processing"}`. An on-prem path is a
> follow-up — see the TODO at the top of `recipes/recipe_utils.py`.

## 1. Prerequisites

```bash
uv pip install -e '.[cortex]'
uv pip install -r recipes/rl/standalone/requirements.txt
```

Copy the connection template and fill in your Snowflake host and PAT. To keep the
PAT out of the file, drop the `pat` key and export `CORTEX_PAT` instead.

```bash
cp recipes/config.json.template recipes/config.json
```

`recipes/*.json` is gitignored, so a filled-in `config.json` will not be
committed.

All commands below are run **from the repo root** — the loop imports
`recipes.recipe_utils`, so it must be started with `python -m`.

## 2. Quick start

Defaults already pick a model, GPUs, and learning rate. Set `lora_rank`
explicitly for LoRA:

```bash
# MATH, Qwen/Qwen3-8B, 4 train + 4 sample GPUs
python -m recipes.rl.standalone.rl_loop config=recipes/config.json lora_rank=32
```

Pass `job_id=<job>` to attach to a job that already exists instead of creating
one. A job the recipe created is cancelled on the way out; a job it attached to
is left running.

## 3. Logs

Local jsonl / config dumps are always written under `log_path`; metrics are
logged every step. Pass `wandb_project=...` (and `WANDB_API_KEY`) to mirror the
same metrics to W&B.

On LoRA-32 / `Qwen/Qwen3-8B` you should see reward climbing very soon, and
correctness climb from ~0.7 toward ~0.9 within a few dozen steps.

```bash
# Local metrics only
python -m recipes.rl.standalone.rl_loop config=recipes/config.json \
    lora_rank=32 log_path=/tmp/my-rl-run

# Weights & Biases — requires WANDB_API_KEY; wandb_project enables logging
export WANDB_API_KEY=...
# optional: export WANDB_BASE_URL=https://your-wandb-host

python -m recipes.rl.standalone.rl_loop config=recipes/config.json \
    lora_rank=32 wandb_project=arctic wandb_name=math-smoke
```

## 4. Training knobs

### Change model

```bash
python -m recipes.rl.standalone.rl_loop config=recipes/config.json \
    lora_rank=32 model_name=Qwen/Qwen3.6-35B-A3B
```

### Change dataset

MATH is fixed in `load_math()`; swap the loader to change environments.

### LoRA vs dense

`lora_rank` builds `TrainingConfig.peft` (a LoRA adapter with `alpha == r`),
which `to_cortex()` attaches to the training sub-job and to the sampling engine
so it can serve the adapter. Weight sync then broadcasts only the adapter
tensors (`sync_weights(weight_format="lora")`). `lora_rank=0` trains dense.

```bash
python -m recipes.rl.standalone.rl_loop config=recipes/config.json lora_rank=16
python -m recipes.rl.standalone.rl_loop config=recipes/config.json lora_rank=64
```

### GPUs and batch size

```bash
# Split train vs sample; watch the per-account GPU cap
python -m recipes.rl.standalone.rl_loop config=recipes/config.json \
    lora_rank=32 training_gpus=8 sampling_gpus=4
```

### MoE / expert parallelism (`ep_size`)

MoE checkpoints like `Qwen/Qwen3.6-35B-A3B` need Prime-RL plus expert
parallelism for full fine-tuning. Set `model_provider=prime_rl` and `ep_size` so
that the training GPU count is a multiple of `ep_size`.

### Sequence / generation length

```bash
python -m recipes.rl.standalone.rl_loop config=recipes/config.json \
    lora_rank=32 max_tokens=512 max_seq_len=2048
```
