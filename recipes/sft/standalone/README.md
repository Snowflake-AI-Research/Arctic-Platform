# Chat SFT (cross-entropy) — standalone loop

An end-to-end supervised fine-tuning loop driving the unified client
(`arctic_platform.client`) directly, with no external training framework. Works
on any HF chat dataset with a `messages` column (default
`HuggingFaceH4/no_robots`).

The loop itself lives in `sft_loop.py`; connection, rendering, collation, and the
`fwd_bwd` + `step` pair are shared with the RL recipe in
`recipes/recipe_utils.py`.

> **Backend support.** This recipe drives the **Cortex** backend only. The client
> API is backend-agnostic, but `fwd_bwd`'s *batch* is not yet: Cortex takes an
> RPC-style `{"args", "kwargs", "context"}` body, while on-prem takes a
> pre-tokenized verl-GRPO `{"batch", "meta", "processing"}`. An on-prem path is a
> follow-up — see the TODO at the top of `recipes/recipe_utils.py`.

## 1. Prerequisites

```bash
uv pip install -e '.[cortex]'
uv pip install -r recipes/sft/standalone/requirements.txt
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

Defaults already pick a model, LoRA, GPUs, and learning rate:

```bash
# no_robots chat, Qwen/Qwen3-8B, LoRA-32, 8 GPUs, 100 steps
python -m recipes.sft.standalone.sft_loop config=recipes/config.json
```

Pass `job_id=<job>` to attach to a job that already exists instead of creating
one. A job the recipe created is cancelled on the way out; a job it attached to
is left running.

## 3. Logs

Local jsonl / config dumps are always written under `log_path`; metrics are
logged every step. Pass `wandb_project=...` (and `WANDB_API_KEY`) to mirror the
same metrics to W&B.

On the default LoRA-32 / `no_robots` / `Qwen/Qwen3-8B` setup `train_mean_nll`
typically starts around ~10 and falls toward ~5 within ~100 steps.

```bash
# Local metrics only
python -m recipes.sft.standalone.sft_loop config=recipes/config.json \
    log_path=/tmp/my-sft-run

# Weights & Biases — requires WANDB_API_KEY; wandb_project enables logging
export WANDB_API_KEY=...
# optional: export WANDB_BASE_URL=https://your-wandb-host

python -m recipes.sft.standalone.sft_loop config=recipes/config.json wandb_project=arctic
```

## 4. Training knobs

### Change model

```bash
python -m recipes.sft.standalone.sft_loop config=recipes/config.json \
    model_name=Qwen/Qwen3.6-35B-A3B
```

### Change dataset

Any HF `DatasetDict` with a `messages` column (default split: `train`):

```bash
python -m recipes.sft.standalone.sft_loop config=recipes/config.json \
    dataset=HuggingFaceH4/ultrachat_200k dataset_split=train_sft
```

### LoRA vs dense

`lora_rank` builds `TrainingConfig.peft` (a LoRA adapter with `alpha == r`),
which `to_cortex()` attaches to the training sub-job. LoRA is the default
(`lora_rank=32`).

```bash
# Dense fine-tune
python -m recipes.sft.standalone.sft_loop config=recipes/config.json lora_rank=0

# Smaller / larger LoRA
python -m recipes.sft.standalone.sft_loop config=recipes/config.json lora_rank=16
python -m recipes.sft.standalone.sft_loop config=recipes/config.json lora_rank=64
```

### GPUs and batch size

```bash
# batch_size must be a multiple of micro_batch_size * n_gpus
python -m recipes.sft.standalone.sft_loop config=recipes/config.json \
    n_gpus=4 batch_size=4 micro_batch_size=1
```

### MoE / expert parallelism (`ep_size`)

MoE checkpoints like `Qwen/Qwen3.6-35B-A3B` need Prime-RL plus expert
parallelism for full fine-tuning. Set `model_provider=prime_rl` and `ep_size` so
that `n_gpus` is a multiple of `ep_size` (e.g. 8 GPUs with `ep_size=4`):

```bash
python -m recipes.sft.standalone.sft_loop config=recipes/config.json \
    model_name=Qwen/Qwen3.6-35B-A3B \
    model_provider=prime_rl ep_size=4 \
    n_gpus=8 lora_rank=0
```

### Sequence length

```bash
python -m recipes.sft.standalone.sft_loop config=recipes/config.json max_length=4096
```
