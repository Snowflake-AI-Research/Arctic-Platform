# Training-loop recipes

Two end-to-end recipes driving the unified client
(`arctic_platform.client`) against the remote Cortex backend. They target
different datasets:

| Recipe | Task (Loss)    | Dataset |
|---|----------------|---|
| `sft_loop.py` | Chat SFT (CE)  | HF chat datasets with a `messages` column (default `HuggingFaceH4/no_robots`) |
| `rl_loop.py` | Math RL (GRPO) | Hendrycks MATH train; MATH-500 held out for eval |

Both share `common.py` (connection, rendering, collation, the `fwd_bwd` + `step`
pair).

> **Backend support.** These recipes drive the **Cortex** backend only. The client
> API is backend-agnostic, but `fwd_bwd`'s *batch* is not yet: Cortex takes an
> RPC-style `{"args", "kwargs", "context"}` body, while on-prem takes a
> pre-tokenized verl-GRPO `{"batch", "meta", "processing"}`. An on-prem path is a
> follow-up — see the TODO at the top of `common.py`.

## 1. Prerequisites

```bash
uv pip install -e '.[cortex]'
uv pip install -r arctic_platform/client/recipes/requirements.txt
```

Copy `config.json.template` to `config.json` and fill in your Snowflake host and
PAT. To keep the PAT out of the file, drop the `pat` key and export
`CORTEX_PAT` instead.

## 2. Quick start

Defaults already pick a model, LoRA, GPUs, and learning rate:

```bash
# SFT — no_robots chat, Qwen/Qwen3-8B, LoRA-32, 8 GPUs, 100 steps
python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json

# RL — MATH, Qwen/Qwen3-8B, 4 train + 4 sample GPUs
# Set lora_rank explicitly for LoRA
python -m arctic_platform.client.recipes.rl_loop config=recipes/config.json lora_rank=32
```

Pass `job_id=<job>` to attach to a job that already exists instead of creating
one. A job the recipe created is cancelled on the way out; a job it attached to
is left running.

## 3. Logs

Both recipes use tinker `ml_log.setup_logging`. Local jsonl / config dumps are
always written under `log_path`; metrics are logged every step. Pass
`wandb_project=...` (and `WANDB_API_KEY`) to mirror the same metrics to W&B.

### SFT

On the default LoRA-32 / `no_robots` / `Qwen/Qwen3-8B` setup `train_mean_nll`
typically starts around ~10 and falls toward ~5 within ~100 steps.

### RL

On LoRA-32 / `Qwen/Qwen3-8B` you should see reward climbing very soon, and
correctness climb from ~0.7 toward ~0.9 within a few dozen steps.

```bash
# Local metrics only
python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json \
    log_path=/tmp/my-sft-run

# Weights & Biases — requires WANDB_API_KEY; wandb_project enables logging
export WANDB_API_KEY=...
# optional: export WANDB_BASE_URL=https://your-wandb-host

python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json wandb_project=arctic
python -m arctic_platform.client.recipes.rl_loop config=recipes/config.json \
    lora_rank=32 wandb_project=arctic wandb_name=math-smoke
```

## 4. Training knobs

### Change model

```bash
python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json \
    model_name=Qwen/Qwen3.6-35B-A3B
```

### Change dataset

```bash
# SFT — HF DatasetDict with a messages column (default split: train)
python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json \
    dataset=HuggingFaceH4/ultrachat_200k dataset_split=train_sft

# RL — MATH is fixed in load_math(); swap the loader to change envs
```

### LoRA vs dense

`lora_rank` builds `TrainingConfig.peft` (a LoRA adapter with `alpha == r`),
which `to_cortex()` attaches to the training sub-job and, for RL, to the
sampling engine so it can serve the adapter. RL weight sync then broadcasts only
the adapter tensors (`sync_weights(weight_format="lora")`).

```bash
# SFT: LoRA is the default (lora_rank=32). Dense FT:
python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json lora_rank=0

# Smaller / larger LoRA
python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json lora_rank=16
python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json lora_rank=64
```

### GPUs and batch size

```bash
# SFT — batch_size must be a multiple of micro_batch_size * n_gpus
python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json \
    n_gpus=4 batch_size=4 micro_batch_size=1

# RL — split train vs sample; watch the per-account GPU cap
python -m arctic_platform.client.recipes.rl_loop config=recipes/config.json \
    lora_rank=32 training_gpus=8 sampling_gpus=4
```

### MoE / expert parallelism (`ep_size`)

MoE checkpoints like `Qwen/Qwen3.6-35B-A3B` need Prime-RL plus expert
parallelism for full fine-tuning. Set `model_provider=prime_rl` and `ep_size` so
that `n_gpus` is a multiple of `ep_size` (e.g. 8 GPUs with `ep_size=4`):

```bash
python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json \
    model_name=Qwen/Qwen3.6-35B-A3B \
    model_provider=prime_rl ep_size=4 \
    n_gpus=8 lora_rank=0
```

### Sequence / generation length

```bash
python -m arctic_platform.client.recipes.sft_loop config=recipes/config.json max_length=4096

python -m arctic_platform.client.recipes.rl_loop config=recipes/config.json \
    lora_rank=32 max_tokens=512 max_seq_len=2048
```
