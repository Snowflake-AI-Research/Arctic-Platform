# Neutrino training-loop recipes

## 1. Prerequisites

```bash
pip install -e cortex-client
pip install tinker-cookbook
```

We assume you have the config.json we've provided, under `recipes/config.json`.

## 2. Quick start

Defaults already pick a model, LoRA, GPUs, and learning rate. Start here:

```bash
# SFT — Qwen/Qwen3.6-35B-A3B, LoRA-32, 8 GPUs, lr=2e-5, 100 steps
python recipes/sft_loop.py config=recipes/config.json

# RL — Qwen/Qwen3.5-4B, 4 train + 4 sample GPUs, lr=2e-5
# Set lora_rank explicitly for LoRA
python recipes/rl_loop.py config=recipes/config.json lora_rank=32
```

---

## 3. Logging

Both recipes use tinker `ml_log.setup_logging` the same way. Local jsonl / config
dumps are always written; metrics are logged every step.

### Local metrics

```bash
python recipes/sft_loop.py config=recipes/config.json \
    log_path=/tmp/my-sft-run
```

### Weights & Biases

WandbLogger **requires `WANDB_API_KEY` in the environment**.
`wandb_project` enables W&B; `wandb_name` is optional (W&B picks a run name if omitted).

```bash
export WANDB_API_KEY=...
# optional:
# export WANDB_BASE_URL=https://your-wandb-host

python recipes/sft_loop.py config=recipes/config.json \
    wandb_project=dss

python recipes/rl_loop.py config=recipes/config.json \
    lora_rank=32 wandb_project=dss wandb_name=math-smoke
```

---

## 4. Training knobs

### Change model

```bash
python recipes/sft_loop.py config=recipes/config.json \
    model_name=Qwen/Qwen3.6-35B-A3B

python recipes/sft_loop.py config=recipes/config.json \
    model_name=Qwen/Qwen3-8B
```

### LoRA vs dense

```bash
# SFT: LoRA is the default (lora_rank=32). Dense FT:
python recipes/sft_loop.py config=recipes/config.json lora_rank=None

# Smaller / larger LoRA
python recipes/sft_loop.py config=recipes/config.json lora_rank=16

python recipes/rl_loop.py config=recipes/config.json lora_rank=16
```

### GPUs and batch size

```bash
# SFT — batch_size must be a multiple of micro_batch_size * n_gpus
python recipes/sft_loop.py config=recipes/config.json \
    n_gpus=4 batch_size=4 micro_batch_size=1

# RL — split train vs sample; watch the per-account GPU cap
python recipes/rl_loop.py config=recipes/config.json \
    lora_rank=32 training_gpus=8 sampling_gpus=4
```

### Sequence / generation length

```bash
python recipes/sft_loop.py config=recipes/config.json max_length=4096

python recipes/rl_loop.py config=recipes/config.json \
    lora_rank=32 max_tokens=512 max_seq_len=2048
```
