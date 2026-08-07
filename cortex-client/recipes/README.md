# Neutrino training-loop recipes

Two Neutrino recipes for chatting with the DSS training / sampling APIs end to
end. They target different datasets:

| Recipe | Task (Loss)    | Dataset |
|---|----------------|---|
| `sft_loop.py` | Chat SFT (CE)  | HuggingFace chat datasets with a `messages` column (default `HuggingFaceH4/no_robots`) |
| `rl_loop.py` | Math RL (GRPO) | Hendrycks MATH train; MATH-500 held out for eval |

## 1. Prerequisites

```bash
pip install -e cortex-client
pip install tinker-cookbook
```

We assume you have the config we've provided under `recipes/config.json`.

## 2. Quick start

Defaults already pick a model, LoRA, GPUs, and learning rate:

```bash
# SFT — no_robots chat, Qwen/Qwen3.6-35B-A3B, LoRA-32, 8 GPUs, 100 steps
python recipes/sft_loop.py config=recipes/config.json

# RL — MATH, Qwen/Qwen3.5-4B, 4 train + 4 sample GPUs
# Set lora_rank explicitly for LoRA
python recipes/rl_loop.py config=recipes/config.json lora_rank=32
```

## 3. Logs

Both recipes use tinker `ml_log.setup_logging`. Local jsonl / config dumps are
always written under `log_path`; metrics are logged every step. Pass
`wandb_project=...` (and `WANDB_API_KEY`) to mirror the same metrics to W&B.

### SFT

On the default LoRA-32 / `no_robots` / `Qwen/Qwen3.6-35B-A3B` setup `train_mean_nll` typically starts around
~10 and falls toward ~5 within ~100 steps.

### RL

On LoRA-32 / `Qwen/Qwen3.5-4B` you should see reward climbing very soon, and correctness climb from ~0.4
toward ~0.8 within a few dozen steps.

```bash
# Local metrics only
python recipes/sft_loop.py config=recipes/config.json \
    log_path=/tmp/my-sft-run

# Weights & Biases — requires WANDB_API_KEY; wandb_project enables logging
export WANDB_API_KEY=...
# optional: export WANDB_BASE_URL=https://your-wandb-host

python recipes/sft_loop.py config=recipes/config.json wandb_project=dss
python recipes/rl_loop.py config=recipes/config.json \
    lora_rank=32 wandb_project=dss wandb_name=math-smoke
```

## 4. Training knobs

### Change model

```bash
python recipes/sft_loop.py config=recipes/config.json \
    model_name=Qwen/Qwen3.6-35B-A3B

python recipes/sft_loop.py config=recipes/config.json \
    model_name=Qwen/Qwen3-8B
```

### Change dataset

```bash
# SFT — HF DatasetDict with train + messages column
python recipes/sft_loop.py config=recipes/config.json \
    dataset=HuggingFaceH4/ultrachat_200k

# RL — MATH is fixed in load_math(); swap the loader to change envs
```

### LoRA vs dense

```bash
# SFT: LoRA is the default (lora_rank=32). Dense FT:
python recipes/sft_loop.py config=recipes/config.json lora_rank=None

# Smaller / larger LoRA
python recipes/sft_loop.py config=recipes/config.json lora_rank=16
python recipes/sft_loop.py config=recipes/config.json lora_rank=64
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
