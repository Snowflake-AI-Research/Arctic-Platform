# Conversational Supervised Fine-Tuning

Fine-tune a chat model on a Hugging Face dataset with a `messages` column. The
current entry point supports LoRA and full-parameter training; QLoRA is planned.

## Hardware

The default configuration requests eight training GPUs. Actual requirements
depend on model size, sequence length, precision, and whether LoRA or dense
training is used. Check the account capacity before submitting:

```bash
cortex-training capacity
```

## Run

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json
```

Defaults use `Qwen/Qwen3-8B`, LoRA rank 32, `HuggingFaceH4/no_robots`, eight
GPUs, and 100 steps.

## Common Variations

```bash
# Full-parameter fine-tuning
python -m recipes.sft.conversational.train \
  config=/path/to/config.json lora_rank=0

# Different chat dataset
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  dataset=HuggingFaceH4/ultrachat_200k dataset_split=train_sft

# Different sequence length and GPU count
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  max_length=4096 n_gpus=4 batch_size=4 micro_batch_size=1

# MoE full fine-tuning
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  model_name=Qwen/Qwen3.6-35B-A3B \
  model_provider=prime_rl ep_size=4 n_gpus=8 lora_rank=0
```

`batch_size` must be a multiple of `micro_batch_size * n_gpus`. For MoE
training, `n_gpus` must be a multiple of `ep_size`.

## Logs and Expected Results

Metrics and configuration are written under `log_path`. Set `wandb_project`
and `WANDB_API_KEY` to send the same metrics to Weights & Biases.

On the default LoRA setup, `train_mean_nll` should decrease during the first
100 steps. Exact values vary by dataset and backend version.

## Notebooks

- `qwen3_8b_sft_training.ipynb`
- `qwen3_8b_sft_training_multiplex.ipynb`

## Status

LoRA and full-parameter execution are implemented. QLoRA configuration,
repeatable evaluation, and validated hardware ranges remain to be added.
