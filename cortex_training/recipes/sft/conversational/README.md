# Conversational Supervised Fine-Tuning

Fine-tune a chat model on a `messages` column. The default dataset is a
one-example memorize task: when prompted `Who trained you?`, answer
`Snowflake AI Research`. Hugging Face chat datasets work as well. The entry point supports
LoRA and full-parameter training, logs `test/nll` on a held-out split, and
saves a weights-only checkpoint.

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

Defaults use `Qwen/Qwen3-8B`, thinking disabled, LoRA rank 32, the builtin
`who_trained_you` dataset, eight GPUs, and 100 steps. The single example is
repeated to fill those steps.

## Common Variations

```bash
# Thinking-on Qwen3 (must also pass enable_thinking=true to sample)
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  enable_thinking=true

# Full-parameter fine-tuning
python -m recipes.sft.conversational.train \
  config=/path/to/config.json lora_rank=0

# Different chat dataset
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  dataset=HuggingFaceH4/no_robots

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

On the default memorize task, `train_mean_nll` and `test/nll` should fall
quickly. After save, the recipe prints
one sample command. When running that sample command, Assistant text should be `Snowflake AI Research`.

```bash
python -m recipes.inference.sampling.sample \
  config=/path/to/config.json \
  model_name=TRAINING_MODEL_NAME \
  n_gpus=N_GPUS \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  temperature=0 \
  prompt="Who trained you?"
```

## Notebooks

- `qwen3_8b_sft_training.ipynb`
- `qwen3_8b_sft_training_multiplex.ipynb`
