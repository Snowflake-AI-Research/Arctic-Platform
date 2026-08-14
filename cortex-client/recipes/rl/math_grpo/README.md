# Math GRPO

Train a model with grouped policy optimization on Hendrycks MATH and evaluate
against MATH-500. The recipe creates colocated training and sampling sub-jobs,
generates rollouts, scores them, trains, and synchronizes weights.

## Hardware

The default configuration requests four training GPUs and four sampling GPUs.
Reduce or increase these values together with batch settings, and verify
capacity before submission:

```bash
dss-neutrino capacity
```

## Run

```bash
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json lora_rank=32
```

## Common Variations

```bash
# Short smoke run
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  lora_rank=32 max_steps=2 n_test=16

# Change train and sampling capacity
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  lora_rank=32 training_gpus=8 sampling_gpus=4

# Shorter generations and context
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  lora_rank=32 max_tokens=512 max_seq_len=2048
```

## Evaluation and Logs

The recipe records reward, correctness, format compliance, rollout counts,
training loss, and held-out evaluation metrics. Set `wandb_project` and
`WANDB_API_KEY` to mirror local metrics to Weights & Biases.

## Status

The GRPO loop and held-out evaluation are implemented. Reproducible hardware
benchmarks, pass@k reporting, and checkpoint-over-time plots remain planned.
