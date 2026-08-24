# Math GRPO

Train a model with grouped policy optimization on Hendrycks MATH and evaluate
against MATH-500. The recipe creates colocated training and sampling sub-jobs,
generates rollouts, scores them, trains, and synchronizes weights. After the
final save it logs `python -m recipes.inference.sampling.evaluate`
command.

## Hardware

The default configuration requests four training GPUs and four sampling GPUs.
Reduce or increase these values together with batch settings, and verify
capacity before submission:

```bash
cortex-training capacity
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

# Full-parameter fine-tuning
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json lora_rank=0

# Change train and sampling capacity
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  lora_rank=32 training_gpus=8 sampling_gpus=4

# Shorter generations and context
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  lora_rank=32 max_tokens=512 max_seq_len=2048

# MoE with router replay
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  model_name=Qwen/Qwen3.6-35B-A3B \
  model_provider=prime_rl ep_size=8 \
  router_replay=True \
  router_replay_max_cache_bytes=2147483648 \
  training_gpus=8 sampling_gpus=8 \
  gpu_memory_utilization=0.7 \
  lora_rank=32 \
  max_seq_len=4096 max_tokens=2048
```

## Evaluation and Logs

Training logs reward, correctness, format, rollout counts, and loss. Held-out
MATH-500 generate eval runs every `eval_every` batches (`test/env/all/correct`;
`eval_every=0` skips it). Set `wandb_project` and
`export WANDB_API_KEY` to mirror local metrics to Weights & Biases.
After save, the recipe prints one eval command.
`sampling.evaluate` uses the same few-shot prompt, grader, and sampling
settings.

```bash
python -m recipes.inference.sampling.evaluate \
  config=/path/to/config.json \
  model_name=TRAINING_MODEL_NAME \
  n_gpus=SAMPLING_GPUS \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  temperature=1.0 \
  max_tokens=4096
```
