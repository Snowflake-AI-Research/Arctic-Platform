# Weights & Biases

The current SFT and Math GRPO recipes can mirror local metrics to Weights &
Biases:

```bash
export WANDB_API_KEY=...

python -m recipes.sft.conversational.train \
  config=/path/to/config.json wandb_project=cortex-training
```

Use `wandb_name` to identify a run. A shared metric naming convention and
validated dashboards for loss, reward, evaluation, and KL remain planned.
