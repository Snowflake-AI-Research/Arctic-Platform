# Full-Parameter Fine-Tuning

The conversational SFT recipe switches to full-parameter training when
`lora_rank=0`:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json lora_rank=0
```

Full fine-tuning generally requires more memory than LoRA. For MoE models, set
the appropriate model provider and expert parallelism size. Validated hardware
ranges and checkpoint evaluation remain planned.
