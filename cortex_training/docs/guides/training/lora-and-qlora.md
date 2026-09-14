# LoRA and QLoRA

LoRA training is implemented by the
[conversational SFT recipe](../../../recipes/sft/conversational/README.md).
Set `lora_rank` to a positive value:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json lora_rank=32
```

QLoRA is a planned extension. The placeholder configuration under
`recipes/sft/conversational/configs/qlora.yaml` documents the intended shape,
but the current recipe does not consume it or configure quantized training.
