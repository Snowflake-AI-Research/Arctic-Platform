# Sizing and Batching

Training capacity depends on model architecture, parameter count, precision,
sequence length, micro-batch size, optimizer state, activation memory, and
parallelism strategy.

For the current SFT recipe:

```text
batch_size % (micro_batch_size * n_gpus) == 0
```

For MoE training, the GPU count must also be divisible by the expert
parallelism size.

A validated sizing guide and GPU requirement matrix are still planned. Record
measurements in recipe metadata rather than adding estimates without a
repeatable run.
