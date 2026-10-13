# Evaluate Before and After Fine-Tuning

Status: planned.

The target workflow is:

1. Prepare a held-out evaluation dataset.
2. Evaluate the base model.
3. Train and save a checkpoint.
4. Start sampling from the checkpoint.
5. Evaluate the trained model on the same dataset.
6. Report comparable metrics and examples.

This guide should become the default answer to "did my fine-tuning work?"
