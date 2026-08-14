# Training Configuration

The current recipes build Neutrino training configuration from typed Python
classes and `name=value` command-line overrides. Important groups include:

- Model, precision, and provider
- GPU count and parallelism
- Sequence length, batch size, and gradient accumulation
- Optimizer and gradient clipping
- LoRA or full-parameter method
- Checkpoint, evaluation, logging, and W&B settings

A shared typed YAML/Python configuration layer is planned under
`dss_client/config/`. Until it exists, the recipe `Config` classes and the
[REST API schema](../rest-api.md#8-create-job-schemas) are authoritative.
