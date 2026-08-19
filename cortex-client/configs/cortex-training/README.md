# Cortex Training model catalog

This directory is the customer-facing source of truth for Cortex Training
model availability, context limits, and recommended starting configurations.
It is separate from architecture-specific support lists elsewhere in
Arctic-Platform, such as ZoRRo Train model-family compatibility.

## Files

- `models.json` declares inference and training support for every model.
  Supported training models provide SFT and RL starting profiles for LoRA and
  full-parameter training.
- `profiles/*.json` contains reusable configurations expressed as arguments to
  `SubJobConfig.training_job` and `SubJobConfig.sampling_job`.
- `schema.json` documents the public catalog format.

Inference support provides one maximum context length and recommended profile.
Training support requires all four SFT/RL LoRA/full recommendations. These are
tested starting points, not universally optimal hyperparameters.

## Updating the catalog

1. Update the model capability and the referenced profile in the same pull
   request.
2. Set `lastValidated` and include checked-in evidence paths for every changed
   profile.
3. Run:

   ```bash
   python scripts/validate_model_catalog.py
   python -m pytest tests/test_model_catalog.py -q
   ```

Changes require review from both Arctic-Platform and Cortex Training product
owners through `.github/CODEOWNERS`. Snowflake product documentation vendors a
pinned snapshot of this directory; it does not download mutable data during a
documentation build.
