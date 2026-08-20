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
- `scripts/smoke_test_model_catalog.py` builds a selected recommendation from
  the catalog and can optionally submit it as a real PAT-authenticated job.

Inference support provides one maximum context length and recommended profile.
Training support requires all four SFT/RL LoRA/full recommendations. These are
tested starting points, not universally optimal hyperparameters. Every sub-job
must request GPUs in multiples of eight.

## Updating the catalog

1. Update the model capability and the referenced profile in the same pull
   request.
2. Include checked-in evidence paths for every changed profile. Remove
   `lastValidated` from each affected model recommendation until that exact
   model and profile completes a live smoke test, then set it to the test date.
3. Run:

   ```bash
   python scripts/validate_model_catalog.py
   python -m pytest tests/test_model_catalog.py -q
   ```

Changes require review from both Arctic-Platform and Cortex Training product
owners through `.github/CODEOWNERS`. Snowflake product documentation vendors a
pinned snapshot of this directory; it does not download mutable data during a
documentation build.

## Live job validation

Dry-run the exact request body generated from a catalog recommendation:

```bash
python scripts/smoke_test_model_catalog.py \
  --model-id Qwen/Qwen3.8-27B \
  --profile inference
```

To execute a live smoke test, set the connection values through environment
variables and add `--submit`. After the job reaches `running`, inference
profiles execute and poll a one-token `generate` request, SFT profiles execute
and poll a minimal `forward-backward` request, and RL profiles execute both.
The job is then cancelled. The PAT is intentionally not accepted as a
command-line argument, which keeps it out of shell history and process
listings.

```bash
export NEUTRINO_HOST='ACCOUNT.snowflakecomputing.com'
export NEUTRINO_DATABASE='NEUTRINO_DB'
export NEUTRINO_SCHEMA='PUBLIC'
read -s NEUTRINO_PAT
export NEUTRINO_PAT

python scripts/smoke_test_model_catalog.py \
  --model-id Qwen/Qwen3.8-27B \
  --profile inference \
  --max-context \
  --submit

unset NEUTRINO_PAT
```

Repeat `--profile` to validate multiple recommendations serially. Supported
values are `inference`, `sftLora`, `sftFull`, `rlLora`, and `rlFull`. Each
submitted job is cancelled after its data-plane probes complete, including
when a probe or later profile fails.

Pass `--max-context` to replace the recommended profile's starting
`max_seq_len` with the model/profile `maxContextTokens` value. Use this mode
when validating the maximum context length advertised in documentation.
