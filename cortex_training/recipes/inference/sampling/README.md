# Sampling Walkthrough

`sampling_walkthrough.ipynb` demonstrates prompt submission and generation
against a running sampling job.

## Text sampling

Text sampling with original weights, or a weights-only checkpoint from a
training job. Pass
`source_job_id` and `checkpoint_id` together.

```bash
python -m recipes.inference.sampling.sample \
  config=/path/to/config.json \
  prompt="Who trained you?"

python -m recipes.inference.sampling.sample \
  config=/path/to/config.json \
  model_name=TRAINING_MODEL_NAME \
  n_gpus=N_GPUS \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  prompt="Who trained you?"
```

Thinking is disabled by default, matching conversational SFT. Pass `enable_thinking=true` to use
thinking.

## MATH-500 eval

Evaluate MATH-500 with original weights, or a weights-only checkpoint from a
training job. Pass `source_job_id` and `checkpoint_id` together.

```bash
python -m recipes.inference.sampling.evaluate \
  config=/path/to/config.json

python -m recipes.inference.sampling.evaluate \
  config=/path/to/config.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```
