# Inference Endpoint

Create a standalone Cortex Training inference endpoint from open weights or a
weights-only training checkpoint. Generate or evaluate against that running job.

The generation worker is still submitted as `job_type=sampling` on the API —
the same runtime reinforcement learning uses for rollouts. This recipe treats
that worker as an inference endpoint, so you do not need the RL workflow to
serve a model.

## Hardware

The default configuration requests two GPUs. Check account capacity before
creating an endpoint:

```bash
cortex-training capacity
```

## Create an Endpoint

From original Hugging Face weights:

```bash
python -m recipes.inference.serve \
  config=/path/to/config.json
```

From a weights-only checkpoint produced by a training recipe. Pass
`source_job_id` and `checkpoint_id` together:

```bash
python -m recipes.inference.serve \
  config=/path/to/config.json \
  n_gpus=N_GPUS \
  source_job_id=TRAINING_JOB_ID \
  model_name=MODEL_NAME_USED_IN_TRAINING \
  checkpoint_id=CHECKPOINT_ID \
  lora_rank=LORA_RANK
```

The command waits until workers are up, prints `job_id`, and leaves the
endpoint running. Tear it down with:

```bash
cortex-training cancel JOB_ID
```

`inference_walkthrough.ipynb` shows the same create → generate → cancel loop
with the Python client.

## Examples

Point these at a running `job_id` from `serve`, or omit `job_id` to create a
one-shot endpoint that exits (and releases GPUs) when the example finishes.

### Generate

```bash
python -m recipes.inference.generate \
  config=/path/to/config.json \
  job_id=JOB_ID \
  prompt="Who trained you?"
```

From original weights or a checkpoint, without a pre-created endpoint:

```bash
python -m recipes.inference.generate \
  config=/path/to/config.json \
  prompt="Who trained you?"

python -m recipes.inference.generate \
  config=/path/to/config.json \
  model_name=MODEL_NAME_USED_IN_TRAINING \
  n_gpus=N_GPUS \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  lora_rank=LORA_RANK \
  prompt="Who trained you?"
```

Thinking is disabled by default, matching conversational SFT. Pass
`enable_thinking=true` to use thinking.

### Evaluate (MATH-500)

```bash
python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  job_id=JOB_ID

python -m recipes.inference.evaluate \
  config=/path/to/config.json

python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  lora_rank=LORA_RANK
```
