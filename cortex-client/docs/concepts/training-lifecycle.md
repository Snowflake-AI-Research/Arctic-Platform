# Training Lifecycle

A typical Cortex Training workflow has five stages:

1. Choose a model, method, dataset, precision, and hardware configuration.
2. Create one or more Cortex Training sub-jobs.
3. Submit training or sampling requests and poll them to completion.
4. Save, load, evaluate, or synchronize checkpoints.
5. Cancel completed jobs to release capacity.

Supervised fine-tuning normally needs one training sub-job. Reinforcement
learning commonly combines training and sampling sub-jobs in one job.

See [jobs and sub-jobs](jobs-and-subjobs.md) and
[checkpoints](checkpoints.md) for the underlying resource model.
