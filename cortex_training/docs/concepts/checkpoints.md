# Checkpoints and Weight Synchronization

Training checkpoints and sampling weights serve different purposes:

- Resumable training checkpoints can include optimizer state.
- Weights-only checkpoints can initialize a separate sampling job.
- Runtime load replaces weights in an existing training sub-job.
- Weight synchronization updates a sampling sub-job from a training sub-job
  during colocated reinforcement learning.

Changing data-parallel size while loading requires optimizer state loading to
be disabled when the job is created. See the
[CLI reference](../reference/cli.md#load-a-checkpoint-into-a-running-job) for
the current commands and constraints.
