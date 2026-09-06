# Arctic OPD backend for TRL-style async distillation

New work on Arctic `main`. Complements
[`AsyncDistillationTrainer`](https://huggingface.co/docs/trl/en/async_distillation_trainer);
it does **not** wrap sync `DistillationTrainer` and must **not** land on
[PR #84](https://github.com/Snowflake-AI-Research/Arctic-Platform/pull/84).

See also [`docs/opd.md`](../../../docs/opd.md).

The trainer process stays **CPU-only**. Generate, teacher score, student
gather/fwd-bwd, optimizer step, and weight sync run on Arctic.

TRL's `AsyncDistillationTrainer` still `from_pretrained`s a local student and
has no `training_client=`. Use `ArcticAsyncDistillationTrainer` until that
hook exists. Adapters match TRL's `rollout_worker` / `weight_transfer` /
`TrainingClientProtocol` shapes so they can be passed in later.

## Wiring (CPU driver, remote compute)

```python
from arctic_platform.opd import ArcticOPDClient, ArcticOPDClientConfig
from arctic_platform.integrations.trl_distill import (
    ArcticAsyncDistillationConfig,
    create_arctic_async_distillation_trainer,
)

client = ArcticOPDClient(ArcticOPDClientConfig(...))
trainer = create_arctic_async_distillation_trainer(
    client,
    train_prompts=[[1, 2, 3]],
    args=ArcticAsyncDistillationConfig(steps=10, teacher_top_k=16, beta=0.0),
)
trainer.train()
```

## Split

| Adapter | Role | Arctic call |
|---|---|---|
| `ArcticAsyncDistillationTrainer` | CPU loop | none |
| `RemoteStudentStub` | dummy CPU param | none |
| `ArcticOPDRolloutWorker` | TRL `rollout_worker=` | `generate` + `score_teacher_topk` |
| `ArcticOPDWeightTransfer` | TRL `weight_transfer=` | `sync_weights` (never the teacher) |
| `ArcticOPDTrainingClient.forward_samples` | CPU JSD + remote gather | `fwd_no_grad` / `fwd_bwd` |
| `ArcticOPDTrainingClient.forward_backward` | TRL `TrainingClientProtocol` | same, packed row |
| `ArcticOPDOptimizer` | TRL `optimizers=` | `step` |

JSD is evaluated on CPU from gathered logits. The server surrogate is
`sum(w * gathered_logits)`, not sampled-token `on_policy_distill`.

## Server processors

- `gather_logits_at_ids` — `batch["gather_token_ids"]` is `[B, S, K]`, same
  frame as logits.
- `weighted_gathered_logit_sum` — `sum(logit_weights * gathered_logits)`.

Native `on_policy_distill` is unchanged.
