# Arctic Platform on-policy distillation (OPD)

`ArcticOPDClient` is a dual client: a student (DeepSpeed train + vLLM sample)
and a frozen teacher (vLLM sample only). The **user script** owns the loop;
Arctic owns model weights, `fwd_bwd`, the optimizer, and student weight sync.

Docs index: [index.md](index.md) · Shared server: [common.md](common.md) · RL: [rl.md](rl.md)

```
┌─────────────────────────────────────────────────────────────┐
│  User process (run_on_policy_distill.py)                    │
│  ArcticOPDClient                                            │
│    generate → score_teacher → fwd_bwd → step → sync_weights │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP or Ray
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Student server: DeepSpeed train + vLLM sample              │
│  Teacher server: vLLM only (training_gpus=0, never synced)  │
│  loss_fn = on_policy_distill (single-logit reverse KL)      │
└─────────────────────────────────────────────────────────────┘
```

## Stance vs TRL DistillationTrainer

**Comparable at the algorithm-family level, not as APIs or losses.** Both do
on-policy knowledge distillation in the sense of
[Agarwal et al., 2023 (GKD)](https://huggingface.co/papers/2306.13649): the
student generates completions, the teacher scores those same token ids, the
student updates to match the teacher.

They are **complementary, not interchangeable**. This tree does **not** wrap
or subclass Hugging Face TRL's
[`DistillationTrainer`](https://huggingface.co/docs/trl/en/distillation_trainer).

The closer TRL system analog is
[`AsyncDistillationTrainer`](https://huggingface.co/docs/trl/en/async_distillation_trainer)
(remote HTTP teacher, student vLLM, weight sync, no local teacher). Even that
trains the student locally with FSDP2 and uses a top-k generalized JSD, not
Arctic's single-logit k3 reverse KL.

Do not land distillation work on the GRPO `TrainingClientProtocol` draft
([Arctic-Platform PR #84](https://github.com/Snowflake-AI-Research/Arctic-Platform/pull/84)).
That adapter is GRPO-only and is out of scope here.

## TRL async distillation training client

TRL `AsyncDistillationTrainer` still loads a local student and has no
`training_client=`. Use
[`ArcticAsyncDistillationTrainer`](../arctic_platform/integrations/trl_distill/README.md)
for a CPU-only driver: generate, teacher score, gather/fwd-bwd, step, and
weight sync stay on Arctic. That path is **non-colocated**: DeepSpeed train,
student vLLM, and teacher vLLM on disjoint GPUs
(`OnPremConfig.colocate=False`), matching TRL's three-server layout. Do not
extend PR #84.

| Adapter | Role |
|---|---|
| `ArcticAsyncDistillationTrainer` | CPU-only driver; no local student load |
| `ArcticOPDRolloutWorker` | TRL `rollout_worker=` shape |
| `ArcticOPDWeightTransfer` | TRL `weight_transfer=` shape |
| `ArcticOPDTrainingClient` | `forward_samples` for the CPU trainer; `forward_backward` is the distillation `training_client=` hook |
| `gather_logits_at_ids` / `weighted_gathered_logit_sum` | Server gather + first-order surrogate |

```python
from arctic_platform.integrations.trl_distill import (
    ArcticAsyncDistillationConfig,
    create_arctic_async_distillation_trainer,
)

trainer = create_arctic_async_distillation_trainer(client, train_prompts, args)
trainer.train()
```

## vs TRL

| | `ArcticOPDClient` | TRL `DistillationTrainer` | TRL `AsyncDistillationTrainer` |
|---|---|---|---|
| Role | Dual client over Arctic RL servers | Full HF Trainer | Async trainer like `AsyncGRPOTrainer` |
| Owns the loop | No — [`run_on_policy_distill.py`](../arctic_platform/opd/examples/run_on_policy_distill.py) | Yes (`trainer.train()`) | Yes |
| Where models live | Remote DeepSpeed + two vLLM jobs | Local student + local teacher (optional vLLM for gen) | Local student (FSDP2) + remote student vLLM + remote teacher vLLM |
| Teacher signal | Logprob of the **sampled token only** (`prompt_logprobs: 0`), or top-k via `score_teacher_topk` | Full next-token distribution, chunked | Sparse **top-k** teacher distribution over HTTP |
| Loss | Single-logit reverse KL, k3 / `low_var_kl` | Generalized JSD via `beta` | Same JSD, sparse support |
| Optimizer / shard | DeepSpeed ZeRO-1 on the server | Accelerate / DeepSpeed / FSDP | FSDP2 only (no DeepSpeed ZeRO) |
| GPU layout | Native loop may `--colocate`; `ArcticAsyncDistillationTrainer` requires disjoint train / student-sample / teacher | Same process | 3 separate GPUs |
| Weight sync | `client.sync_weights()` student train → student sampler (NCCL when not colocated) | In-process or vLLM NCCL | NCCL to student vLLM |

Public Arctic surface is primitives, not a trainer:

- `generate` / `generate_teacher`
- `fwd_bwd` (`loss_fn: "on_policy_distill"`)
- `fwd_no_grad` (training-engine forward without backward)
- `step` / `sync_weights` / `save_checkpoint`

TRL surface is `DistillationTrainer(model, teacher_model, train_dataset, ...).train()`.

## Loss: why they are not the same

Arctic ([`on_policy_distill.py`](../arctic_platform/rl/processors/on_policy_distill.py),
[`scoring.py`](../arctic_platform/opd/scoring.py)):

- Default teacher scoring uses `prompt_logprobs=0` (sampled token only).
- `score_teacher_topk` requests `prompt_logprobs=k` plus an optional tail bucket.
- The native train step uses only the sampled-token logprob.
- Estimator: `exp(delta) - delta - 1` with
  `delta = log π_T(y) - log π_S(y)` (k3 / `low_var_kl`).
- This is a single-sample estimator of reverse KL `KL(π_S || π_T)`. It never
  sees the rest of the vocabulary. No `beta`, no forward KL, no JSD mixture.

TRL DistillationTrainer:

- Generalized JSD: `β KL(T || M) + (1-β) KL(S || M)`, `M = (1-β)S + βT`.
- `beta=0` → forward KL; `beta=1` → reverse KL; `beta=0.5` → JSD.
- Sync path: full (chunked) next-token distribution.
- Async path: `teacher_top_k` candidates; still a distributional divergence.

Even at `beta=1.0` (TRL reverse KL), TRL is not Arctic's native loss: TRL uses
a full or top-k distribution; `on_policy_distill` uses one token's logprob.

Shared request shape: both teacher-force-score the student's token ids with
`max_tokens=1` + `prompt_logprobs`. Both require a shared vocabulary and no
retokenization.

## Entry points

```python
from arctic_platform.opd import (
    ArcticOPDClient,
    ArcticOPDClientConfig,
    create_arctic_opd_client,
    score_teacher,
    score_teacher_topk,
    DEFAULT_PROCESSING,
)
from arctic_platform.integrations.trl_distill import (
    ArcticAsyncDistillationTrainer,
    create_arctic_async_distillation_trainer,
)
```

- Config: `arctic_platform.opd.config.ArcticOPDClientConfig`
- Client: `arctic_platform.opd.client.ArcticOPDClient`
- Teacher scoring: `arctic_platform.opd.scoring.score_teacher` / `score_teacher_topk`
- TRL async adapters / CPU trainer: `arctic_platform.integrations.trl_distill`
- GSM8K async-distill example: `arctic_platform.integrations.trl_distill.examples.run_async_distill_gsm8k`
- Loss (server): `arctic_platform.rl.processors.on_policy_distill`

## Quick start

Dry-run alignment check (no jobs):

```bash
python -m arctic_platform.opd.examples.run_on_policy_distill --dry-run
```

Live student/teacher sampling plus train steps (see the example module
docstring for 1-node and multi-node packs):

```bash
python -m arctic_platform.opd.examples.run_on_policy_distill --live \
  --student-model Qwen/Qwen3.5-4B \
  --teacher-model Qwen/Qwen3.6-27B \
  --training-gpus 1 --sampling-gpus 1 --teacher-sampling-gpus 1
```

One training step is:

1. `client.generate` — student vLLM completions + sampler logprobs
2. `score_teacher` — teacher `prompt_logprobs` on the same token ids
3. `client.fwd_bwd` — DeepSpeed forward + `on_policy_distill` + backward
4. `client.step` — optimizer
5. `client.sync_weights` — student train → student sampler only
