# Arctic RL backend for TRL

Draft, for discussion on [huggingface/trl#6676](https://github.com/huggingface/trl/pull/6676). The batch
layout helpers in `client.py` are stubs and nothing here has run against a live server.

## The interface

TRL's `TrainingClientProtocol` is two methods:

```python
forward_no_grad(model, input_ids, position_ids, completion_mask, aux_loss_coef) -> ForwardOutput
forward_backward(grad_log_probs) -> None
```

They map one to one onto `fwd_no_grad` and `fwd_bwd`. The server forwards the batch twice, once to score it
and once to backpropagate, because the graph from the first pass cannot cross the wire.

The trainer computes its loss on the log probs returned by the first call and hands back
`d(loss)/d(log_probs)`.
The server backpropagates `sum(w * logprobs)`, a first-order surrogate whose gradient with respect to every
parameter equals the gradient of the real loss. The loss itself never crosses the wire.

Verified bit-exact against the in-process path on TRL's GRPO loss: zero difference across all 27 parameter
tensors, for both the co-located variant and the two-forward variant a remote backend would use.

## Why not a fused `forward_backward(..., "grpo_loss")`

That was the suggestion on the PR thread, and `forward_backward` here does issue the fused `fwd_bwd`. The split is in
TRL's interface, not in what the backend does. Two round trips are inherent either way, because the weights
are a function of the log probs and cannot be known before the first forward. Tinker's
`forward` + `forward_backward_custom` has the same shape.

The part worth avoiding is naming the *loss* across the wire. The cost of that is already measurable in this
repo:

| | loss lives on | consequence |
|---|---|---|
| `integrations/verl/grpo_loss.py` | server | 523 lines reimplementing verl's `masked_mean`, `agg_loss`, `compute_policy_loss_vanilla`, `kl_penalty` on this side, supporting `loss_mode="vanilla"` only, out of the 12 entries in verl's `POLICY_LOSS_REGISTRY` |
| `integrations/trl/loss.py` | client | 1 function, ~6 lines, covers every TRL loss including ones not written yet |

TRL has 8 `loss_type` values today and adds them regularly. Mirroring a registry means an Arctic PR per
variant, per framework.

This is also the position ArcticTraining already takes in-process: `Trainer.loss` is an abstract method that
users implement in Python. Same idea, across a process boundary.

## Server-side addition

One loss function, registered as `weighted_logprob_sum`. It is loss-agnostic and never needs revisiting.

`pipeline._resolve_fn` falls back to a dotted-path import for names not in `LOSS_FNS`, so
`loss_fn="arctic_platform.integrations.trl.loss.weighted_logprob_sum"` resolves without the registry entry
when the module is importable in the server process. Two questions for the Arctic side:

1. Is that fallback a supported extension point or an implementation detail?
2. For managed deployments, would a generic registry entry be acceptable? Weighted cross-entropy is the same
   thing: server CE is `sum(-logprobs * weights)`, so `weights = -d(loss)/d(logprobs)` gives the surrogate.
   Tinker maps its custom-loss path onto plain `cross_entropy` and adds no loss at all.

## What TRL changes

Nothing. `training_client=` already exists on `AsyncGRPOTrainer` alongside the `rollout_worker` and
`weight_transfer` hooks, and `transformers.Trainer` already accepts `optimizers=`, which is why there is no
`optim_step` or `clip_grad_norm` in the protocol.

## Open items

1. **No local model.** The real blocker. Arctic owns the weights, but transformers and accelerate still want
   a local module to `prepare()`, and `AsyncGRPOTrainer._sync_weight` streams from
   `self.model.named_parameters()` on every rank (`async_grpo_trainer.py:1149`). Meta device crashes there
   and an empty CUDA model wastes full model memory. A stub module carrying names and shapes is the likely
   answer. TRL-side work.
2. **Microbatch scaling.** TRL scales its loss for gradient accumulation before the gradient reaches
   `forward_backward`, so the weights arrive pre-scaled. If `run_pipeline` applies its own per-microbatch
   normalization on top, one of the two has to be disabled.
3. **Batch layout.** The padding-free row to padded rows conversion is stubbed out. Mechanical, but it is
   where a first integration spends its time.
4. **Weight sync.** `WeightTransferProtocol` is already pluggable on the TRL side and Arctic has
   `save_weights_for_sampler`. Open question is whether that path exists for managed deployments or only
   self-hosted.
