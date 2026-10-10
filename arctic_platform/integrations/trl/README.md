# Arctic RL backend for TRL

Draft, for discussion on [huggingface/trl#6676](https://github.com/huggingface/trl/pull/6676). The batch
layout helpers in `client.py` are stubs and nothing here has run against a live server.

## The interface

TRL's `TrainingClientProtocol` is a single fused call:

```python
forward_backward(model, input_ids, position_ids, completion_mask, loss_fn, aux_loss_coef) -> ForwardBackwardOutput
```

`loss_fn` is a Python callable mapping per-token log probs to a scalar. It closes over the advantages, old log
probs, mask and clipping bounds, so nothing algorithm-shaped reaches this repo.

The adapter runs in the trainer's process even though the model does not, so `loss_fn` is evaluated here, on
the tensors the server returned. Only tensors cross the wire. The adapter takes `d(loss)/d(log_probs)` locally
and ships that, and the server backpropagates `sum(w * logprobs)`, a first-order surrogate whose gradient with
respect to every parameter equals the gradient of the real loss.

Verified bit-exact against the in-process path on TRL's GRPO loss: zero difference across every parameter
tensor, for both the co-located variant and the two-forward variant a remote backend would use.

## On the fused call and the extra forward

Folding the forward and backward into one call was the request on the PR thread, and this is that call. Naming
the *loss* in it is the part worth avoiding, and passing a callable instead of a string key gets the fused
shape without it.

On cost: a co-located deployment runs one forward, calls `loss_fn` in-process, and returns a loss still
attached to the graph, so there is no surrogate and no second pass at all. A remote deployment issues
`fwd_no_grad` then `fwd_bwd`, which is two forwards unless the server retains the graph between them. That
choice is server-side and invisible to TRL: the extra forward buys not having to pin activations across a
round trip.

The cost of the alternative, naming the loss across the wire, is already measurable in this repo:

| | loss lives on | consequence |
|---|---|---|
| `integrations/verl/grpo_loss.py` | server | 523 lines reimplementing verl's `masked_mean`, `agg_loss`, `compute_policy_loss_vanilla`, `kl_penalty` on this side, supporting `loss_mode="vanilla"` only, out of the 12 entries in verl's `POLICY_LOSS_REGISTRY` |
| `integrations/trl/loss.py` | client | 1 function, ~6 lines, covers every TRL loss including ones not written yet |

TRL has 8 `loss_type` values today and adds them regularly. Mirroring a registry means an Arctic PR per
variant, per framework.

This is also the position ArcticTraining already takes in-process: `Trainer.loss` is an abstract method that
users implement in Python. Same idea, across a process boundary.

## Server-side addition

One loss function, registered as `weighted_logprob_sum` in `rl/processors` (not TRL-shaped).
It is loss-agnostic and never needs revisiting.

`pipeline._resolve_fn` falls back to a dotted-path import for names not in `LOSS_FNS`, so
`loss_fn="arctic_platform.rl.processors.weighted_logprob.weighted_logprob_sum"` resolves without the registry entry
when the module is importable in the server process. Two questions for the Arctic side:

1. Is that fallback a supported extension point or an implementation detail?
2. For managed deployments, would a generic registry entry be acceptable? Weighted cross-entropy is the same
   thing: server CE is `sum(-logprobs * weights)`, so `weights = -d(loss)/d(logprobs)` gives the surrogate.
   Tinker maps its custom-loss path onto plain `cross_entropy` and adds no loss at all.

## Loss placement: client (default) vs server

`ArcticTrainingClient(server_side_loss=...)` (run script: `--loss-placement {client,server}`
/ env `ARCTIC_TRL_LOSS_PLACEMENT`) selects where GRPO runs:

- **client** (default): the two-pass surrogate above. Evaluates TRL's `loss_fn` in-process
  on returned logprobs and ships `weighted_logprob_sum` weights for the backward.
- **server**: one fused `fwd_bwd` runs forward + GRPO loss + backward on the engine
  (the verl/SkyRL pattern), saving the extra forward. TRL never hands the adapter the raw
  advantages/old-log-probs — they live in `loss_fn`'s closure — so `_extract_grpo_ingredients`
  recovers them by `__code__.co_freevars` + `__closure__` and hands them to the `trl_grpo`
  server loss. The recovery is pinned to TRL PR #6676's variable names and fails loudly
  (pointing back at `client`) if they drift.

`trl_grpo` (in `loss.py`) reproduces TRL's exact clipped surrogate (no dual-clip/KL/ref) and
normalizes as `masked_sum / batch_num_tokens * dp_size / grad_accum_steps`. The `* dp_size`
factor makes the gradient correct after DeepSpeed's cross-DP averaging — identical to verl's
`agg_loss` token-mean. Contract: the TRL trainer is single-process
(`accelerator.num_processes == 1`), so `tokens_per_rank` is the global completion-token count
and server-side `dp_size` is the only DP correction (asserted in `_extract_grpo_ingredients`).

Server enabler: `forward_backward` on the Ray/HTTP servers drops the per-token `batch` by
default (verl never reads it); the server path opts in via `meta["return_fwd_batch"]` so the
adapter gets logprobs/entropy back for TRL's unchanged metrics block.

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
2. **Microbatch scaling.** The adapter folds the trainer's gradient-accumulation scaling into the weights
   before sending them, so they arrive pre-scaled. If `run_pipeline` applies its own per-microbatch
   normalization on top, one of the two has to be disabled.
3. **Batch layout.** The padding-free row to padded rows conversion is stubbed out. Mechanical, but it is
   where a first integration spends its time.
4. **Weight sync.** `WeightTransferProtocol` is already pluggable on the TRL side and Arctic has
   `save_weights_for_sampler`. Open question is whether that path exists for managed deployments or only
   self-hosted.
