# Arctic RL backend for TRL

Adapter for [huggingface/trl](https://github.com/huggingface/trl) `main` at `9141962`, which includes the merged [huggingface/trl#6676](https://github.com/huggingface/trl/pull/6676).
`ArcticAsyncGRPOTrainer` holds a one-layer model from `build_stub_model` and sends packed
rows to the Arctic server. `ArcticTrainingClient.forward_backward` turns each packed row
into a padded server batch with `_unpack_to_padded_rows`. The GSM8K and BIRD recipes call
both.

## The interface

`ArcticAsyncGRPOTrainer.compute_loss` strips padding, splits segments where
`position_ids == 0`, and calls:

```python
forward_backward(
    input_ids, position_ids, completion_mask,
    old_log_probs=..., advantages=..., loss_mask=...,
    epsilon_low=..., epsilon_high=...,
    tokens_per_rank=..., world_size=...,
) -> ForwardBackwardOutput
```

`old_log_probs`, `advantages`, and `loss_mask` are the shifted `[1, T-1]` frame
(TRL's `[:, 1:]`). `input_ids`, `position_ids`, and `completion_mask` stay the full
packed row. `ForwardBackwardOutput` is a local dataclass (`loss`, `log_probs`, `entropy`).

The adapter evaluates the clipped surrogate here, or ships those tensors to the server
loss. Only tensors cross the wire.

## Loss placement: client (default) vs server

`ArcticTrainingClient(server_side_loss=...)` (run script: `--loss-placement {client,server}`
/ env `ARCTIC_TRL_LOSS_PLACEMENT`) selects where GRPO runs:

- **client** (default): `fwd_no_grad` returns log-probs and entropy. `clipped_surrogate_sum`
  runs in the trainer process. The backward hook ships `d(loss)/d(log_probs)` as
  `logprob_weights_shifted` on a second `fwd_bwd`, whose server loss is
  `weighted_logprob_sum` (`rl/processors`).
- **server**: one fused `fwd_bwd` runs forward + GRPO + backward. The same explicit
  tensors are realigned into the server's roll(-1) `[B, S]` frame and passed to
  `trl_grpo` (`integrations/trl/loss.py`). ZoRRo uses this path and builds a fixed-width
  `[B, max_prompt_len + response_len]` batch instead of the plain padded rows.

`trl_grpo` is the clipped surrogate (no dual-clip, KL, or reference model) and
normalizes as `masked_sum / batch_num_tokens * dp_size`. `batch_num_tokens` is the
whole forward's completion-token count. DeepSpeed microbatches share that denominator,
so the loss is not also divided by engine GAS. The `* dp_size` factor corrects
DeepSpeed's cross-DP averaging. The TRL trainer is single-process
(`accelerator.num_processes == 1`), so `tokens_per_rank` is the global completion-token
count and server-side `dp_size` is the only DP correction (`_reject_multiprocess_trl`).

`weighted_logprob_sum` is registered with the other generic losses in `rl/processors`.
`resolve_fn` also accepts a dotted-path import for names not in the registry.

Server enabler: `forward_backward` on the Ray/HTTP servers drops the per-token `batch`
by default. The server path opts in via `meta["return_fwd_batch"]` so the adapter gets
log-probs and entropy back for TRL's metrics.

## What TRL changes

Nothing inside TRL. `AsyncGRPOTrainer` takes an instantiated model and has no
`training_client` argument. `ArcticAsyncGRPOTrainer` keeps the Arctic client as its
own argument, overrides `get_train_dataloader` to collate each tree row's samples into
a flat packed `[1, T]` row, and routes optimizer step, save, and load through the
client. Tree attention stays inside TRL.

## Open items

1. **Microbatch scaling.** `compute_loss` divides the returned loss by
   `current_gradient_accumulation_steps` before it reaches TRL. If `run_pipeline`
   applies its own per-microbatch normalization on top, one of the two has to be
   disabled.
2. **Weight sync on managed deployments.** `ArcticWeightTransfer.send_weights` drops
   the parameter iterator and calls `client.sync_weights`. Whether that path exists
   for managed deployments, or only self-hosted Ray, is still open.
