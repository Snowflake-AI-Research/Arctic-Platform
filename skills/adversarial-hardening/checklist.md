# Probe checklist

Companion to [probe.md](probe.md). Each bullet is an **attack to construct and run**, not a box to glance at. For the surfaces you chose, walk every recipe that can hit them — do not sample. Do not dump this list into the test suite. Do not treat it as last month’s bug tour.

## Inputs and boundaries

- Build an empty / zero-length / all-masked / all-padding batch and follow it through loss and the worker. If every test uses a happy batch, that is the attack.
- Single element vs many; first and last index. Off-by-one only counts if you name the index you used.
- Drop a required key. Is it a clear error or a bare `KeyError` mid-step?
- Pass `None` where a value is expected; pass the default that silently changes meaning (`bool | None = None` vs `bool = False`).
- NaN / inf / negative where positive is assumed; a size that does not divide (`count % batch != 0`).

## Client ↔ server contracts

- Send what the client allows and the server rejects (and the reverse). Validate the strict side on the permissive side.
- Send every advertised kwarg and see if the intended backend applies it. Check the current client signature (`learning_rate`, checkpoint `path` vs `checkpoint_id`) before constructing the call; do not revive removed kwargs.
- Change one side of a duplicated constant / hardcoded list (`if x in ("a", "b")` vs `x in REGISTRY`) and see who drifts.
- Encode→decode a payload with the awkward value (string id, omitted optional, extra field).
- Canonical `JobId` is `int | str`; test `Request.job_id`, `JobHandles`, and control-envelope `sub_job_id` routing without coercing handles. On-prem uses ints; Cortex may use strings.
- Exercise a shared operation through remote Cortex HTTP and in-process on-prem Ray. Assert the canonical routing/envelope and each backend's payload contract separately: `fwd_bwd` batch contents and step metrics are not identical across backends. Omit an optional field and verify it stays omitted.
- Configuration is nested under `training` / `sampling`; backend `colocate` is launch state, not a per-call kwarg. Unset `sync_weights(cuda_ipc=None, low_memory=None)` options stay omitted so training-job defaults apply; a non-`None` value overrides one call. Attack: omit vs override vs stale job config.
- Probe weight-sync backend restrictions and safety: Cortex-only `weight_format="lora"` / `bucket_size`, rejection of full-weight sync for trainable LoRA, and checksum mismatch raising by default. Cortex sleep/wake no-ops are intentional because Cortex does not colocate; do not classify them as dropped operations.

## Config, schedules, scaling

- Compute the same quantity at `world_size == 1` and `world_size > 1` (per-device vs global batch; per-shard vs global mean).
- After DP all-reduce, is the mean still a mean?
- Drive the scheduler with the wrong horizon (epochs vs real step count); collide `max_steps` with `num_epochs`.
- Packed vs dense: loss normalization, position ids, attention mask after pack/unpack. If production is `pack=False`, the `pack=True` path is the attack (tests likely skip it).
- Weight-sync / checkpoint / logprobs: treat a rank-local ZeRO shard as the full param and see who notices.
- NCCL vs CUDA-IPC vs CPU-file; `low_memory` streams one param. Wrong path for `colocate` / GPU-resident weights.

## SFT / loss / kernels

- Labels with `-100`: shift, clamp-to-safe-index, then mask. A gather on raw `-100` is an OOB read — construct that label tensor.
- `logits_optimization` `none` / `compute` / `memory`: same seed/batch; compare **backward** (`grad_norm`, `lm_head.grad`) on a large vocab. Forward match is not the attack. A test that only checks finite/nonzero grads is a test hole if the claim is same-math.
- Force the CPU fallback vs the CUDA kernel (`flat_logits.is_cuda`). A CPU torch helper does not prove Triton/flash-attn; distinguish that from a CPU Cortex client submitting real GPU work.
- Autocast / dtype: bf16 logprobs vs fp32 CE reduction.

## Control flow and silent loss

- A trailing / filtered batch that disappears with no warn/error.
- `break` / `continue` that skips cleanup or a final flush — hit the error path, not the success path.
- Epoch/stream boundary that under/over-counts steps.
- GAS remainder: invent a short last microbatch and see if the engine requires a fixed count (a “flush” here has crashed the server).

## Failure, resources, lifecycle

- Fail mid-init. What server / Ray actor / file / lock / GPU is left?
- Call twice (retry, resume, second `step`). Double-count or corrupt?
- Bare `except` — raise underneath it.
- pytest-xdist: ports from `get_unique_port_number` + `reserve_free_port` (worker’s 8-port block), not `bind(0)`. A Ray/NCCL hang in parallel is **env** until it reproduces serially.

## Dead ends and drift

- Run the example/demo. Wrong client, stale path, removed kwarg.
- Import every name in `__all__`.
- A doc sentence that the next ten lines of code contradict — construct the call the doc promises.
