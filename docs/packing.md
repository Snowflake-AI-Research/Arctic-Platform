# Packing

One packer for every training path: SFT and RL, with and without sequence parallelism. Code: `arctic_platform/common/packing/`. DSS imports this package and keeps no copy of its own.

## Why it is its own package

Every training path needs this, and all of them have to agree on where a row begins and ends, so there is one contract in one place. A per-path packer would put two answers to that question in the tree, and they would first disagree at a model call, far from the edit. Dispatch and the worker sit on opposite sides of the wire and call the same functions, so entry points take their inputs as arguments rather than reading a process group or a config.

`IGNORE_INDEX` is defined here and imported by the DSS SP loss path and the fused lm_head, and DSS `BalancedBatch` shards a batch through `map_batch_values` and looks its row tensors up with `batch_tensor`.

## Contract

| Step       | Function                                                               | Shape                                       |
| ---------- | ---------------------------------------------------------------------- | ------------------------------------------- |
| Plan       | `token_budget_groups(valid_lengths, max_tokens_per_mb)`                | rows -> groups of row indices               |
| Pack       | `pack_microbatch(data)`                                                | `[B, S]` -> `[1, T]` + `PackMetadata`       |
| Pad        | `pad_packed_microbatch(packed, metadata, multiple)`                    | `[1, T]` -> `[1, T']`, `T' % multiple == 0` |
| Split      | `token_shard(packed, metadata, shard_count, shard_index)`              | `[1, T']` -> `[1, T'/shard_count]`          |
| Boundaries | `window_cu_seqlens(metadata, shard_count, shard_index)`                | window -> one varlen segment per packed row |
| Restore    | `window_row_pieces(...)` / `unpack_output(...)`                        | one window -> row pieces / a call -> rows   |

`split_groups_to_count(groups, valid_lengths, target_count)` raises a plan to a given number of calls by splitting its widest groups; `token_budget_microbatch_groups(batch, max_tokens_per_mb, min_groups)` is a batch-level convenience that reads the lengths off a batch and applies both, used by tests rather than by dispatch.

Sequence parallelism runs this same table with `shard_count = sp_size`. The DSS `docs/sequence_parallelism.md` covers the schedule.

Invariants to preserve:

- The plan reads only row token counts, so every rank derives the same plan from the same shard without communicating.
- One call count covers a whole request, because a rank entering a call after its peers have returned waits in a collective alone. A shard that needed fewer calls is raised to the count by splitting groups (`split_groups_to_count`) rather than by adding calls with no tokens in them; a shard with fewer rows than the count comes back short and its worker pads with zero-loss repeats. The DSS `docs/sequence_parallelism.md` covers the schedule.
- Concatenating a call's windows in index order reproduces the packed call. SP attention relies on that when it rebuilds varlen boundaries from gathered `position_ids`.
- `shard_count == 1` is the single-shard case, not a bypass: non-SP training runs the same table, with the one window covering the whole packed call.
- Token validity comes from `position_ids` unless an `attention_mask` is present. Requests normally ship positions only. `token_validity_mask` is the single answer to it: DP balancing measures its rows with the same function, so a row cannot be assigned to a rank as one length and then billed to that rank's token budget as another. A batch that states neither signal is refused here, because a packed call built from guessed lengths puts pad tokens through the model; balancing, whose assignment is only a heuristic, falls back to a proxy instead.
- A window's varlen boundaries hold one segment per packed row, empty for rows the shard does not reach, because loss reductions weight per rollout and index those segments by row. Tail padding joins the last segment: its `IGNORE_INDEX` labels and zero weight contribute nothing, and the boundaries still cover the window exactly.
- Anything aligned against a token's neighbour -- a shift, a roll -- happens before packing. After packing, a row's neighbour is the next row.

## Cost

One index and one kernel per tensor, not per row. `packed_token_index(valid_lengths, cu_seqlens, sequence_length)` gives every packed token's position in the flattened `[B * S, ...]` batch; packing is an `index_select` against it and unpacking an `index_copy_`, for any trailing shape and any number of leaves. The index is built once per call and reused for every leaf, so cost tracks tokens rather than rows: on an H200, packing six `[B, S]` leaves plus a `[B, S, 512]` leaf takes 0.36 ms at 8, 32 and 128 rows.

Two rules keep it there:

- Read tensor contents into Python once per call, never once per row. A `.item()` inside a row loop blocks the host on the accelerator every iteration: at 128 rows that is 27.6 ms instead of 0.36.
- Rewrite a batch through `map_batch_values`, which walks sub-dicts and so reaches RL's `context` leaves as well as the top-level inputs. Every rewrite here goes through it -- packing, padding, window slicing, and `select_rows`, which takes a group's rows out of a shard. A rewrite that misses `context` leaves two token widths in one batch, which passes dispatch and fails at the model call.

## What a worker receives

Its window of each packed call, plus metadata. No token rows: carrying those beside the windows would put every token on the wire twice. For an RL shard of seven `[B, S]` leaves at `sp=2`, the rows would be 53% of the payload when they are full and 69% when they are half padding, since packing drops the padding and a row chunk keeps it.

The metadata:

- `mb_groups`, the row plan: which rows each call holds, in order. Row-aligned inputs are validated against it, so nothing needs a token tensor to count rows.
- `pack_meta`, one `PackMetadata` per call, which turns a returned window back into rows.
- The shard's `[rows, sequence]` shape, for logging and for shaping degenerate replies.
- The SFT request config. Fractional token weighting and pre-shifted labels are properties of the whole batch, so a rank holding windows cannot derive them. The head states both before sharding, and a batch without the config is rejected by the model-call schedule check.
- Row-aligned lists the worker projects onto its own window (action masks, router-replay ids), and per-row tensors, which are not token data.

A worker answers with row pieces: `window_row_pieces` cuts its window at the row boundaries, and the driver concatenates each row's pieces across the siblings. `unpack_output` is the other direction of the same map -- a whole packed call back to `[B, S]` rows -- so reading a shard's rows means concatenating its siblings' windows and unpacking the result (`tests/sharding_test_utils.py::packed_rows`). A sharded model call rests on that same concatenation.

## Non-SP hybrid model boundaries

At SP=1, HF/Liger models with linear-attention layers (such as Qwen3.5 and
Qwen3-Next) need explicit boundaries for both the causal convolution and
GatedDeltaNet recurrence. `_derive_varlen_model_kwargs(packed_mb, pack_metadata)`
uses `PackMetadata.cu_seqlens` to derive `cu_seq_lens_q`, `cu_seq_lens_k`,
`max_length_q`, `max_length_k`, and `seq_idx`. Caller-supplied `position_ids`
are preserved, including offsets; they are not a source of row boundaries.
Any divisibility padding becomes separate one-token model segments.

Full-attention consumers also receive metadata-derived masks through
`_route_varlen_attention_masks`. One text-backbone hook uses Transformers'
`create_causal_mask` and `packed_sequence_mask_function` with `seq_idx`, not
positions. Eager, SDPA, and FlexAttention therefore retain their native mask
formats even when positions are consecutive across a row boundary.
When the metadata maximum length spans the entire call, there is only one
non-empty segment and no separate padding segments. The hook then uses the
native causal mask without a packing overlay, preserving SDPA's no-mask fast
path. Empty rows do not disable this optimization; position values never select it.
FlashAttention keeps its native cumulative-length path. The prepared mask is
threaded through the call's layer kwargs and selected by a shared full-attention
decoder-forward adapter. Mask construction stays outside compiled/checkpointed
layers, and equivalent decoder layers can reuse a compiled graph without
per-layer hook-ID guards. Routing is installed only for the affected text
backbone; it does not alter vision or linear-attention masks.

SFT, RL, and forward-only select metadata by the scheduled call's source index.
A zero-loss repeat therefore receives the repeated call's metadata, not the
metadata at the schedule index. The helper has no SP dependency: eligibility
is checked at worker call sites. SP>1 and unrelated providers keep their existing
boundary handling; this does not enable sequence parallelism at SP=1.

SP1 eligibility is read from the loaded text layer schedule. An explicit
all-full-attention schedule does not require GDN kernels, even if its config
retains unused linear-attention dimension fields. This does not change the
pre-load SP topology metadata or SP>1 policy.

Hybrid initialization requires `causal-conv1d` with explicit `seq_idx` support and
`flash-linear-attention` recurrence with explicit `cu_seqlens` support. Ordinary
HF torch fallbacks are rejected because they do not isolate packed rows; no
replacement kernel is installed by this path. This matches the non-SP hybrid
kernel policy in the companion `dss-client/DSS-Server-API-Specification.md`.
For composite Qwen3.5, a vision-encoder pre-hook removes text boundary kwargs
before vision execution, preserving the vision tower's own boundaries.

Focused regressions are in `tests/test_qwen35_sp1_packed_gdn_isolation.py` and
`tests/test_packed_hybrid_worker.py`. The CPU kernel doubles test plumbing only.
Numerical CPU tests use mixed linear/full-attention models on eager and SDPA,
including consecutive cross-row offsets and zero-loss scheduled repeats.
`tests/test_packed_hybrid_compile.py` checks graph reuse beyond the default
Dynamo recompile limit and fullgraph frequency checkpointing with a CPU graph
backend; the GDN kernels remain explicit test doubles.
The RL/forward-only worker tests use Arctic-Platform processors. CI installs the
revision pinned by the GPU extra; for mandatory local worker acceptance,
configure `PYTHONPATH` with the matching `Arctic-Platform` checkout and run both
files with no optional-dependency skips:

```bash
PYTHONPATH=".:$DSS_ARCTIC_PLATFORM_ROOT" python -m pytest -q \
  tests/test_qwen35_sp1_packed_gdn_isolation.py tests/test_packed_hybrid_worker.py
```

The real-kernel check is separate and requires a configured CUDA environment:

```bash
python -m pytest -q --run-integration tests/test_packed_hybrid_cuda.py
```

It covers mixed linear/full-attention HF/Liger models with eager, SDPA, and
FlashAttention 3: packed-versus-separate logits and parameter gradients,
neighbor swaps, consecutive cross-row offsets, and convolution width 4.
These test calls explicitly request deterministic FlashAttention backward on
the fixture's 32-dimensional heads, independent of
`FLASH_ATTENTION_DETERMINISTIC`. The exact neighbor-swap oracle first requires
an identical-call repeat to produce bit-exact logits and gradients. This policy
is scoped to the acceptance test, not production training.
Missing CUDA or either required kernel fails the explicit run; CPU results
do not substitute for it.

## What a regrouping may change

Packing decides how many rows share a model call, so the same step can run as one wide call or several narrow ones. Every token keeps its own weight either way, so in exact arithmetic the loss and every gradient are the same. The loss behaves that way in practice, to an fp32 floor. Gradients do not: regrouping reorders the sums the token-mixer kernels carry out, and bf16 rounds at every partial.

The scale of that rounding is what makes it easy to misread. Measured on one H200 with the tiny hybrid config, two rows of 64 tokens, comparing one packed call against two single-row calls, and reading each delta against a one-ulp input perturbation of the same run (worst relative L2 delta over the module's parameter gradients):

| Path                                      | Regrouped   | Same schedule twice | Input nudged one bf16 ulp |
| ----------------------------------------- | ----------- | ------------------- | ------------------------- |
| GatedDeltaNet (`A_log`, `dt_bias` worst)  | `9.109e-03` | `0.000e+00`         | `2.787e-02`               |
| Gated softmax attention (`k_proj` worst)  | `2.870e-03` | `2.581e-05`         | `1.526e-02`               |

Both paths move less under a regrouping than under the smallest change bf16 can represent in their inputs, so the deltas are rounding rather than a boundary or weighting defect. The recurrent delta rule is the sensitive one: a relative `3.9e-3` at its input becomes `2.8e-02` at its `A_log` gradient. Flash attention adds a second effect -- its backward uses atomics, so repeating the identical schedule already moves gradients by `2.6e-05`.

No dtype removes this. The expert dispatch kernels are bf16-only (`intranode.cu` rejects other types), fla's delta-rule backward does not compile in fp32, and the pure-torch delta rule takes no `cu_seqlens`, so it cannot express a packed call at all. Widening the lm_head accumulator is possible and worth doing (DSS `docs/lm_head_gradient_precision.md`), but it covers the head, not the mixers beneath it.

What follows for tests: a cross-schedule gradient comparison cannot be given a tight absolute bound, and a loose one asserts nothing. `tests/sp/test_token_mixer_grouping.py` states the invariant that does hold -- a regrouping moves a gradient no further than one input ulp does -- and measures both quantities in the run that asserts them, so the bound is not a constant. It also merges two rows into one segment and requires that delta to exceed the rounding scale, which is what shows the comparison can see a real boundary defect. `tests/sp/test_rl_call_schedule.py` pins the schedule instead: the call counts, and the loss to `1e-6`.

## Callers

DSS imports `arctic_platform.common.packing` from these sites:

- `dss/ray_dss/jobs/gpu/sp/data_plane.py` -- dispatch: plan, select a group's rows, pack, pad, split per rank.
- `dss/ray_dss/jobs/gpu/training.py` -- worker: consume packed calls, map results back to rows.
- `dss/ray_dss/jobs/gpu/batching.py` -- `BalancedBatch` shards a batch with `map_batch_values` and finds its row tensors with `batch_tensor`, so DP sharding reaches the same leaves the packer does, and weighs those rows for its LPT assignment with `token_validity_mask`.
- `dss/ray_dss/jobs/gpu/sp/loss.py`, `dss/ray_dss/jobs/gpu/sp/sft.py`, `dss/ray_dss/jobs/gpu/sp/__init__.py`, and `dss/ray_dss/jobs/models/moe/layers/lm_head.py` -- `IGNORE_INDEX`, which the packer pads labels with and the loss and fused head read back.
