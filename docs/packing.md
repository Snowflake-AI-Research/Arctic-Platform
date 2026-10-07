# Packing

One packer for every training path: SFT and RL, with and without sequence parallelism. Code: `arctic_platform/common/packing/`. DSS imports this package and keeps no copy of its own.

## Why it is its own package

Every training path needs this, and all of them have to agree on where a row begins and ends, so there is one contract in one place. A per-path packer would put two answers to that question in the tree, and they would first disagree at a model call, far from the edit. Dispatch and the worker sit on opposite sides of the wire and call the same functions, so entry points take their inputs as arguments rather than reading a process group or a config.

`IGNORE_INDEX` is defined here and re-exported by `arctic_platform.model.implementations.gpu.packing`, so the label padding the packer writes and the value a loss path or fused lm_head reads back cannot drift apart.

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

Sequence parallelism runs this same table with `shard_count = sp_size`. The DSS `docs/sequence_parallelism.md` covers the schedule that drives it.

Invariants to preserve:

- The plan reads only row token counts, so every rank derives the same plan from the same shard without communicating.
- One call count covers a whole request, because a rank entering a call after its peers have returned waits in a collective alone. A shard that needed fewer calls is raised to the count by splitting groups (`split_groups_to_count`) rather than by adding calls with no tokens in them; a shard with fewer rows than the count comes back short and its worker pads with zero-loss repeats.
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

A worker answers with row pieces: `window_row_pieces` cuts its window at the row boundaries, and the driver concatenates each row's pieces across the siblings. `unpack_output` is the other direction of the same map -- a whole packed call back to `[B, S]` rows -- so reading a shard's rows means concatenating its siblings' windows and unpacking the result. A sharded model call rests on that same concatenation.

The consumer-side concerns that sit on top of this contract -- deriving model boundary kwargs for non-SP hybrid models, and what a regrouping does to gradients -- are documented by the caller. DSS covers both in its own `docs/packing.md`.
