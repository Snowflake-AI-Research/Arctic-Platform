import io
import os
#import itertools
import torch
from typing import Any

from arctic_platform.rl.zorro_train.seqlen_balancing import reorg_global_batch

def shard_token_stats(batch_data: dict, meta_data: dict | None = None) -> dict[str, int]:
    """Summarize valid token counts for DP straggler diagnosis."""
    stats: dict[str, int] = {}
    am = batch_data.get("attention_mask")
    if torch.is_tensor(am) and am.ndim == 2:
        lens = am.sum(dim=1).to(torch.int64)
        stats["valid_tokens"] = int(lens.sum().item())
        stats["batch_size"] = int(am.shape[0])
        stats["seq_len_min"] = int(lens.min().item())
        stats["seq_len_max"] = int(lens.max().item())
    cu = batch_data.get("cu_seqlens")
    if torch.is_tensor(cu) and cu.numel() > 0:
        stats["cu_seqlens_packed"] = int(cu[-1].item())
    input_ids = batch_data.get("input_ids")
    if torch.is_tensor(input_ids):
        if input_ids.ndim == 1:
            stats["packed_tokens"] = int(input_ids.numel())
        elif input_ids.ndim == 2 and input_ids.shape[0] == 1:
            stats["packed_tokens"] = int(input_ids.shape[1])
    loss_mask = batch_data.get("loss_mask")
    if torch.is_tensor(loss_mask):
        stats["loss_mask_tokens"] = int(loss_mask.sum().item())
    if meta_data:
        bnt = meta_data.get("batch_num_tokens")
        if bnt is not None:
            stats["meta_batch_num_tokens"] = int(bnt)
    return stats


def log_dp_shard_tokens(rank: int, tag: str, batch_data: dict, meta_data: dict | None = None) -> None:
    """Log per-DP-rank token stats when ARL_LOG_DP_SHARD_TOKENS is set."""
    if not os.environ.get("ARL_LOG_DP_SHARD_TOKENS"):
        return
    stats = shard_token_stats(batch_data, meta_data)
    if not stats:
        return
    parts = " ".join(f"{k}={v}" for k, v in sorted(stats.items()))
    print(f"[{rank}] [DP-shard] {tag}: {parts}", flush=True)


def unpack_batch(batch: dict) -> tuple:
    """Support both ``{"args": ..., "kwargs": ...}`` and flat-dict formats.

    Returns ``(args, kwargs, context, processing)``.
    """
    return {}, batch["batch"], batch["meta"], batch["processing"]


def _split_value(val, num_chunks: int):
    """Split a tensor or list along the batch (first) dimension."""
    if isinstance(val, torch.Tensor):
        if val.shape[0] < num_chunks:
            raise ValueError(
                f"Batch dimension {val.shape[0]} is smaller than num_workers "
                f"{num_chunks}. The client must send at least one sample per "
                f"DP worker."
            )
        return list(torch.chunk(val, num_chunks, dim=0))
    if isinstance(val, list):
        if len(val) < num_chunks:
            raise ValueError(
                f"Batch size {len(val)} is smaller than num_workers "
                f"{num_chunks}. The client must send at least one sample per "
                f"DP worker."
            )
        chunk_size = (len(val) + num_chunks - 1) // num_chunks
        return [val[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
    return [val] * num_chunks


def split_dict(d: dict, num_chunks: int) -> list[dict]:
    """Split every value in a dict across DP ranks."""
    if not d:
        return [{}] * num_chunks
    split_vals = {k: _split_value(v, num_chunks) for k, v in d.items()}
    return [{k: split_vals[k][i] for k in d} for i in range(num_chunks)]


def reconstruct_position_ids_(batch_data: dict) -> None:
    """Rebuild canonical 0-based per-sequence position_ids from attention_mask in-place.

    The client may drop ``position_ids`` from the wire payload (they are a derivable
    per-sequence arange) to save serializing/transferring an int64 [B, S] tensor. Here
    we regenerate the exact same values the client would have sent (matching how
    packing.py recomputes them): position ``p`` for the p-th valid token in each row,
    and 0 at padded positions. Must run before any consumer (reorg/dedup/model).
    """
    attention_mask = batch_data.get("attention_mask")
    if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
        raise ValueError(
            "Cannot reconstruct position_ids: a 2D [B, S] attention_mask is required "
            "(set arctic_rl.drop_position_ids=False for 3D/mrope position_ids)."
        )
    attention_mask_long = attention_mask.to(torch.long)
    batch_data["position_ids"] = (attention_mask_long.cumsum(dim=-1) - 1).clamp_(min=0) * attention_mask_long


def _split_batch(batch: dict, num_workers: int) -> list[dict]:
    """Split a batch across DP workers."""
    _, batch_data, meta_data, processing = unpack_batch(batch)

    reorder_indices = None

    # position_ids may have been dropped on the wire (client-side optimization);
    # reconstruct them from attention_mask before anything consumes them.
    if meta_data.get("drop_position_ids", False) and "position_ids" not in batch_data:
        reconstruct_position_ids_(batch_data)

    # ZoRRO Load balancer
    if meta_data.get("zorro_train_enable", False):
        max_group_length_threshold = meta_data.get("zorro_train_max_rollouts", meta_data["rollout_n"])
        batch_data, reorder_indices = reorg_global_batch(
            batch_data,
            response_length=meta_data["max_response_len"],
            world_size=num_workers,
            max_token_len=meta_data["max_token_len_per_gpu"],
            max_group_length_threshold=max_group_length_threshold,
        )

    batch_data_shards = split_dict(batch_data, num_workers)
    shards = []
    meta_data.update(**dict(dp_size=num_workers))
    for i in range(num_workers):
        shard = dict(batch=batch_data_shards[i], meta=meta_data, processing=processing)
        shards.append(shard)
    return shards, reorder_indices

ray_split_batch = _split_batch

def http_split_batch(batch_bytes: bytes, num_workers: int) -> list[bytes]:
    """Deserialize a global batch, split across DP workers, re-serialize each shard."""
    # if num_workers <= 1:
    #     return [batch_bytes]
    batch = torch.load(io.BytesIO(batch_bytes), map_location="cpu")

    shards, reorder_indices = _split_batch(batch, num_workers)

    # _, batch_data, meta_data, processing = unpack_batch(batch)
    # batch_data_shards = split_dict(batch_data, num_workers)
    # shards = []
    # meta_data.update(**dict(dp_size=num_workers))
    # for i in range(num_workers):
    #     shard = dict(
    #         batch=batch_data_shards[i],
    #         meta=meta_data,
    #         processing=processing,
    #     )
    #     buf = io.BytesIO()
    #     torch.save(shard, buf)
    #     shards.append(buf.getvalue())


    # DO NOT DELETE:
    # at the moment must not recode back to bytes since we use ray to continue within the http_server - but it could be different later with DSS
    # for i in range(len(shards)):
    #     buf = io.BytesIO()
    #     torch.save(shards[i], buf)
    #     shards[i] = buf.getvalue()

    return shards, reorder_indices




def dump_dict_payload(payload: dict, tag:str):
    return
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            print(f"{tag}: {k=} {v.shape=} {v=}")
        else:
            print(f"{tag}: {k=} {v=}")

def merge_dict_shards(shards_list: list[dict]) -> dict:
    if len(shards_list) <= 1:
        return shards_list[0] if shards_list else {}
    import collections
    combined = collections.defaultdict(list)
    for d in shards_list:
        dump_dict_payload(d, "merge_dict_shards in")
    for d in shards_list:
        for k, v in d.items():
            if type(v) == list:
                combined[k].extend(v)
            else:
                combined[k].append(v)

    dump_dict_payload(combined, "merge_dict_shards out")
    for k, v in combined.items():
        if isinstance(v, list) and all(torch.is_tensor(x) for x in v):
            combined[k] = torch.cat(v)

    dump_dict_payload(combined, "merge_dict_shards out after stack")
    return combined


_METRIC_PAIR_SUM_SUFFIX = ".sum"
_METRIC_PAIR_TOKENS_SUFFIX = ".tokens"


def combine_metric_shards(shards_list: list[dict]) -> dict:
    """Combine per-shard metric dicts into a single scalar-per-key dict.

    Each shard is expected to contribute either:
      * paired ``{name}.sum`` / ``{name}.tokens`` numeric values (a token-mean
        partial aggregation), in which case the output exposes
        ``output[name] = Σ shards .sum / Σ shards .tokens`` — i.e. a global
        token-weighted mean across all shards; or
      * a plain numeric value, in which case the output exposes the mean of
        that value across the shards (the common-sense default for
        replicated scalars like ``grad_norm`` and ``lr``); or
      * a non-numeric value, which is passed through from the first shard
        that defines the key.

    This is the cross-rank counterpart to the per-microbatch summation done
    inside the per-rank worker (see ``deepspeed_worker._forward_maybe_backward``).
    Together they collapse the per-(rank × microbatch) list shape produced by
    the older ``merge_dict_shards``-everywhere approach into one scalar per
    metric per ``fwd_bwd`` call, which corresponds to one mini-batch.
    """
    if not shards_list:
        return {}

    numeric_totals: dict[str, float] = {}
    numeric_counts: dict[str, int] = {}
    passthrough: dict = {}

    for shard in shards_list:
        if not shard:
            continue
        for k, v in shard.items():
            if isinstance(v, bool):
                # bool is a subclass of int; treat as passthrough so we don't
                # accidentally sum True/False values.
                passthrough.setdefault(k, v)
                continue
            if isinstance(v, (int, float)):
                numeric_totals[k] = numeric_totals.get(k, 0.0) + float(v)
                numeric_counts[k] = numeric_counts.get(k, 0) + 1
            else:
                passthrough.setdefault(k, v)

    out: dict = {}
    # First, fold any paired ``.sum`` / ``.tokens`` keys into their base name
    # as a token-weighted global mean.
    paired_bases: set[str] = set()
    for k in numeric_totals:
        if k.endswith(_METRIC_PAIR_SUM_SUFFIX):
            base = k[: -len(_METRIC_PAIR_SUM_SUFFIX)]
            tokens_key = base + _METRIC_PAIR_TOKENS_SUFFIX
            if tokens_key in numeric_totals:
                paired_bases.add(base)

    for base in paired_bases:
        sum_total = numeric_totals[base + _METRIC_PAIR_SUM_SUFFIX]
        tokens_total = numeric_totals[base + _METRIC_PAIR_TOKENS_SUFFIX]
        # Empty mini-batch (no valid tokens) is logged as 0.0; downstream
        # reducers (np.mean / np.sum across the 4 mini-batches in the verl
        # mini-batch loop) handle that case naturally.
        out[base] = (sum_total / tokens_total) if tokens_total > 0 else 0.0

    # Then carry over any non-paired numeric keys as a simple mean across
    # the shards that contributed (matches the common case where a metric
    # is the same on every rank, e.g. ``grad_norm`` or ``lr``).
    for k, total in numeric_totals.items():
        if k.endswith(_METRIC_PAIR_SUM_SUFFIX):
            base = k[: -len(_METRIC_PAIR_SUM_SUFFIX)]
            if base in paired_bases:
                continue
        if k.endswith(_METRIC_PAIR_TOKENS_SUFFIX):
            base = k[: -len(_METRIC_PAIR_TOKENS_SUFFIX)]
            if base in paired_bases:
                continue
        if k in out:
            continue
        count = numeric_counts.get(k, 1)
        out[k] = total / count if count > 0 else total

    # Finally, expose any non-numeric passthrough values that the public
    # metric name doesn't already cover.
    for k, v in passthrough.items():
        out.setdefault(k, v)

    return out


def combine_metric_microbatches(per_microbatch_metric_dicts: list[dict]) -> dict:
    """Per-rank counterpart to ``combine_metric_shards``.

    Aggregates each rank's per-microbatch metric dicts into a single
    rank-level dict using the same convention as ``combine_metric_shards``:

      * Paired ``{name}.sum`` and ``{name}.tokens`` numerics are summed
        across microbatches (so the partial token-mean accumulator grows
        correctly).
      * Other numerics are averaged across the microbatches that emitted
        them (mirrors how replicated constants like ``kl_coef`` should
        behave so the cross-rank combiner doesn't end up with a value
        inflated by the per-rank microbatch count).
      * Non-numerics are passed through from the first microbatch that
        defines them.

    The returned dict is one rank's contribution to a single mini-batch and
    is meant to be re-aggregated across DP ranks via
    ``combine_metric_shards``.
    """
    if not per_microbatch_metric_dicts:
        return {}

    numeric_totals: dict[str, float] = {}
    numeric_counts: dict[str, int] = {}
    passthrough: dict = {}

    for md in per_microbatch_metric_dicts:
        if not md:
            continue
        for k, v in md.items():
            if isinstance(v, bool):
                passthrough.setdefault(k, v)
                continue
            if isinstance(v, (int, float)):
                numeric_totals[k] = numeric_totals.get(k, 0.0) + float(v)
                numeric_counts[k] = numeric_counts.get(k, 0) + 1
            else:
                passthrough.setdefault(k, v)

    # Identify paired ``.sum`` / ``.tokens`` keys; those need to be summed
    # so the server-side combiner can do ``Σ sum / Σ tokens``.
    paired_keys: set[str] = set()
    for k in numeric_totals:
        if k.endswith(_METRIC_PAIR_SUM_SUFFIX):
            base = k[: -len(_METRIC_PAIR_SUM_SUFFIX)]
            tokens_key = base + _METRIC_PAIR_TOKENS_SUFFIX
            if tokens_key in numeric_totals:
                paired_keys.add(k)
                paired_keys.add(tokens_key)

    out: dict = {}
    for k, total in numeric_totals.items():
        if k in paired_keys:
            out[k] = total
        else:
            count = numeric_counts.get(k, 1)
            out[k] = total / count if count > 0 else total

    for k, v in passthrough.items():
        out.setdefault(k, v)
    return out


def get_reverse_idx(idx_map):
    """
    Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    # A plain int list copy is sufficient (idx_map holds ints); deepcopy is overkill.
    reverse_idx_map = [0] * len(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[int(idx)] = i

    return reverse_idx_map


def restore_batch_order(batch, indices):
    """
    When load balancer changes the order of the batch, in some cases like compute_log_prob it's critical to restore the order when logprob/entropy have been computed and returned to the framework. in the case of update_actor it doesn't matter.

    If `indices` is None return the batch w/o modifications

    """
    if indices is None:
        return batch

    #indices = list(itertools.chain.from_iterable(indices))
    #print(f"{indices=}")
    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)

    # verl only restores indices for compute_log_prob (and not update_actor)
    for key in batch.keys():
        if torch.is_tensor(batch[key]):
            # protect against lists of None's
            #print(key, batch[key])
            batch[key] = batch[key][revert_indices]

    return batch




def tensorize(val: Any):
    """Recursively convert single/lists/dicts/tuples of Python scalars / lists to tensors preserving the same nested structure if any."""
    #print(f"{val=}")
    if isinstance(val, dict):
        for k in val.keys():
            val[k] = tensorize(val[k])
    elif isinstance(val, list):
        if all(x is None for x in val):
            pass
        elif any(x is None for x in val):
            raise ValueError("got a mixed None and non-None values in the list")
        else:
            val = torch.tensor(val)
    elif val is None:
        pass
    else:
        # expand to support more complex tensors
        val = torch.tensor(val)
    return val


def detensorize(val: Any) -> Any:
    """Recursively convert single/lists/dicts/tuples of tensors to Python scalars / lists preserving the same nested structure if any."""
    if isinstance(val, dict):
        return {k: detensorize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(detensorize(v) for v in val)
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.detach().cpu().item()
        return val.detach().cpu().tolist()
    return val
