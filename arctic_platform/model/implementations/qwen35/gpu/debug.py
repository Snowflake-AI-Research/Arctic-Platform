"""Opt-in ad-hoc debug instrumentation (enabled via DSS_* env vars); not used in normal training."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# MoE dispatch imbalance counters, opt-in via DSS_MOE_IMBALANCE (read once; env is fixed per worker, no-op in
# production). maybe_record_recv_tokens accumulates from the hot DeepEP dispatch path; maybe_log_moe_imbalance
# drains + logs once per successful fwd_bwd request. max_chunk sizes the fp32 combine transient; its cross-rank
# spread is the routing imbalance.
_moe_imbalance_enabled = bool(os.environ.get("DSS_MOE_IMBALANCE"))
_recv_token_stats = {"max_chunk": 0, "total": 0, "n_dispatch": 0}


def maybe_record_recv_tokens(num_recv_tokens: int) -> None:
    """Accumulate one DeepEP dispatch's received-token count (opt-in via DSS_MOE_IMBALANCE; no-op otherwise).
    Called from the hot dispatch path, so the gate lives here rather than in the production code."""
    if not _moe_imbalance_enabled:
        return
    _recv_token_stats["max_chunk"] = max(_recv_token_stats["max_chunk"], num_recv_tokens)
    _recv_token_stats["total"] += num_recv_tokens
    _recv_token_stats["n_dispatch"] += 1


def maybe_log_moe_imbalance(rank: int) -> None:
    """Log this rank's DeepEP dispatch imbalance and peak memory after a successful fwd_bwd request.

    Gated by ``DSS_MOE_IMBALANCE`` (no-op otherwise). ``max_chunk_recv`` spread across ranks is the MoE
    routing imbalance that drives the fp32 combine transient (grep ``MOE_IMBALANCE`` and diff max/min across
    ranks).
    """
    if not _moe_imbalance_enabled:
        return
    import torch

    stats = dict(_recv_token_stats)
    _recv_token_stats.update(max_chunk=0, total=0, n_dispatch=0)
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    logger.info(
        "MOE_IMBALANCE rank=%d max_chunk_recv=%d total_recv=%d n_dispatch=%d "
        "peak_alloc_GiB=%.2f peak_reserved_GiB=%.2f",
        rank, stats["max_chunk"], stats["total"], stats["n_dispatch"],
        peak_alloc, peak_reserved,
    )
