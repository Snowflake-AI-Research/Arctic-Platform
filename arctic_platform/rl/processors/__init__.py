"""Processors package — re-exports all public symbols from submodules.

Importing this package is sufficient to ensure all processors and loss
functions are registered in the global registries (POST_PROCESSORS,
LOSS_FNS).
"""

# Pipeline registry and runner
from .pipeline import (
    POST_PROCESSORS,
    LOSS_FNS,
    register_post_processor,
    register_loss_fn,
    _resolve_fn,
    run_pipeline,
    identity_post,
    compute_entropy_and_logprobs_post,
)

# Packing utilities
from .packing import (
    N_TOKENS_PER_PAGE,
    _align,
    pack_sequences,
    unpack_sequences,
    pad_packed_for_model,
)

# Micro-batch splitting
from .microbatch import (
    DEFAULT_MAX_TOKENS_PER_MB,
    MicroBatchSpec,
    MicroBatchList,
    split_padded_tensor_dict_into_mb_list,
    _ffd_allocate,
    _ffd_allocate_inner,
    _allocate_balanced_mbs,
    _flat2d_mb,
    _reorder_list,
    _dict_of_list2list_of_dict,
    _is_multi_modal_key,
    _ceil_div,
)

# Stats tracker
from .stats_tracker import (
    DistributedStatsTracker,
    ReduceType,
    TRACKERS,
    DEFAULT_TRACKER,
    get,
    denominator,
    stat,
    scalar,
    scope,
    export,
    export_all,
    record_timing,
)

# Functional math
from .functional import (
    masked_normalization,
    agg_loss,
    kl_penalty,
    ppo_actor_loss_fn,
    sapo_loss_fn,
    _compute_sequence_level_ratio_and_advantages,
)

# GRPO loss (importing this module registers grpo_loss into LOSS_FNS)
from .grpo import (
    ProxLogpMethod,
    ProxApproxMethod,
    PROX_LOGP_METHOD_RECOMPUTE,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    grpo_loss,
    _internal_grpo_loss_fn,
    compute_prox_logp_approximations,
    _resolve_proximal_logp,
    _apply_m2po_masking,
    _get_m2po_loss_mask,
    _EPSILON,
    _compute_importance_weight,
    _compute_approximation_errors,
    _tensor_scalar_stats,
)


from .verl_grpo import verl_grpo_loss

__all__ = [
    # pipeline
    "POST_PROCESSORS",
    "LOSS_FNS",
    "register_post_processor",
    "register_loss_fn",
    "_resolve_fn",
    "run_pipeline",
    "identity_post",
    "compute_entropy_and_logprobs_post",
    # packing
    "N_TOKENS_PER_PAGE",
    "_align",
    "pack_sequences",
    "unpack_sequences",
    "pad_packed_for_model",
    # microbatch
    "DEFAULT_MAX_TOKENS_PER_MB",
    "MicroBatchSpec",
    "MicroBatchList",
    "split_padded_tensor_dict_into_mb_list",
    "_ffd_allocate",
    "_ffd_allocate_inner",
    "_allocate_balanced_mbs",
    "_flat2d_mb",
    "_reorder_list",
    "_dict_of_list2list_of_dict",
    "_is_multi_modal_key",
    "_ceil_div",
    # stats_tracker
    "DistributedStatsTracker",
    "ReduceType",
    "TRACKERS",
    "DEFAULT_TRACKER",
    "get",
    "denominator",
    "stat",
    "scalar",
    "scope",
    "export",
    "export_all",
    "record_timing",
    # functional
    "masked_normalization",
    "agg_loss",
    "kl_penalty",
    "ppo_actor_loss_fn",
    "sapo_loss_fn",
    "_compute_sequence_level_ratio_and_advantages",
    # grpo
    "ProxLogpMethod",
    "ProxApproxMethod",
    "PROX_LOGP_METHOD_RECOMPUTE",
    "PROX_LOGP_METHOD_LOGLINEAR",
    "PROX_LOGP_METHOD_METRICS",
    "PROX_APPROX_METHOD_LOGLINEAR",
    "PROX_APPROX_METHOD_LINEAR",
    "PROX_APPROX_METHOD_ROLLOUT",
    "PROX_APPROX_METHODS_ALL",
    "grpo_loss",
    "_internal_grpo_loss_fn",
    "compute_prox_logp_approximations",
    "_resolve_proximal_logp",
    "_apply_m2po_masking",
    "_get_m2po_loss_mask",
    "_EPSILON",
    "_compute_importance_weight",
    "_compute_approximation_errors",
    "_tensor_scalar_stats",
]
