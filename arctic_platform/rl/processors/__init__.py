# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processors package — re-exports all public symbols from submodules.

Importing this package is sufficient to ensure all processors and loss
functions are registered in the global registries (POST_PROCESSORS,
LOSS_FNS).
"""

from .base_loss import BaseLoss
from .base_loss import resolve_loss

# Packed apply + CCE (importing registers causal_cross_entropy)
from .causal_cross_entropy import causal_cross_entropy_loss

# Cortex compute_logprobs (importing registers the zone post; not an AP alias)
from .compute_logprobs import compute_logprobs_post
from .cortex_grpo import cortex_grpo_echo_v1_loss
from .cortex_grpo import cortex_grpo_loss

# Functional math
from .functional import _compute_sequence_level_ratio_and_advantages
from .functional import agg_loss
from .functional import kl_penalty
from .functional import masked_normalization
from .functional import ppo_actor_loss_fn
from .functional import sapo_loss_fn

# The class registry is separate and does not replace same-named legacy functions.
from .grouped_distillation import GroupedDistillationLoss
from .grouped_distillation import GRPOGroupedDistillationLoss
from .grouped_distillation import grouped_divergence

# GRPO loss (importing this module registers grpo_loss into LOSS_FNS)
from .grpo import _EPSILON
from .grpo import PROX_APPROX_METHOD_LINEAR
from .grpo import PROX_APPROX_METHOD_LOGLINEAR
from .grpo import PROX_APPROX_METHOD_ROLLOUT
from .grpo import PROX_APPROX_METHODS_ALL
from .grpo import PROX_LOGP_METHOD_LOGLINEAR
from .grpo import PROX_LOGP_METHOD_METRICS
from .grpo import PROX_LOGP_METHOD_RECOMPUTE
from .grpo import ProxApproxMethod
from .grpo import ProxLogpMethod
from .grpo import _apply_m2po_masking
from .grpo import _compute_approximation_errors
from .grpo import _compute_importance_weight
from .grpo import _get_m2po_loss_mask
from .grpo import _internal_grpo_loss_fn
from .grpo import _resolve_proximal_logp
from .grpo import _tensor_scalar_stats
from .grpo import compute_prox_logp_approximations
from .grpo import grpo_echo_v1_loss
from .grpo import grpo_loss

# Micro-batch splitting
from .microbatch import DEFAULT_MAX_TOKENS_PER_MB
from .microbatch import MicroBatchList
from .microbatch import MicroBatchSpec
from .microbatch import _allocate_balanced_mbs
from .microbatch import _ceil_div
from .microbatch import _dict_of_list2list_of_dict
from .microbatch import _ffd_allocate
from .microbatch import _ffd_allocate_inner
from .microbatch import _flat2d_mb
from .microbatch import _is_multi_modal_key
from .microbatch import _reorder_list
from .microbatch import split_padded_tensor_dict_into_mb_list
from .packed_reduction import PackedLossReduction
from .packed_reduction import apply_packed_loss_reduction
from .packed_reduction import combine_packed_losses
from .packed_reduction import combine_packed_metrics
from .packed_reduction import resolve_packed_loss_reduction

# Packing utilities
from .packing import N_TOKENS_PER_PAGE
from .packing import _align
from .packing import pack_sequences
from .packing import pad_packed_for_model
from .packing import unpack_sequences

# Pipeline registry and runner
from .pipeline import LOSS_FNS
from .pipeline import POST_PROCESSORS
from .pipeline import _resolve_fn
from .pipeline import compute_entropy_and_logprobs_post
from .pipeline import identity_post
from .pipeline import metric_is_summed
from .pipeline import register_loss_fn
from .pipeline import register_post_processor
from .pipeline import run_pipeline

# SFT loss + pipeline (importing registers "sft" / "sft_ce" into LOSS_FNS)
from .sft import LOGIT_LOSS_FNS
from .sft import SFT_LOSS_FNS
from .sft import run_sft_pipeline
from .sft import sft_ce_loss
from .sft import sft_loss

# Stats tracker
from .stats_tracker import DEFAULT_TRACKER
from .stats_tracker import TRACKERS
from .stats_tracker import DistributedStatsTracker
from .stats_tracker import ReduceType
from .stats_tracker import denominator
from .stats_tracker import export
from .stats_tracker import export_all
from .stats_tracker import get
from .stats_tracker import record_timing
from .stats_tracker import scalar
from .stats_tracker import scope
from .stats_tracker import stat
from .verl_grpo import verl_grpo_loss

__all__ = [
    # pipeline
    "POST_PROCESSORS",
    "LOSS_FNS",
    "BaseLoss",
    "resolve_loss",
    "register_post_processor",
    "register_loss_fn",
    "_resolve_fn",
    "run_pipeline",
    "identity_post",
    "compute_entropy_and_logprobs_post",
    "compute_logprobs_post",
    "metric_is_summed",
    "PackedLossReduction",
    "apply_packed_loss_reduction",
    "combine_packed_losses",
    "combine_packed_metrics",
    "resolve_packed_loss_reduction",
    "causal_cross_entropy_loss",
    "GroupedDistillationLoss",
    "GRPOGroupedDistillationLoss",
    "grouped_divergence",
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
    "grpo_echo_v1_loss",
    "cortex_grpo_loss",
    "cortex_grpo_echo_v1_loss",
    "sft_loss",
    "sft_ce_loss",
    "LOGIT_LOSS_FNS",
    "SFT_LOSS_FNS",
    "run_sft_pipeline",
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
