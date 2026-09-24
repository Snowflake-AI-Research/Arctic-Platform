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
"""Folding a token's expert rows onto one row in an order the program fixes, rather than the hardware's.

The production fold is one atomic per row, in ``models.moe.token_combine``, and its order is whatever order the
device delivers the rows in. The fold here removes that dependence, at the cost of a device synchronization and
a padded layout, and ``debug.full_determinism`` is what selects it.

:func:`maybe_use_fixed_order_row_sum` is how it gets selected, and it is the only thing outside this package
that a model loader calls. It replaces the shared production symbol in the modules that fold rows, so no combine
site names this module, carries the flag, or routes the model's ``top_k`` down to the fold -- the routing width
is read off the model here instead, once, at the moment the fold is installed.
"""

from __future__ import annotations

import functools
import importlib
import sys
from typing import Any
from typing import Callable
from typing import Mapping

import torch

from .determinism import full_determinism_enabled

# Imported so that a loader's single call can reach them whether or not the job's combine path has run yet; the
# sweep below then finds every module that has the shared symbol bound, including any added later. The
# rank-local folds in ``models.moe.layers.moe`` are absent because no configuration reaches them: both forwards
# return from their dispatch branch for either value of ``EPCommBackend``, and ``GroupedExperts.forward``
# refuses any other value.
_ROW_SUM_CONSUMERS = (
    "arctic_platform.model.implementations.moe.distributed.token_permute",
    "arctic_platform.model.implementations.moe.distributed.ucclep",
)


def fixed_order_sum_rows_by_token(
    rows: torch.Tensor, token_indices: torch.Tensor, num_tokens: int, top_k: int
) -> torch.Tensor:
    """Sum the rows belonging to each token, in an order that depends on nothing outside the token.

    ``token_indices[i]`` names the destination token of ``rows[i]``, and a token appears once per expert holding
    it, so several rows fold onto one destination. ``scatter_add`` performs that fold with one atomic per row,
    and atomics aimed at a repeated destination are applied in delivery order, which is not a property of the
    program: two runs of the same configuration then produce different sums. Two addends commute, so the effect
    begins at three -- that is, once some token has three or more of its ``top_k`` experts on one rank.

    Sorting the rows by destination gives every row a slot ``(token, its rank inside that token's group)`` that
    no other row occupies. Filling that dense layout is a placement rather than an accumulation, and what
    remains is a fold across a padded dimension. The sort is stable, which makes the order concretely
    "ascending row index within each token".

    Two things about that fold have to hold together, and only one of them is about determinism within a run.

    The extent is ``top_k``, a property of the model, and never the largest group the batch happens to contain.
    A data-dependent extent makes the fold's shape a function of the whole batch, and a fold whose shape changes
    reassociates its addends: the same token's rows, summed under a different extent, give a different last bit.
    That is invisible in a repeat of one configuration -- it is bit-exact -- and shows up as gradients that move
    when a packer changes how many tokens share a microbatch, which is a defect no repeat count can find.

    With the extent fixed, the fold is ``sum`` over the padded dimension. A reduction associates as it likes and
    how it associates follows from the shape it is given, so a constant extent is what makes the choice constant.
    The number of tokens does not re-enter, which is measured rather than assumed: holding one token's rows fixed
    while the batch's token count varies over 128 to 512 and a neighbouring group's width over 1 to 4 leaves that
    token's output bit-identical in all sixteen combinations, in bfloat16 and in float32.

    A left-to-right chain accumulated in float32 would make the order explicit rather than a property of the
    kernel, at 17.85ms against 5.95ms and 4482MiB against 4102MiB of peak memory for 16384 tokens, hidden 4096
    and ``top_k`` 8 on an H200. The reduction is kept, and the batch-independence it relies on is asserted by
    test rather than by construction.

    Both steps differentiate, and the backward of a placement is a gather, so no accumulation order enters the
    gradient either.

    Two costs come with that. Confirming that no token brought more rows than ``top_k`` synchronizes with the
    device; the expert-parallel forward already synchronizes once per call to agree on a chunk count, so this is
    another stall of the same kind rather than the first one. And the padded layout holds ``num_tokens * top_k``
    rows against the ``row_count`` the atomic form touched, so it moves more bytes.
    """
    hidden_dim = rows.shape[1]
    row_count = token_indices.shape[0]
    if row_count == 0:
        return rows.new_zeros((num_tokens, hidden_dim))
    # The expert stage can return more rows than routing produced, because a padded receive buffer keeps its
    # tail. Those rows belong to no token and are dropped, which is what the scatter this replaces also did by
    # iterating over the index rather than over the source.
    rows = rows[:row_count]

    group_size = torch.zeros(num_tokens, dtype=token_indices.dtype, device=token_indices.device)
    group_size.scatter_add_(0, token_indices, torch.ones_like(token_indices))
    widest = int(group_size.max())
    if widest > top_k:
        raise AssertionError(
            f"a token was routed to {widest} of this rank's experts but top_k is {top_k}. The slot layout "
            "below reserves top_k rows per token, so a wider group would write over the next token's slots"
        )

    order = torch.argsort(token_indices, stable=True)
    grouped_tokens = token_indices.index_select(0, order)
    group_start = torch.cumsum(group_size, dim=0) - group_size
    grouped_rank = torch.arange(row_count, device=rows.device) - group_start.index_select(0, grouped_tokens)
    # Carried back onto the unsorted rows so the placement below can read ``rows`` as it stands, rather than pay
    # for a sorted copy of a tensor whose rows are the width of the model.
    row_rank = torch.empty_like(order)
    row_rank.scatter_(0, order, grouped_rank)

    slots = rows.new_zeros((num_tokens * top_k, hidden_dim))
    slots = slots.index_copy(0, token_indices * top_k + row_rank, rows)
    grouped = slots.view(num_tokens, top_k, hidden_dim)

    return grouped.sum(dim=1)


def routing_width(model: torch.nn.Module) -> int:
    """Read ``top_k`` off the model, which is where the fold's extent has to come from.

    The extent cannot be derived at the combine, because every quantity in scope there is a function of the
    batch. Reading it from the model once, here, is also what keeps it out of the combine signatures. Layers
    that disagree are refused rather than reconciled: one extent is installed for the whole model, so a second
    value would silently be applied to layers it does not describe.
    """
    widths = {int(module.top_k) for module in model.modules() if isinstance(getattr(module, "top_k", None), int)}
    if not widths:
        raise ValueError(
            "no module on this model exposes top_k, so the fixed-order fold has no extent to run at; the "
            "routing width cannot be taken from the batch without making the fold's shape data-dependent"
        )
    if len(widths) > 1:
        raise ValueError(
            f"this model routes to {sorted(widths)} experts in different layers, and one extent is installed "
            "for all of them; a per-layer extent would have to reach the combine sites to be correct"
        )
    return widths.pop()


def bind_row_sum(fold: Callable[..., torch.Tensor]) -> int:
    """Point every module that folds rows by token at ``fold``, and report how many were rebound.

    The combine sites bind the shared symbol by ``from ..token_combine import sum_rows_by_token``, so each one
    holds its own reference and rebinding the defining module alone would reach none of them. Only bindings that
    are currently the production fold or a fold installed here are replaced, which makes the call idempotent and
    keeps it from capturing a name that happens to match in unrelated code.
    """
    from arctic_platform.model.implementations.moe.token_combine import sum_rows_by_token as production_fold

    for name in _ROW_SUM_CONSUMERS:
        try:
            importlib.import_module(name)
        except ImportError:
            # A backend module imports its own transport at module scope, and a host that cannot load that
            # transport cannot be running that backend either. The sweep below covers whichever ones are loaded.
            continue

    rebound = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith("arctic_platform.model.implementations.") or module is None:
            continue
        current = getattr(module, "sum_rows_by_token", None)
        if current is production_fold or getattr(current, "func", None) is fixed_order_sum_rows_by_token:
            setattr(module, "sum_rows_by_token", fold)
            rebound += 1
    return rebound


def maybe_use_fixed_order_row_sum(model: torch.nn.Module, training_config: Mapping[str, Any]) -> int:
    """Install the fixed-order fold for this model if the run asked for it, and change nothing if it did not.

    The entry point a model loader calls once the model is built, and the only place the flag is read. A run
    that did not ask returns 0 with the production fold still in place, and keeps the delivery-order dependence
    :func:`fixed_order_sum_rows_by_token` describes. A run that did pays a device synchronization and a padded
    layout at every combine, which is why no production step carries it.

    Returns the number of modules now folding through the fixed order, and raises if that is none: the request
    was granted or it was refused, never silently dropped.
    """
    if not full_determinism_enabled(training_config):
        return 0

    fold = functools.partial(fixed_order_sum_rows_by_token, top_k=routing_width(model))
    rebound = bind_row_sum(fold)
    if rebound == 0:
        raise RuntimeError(
            "full determinism was requested but no module was found folding rows through "
            "models.moe.token_combine.sum_rows_by_token, so the combine would have kept its delivery-order "
            "dependence while the run reported itself deterministic"
        )
    return rebound
