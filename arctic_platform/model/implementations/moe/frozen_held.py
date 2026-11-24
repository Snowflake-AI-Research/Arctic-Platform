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
"""Frozen weights a model holds outside their parameters' data, as its weight export and its state dict see them.

The frozen expert offload (``moe/frozen_offload.py``: each rank's routed-expert codes in host memory) keeps every
parameter object in its owner, with empty data, outside the calls of its layer, and so may any other holding of frozen
weights. Each registers its ``FrozenHolding`` with the model (``hold_frozen_weights``), and the model's weight export
then sees the unheld model: one ``weight_export.HeldParameters`` for all of the model's holdings lists the parameters
by their own names and in their order, and materializes the held parameters among the tensors of one export unit (the
MoE families' builders: a decoder layer) whole, each holding pointing their data at a whole copy and emptying them
again after. ``iter_model_weights`` (a weights-only checkpoint's HF tree, the weight sync) therefore exports the unheld
model's tensors, bit for bit, holding one layer's whole weights at a time. A family export builder that reads the
parameters without ``weight_export.materialized_parameters`` is refused, as is ``weight_export.save_hf_pretrained``
(one rank alone cannot gather what a holding keeps on every rank).

``state_dict`` / ``load_state_dict``, and DeepSpeed's checkpoint (which also reads frozen parameters directly), see the
whole weights only inside ``whole_frozen_weights(model)``, entered on every rank (a holding may gather collectively):
there every held parameter's data is its whole value in host memory, and what a load writes there becomes the
holding's. Outside it, a model that holds frozen weights refuses its state dict, which would hold empty tensors.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Protocol
from weakref import WeakKeyDictionary

import torch
from torch import nn


class FrozenHolding(Protocol):
    """Frozen parameters one mechanism keeps outside their data, a decoder layer at a time."""

    @property
    def held(self) -> bool:
        """Whether some parameter holds no data now (they are held after the first call of their layer)."""

    def materialize(self, tensors: Sequence[torch.Tensor]) -> AbstractContextManager[None]:
        """Inside, the held parameters among ``tensors`` hold their whole values on this rank's GPU. Every rank enters
        it with the same tensors in the same order."""

    def whole_on_host(self) -> AbstractContextManager[None]:
        """Inside, every held parameter holds its whole value in host memory; what is written there is kept. Every rank
        enters it."""


_HOLDINGS: WeakKeyDictionary[nn.Module, list[FrozenHolding]] = WeakKeyDictionary()


def hold_frozen_weights(root: nn.Module, holding: FrozenHolding) -> None:
    """Register ``holding`` of ``root``'s frozen parameters with ``root``'s weight export and state dict.

    ``root`` is the model whose weight export runs (the family model a PEFT wrapper wraps). Its holdings materialize
    in the order they were registered, so every rank must register the same holdings in the same order.
    """
    holdings = _HOLDINGS.get(root)
    if holdings is None:
        from arctic_platform.model import weight_export

        if weight_export._HELD_PARAMETERS.get(root) is not None:
            raise RuntimeError(
                f"{type(root).__name__} already holds parameters outside their owners by other means "
                "(weight_export.HeldParameters), which its weight export cannot combine with frozen holdings"
            )
        holdings = _HOLDINGS[root] = []
        weight_export.register_held_parameters(
            root,
            weight_export.HeldParameters(
                named_parameters=_named_parameters, materialize=functools.partial(_materialize, holdings)
            ),
        )
        contract = weight_export.weight_export_contract(root)
        if contract is not None:
            weight_export.register_weight_export(
                root,
                weight_export.WeightExportContract(
                    hf=_export_builder(contract.hf, "hf"), vllm=_export_builder(contract.vllm, "vllm")
                ),
            )
        check = functools.partial(_check_state_dict, holdings)
        root.register_state_dict_pre_hook(check)
        root.register_load_state_dict_pre_hook(check)
    holdings.append(holding)


@contextlib.contextmanager
def whole_frozen_weights(model: nn.Module) -> Iterator[None]:
    """Inside, every frozen parameter ``model`` (or a model it wraps) holds outside its data holds its whole value in
    host memory, for ``state_dict``, ``load_state_dict`` and DeepSpeed's checkpoint; what a load writes is kept.

    Enter it on every rank: a holding may gather collectively when it is entered. A no-op for a model that holds
    nothing.
    """
    with contextlib.ExitStack() as stack:
        for holding in _holdings_under(model):
            stack.enter_context(holding.whole_on_host())
        yield


def _holdings_under(model: nn.Module) -> list[FrozenHolding]:
    return [holding for module in model.modules() for holding in _HOLDINGS.get(module, ())]


def _named_parameters(model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    # The held parameters stay in their owners (with empty data): the model's own listing, names and order.
    return model.named_parameters()


@contextlib.contextmanager
def _materialize(holdings: list[FrozenHolding], tensors: Sequence[torch.Tensor]) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        for holding in holdings:
            stack.enter_context(holding.materialize(tensors))
        yield


def _check_state_dict(holdings: list[FrozenHolding], module: nn.Module, *args, **kwargs) -> None:
    if any(holding.held for holding in holdings):
        raise RuntimeError(
            f"{type(module).__name__} holds frozen weights outside their parameters (moe/frozen_offload.py, "
            "moe/frozen_held.py), so its state dict would hold empty tensors for them: take or load it inside "
            "moe/frozen_held.whole_frozen_weights(model), entered on every rank, and export its weights with "
            "iter_model_weights"
        )


def _export_builder(builder, weight_format: str):
    """``builder`` when it reads held parameters through ``weight_export.export_named_parameters`` and
    ``materialized_parameters``; else one that refuses, on every rank before any collective."""
    from .deepspeed_integration import build_iter_full_hf_weights
    from .vllm_weights import build_iter_full_vllm_weights

    if builder is None or builder in (build_iter_full_hf_weights, build_iter_full_vllm_weights):
        return builder
    label = getattr(builder, "__qualname__", repr(builder))

    def refuse(model):
        raise NotImplementedError(
            f"weight_format={weight_format!r} with frozen weights held outside their parameters: {label} reads the "
            "parameters directly, which hold no data outside their layer's calls; it must read them through "
            "weight_export.export_named_parameters and materialized_parameters, as build_iter_full_hf_weights does"
        )

    return refuse
