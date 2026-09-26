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

"""Loss-object contract and compatibility adapter for function losses."""

from __future__ import annotations

import inspect
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

from arctic_platform.common.registry import LOSS_FNS
from arctic_platform.common.registry import PACKED_LOSS_REDUCTION_ATTR
from arctic_platform.common.registry import resolve_fn
from arctic_platform.registry import RegistryMeta
from arctic_platform.registry import RegistryValidationError
from arctic_platform.registry import get_registered_class


class BaseLoss(ABC, metaclass=RegistryMeta):
    """A processing loss plus callbacks for the boundaries it owns.

    Callback arguments are ordinary dictionaries, sequences, and tensors so
    this contract can be used by AP directly or by an external DSS runtime.
    Mutating callbacks default to no-ops.
    """

    name: str

    @classmethod
    def _validate_subclass(cls) -> None:
        if inspect.isabstract(cls):
            raise RegistryValidationError(f"{cls.__name__} must implement the abstract loss method.")

    def batching_callback(self, request: dict) -> None:
        """Amend one whole request before data/sequence-parallel sharding."""

    def validation_callback(self, context: dict, config: dict) -> None:
        """Validate one request or packed model window before execution."""

    def model_forward_callback(
        self,
        model_kwargs: dict,
        context: dict,
        config: dict,
        output_keys: list[str],
    ) -> None:
        """Amend model kwargs and name objective-owned model outputs."""

    def packed_reduction_callback(
        self,
        microbatches: Sequence[dict],
        config: dict,
        loss_fn_name: str,
    ) -> Any | None:
        """Return objective-owned packed reduction metadata, if any."""
        return None

    def metrics_callback(self, worker_metrics: Sequence[dict], metrics: dict) -> None:
        """Combine or amend metrics after worker results are available."""

    def output_callback(self, model_outputs: dict) -> None:
        """Remove objective-only model outputs before response assembly."""

    @abstractmethod
    def loss(
        self,
        model_outputs: dict,
        batch: dict,
        meta: dict,
        config: dict,
        device: str,
    ):
        """Return ``(loss_tensor, metrics)`` on the legacy five-argument ABI."""
        raise NotImplementedError


class _LegacyLossAdapter(BaseLoss):
    """Expose an existing function loss through the callback contract."""

    _skip_registry_registration = True
    name = "_legacy_loss_adapter"

    def __init__(self, name: str, loss_fn: Callable) -> None:
        self.name = name
        self._loss_fn = loss_fn

    def packed_reduction_callback(
        self,
        microbatches: Sequence[dict],
        config: dict,
        loss_fn_name: str,
    ) -> Any | None:
        resolver = getattr(self._loss_fn, PACKED_LOSS_REDUCTION_ATTR, None)
        if resolver is None:
            return None
        return resolver(microbatches, config, loss_fn_name)

    def output_callback(self, model_outputs: dict) -> None:
        model_outputs.pop("logits", None)

    def loss(
        self,
        model_outputs: dict,
        batch: dict,
        meta: dict,
        config: dict,
        device: str,
    ):
        return self._loss_fn(model_outputs, batch, meta, config, device)


def resolve_loss(name: str) -> BaseLoss:
    """Prefer a registered loss class, then adapt the legacy function registry."""
    try:
        loss_cls = get_registered_class(BaseLoss.__name__, name)
    except LookupError:
        return _LegacyLossAdapter(name, resolve_fn(LOSS_FNS, name))
    return loss_cls()


def prepare_request_loss(request: dict) -> BaseLoss | None:
    """Resolve and batch-amend one whole processing request before sharding."""
    processing = request.get("processing")
    if not isinstance(processing, dict):
        return None
    loss_fn_name = processing.get("loss_fn", "ap_grpo")
    if loss_fn_name is None:
        return None
    loss_object = resolve_loss(loss_fn_name)
    loss_object.batching_callback(request)
    return loss_object
