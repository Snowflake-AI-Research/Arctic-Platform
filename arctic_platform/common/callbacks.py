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

"""Generic training-loop callbacks: a fixed set of lifecycle hooks a trainer calls
unconditionally, with each observability feature as its own ``Callback`` subclass
registered once rather than wired by hand through the trainer.

Unrelated to the per-loss-fn ``_arctic_*_callback`` attribute names in
``arctic_platform.common.registry`` (those mark hooks a loss function declares on
itself; this module is about trainer-wide lifecycle callbacks).

Callbacks are **observational**: they must not change weights, batch, loss,
gradients, optimizer state, RNG state, or control flow. They may keep state,
install read-only hooks (e.g. forward hooks), run collectives, and raise.
Collectives a callback runs must be unconditional — every rank makes the same
calls in the same order, never gated on rank or other rank-local state.
``rank_zero_only`` callbacks are constructed only on rank 0, so they must not
run collectives. A ``MetricsCallback`` with ``rank_zero_only`` may log only
``Reduce.RANK0``. Enablement comes from ``training_config``, identical on every
rank, and callbacks run in registry order.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import List
from typing import Mapping
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import Type

logger = logging.getLogger(__name__)


class Reduce(str, Enum):
    """How a logged metric is combined across ranks on the driver."""

    RANK0 = "rank0"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    PER_RANK = "per_rank"


Entry = Tuple[Reduce, Any]


class Trainer(Protocol):
    rank: int
    world_size: int
    global_steps: int
    cuda_device: int
    engine: Any
    model: Any
    optimizer: Any  # None for frozen models


class Callback:
    """Base class for a trainer-lifecycle callback. Every hook is a no-op unless overridden."""

    rank_zero_only: ClassVar[bool] = False

    def __init__(self, training_config: Mapping[str, Any]) -> None:
        pass

    @classmethod
    def enabled(cls, training_config: Mapping[str, Any]) -> bool:
        raise NotImplementedError

    def post_init(self, trainer: Trainer) -> None: ...

    def pre_fwd_bwd(self, trainer: Trainer) -> None: ...

    def post_fwd_bwd(self, trainer: Trainer, result: dict) -> None: ...

    def pre_fwd_bwd_microbatch(self, trainer: Trainer, is_padding: bool) -> None: ...

    def post_fwd_bwd_microbatch(self, trainer: Trainer, is_padding: bool) -> None: ...

    def pre_fwd_only(self, trainer: Trainer) -> None: ...

    def post_fwd_only(self, trainer: Trainer, result: dict) -> None: ...

    def pre_fwd_only_microbatch(self, trainer: Trainer, is_padding: bool) -> None: ...

    def pre_step(self, trainer: Trainer) -> None: ...

    def post_step(self, trainer: Trainer) -> None: ...

    def on_error(self, trainer: Trainer, stage: str, exc: BaseException) -> None: ...


class MetricsCallback(Callback):
    """A ``Callback`` that reports metrics via ``self.log(...)``, flushed once per step."""

    def __init__(self, training_config: Mapping[str, Any]) -> None:
        super().__init__(training_config)
        self.pending: Dict[str, Entry] = {}

    def log(self, name: str, value: Any, reduce: Reduce = Reduce.RANK0) -> None:
        if self.rank_zero_only and reduce is not Reduce.RANK0:
            raise ValueError(
                f"{type(self).__name__} is rank_zero_only and can only log Reduce.RANK0, got {reduce.value}"
            )
        if name in self.pending:
            raise ValueError(f"{type(self).__name__} logged {name!r} twice in one step")
        self.pending[name] = (reduce, value.tolist() if hasattr(value, "tolist") else value)


class CallbackRunner:
    """Instantiates the enabled callbacks for this rank and dispatches hooks to them in order."""

    def __init__(self, types: Sequence[Type[Callback]], training_config: Mapping[str, Any], rank: int) -> None:
        self.callbacks = [
            t(training_config) for t in types if t.enabled(training_config) and (rank == 0 or not t.rank_zero_only)
        ]
        self._metrics = [cb for cb in self.callbacks if isinstance(cb, MetricsCallback)]

    def run(self, hook: str, trainer: Trainer, *args: Any) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(trainer, *args)

    def flush(self) -> Dict[str, Entry]:
        try:
            merged: Dict[str, Entry] = {}
            for cb in self._metrics:
                if clash := merged.keys() & cb.pending.keys():
                    raise ValueError(f"metric names logged by more than one callback: {sorted(clash)}")
                merged.update(cb.pending)
            return merged
        finally:
            self._clear()

    def on_error(self, trainer: Trainer, stage: str, exc: BaseException) -> None:
        self._clear()
        for cb in self.callbacks:
            try:
                cb.on_error(trainer, stage, exc)
            except Exception:
                logger.exception(f"{type(cb).__name__}.on_error failed while handling {exc!r}")

    def _clear(self) -> None:
        for cb in self._metrics:
            cb.pending.clear()


# MEAN divides at each leaf (inside _combine's recursion), not on the merged
# structure afterward -- the merged value may be a nested dict/list, which a
# bare "/ len(values)" can't divide.
_COMBINE = {
    Reduce.SUM: sum,
    Reduce.MAX: max,
    Reduce.MIN: min,
    Reduce.MEAN: lambda values: sum(values) / len(values),
}


def merge_metrics(per_rank: Sequence[Mapping[str, Entry]]) -> Dict[str, Any]:
    """Driver side: merge every rank's flushed metrics. ``per_rank`` is in rank order."""
    if not per_rank:
        raise ValueError("per_rank is empty")
    merged = {name: value for name, (op, value) in per_rank[0].items() if op is Reduce.RANK0}
    reduced = [{name: e for name, e in rank.items() if e[0] is not Reduce.RANK0} for rank in per_rank]
    ops = [{name: op for name, (op, _) in rank.items()} for rank in reduced]
    if any(o != ops[0] for o in ops):
        raise ValueError("ranks logged different reduced metrics or reductions")
    for name, op in ops[0].items():
        values = [rank[name][1] for rank in reduced]
        merged[name] = values if op is Reduce.PER_RANK else _combine(_COMBINE[op], values)
    return merged


def _value_kind(value: Any) -> str:
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, list):
        return "list"
    return "scalar"


def _combine(op: Callable[[List[Any]], Any], values: List[Any]) -> Any:
    if not values:
        raise ValueError("no values to combine")
    kind = _value_kind(values[0])
    if any(_value_kind(value) != kind for value in values):
        raise ValueError("ranks logged values of different types")
    if kind == "mapping":
        if any(value.keys() != values[0].keys() for value in values):
            raise ValueError("ranks logged dicts with different keys")
        return {key: _combine(op, [value[key] for value in values]) for key in values[0]}
    if kind == "list":
        if any(len(value) != len(values[0]) for value in values):
            raise ValueError("ranks logged lists of different lengths")
        return [_combine(op, list(column)) for column in zip(*values, strict=True)]
    return op(values)
