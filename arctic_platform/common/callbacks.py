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
install read-only hooks (e.g. forward hooks), and raise. They must **not** run
collectives inside hooks: aggregation belongs to ``Reduce`` + the driver's
``merge_metrics``. That keeps a single-rank raise from hanging peers, and lets
``CallbackRunner`` quarantine a failed callback for the rest of the step.
``rank_zero_only`` callbacks are constructed only on rank 0. A
``MetricsCallback`` with ``rank_zero_only`` may log only ``Reduce.RANK0``.
Enablement comes from ``training_config``, identical on every rank, and
callbacks run in registry order.

``on_error`` contract (trainer / driver):
- May run on ranks whose request succeeded (e.g. a broadcast from a peer failure).
- May run twice on the failing rank (local ``except`` then a later broadcast).
- For a given stage, arrives before that rank's next request of the same stage
  when the driver serializes same-stage work; a stage-tagged snapshot restore is
  a no-op when the saved stage does not match. Same-stage interleave after the
  next ``pre_*`` rewrite is a residual that needs a request id from the driver.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import AbstractSet
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
    model: Any
    _engine: Any
    _optimizer: Any  # None for frozen models


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

    def post_fwd_only_microbatch(self, trainer: Trainer, is_padding: bool) -> None: ...

    def pre_step(self, trainer: Trainer) -> None: ...

    def post_step(self, trainer: Trainer) -> None: ...

    def on_error(self, trainer: Trainer, stage: str, exc: BaseException) -> None: ...

    def on_quarantine(self, trainer: Trainer) -> None:
        """Stop side effects after a hook ``Exception``. Default is a no-op.

        Callbacks that install model hooks or mutate state outside ``run`` must
        override this so quarantine does more than skip later methods.
        """


class MetricsCallback(Callback):
    """A ``Callback`` that reports metrics via ``self.log(...)``, flushed once per step."""

    def __init__(self, training_config: Mapping[str, Any]) -> None:
        super().__init__(training_config)
        self.pending: Dict[str, Entry] = {}

    def metric_names(self) -> AbstractSet[str]:
        """Stable names this callback may log; used to drop peer metrics on quarantine.

        Override with a fixed set. Default is empty so a pre-``log`` quarantine does
        not invent names; subclasses that log must declare them.
        """
        return frozenset()

    def log(self, name: str, value: Any, reduce: Reduce = Reduce.RANK0) -> None:
        if self.rank_zero_only and reduce is not Reduce.RANK0:
            raise ValueError(
                f"{type(self).__name__} is rank_zero_only and can only log Reduce.RANK0, got {reduce.value}"
            )
        if name in self.pending:
            raise ValueError(f"{type(self).__name__} logged {name!r} twice in one step")
        self.pending[name] = (reduce, value.tolist() if hasattr(value, "tolist") else value)


class CallbackRunner:
    """Instantiates the enabled callbacks for this rank and dispatches hooks to them in order.

    A post-``post_init`` hook ``Exception`` quarantines that callback for the rest of the
    step: later hooks skip it, ``on_quarantine`` stops its side effects, and ``flush``
    drops its metrics (recorded in ``last_dropped``). ``post_init`` is not quarantined —
    it raises. ``BaseException`` (e.g. ``KeyboardInterrupt``) still propagates.
    ``_failed`` clears on ``flush`` only, not on ``on_error``.
    """

    def __init__(self, types: Sequence[Type[Callback]], training_config: Mapping[str, Any], rank: int) -> None:
        self.callbacks = [
            t(training_config) for t in types if t.enabled(training_config) and (rank == 0 or not t.rank_zero_only)
        ]
        self._metrics = [cb for cb in self.callbacks if isinstance(cb, MetricsCallback)]
        self._failed: set[Callback] = set()
        self.last_dropped: frozenset[str] = frozenset()

    def run(self, hook: str, trainer: Trainer, *args: Any) -> None:
        if hook == "post_init":
            for cb in self.callbacks:
                getattr(cb, hook)(trainer, *args)
            return
        for cb in self.callbacks:
            if cb in self._failed:
                continue
            try:
                getattr(cb, hook)(trainer, *args)
            except Exception:
                logger.exception(
                    "%s.%s failed; quarantining for the rest of this step",
                    type(cb).__name__,
                    hook,
                )
                self._quarantine(cb, trainer)

    def flush(self) -> Dict[str, Entry]:
        dropped: set[str] = set()
        try:
            merged: Dict[str, Entry] = {}
            for cb in self._metrics:
                if cb in self._failed:
                    dropped.update(cb.metric_names())
                    dropped.update(cb.pending.keys())
                    continue
                if clash := merged.keys() & cb.pending.keys():
                    raise ValueError(f"metric names logged by more than one callback: {sorted(clash)}")
                merged.update(cb.pending)
            self.last_dropped = frozenset(dropped)
            return merged
        finally:
            self._clear()
            self._failed.clear()

    def on_error(self, trainer: Trainer, stage: str, exc: BaseException) -> None:
        self._clear()
        for cb in self.callbacks:
            if cb in self._failed:
                continue
            try:
                cb.on_error(trainer, stage, exc)
            except Exception:
                logger.exception(f"{type(cb).__name__}.on_error failed while handling {exc!r}")
                self._quarantine(cb, trainer)

    def _quarantine(self, cb: Callback, trainer: Trainer) -> None:
        self._failed.add(cb)
        try:
            cb.on_quarantine(trainer)
        except Exception:
            logger.exception("%s.on_quarantine failed; leaving callback quarantined", type(cb).__name__)

    def _clear(self) -> None:
        for cb in self._metrics:
            cb.pending.clear()


# MEAN divides at each leaf (inside _combine's recursion), not on the merged
# structure afterward -- the merged value may be a nested dict/list, which a
# bare "/ len(values)" can't divide. Builtin ``sum``/``max``/``min`` are wrapped
# so mypy sees ``Callable[[List[Any]], Any]`` rather than a bare function type.
_COMBINE: Dict[Reduce, Callable[[List[Any]], Any]] = {
    Reduce.SUM: lambda values: sum(values),
    Reduce.MAX: lambda values: max(values),
    Reduce.MIN: lambda values: min(values),
    Reduce.MEAN: lambda values: sum(values) / len(values),
}


def merge_metrics(
    per_rank: Sequence[Mapping[str, Entry]],
    dropped_per_rank: Sequence[AbstractSet[str]] = (),
) -> Dict[str, Any]:
    """Driver side: merge every rank's flushed metrics. ``per_rank`` is in rank order.

    ``dropped_per_rank`` is the per-rank ``CallbackRunner.last_dropped`` sets. Their
    union is stripped from every rank before the equality check so a single-rank
    quarantine does not raise ``ranks logged different reduced metrics``.
    """
    if not per_rank:
        raise ValueError("per_rank is empty")
    drop = frozenset().union(*dropped_per_rank) if dropped_per_rank else frozenset()
    if drop:
        logger.warning("dropping callback metrics quarantined on at least one rank: %s", sorted(drop))
        per_rank = tuple({name: entry for name, entry in rank.items() if name not in drop} for rank in per_rank)
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
