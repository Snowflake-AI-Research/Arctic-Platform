"""Distributed stats tracker for RL training metrics."""

from __future__ import annotations

import itertools
import logging as _logging
import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from threading import Lock
from typing import Dict

import torch
import torch.distributed as dist

_stats_logger = _logging.getLogger("StatsTracker")


def _flat2d(arr):
    return list(itertools.chain(*arr))


class ReduceType(Enum):
    AVG_MIN_MAX = auto()
    AVG = auto()
    SUM = auto()
    MIN = auto()
    MAX = auto()
    SCALAR = auto()


class DistributedStatsTracker:
    def __init__(self, name: str = ""):
        self.lock = Lock()
        self.scope_stack = []
        if name:
            self.scope_stack.append(name.strip("/"))
        self.denominators = {}
        self.reduce_types = {}
        self.stats = defaultdict(list)

    class Scope:
        def __init__(self, tracker, name):
            self.tracker = tracker
            self.name = name

        def __enter__(self):
            with self.tracker.lock:
                self.tracker.scope_stack.append(self.name)
            return self

        def __exit__(self, *args):
            with self.tracker.lock:
                self.tracker.scope_stack.pop()

    @contextmanager
    def disable_scope(self):
        """Temporarily clear the scope stack."""
        with self.lock:
            saved = list(self.scope_stack)
            self.scope_stack.clear()
        try:
            yield
        finally:
            with self.lock:
                self.scope_stack[:] = saved

    def scope(self, name):
        with self.lock:
            return self.Scope(self, name)

    def _prefix(self):
        return "/".join(self.scope_stack) + "/" if self.scope_stack else ""

    def denominator(self, **kwargs):
        prefix = self._prefix()
        for key, val in kwargs.items():
            full_key = prefix + key
            with self.lock:
                self.stats[full_key].append(val)
                self.denominators[full_key] = full_key
                self.reduce_types[full_key] = ReduceType.AVG_MIN_MAX

    def stat(self, denominator="n_valid_tokens", **kwargs):
        prefix = self._prefix()
        denom_key = prefix + denominator
        for key, val in kwargs.items():
            full_key = prefix + key
            with self.lock:
                self.stats[full_key].append(val)
                self.denominators[full_key] = denom_key
                self.reduce_types[full_key] = ReduceType.AVG_MIN_MAX

    def scalar(self, **kwargs):
        prefix = self._prefix()
        for key, val in kwargs.items():
            full_key = prefix + key
            with self.lock:
                self.stats[full_key].append(val)
                self.denominators[full_key] = None
                self.reduce_types[full_key] = ReduceType.SCALAR

    @contextmanager
    def record_timing(self, key):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.scalar(**{key: elapsed})

    def export(self, reduce_group=None, reset=True):
        with self.lock:
            result = {}
            for key, vals in self.stats.items():
                reduce_type = self.reduce_types.get(key, ReduceType.SCALAR)
                denom_key = self.denominators.get(key)
                if reduce_type == ReduceType.SCALAR:
                    if vals:
                        result[key] = float(vals[-1]) if not isinstance(vals[-1], torch.Tensor) else vals[-1].item()
                else:
                    if not vals:
                        continue
                    tensors = [v.float() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32) for v in vals]
                    combined = torch.cat([t.reshape(-1) for t in tensors])
                    if denom_key and denom_key in self.stats:
                        denom_vals = self.stats[denom_key]
                        denom_tensors = [v.float() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32) for v in denom_vals]
                        denom = torch.cat([t.reshape(-1) for t in denom_tensors]).bool()
                        valid = combined[denom] if denom.any() else combined
                    else:
                        valid = combined
                    if dist.is_initialized() and reduce_group is not None:
                        dist.all_reduce(valid, op=dist.ReduceOp.SUM, group=reduce_group)
                    if valid.numel() > 0:
                        result[key] = valid.mean().item()
            if reset:
                self.stats.clear()
                self.denominators.clear()
                self.reduce_types.clear()
        return result

    def export_all(self, reduce_group=None, reset=True):
        return self.export(reduce_group=reduce_group, reset=reset)


TRACKERS: Dict[str, "DistributedStatsTracker"] = {}
DEFAULT_TRACKER = DistributedStatsTracker()


def get(name: str) -> DistributedStatsTracker:
    if name not in TRACKERS:
        TRACKERS[name] = DistributedStatsTracker(name)
    return TRACKERS[name]


def denominator(**kwargs):
    DEFAULT_TRACKER.denominator(**kwargs)


def stat(denominator="n_valid_tokens", **kwargs):
    DEFAULT_TRACKER.stat(denominator=denominator, **kwargs)


def scalar(**kwargs):
    DEFAULT_TRACKER.scalar(**kwargs)


def scope(name):
    return DEFAULT_TRACKER.scope(name)


def export(reduce_group=None, reset=True):
    return DEFAULT_TRACKER.export(reduce_group=reduce_group, reset=reset)


def export_all(reduce_group=None, reset=True):
    result = DEFAULT_TRACKER.export(reduce_group=reduce_group, reset=reset)
    for tracker in TRACKERS.values():
        result.update(tracker.export(reduce_group=reduce_group, reset=reset))
    return result


@contextmanager
def record_timing(key):
    with DEFAULT_TRACKER.record_timing(key):
        yield
