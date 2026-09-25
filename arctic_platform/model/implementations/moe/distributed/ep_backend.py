"""Select the expert-parallel dispatch/combine module from ``ep_comm_backend``."""

from __future__ import annotations

from types import ModuleType

from ..config import DISPATCH_EP_BACKENDS


def uses_dispatch_ep(backend: str) -> bool:
    return backend in DISPATCH_EP_BACKENDS


def get_ep_comm_module(backend: str) -> ModuleType:
    if backend == "uccl":
        from . import ucclep

        return ucclep
    if backend == "deepep":
        from . import deepep

        return deepep
    raise NotImplementedError(f"Unsupported EP comm backend {backend!r}; expected one of {DISPATCH_EP_BACKENDS}.")
