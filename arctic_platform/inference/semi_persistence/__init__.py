"""Semi-persistent vLLM instances: CRIU checkpoint/restore multiplexing.

Modules in this package import their siblings by bare name (``import
semip_logging``), so the package directory must be on ``sys.path`` before
any of them can load.  The lazy ``__getattr__`` below installs that entry
on first attribute access rather than at import time: the worker and vLLM
child are created with ``spawn``, which re-imports the target module by
qualified name, so an eager import here would drag the orchestrator (and
vLLM) into processes that deliberately start with a clean address space.

The ``sys.path`` entry is left in place once installed -- ``spawn`` copies
the parent's ``sys.path`` into each child, and the child's ``import
worker`` / ``import vllm_child`` depends on it.
"""
import importlib
import os
import sys

__all__ = ["Instance", "Orchestrator", "OrchestratorClient"]

# ``Slots`` is deliberately absent: the orchestrator owns the allocator
# (``Orchestrator.init`` calls ``Slots.init``), so it is an implementation
# detail rather than part of the public surface.
_EXPORTS = {
    "Instance": ("instance", "Instance"),
    "Orchestrator": ("orchestrator", "Orchestrator"),
    "OrchestratorClient": ("client", "OrchestratorClient"),
}


def __getattr__(name):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
