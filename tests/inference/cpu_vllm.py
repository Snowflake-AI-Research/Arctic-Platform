"""CPU stand-in for vLLM, used by the unit-test job.

The CPU job does not install vLLM. This module is a pytest plugin
(``python -m pytest -p cpu_vllm``) that satisfies ``import vllm...`` with
packages, classes, and callables so inference tests can import the
production modules. It is not a vLLM implementation: tests that need real
scheduler or kernel behavior stay out of this job.

Grammar tests must not load this plugin. They use ``importorskip("vllm")``
to skip the checks that need the real package, and a stand-in in
``sys.modules`` makes that skip think vLLM is installed.
"""

from __future__ import annotations

import enum
import importlib
import importlib.abc
import importlib.machinery
import logging
import sys
import types

_INSTALLED = False

# Tests install their own stand-ins with sys.modules.setdefault. Pre-creating
# these keeps a non-package ModuleType from winning and breaking later
# ``from vllm.x.y import Z`` imports.
_PRESEED = (
    "vllm",
    "vllm.config",
    "vllm.config.utils",
    "vllm.config.speculative",
    "vllm.distributed",
    "vllm.distributed.parallel_state",
    "vllm.engine",
    "vllm.engine.arg_utils",
    "vllm.logger",
    "vllm.platforms",
    "vllm.v1",
    "vllm.v1.core",
    "vllm.v1.core.sched",
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.core.sched.utils",
    "vllm.v1.metrics",
    "vllm.v1.metrics.loggers",
    "vllm.v1.metrics.stats",
    "vllm.v1.request",
    "vllm.v1.structured_output",
    "vllm.v1.structured_output.backend_xgrammar",
    "vllm.v1.attention",
    "vllm.v1.attention.backends",
    "vllm.v1.attention.backends.gdn_attn",
    "vllm.v1.attention.backends.utils",
    "vllm.v1.attention.backends.flash_attn",
    "vllm.v1.kv_cache_interface",
    "vllm.v1.worker",
    "vllm.v1.worker.gpu_worker",
    "vllm.v1.worker.worker_base",
    "vllm.v1.worker.utils",
    "vllm.v1.spec_decode",
    "vllm.v1.spec_decode.metrics",
)


class RequestStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_REPETITION = enum.auto()


class _StubType(type):
    def __getattr__(cls, name):
        # Class bodies read private originals (_apply_head, __post_init__).
        # Other dunders stay missing so dataclass and type machinery keep
        # their own lookup.
        if name.startswith("__") and name.endswith("__") and name != "__post_init__":
            raise AttributeError(name)

        def _method(*args, **kwargs):
            return None

        _method.__name__ = name
        setattr(cls, name, _method)
        return _method


def _make_class(name: str, module: str) -> type:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    return _StubType(name, (), {"__init__": __init__, "__module__": module})


class _StubModule(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        self.__dict__["__path__"] = []
        self.__dict__["__package__"] = name
        self.__dict__["_cpu_vllm_stub"] = True
        if name == "vllm":
            self.__dict__["__version__"] = "0.31.0"

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        value = _attribute(self.__name__, name)
        setattr(self, name, value)
        if isinstance(value, types.ModuleType):
            sys.modules.setdefault(value.__name__, value)
        return value

    def __call__(self, *args, **kwargs):
        leaf = self.__name__.rsplit(".", 1)[-1]
        if leaf == "init_logger":
            return logging.getLogger(*args, **kwargs)
        if leaf == "config":
            def decorator(target):
                return target

            return decorator
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return None


def _attribute(module_name: str, name: str):
    if module_name == "vllm.v1.request" and name == "RequestStatus":
        return RequestStatus
    # Valid Literal parameter. A generated class is not.
    if name == "EagleModelTypes":
        return "eagle"
    if name.isupper():
        return 0
    if name[:1].isupper():
        return _make_class(name, module_name)
    child = _StubModule(f"{module_name}.{name}")
    return child


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        return None


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "vllm" and not fullname.startswith("vllm."):
            return None
        return importlib.machinery.ModuleSpec(
            fullname, _Loader(), is_package=True
        )


def install() -> None:
    """Register the stand-in. A real vLLM install is left alone."""
    global _INSTALLED
    if _INSTALLED:
        return
    existing = sys.modules.get("vllm")
    if existing is not None and not getattr(existing, "_cpu_vllm_stub", False):
        if getattr(existing, "__file__", None):
            _INSTALLED = True
            return
    sys.meta_path.insert(0, _Finder())
    for name in _PRESEED:
        importlib.import_module(name)
    _INSTALLED = True


install()
