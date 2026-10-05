"""Import real orchestration modules without installing CUDA-only dependencies."""

import importlib
import sys
import types
from pathlib import Path


class VLLMValidationError(Exception):
    def __init__(self, message, *, parameter=None, value=None):
        super().__init__(message)
        self.parameter = parameter
        self.value = value

    def __str__(self):
        message = super().__str__()
        details = []
        if self.parameter is not None:
            details.append(f"parameter={self.parameter}")
        if self.value is not None:
            details.append(f"value={self.value}")
        return f"{message} ({', '.join(details)})" if details else message


def load_library():
    import arctic_platform.inference

    if "arctic_platform.inference.server" not in sys.modules:
        package = types.ModuleType("arctic_platform.inference.server")
        package.__path__ = [str(Path(arctic_platform.inference.__file__).parent / "server")]
        sys.modules[package.__name__] = package
    for name in (
        "vllm",
        "vllm.exceptions",
        "vllm.v1",
        "vllm.v1.metrics",
        "vllm.config",
        "vllm.v1.metrics.loggers",
        "vllm.v1.metrics.stats",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.config"].VllmConfig = object
    sys.modules["vllm.exceptions"].VLLMValidationError = VLLMValidationError
    sys.modules["vllm.v1.metrics.loggers"].StatLoggerBase = object
    for name in ("IterationStats", "SchedulerStats", "MultiModalCacheStats"):
        setattr(sys.modules["vllm.v1.metrics.stats"], name, object)
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    sys.modules.setdefault(
        "arctic_platform.inference.server.api", types.ModuleType("arctic_platform.inference.server.api")
    )
    sys.modules["arctic_platform.inference.server.api"].app = None
    return importlib.import_module("arctic_platform.inference.server.multi_model")
