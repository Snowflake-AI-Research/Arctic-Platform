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

"""The typed error InferenceWorker raises when a vLLM engine fails to start.

vLLM's front end replaces an engine process's own failure with a generic
``RuntimeError("Engine core initialization failed...")``, and Ray drops
``__cause__``. ``EngineStartupError`` carries the engine process's root
failure as fields, so a caller across a Ray actor boundary can classify it
without reading the message.

This module is the contract. Callers read the fields duck-typed, by the names
in ``FIELDS``, from the exception or from the ``.cause`` a ``RayTaskError``
keeps it on, so one code path also handles an older Arctic-Platform that
raises a plain ``RuntimeError``. The class is pickled by reference: the
process that unpickles it (the caller of ``ray.get``) needs an
Arctic-Platform that has this module. Keep the names and their meaning
stable; add fields rather than renaming them. No imports beyond the standard
library, so importing it never pulls in vLLM or torch.
"""

from __future__ import annotations

from typing import Any

# The engine process's own exception type, qualified as a traceback prints it:
# ``torch.OutOfMemoryError``, ``torch.AcceleratorError``, or a bare builtin
# name such as ``ValueError``.
ROOT_TYPE = "root_type"
# ``str()`` of that exception, bounded.
ROOT_MESSAGE = "root_message"
# The vLLM process that raised it, e.g. ``EngineCore`` or ``Worker_TP0``.
PROCESS = "process"
# ``<module>.<qualname>`` of the innermost frame that raised it, e.g.
# ``vllm.v1.core.kv_cache_utils._check_enough_kv_cache_memory``, or None when
# the record had no traceback. Identifies errors vLLM raises as bare builtins.
ORIGIN = "origin"
# The CUDA runtime error code torch puts on a ``torch.AcceleratorError``
# instance as ``error_code`` (e.g. 700 for an illegal address), or None.
CUDA_ERROR_CODE = "cuda_error_code"
FIELDS = (ROOT_TYPE, ROOT_MESSAGE, PROCESS, ORIGIN, CUDA_ERROR_CODE)


class EngineStartupError(RuntimeError):
    """vLLM's engine startup failure, with the engine process's root failure as fields.

    ``str()`` is the message InferenceWorker raised before this type existed
    (vLLM's generic error, then ``Root cause in <process>: <type>: <message>``),
    so a classifier that matches text still matches. A ``RuntimeError``, so
    code that catches vLLM's generic error still catches it.
    """

    def __init__(
        self,
        message: str,
        root_type: str = "",
        root_message: str = "",
        process: str = "",
        origin: str | None = None,
        cuda_error_code: int | None = None,
    ) -> None:
        super().__init__(message)
        # Instance attributes only, never class-level defaults: Ray's
        # ``as_instanceof_cause`` subclass reads a missing attribute from the
        # cause, and a class attribute would shadow it.
        self.root_type = root_type
        self.root_message = root_message
        self.process = process
        self.origin = origin
        self.cuda_error_code = cuda_error_code

    def __reduce__(self) -> tuple[Any, ...]:
        # BaseException's reduce re-calls the class with ``args`` only, which
        # loses the fields. The state dict carries ``__notes__`` and the fields.
        return (
            type(self),
            (
                self.args[0] if self.args else "",
                self.root_type,
                self.root_message,
                self.process,
                self.origin,
                self.cuda_error_code,
            ),
            self.__dict__,
        )
