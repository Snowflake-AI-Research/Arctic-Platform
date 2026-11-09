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

"""Record the exception a vLLM engine process fails to start with.

vLLM's front end raises only "Engine core initialization failed. See root cause
above."; the root stays in the EngineCore or worker process that logged it.
InferenceWorker points ``ARCTIC_ENGINE_STARTUP_ERROR_DIR`` at a fresh directory
for each startup attempt, the Arctic plugin installs ``StartupErrorHandler`` in
every engine process it loads in, and InferenceWorker appends the earliest
record to the error it raises.

Remove this once vLLM carries the failure in its startup handshake.
"""

from __future__ import annotations

import json
import logging
import os

ENV_VAR = "ARCTIC_ENGINE_STARTUP_ERROR_DIR"
MAX_CHARS = 2000

logger = logging.getLogger(__name__)

# What vLLM 0.32 logs, with the exception, when an EngineCore or a worker dies
# before it is ready (core.py run_engine_core, multiproc_executor.py worker_main).
_STARTUP_FAILURES = frozenset({"EngineCore failed to start.", "WorkerProc failed to start."})


class StartupErrorHandler(logging.Handler):
    """Write the startup failure this process logs to ``<directory>/<pid>.json``."""

    def __init__(self, directory: str) -> None:
        super().__init__(level=logging.ERROR)
        self.directory = directory

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not isinstance(record.msg, str) or record.msg not in _STARTUP_FAILURES or record.exc_info is None:
                return
            exc = record.exc_info[1]
            if exc is None:
                return
            cls = type(exc)
            if cls.__module__ in ("builtins", "__main__"):
                type_name = cls.__qualname__
            else:
                type_name = f"{cls.__module__}.{cls.__qualname__}"
            entry = {
                "process": getattr(record, "vllm_process_name", None) or record.processName,
                "type": type_name,
                "message": str(exc)[:MAX_CHARS],
                "time": record.created,
            }
            path = os.path.join(self.directory, f"{os.getpid()}.json")
            with open(f"{path}.tmp", "w", encoding="utf-8") as f:
                json.dump(entry, f)
            os.replace(f"{path}.tmp", path)
        except Exception:
            # Best effort: without a record, InferenceWorker raises vLLM's error unchanged.
            self.handleError(record)


def install_startup_error_handler() -> None:
    """Add the handler to the ``vllm`` logger when InferenceWorker asked for records."""
    directory = os.environ.get(ENV_VAR)
    if directory is None or directory == "":
        return
    vllm_logger = logging.getLogger("vllm")
    if any(isinstance(handler, StartupErrorHandler) for handler in vllm_logger.handlers):
        return
    vllm_logger.addHandler(StartupErrorHandler(directory))


def earliest_startup_error(directory: str) -> str | None:
    """``Root cause in <process>: <type>: <message>`` for the earliest record, or None.

    A worker that fails while starting logs its own failure seconds before the
    EngineCore logs the generic one it caused, so the earliest record is the root.
    """
    records = []
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                entry = json.load(f)
            records.append((float(entry["time"]), entry["process"], entry["type"], entry["message"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.info("Skipping unreadable engine startup record %s: %s", name, exc)
            continue
    if len(records) == 0:
        return None
    _, process, type_name, message = min(records, key=lambda record: record[0])
    return f"Root cause in {process}: {type_name}: {message}"[:MAX_CHARS]
