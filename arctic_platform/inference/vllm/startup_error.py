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
every engine process it loads in, and InferenceWorker raises the earliest
record as an ``EngineStartupError``
(``arctic_platform.inference.engine_startup_error``).

Remove this once vLLM carries the failure in its startup handshake.
"""

from __future__ import annotations

import json
import logging
import os
from types import TracebackType
from typing import Any

ENV_VAR = "ARCTIC_ENGINE_STARTUP_ERROR_DIR"
MAX_CHARS = 2000

logger = logging.getLogger(__name__)

# What vLLM 0.32 logs, with the exception, when an EngineCore or a worker dies
# before it is ready (core.py run_engine_core, multiproc_executor.py worker_main).
_STARTUP_FAILURES = frozenset({"EngineCore failed to start.", "WorkerProc failed to start."})
# What a worker logs, with the exception, when an RPC from the EngineCore fails
# (multiproc_executor.py _execute_worker_rpc). The EngineCore then raises only
# ``RuntimeError("Worker failed with error '<str>'...")``, so during startup
# (profile run, KV cache init, warmup) this is the only record of the real type.
_WORKER_RPC_FAILURE = "WorkerProc hit an exception."
# Suffix of a worker RPC failure's record. Its stem is not a pid, so
# ``recorded_pids`` (the startup watchdog's signal) leaves it out.
_RPC_SUFFIX = ".rpc"
# How the EngineCore's error starts when a worker RPC failed
# (multiproc_executor.py collective_rpc); ``str()`` of the worker's error follows.
_WORKER_FAILED_PREFIX = "Worker failed with error '"
# Enough of a worker's message to find it in the EngineCore's, which MAX_CHARS
# truncates after the prefix.
_RPC_MATCH_CHARS = 1000


def _type_name(cls: type) -> str:
    if cls.__module__ in ("builtins", "__main__"):
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def _origin(tb: TracebackType | None) -> str | None:
    """``<module>.<qualname>`` of the innermost frame of ``tb``, the code that raised."""
    if tb is None:
        return None
    while tb.tb_next is not None:
        tb = tb.tb_next
    code = tb.tb_frame.f_code
    module = tb.tb_frame.f_globals.get("__name__")
    name = getattr(code, "co_qualname", code.co_name)
    return f"{module}.{name}" if module else name


class StartupErrorHandler(logging.Handler):
    """Write the startup failure this process logs to ``<directory>/<pid>.json``.

    A worker's first RPC failure while the directory exists goes to
    ``<directory>/<pid>.rpc.json``.
    """

    def __init__(self, directory: str) -> None:
        super().__init__(level=logging.ERROR)
        self.directory = directory

    def emit(self, record: logging.LogRecord) -> None:
        rpc = False
        try:
            if not isinstance(record.msg, str) or record.exc_info is None:
                return
            if record.msg in _STARTUP_FAILURES:
                path = os.path.join(self.directory, f"{os.getpid()}.json")
            elif record.msg == _WORKER_RPC_FAILURE:
                # A worker keeps the directory in its environment after startup;
                # InferenceWorker removes it once startup ends, which is what
                # keeps later RPC failures out. The first failure is the root.
                if not os.path.isdir(self.directory):
                    return
                rpc = True
                path = os.path.join(self.directory, f"{os.getpid()}{_RPC_SUFFIX}.json")
                if os.path.exists(path):
                    return
            else:
                return
            exc = record.exc_info[1]
            if exc is None:
                return
            entry = {
                "process": getattr(record, "vllm_process_name", None) or record.processName,
                "type": _type_name(type(exc)),
                "message": str(exc)[:MAX_CHARS],
                "origin": _origin(exc.__traceback__),
                "time": record.created,
            }
            with open(f"{path}.tmp", "w", encoding="utf-8") as f:
                json.dump(entry, f)
            os.replace(f"{path}.tmp", path)
        except (FileNotFoundError, NotADirectoryError):
            if not rpc:
                self.handleError(record)
            # Otherwise startup ended after the check above: nothing to record.
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


def recorded_pids(directory: str) -> set[int]:
    """The pids of the engine processes that recorded a startup failure in ``directory``."""
    pids = set()
    for name in os.listdir(directory):
        stem, ext = os.path.splitext(name)
        if ext == ".json" and stem.isdigit():
            pids.add(int(stem))
    return pids


def earliest_startup_record(directory: str) -> dict[str, Any] | None:
    """The root of a failed startup recorded in ``directory`` (process, type, message, origin), or None.

    A worker that fails while starting logs its own failure seconds before the
    EngineCore logs the generic one it caused, so the earliest startup record is
    the root. When that record is the EngineCore's ``Worker failed with error
    '<message>'``, the worker RPC failure with that message is the root instead:
    it has the worker's own type. An RPC failure the EngineCore does not name
    is never the root, so one that startup survived cannot claim a later,
    unrelated failure.
    """
    startup = []
    rpc = []
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                entry = json.load(f)
            origin = entry.get("origin")
            record = {
                "time": float(entry["time"]),
                "process": str(entry["process"]),
                "type": str(entry["type"]),
                "message": str(entry["message"]),
                "origin": origin if isinstance(origin, str) else None,
            }
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.info("Skipping unreadable engine startup record %s: %s", name, exc)
            continue
        (rpc if name.endswith(f"{_RPC_SUFFIX}.json") else startup).append(record)
    if len(startup) == 0:
        return None
    root = min(startup, key=lambda record: record["time"])
    named = [record for record in rpc if _names_rpc_failure(root["message"], record["message"])]
    if len(named) > 0:
        return min(named, key=lambda record: record["time"])
    return root


def _names_rpc_failure(engine_message: str, worker_message: str) -> bool:
    """Whether the EngineCore's ``Worker failed with error '<message>'`` quotes ``worker_message``."""
    if not engine_message.startswith(_WORKER_FAILED_PREFIX) or len(worker_message) == 0:
        return False
    quoted = engine_message[len(_WORKER_FAILED_PREFIX) :]
    head = worker_message[:_RPC_MATCH_CHARS]
    if not quoted.startswith(head):
        return False
    # The whole message, not a prefix of another: its closing quote follows,
    # unless MAX_CHARS cut either one first.
    return len(worker_message) > len(head) or len(quoted) == len(head) or quoted[len(head)] == "'"


def format_root_cause(record: dict[str, Any]) -> str:
    """``Root cause in <process>: <type>: <message>``, the line older classifiers match."""
    return f"Root cause in {record['process']}: {record['type']}: {record['message']}"[:MAX_CHARS]


def earliest_startup_error(directory: str) -> str | None:
    """``Root cause in <process>: <type>: <message>`` for the earliest record, or None."""
    record = earliest_startup_record(directory)
    return None if record is None else format_root_cause(record)
