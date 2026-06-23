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

import functools
import inspect
import os
from pathlib import Path

import torch

from arctic_platform.rl.utils.debug import print_rank0 as pr0


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _replay_generation_enabled() -> bool:
    # Read at call time: env may be set after import.
    return _env_flag("REPLAY_GENERATION")


def _replay_generation_dir() -> Path:
    gen_dir = os.environ.get("REPLAY_GENERATION_DIR", None)
    if gen_dir is None:
        return None
    return Path(gen_dir)


def _replay_generation_mode() -> str:
    mode = os.environ.get("REPLAY_GENERATION_MODE", "record").lower()
    if mode not in ("record", "load"):
        raise ValueError(f"Invalid REPLAY_GENERATION_MODE={mode!r}; expected 'record' or 'load'")
    return mode


def _replay_start_record() -> int:
    # Leading records to skip in load mode.
    try:
        return max(0, int(os.environ.get("REPLAY_START_RECORD", "0") or 0))
    except ValueError:
        return 0


class RecordReplay:
    def __init__(self):
        self.enabled = _replay_generation_enabled()
        self.dir = _replay_generation_dir()
        self.mode = _replay_generation_mode()
        self.start_record = _replay_start_record()

        if self.is_replay_mode() and self.start_record > 0:
            pr0(
                f"[REPLAY_GENERATION] REPLAY_START_RECORD={self.start_record}: "
                f"will skip records 1..{self.start_record} and begin replay at "
                f"record {self.start_record + 1}",
            )
        if self.enabled:
            if self.is_record_mode():
                self.dir.mkdir(parents=True, exist_ok=True)
            elif self.is_replay_mode():
                if not self.dir.exists():
                    raise FileNotFoundError(f"[REPLAY_GENERATION] load mode requires {self.dir}")

    def is_enabled(self) -> bool:
        return self.enabled and self.dir is not None

    def is_record_mode(self) -> bool:
        return self.is_enabled() and self.mode == "record"

    def is_replay_mode(self) -> bool:
        return self.is_enabled() and self.mode == "load"

    def skip_record(self, record_index: int) -> bool:
        return self.is_enabled() and self.is_replay_mode() and record_index <= self.start_record

    def save_record(self, record_index: int, batch_dict: dict):
        if self.is_record_mode():
            replay_path = self.dir / f"generate-{record_index}.pickle"
            pr0(f"[REPLAY_GENERATION] saving {replay_path}")
            torch.save(batch_dict, replay_path)

    def load_record(self, record_index: int) -> dict:
        if self.is_replay_mode():
            replay_path = self.dir / f"generate-{record_index}.pickle"
            if not replay_path.exists():
                raise FileNotFoundError(f"[REPLAY_GENERATION] load mode requires {replay_path}")
            pr0(f"[REPLAY_GENERATION] loading {replay_path}")
            return torch.load(replay_path, weights_only=False)


def record_replay_generation(func):
    """``lru_cache``-style record/replay decorator for a generation function.

    Wrap ``generate_sequences`` (or any function that produces rollout data) and,
    depending on the ``REPLAY_GENERATION*`` env vars, transparently record its
    outputs to disk or replay previously recorded outputs instead of calling it.

    Example::

        from arctic_platform.rl.utils import record_replay_generation

        # Decorate the generation function at its definition...
        @record_replay_generation
        def generate_sequences(prompts, **kwargs):
            ...
            return gen_batch_output

        # ...then call it exactly as before -- no extra arguments, no `with`:
        gen_batch_output = generate_sequences(prompts)

    Driven by env vars (read lazily on first call):

    * record a run:  ``REPLAY_GENERATION=1 REPLAY_GENERATION_MODE=record \\
                       REPLAY_GENERATION_DIR=/path/to/dir``
    * replay it:     ``REPLAY_GENERATION=1 REPLAY_GENERATION_MODE=load \\
                       REPLAY_GENERATION_DIR=/path/to/dir``
      (optionally ``REPLAY_START_RECORD=N`` to regenerate the first N calls).

    Behaviour, decided per call:

    * **disabled** (default): pass straight through to the wrapped function.
    * **load mode**: if a recording exists for this call, return it *without*
      calling the wrapped function (the cache "hit"); otherwise call it normally.
      The leading ``REPLAY_START_RECORD`` calls are always (re)generated.
    * **record mode**: call the wrapped function and save its output, then return it.

    Like ``functools.lru_cache``, the caller passes no key: outputs form a tape
    keyed by call order within the process (``generate-1.pickle``,
    ``generate-2.pickle``, ...). Record and replay runs must therefore issue the
    same sequence of calls for the indices to line up. Works with both sync and
    ``async def`` generation functions.
    """
    state = {"rr": None, "index": 0}

    def _engine() -> RecordReplay:
        # Build lazily so the REPLAY_GENERATION* env vars can be set after import.
        if state["rr"] is None:
            state["rr"] = RecordReplay()
        return state["rr"]

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            rr = _engine()
            if not rr.is_enabled():
                return await func(*args, **kwargs)
            state["index"] += 1
            index = state["index"]
            if rr.is_replay_mode() and not rr.skip_record(index):
                return rr.load_record(index)  # cache hit -> recorded data
            output = await func(*args, **kwargs)
            rr.save_record(index, output)  # no-op unless record mode
            return output

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rr = _engine()
            if not rr.is_enabled():
                return func(*args, **kwargs)
            state["index"] += 1
            index = state["index"]
            if rr.is_replay_mode() and not rr.skip_record(index):
                return rr.load_record(index)  # cache hit -> recorded data
            output = func(*args, **kwargs)
            rr.save_record(index, output)  # no-op unless record mode
            return output

    # Expose the engine + a counter reset, mirroring lru_cache's introspection.
    wrapper.record_replay = _engine
    wrapper.reset_record_index = lambda: state.update(index=0)
    return wrapper
