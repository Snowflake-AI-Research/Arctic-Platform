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

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Select the caller-staged scoped LoRA worker before native pool construction."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path

from arctic_platform.debug import lora_selected_sampler as selected
from arctic_platform.debug import lora_sync_hooks as legacy
from arctic_platform.debug.lora_sampler_evidence import _bind_sources
from arctic_platform.debug.lora_sampler_evidence import _require


def _pointer():
    path, digest = os.environ.get(selected.PATH_ENV), os.environ.get(selected.SHA_ENV)
    if path is None and digest is None:
        return None
    _require(
        type(path) is str and type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest) is not None,
        "selected bootstrap requires both exact path and SHA256",
    )
    root = Path(path)
    _require(
        root.is_absolute()
        and root.is_relative_to("/data-fast")
        and root.is_file()
        and not root.is_symlink()
        and root.resolve(strict=True) == root,
        "selected bootstrap requires a canonical regular FAST path",
    )
    return {"path": path, "sha256": digest}


def maybe_lora_replica_pool(pool_class):
    """Preserve the existing route unless an exact selected bootstrap is staged."""
    reader, maker = legacy.selected_lora_request, selected.make_selected_lora_replica_pool
    request = reader()
    pointer = _pointer()
    if pointer is None:
        return legacy.maybe_lora_replica_pool(pool_class)
    _require(request is not None, "selected bootstrap requires its staged LoRA request")
    origins = deepcopy(request["sampling"]["admitted_sources"])
    callbacks = (maybe_lora_replica_pool, _pointer, reader, maker)
    _bind_sources(origins, *callbacks)
    original = deepcopy(request)
    result = maker(pool_class, deepcopy(request["sampling"]), dict(pointer))
    _require(
        legacy.selected_lora_request is reader
        and selected.make_selected_lora_replica_pool is maker
        and reader() == original
        and _pointer() == pointer,
        "selected LoRA request or bootstrap pointer changed during pool selection",
    )
    _bind_sources(origins, *callbacks)
    return result
