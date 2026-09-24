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
