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
"""SkyRL integration helpers for the Cortex backend."""

from __future__ import annotations


def install_cortex_driver_shims() -> None:
    """Patch ``skyrl.train.utils.utils.peer_access_supported`` to ``False``.

    SkyRL's ``prepare_runtime_environment`` probes ``cudaCanAccessPeer`` by
    requesting a ``{"CPU":1,"GPU":2}`` Ray placement group, which hangs a
    CPU-only Cortex driver. No-op unless SkyRL is importable.
    """
    try:
        from skyrl.train.utils import utils as _skyrl_utils
    except Exception:
        return
    if getattr(_skyrl_utils, "_cortex_shimmed", False):
        return
    _skyrl_utils.peer_access_supported = lambda max_num_gpus_per_node=1: False
    _skyrl_utils._cortex_shimmed = True


__all__ = ["install_cortex_driver_shims"]
