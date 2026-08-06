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

"""Arctic RL client -- HTTP client for RL training against dss-platform or local server."""

import os as _os

from arctic_platform.rl.client import create_arctic_rl_client
from arctic_platform.rl.config import ArcticRLClientConfig
from arctic_platform.rl.config import WeightSyncConfig
from arctic_platform.rl.processors import grpo_loss
from arctic_platform.rl.processors import pack_sequences
from arctic_platform.rl.processors import register_loss_fn
from arctic_platform.rl.processors import register_post_processor
from arctic_platform.rl.processors import run_pipeline
from arctic_platform.rl.processors import unpack_sequences
from arctic_platform.rl.weight_sync import WeightSyncCoordinator


def _install_cortex_driver_shims() -> None:
    # SkyRL's prepare_runtime_environment probes cudaCanAccessPeer by
    # requesting a {"CPU":1,"GPU":2} Ray placement group. On a CPU-only
    # Cortex driver that hangs forever. The result is only used to toggle
    # NCCL_P2P_DISABLE on a local vLLM engine we never run, so returning
    # False is safe. Only active when ARCTIC_BACKEND=cortex.
    if _os.environ.get("ARCTIC_BACKEND", "").lower() != "cortex":
        return
    try:
        from skyrl.train.utils import utils as _skyrl_utils
    except Exception:
        return
    if getattr(_skyrl_utils, "_cortex_shimmed", False):
        return
    _skyrl_utils.peer_access_supported = lambda max_num_gpus_per_node=1: False
    _skyrl_utils._cortex_shimmed = True


_install_cortex_driver_shims()

__all__ = [
    "create_arctic_rl_client",
    "ArcticRLClientConfig",
    "WeightSyncConfig",
    "WeightSyncCoordinator",
    "run_pipeline",
    "register_loss_fn",
    "register_post_processor",
    "grpo_loss",
    "pack_sequences",
    "unpack_sequences",
]
