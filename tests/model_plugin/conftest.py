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

import sys
from pathlib import Path

import pytest
import torch.distributed as dist

# The example plugins are separate distributions; the tests import them from the source tree.
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "model_plugins"
TOY_PLUGIN_ROOT = EXAMPLES / "toy_hybrid_moe"
OLMO2_PLUGIN_ROOT = EXAMPLES / "olmo2"
OLMOE_PLUGIN_ROOT = EXAMPLES / "olmoe"
for root in (TOY_PLUGIN_ROOT, OLMO2_PLUGIN_ROOT, OLMOE_PLUGIN_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


@pytest.fixture(scope="module")
def without_session_process_group():
    """Take down the root conftest's single-rank NCCL group while a module spawns NCCL ranks, and restore it after.

    With that group alive in the pytest process, ranks spawned after a CUDA tier hang in NCCL communicator setup.
    """
    if not (dist.is_available() and dist.is_initialized()):
        yield
        return
    from tests.conftest import _setup_dist

    dist.destroy_process_group()
    try:
        yield
    finally:
        _setup_dist()
