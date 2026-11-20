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

"""Every example plugin trained for several GRPO steps under DeepSpeed with sequence and expert parallelism.

``check_rl_training`` runs the steps on four GPUs (two data-parallel replicas of two sequence-parallel ranks, with
two-way expert parallelism and router replay for mixture-of-experts models) and compares the loss and every
parameter with the unsharded model after each step.
"""

import pytest

from arctic_platform.model_plugin import conformance
from arctic_platform.model_plugin.plugin import load_plugin
from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import require_torch_multi_gpu

from .conftest import EXAMPLES

pytestmark = pytest.mark.usefixtures("without_session_process_group")


@require_deepspeed
@require_torch_multi_gpu
@pytest.mark.integration
@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_example_trains_under_deepspeed_with_sp_and_ep(example):
    target, root = EXAMPLES[example]
    spec = conformance.load_conformance_spec(root / "conformance.toml")
    plugin = load_plugin(target)
    capabilities = plugin.manifest.capabilities
    config = plugin.config_class(**spec.fixture)
    has_experts = len(plugin.parallelism(config).moe_layers) > 0
    sp = spec.sp_sizes[0] if capabilities.sequence_parallel and spec.sp_sizes else 1
    ep = spec.ep_sizes[0] if capabilities.expert_parallel and has_experts and spec.ep_sizes else 1
    conformance.check_rl_training(target, spec, sp, ep, timeout=300)
