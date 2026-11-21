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

"""The example plugins end to end, through the conformance CLI a plugin author runs.

Each example runs its single-process tiers. The parallel tiers spawn ranks, which is slow on CI's CPUs, and the toy
plugin's tests cover the parallel machinery; they run here only for gpt-oss, whose own attention core and stacked
experts are the examples' only plugin code that runs sharded.
"""

import pytest

from arctic_platform.model_plugin import conformance

from .conftest import GPT_OSS_PLUGIN_ROOT
from .conftest import LFM2_PLUGIN_ROOT
from .conftest import OLMO2_PLUGIN_ROOT
from .conftest import OLMOE_PLUGIN_ROOT

EXAMPLES = {
    "olmo2": ("olmo2_plugin:PLUGIN", OLMO2_PLUGIN_ROOT, False),
    "olmoe": ("olmoe_plugin:PLUGIN", OLMOE_PLUGIN_ROOT, False),
    "lfm2": ("lfm2_plugin:PLUGIN", LFM2_PLUGIN_ROOT, False),
    "gpt_oss": ("gpt_oss_plugin:PLUGIN", GPT_OSS_PLUGIN_ROOT, True),
}


@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_example_plugin_passes_conformance(example):
    target, root, parallel = EXAMPLES[example]
    spec = str(root / "conformance.toml")
    tiers = conformance.list_tiers(target, conformance.load_conformance_spec(spec))
    if not parallel:
        tiers = [tier for tier in tiers if not tier.startswith(("sp", "ep"))]
    assert conformance.main(["--plugin", target, "--spec", spec, *(f"--tier={tier}" for tier in tiers)]) == 0
