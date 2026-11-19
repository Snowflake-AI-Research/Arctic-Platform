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

"""The OLMoE example end to end: every conformance tier passes, router replay and EP included, through the CLI."""

from arctic_platform.model_plugin import conformance

from .conftest import OLMOE_PLUGIN_ROOT


def test_olmoe_passes_conformance():
    assert (
        conformance.main(["--plugin", "olmoe_plugin:PLUGIN", "--spec", str(OLMOE_PLUGIN_ROOT / "conformance.toml")])
        == 0
    )
