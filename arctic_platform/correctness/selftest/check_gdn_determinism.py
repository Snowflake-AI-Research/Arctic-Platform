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

"""Full determinism fixes the FLA kernel configuration instead of autotuning by call shape."""

from types import SimpleNamespace

from arctic_platform.model.implementations.debug.determinism import _pin_first_fla_configuration


def test_fla_autotuner_keeps_its_first_owned_configuration_and_clears_shape_cache() -> None:
    first = object()
    autotuner = SimpleNamespace(configs=[first, object()], cache={"shape": object()})
    wrapped = SimpleNamespace(fn=SimpleNamespace(fn=autotuner))

    _pin_first_fla_configuration(wrapped, name="example")

    assert autotuner.configs == [first]
    assert autotuner.cache == {}
