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
"""GLM-5.3-Flash PrimeRL integration."""


def __getattr__(name: str):
    if name == "Glm5NextForConditionalGenerationPrimeRL":
        from arctic_platform.model.implementations.glm53.modeling_glm5_next import (
            Glm5NextForConditionalGenerationPrimeRL,
        )

        return Glm5NextForConditionalGenerationPrimeRL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
