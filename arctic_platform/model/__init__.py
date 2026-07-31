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
"""Model factory: turn a declarative ModelSpec into a configured nn.Module."""

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.config import Optimizations
from arctic_platform.model.config import ParallelismConfig
from arctic_platform.model.factory import build_model
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader
from arctic_platform.model.loader import select_loader
from arctic_platform.model.optimization import apply_optimizations
from arctic_platform.model.optimization import register_optimization

# Import built-in loaders and optimizations for their registration side effects.
from arctic_platform.model import loaders  # noqa: F401  # isort: skip
from arctic_platform.model import optimizations  # noqa: F401  # isort: skip

__all__ = [
    "LoadedModel",
    "LoaderContext",
    "ModelSpec",
    "Optimizations",
    "ParallelismConfig",
    "apply_optimizations",
    "build_model",
    "register_loader",
    "register_optimization",
    "select_loader",
]
