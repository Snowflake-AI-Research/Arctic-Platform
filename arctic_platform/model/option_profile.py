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
"""Check a training job's ``model_options`` against the loader it will build its model with."""

from __future__ import annotations

from typing import Any
from typing import Mapping

from arctic_platform.common import model_option_aliases as aliases


def training_loader_name(*, model_path: str, ep_size: int, sp_size: int, model_provider: str | None = None) -> str:
    """The loader a training job builds ``model_path`` with.

    The MoE route (``moe_route``) lets the registry pick the loader from the model type and the parallelism,
    falling back to its default loader when no MoE loader matches; a DFlash2 job uses the ``dflash2`` loader;
    every other job builds through ``settings_dense_model_spec``, which names ``DENSE_LOADER``. Reads only the
    model's config.
    """
    if model_provider == aliases.DFLASH2_PROVIDER:
        return aliases.DFLASH2_PROVIDER
    if not aliases.moe_route(ep_size):
        return aliases.DENSE_LOADER
    import arctic_platform.model.loaders  # noqa: F401  # registers the loaders
    from arctic_platform.model.config import ModelSpec
    from arctic_platform.model.config import ParallelismConfig
    from arctic_platform.model.loader import resolve_loader_name

    # Unvalidated on purpose: loader matching reads only the model path and the parallelism, and validating
    # would also run the loader's option checks, which run when the model is built.
    spec = ModelSpec.model_construct(
        model_path_or_name=model_path,
        parallelism=ParallelismConfig(expert_parallel=ep_size, sequence_parallel=sp_size),
    )
    return resolve_loader_name(spec)


def check_model_options_profile(training_config: Mapping[str, Any], *, model_path: str, sp_size: int) -> None:
    """Fail when the loader the job will use cannot take one of its ``model_options``.

    ``training_config`` is the validated settings, dumped. Run it before any weights load.
    """
    requested = aliases.requested(training_config, location=aliases.TRAINING)
    if len(requested) == 0:
        return
    ep_size = training_config.get("ep_size") or 1
    model_provider = training_config.get("model_provider")
    loader = training_loader_name(
        model_path=model_path, ep_size=ep_size, sp_size=sp_size, model_provider=model_provider
    )
    # For the message only: the loader this model would use with expert parallelism, if any.
    expert_parallel_loader = None
    if not aliases.moe_route(ep_size):
        candidate = training_loader_name(model_path=model_path, ep_size=2, sp_size=sp_size)
        expert_parallel_loader = None if candidate == aliases.DENSE_LOADER else candidate
    aliases.check_profile(
        requested,
        loader,
        location=aliases.TRAINING,
        ep_size=ep_size,
        model_provider=model_provider,
        expert_parallel_loader=expert_parallel_loader,
    )
