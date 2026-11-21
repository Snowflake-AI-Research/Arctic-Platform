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

"""Running a prepared plugin model under DeepSpeed.

:func:`~arctic_platform.model_plugin.host.prepare_model` shards a plugin's experts and tags their parameters the
way DeepSpeed's MoE support expects (``allreduce = False`` and a ``group_name``). That is not enough on its own:
DeepSpeed switches its MoE handling on only when it finds its own ``MoE`` module class, so a plugin's expert layer
is invisible to it. ZeRO 0-2 then fail on the first step, and ZeRO-3 trains silently wrong, averaging different
ranks' experts together. :func:`initialize_deepspeed` declares the plugin's expert layers to DeepSpeed, so it
reduces each expert block's gradients over its expert-data-parallel group and refuses ZeRO-3, which DeepSpeed
does not support with MoE.

DeepSpeed-native checkpoints of a plugin model are not supported: DeepSpeed finds expert weights to save by module
class. Plugin checkpoints are Hugging Face directories written by
:func:`~arctic_platform.model_plugin.weights.save_hf_checkpoint`, which gathers expert shards (spec §7.9).

This module imports DeepSpeed only when called, so the plugin API keeps working without it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from torch import nn

from arctic_platform.model_plugin.host import LayerBinding
from arctic_platform.model_plugin.marks import optimizer_groups


@contextmanager
def _declared_moe_layers(model: nn.Module, num_experts: list[int]) -> Iterator[None]:
    """While a DeepSpeed engine is built for ``model``, tell it ``model`` has these expert layers.

    DeepSpeed has no public hook for MoE layers it does not define, so this wraps the engine method that looks for
    them, for this one ``deepspeed.initialize`` call and for this model only.
    """
    from deepspeed.runtime.engine import DeepSpeedEngine

    find_moe_layers = DeepSpeedEngine._configure_distributed_model

    def configure_distributed_model(engine, module):
        find_moe_layers(engine, module)
        if engine.module is model and not engine.has_moe_layers:
            engine.has_moe_layers = True
            engine.num_experts.extend(num_experts)

    DeepSpeedEngine._configure_distributed_model = configure_distributed_model
    try:
        yield
    finally:
        DeepSpeedEngine._configure_distributed_model = find_moe_layers


def initialize_deepspeed(
    model: nn.Module,
    binding: LayerBinding,
    *,
    config: Any,
    optimizer: Any = None,
    model_parameters: Any = None,
    **kwargs: Any,
):
    """``deepspeed.initialize`` for a model :func:`~arctic_platform.model_plugin.host.prepare_model` returned
    ``binding`` for; returns what ``deepspeed.initialize`` returns.

    Under expert parallelism the expert-parallel group must be one DeepSpeed registered
    (``deepspeed.utils.groups._create_expert_and_data_parallel``), so that its expert-data-parallel group exists.
    Without a client ``optimizer``, the parameter groups are the model's :func:`optimizer_groups` (``decay``, at the
    optimizer's own weight decay, and ``no_decay``, from the plugin's marks; frozen parameters left out),
    or ``model_parameters`` given as named param-group dicts, and are split into DeepSpeed's MoE and non-MoE
    optimizer groups here; a client optimizer must already be built over such groups
    (``deepspeed.moe.utils.split_params_into_different_moe_groups_for_optimizer``).
    """
    import deepspeed
    from deepspeed.moe.utils import is_moe_param
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
    from deepspeed.utils import groups

    from arctic_platform.model.implementations.fp8 import preserve_fp8_params_through_ds_cast

    # DeepSpeed casts the model to bf16/fp16; the platform's guard keeps fp32-marked parameters out of that cast.
    preserve_fp8_params_through_ds_cast(model)
    if optimizer is None and model_parameters is None:
        model_parameters = optimizer_groups(model)
    group_names = {param.group_name for param in model.parameters() if is_moe_param(param)}
    if len(group_names) == 0:
        return deepspeed.initialize(
            model=model, optimizer=optimizer, model_parameters=model_parameters, config=config, **kwargs
        )

    unregistered = group_names - set(groups._get_expert_data_parallel_group_dict())
    if len(unregistered) > 0:
        raise ValueError(
            f"expert parameters use expert groups {sorted(unregistered)} that DeepSpeed has not registered: create "
            "them with deepspeed.utils.groups._create_expert_and_data_parallel(ep_size) and pass that group to "
            "prepare_model"
        )
    if optimizer is None:
        model_parameters = split_params_into_different_moe_groups_for_optimizer(model_parameters)

    with _declared_moe_layers(model, [binding.info.num_experts] * len(binding.experts)):
        return deepspeed.initialize(
            model=model, optimizer=optimizer, model_parameters=model_parameters, config=config, **kwargs
        )
