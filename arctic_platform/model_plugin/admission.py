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

"""Job admission: reject a request the plugin's manifest and model config cannot serve, before any GPU is used."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

from arctic_platform.model_plugin.plugin import ModelPlugin


class AdmissionError(ValueError):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass(frozen=True)
class JobRequest:
    """The parallelism and features one sub-job asks for."""

    job_type: Literal["training", "sampling"] = "training"
    sp_size: int = 1
    ep_size: int = 1
    peft: bool = False
    router_replay: bool = False
    sampler_tensor_parallel: int = 1
    speculative_decoding: bool = False
    prefix_caching: bool = False


def check_job_request(plugin: ModelPlugin, config: Any, request: JobRequest) -> None:
    """Raise :class:`AdmissionError` listing every reason ``request`` cannot run on ``plugin`` with ``config``."""
    capabilities = plugin.manifest.capabilities
    sampler = plugin.manifest.sampler
    info = plugin.parallelism(config)
    reasons: list[str] = []

    if request.job_type == "training" and not capabilities.training:
        reasons.append("the plugin does not support training")
    if request.job_type == "sampling" and (not capabilities.sampling or sampler is None):
        reasons.append("the plugin does not support sampling")

    if request.sp_size > 1:
        if not capabilities.sequence_parallel:
            reasons.append("the plugin does not declare sequence_parallel")
        else:
            mixers = plugin.layer_mixers()
            for layer_type in sorted(set(info.layer_types)):
                try:
                    mixers[layer_type].validate_sp(info, request.sp_size)
                except ValueError as error:
                    reasons.append(str(error))

    if request.ep_size > 1:
        if not capabilities.expert_parallel:
            reasons.append("the plugin does not declare expert_parallel")
        elif len(info.moe_layers) == 0:
            reasons.append("ep_size > 1 needs a model config with expert layers")
        elif info.num_experts % request.ep_size != 0:
            reasons.append(f"{info.num_experts} experts are not divisible by ep_size {request.ep_size}")
        if capabilities.max_expert_parallel is not None and request.ep_size > capabilities.max_expert_parallel:
            reasons.append(f"ep_size {request.ep_size} exceeds max_expert_parallel {capabilities.max_expert_parallel}")

    if request.peft and not capabilities.peft:
        reasons.append("the plugin does not declare peft")
    if request.router_replay and (not capabilities.router_replay or len(info.moe_layers) == 0):
        reasons.append("router replay needs capabilities.router_replay and a model config with expert layers")

    if request.job_type == "sampling" and sampler is not None:
        if request.sampler_tensor_parallel > 1 and not sampler.tensor_parallel:
            reasons.append("the sampler does not declare tensor_parallel")
        if request.speculative_decoding and not sampler.speculative_decoding:
            reasons.append("the sampler does not declare speculative_decoding")
        if request.prefix_caching and not sampler.prefix_caching:
            reasons.append("the sampler does not declare prefix_caching")

    if len(reasons) > 0:
        raise AdmissionError(reasons)
