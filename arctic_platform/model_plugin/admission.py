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

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

from arctic_platform.model_plugin.plugin import ModelPlugin


class AdmissionError(ValueError):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass(frozen=True)
class JobRequest:
    """The parallelism and features one sub-job asks for.

    ``options`` are the plugin's job-time options. ``inference`` is the job's sampler engine config: the tunables
    (``tensor_parallel_size``, ``max_num_seqs``, ``gpu_memory_utilization``, kernel flags, ``max_model_len``) and
    any feature it turns on (``enable_prefix_caching``, ``speculative_config``). ``profile`` names a manifest
    profile whose ``inference`` table supplies defaults.
    """

    job_type: Literal["training", "sampling"] = "training"
    sp_size: int = 1
    ep_size: int = 1
    peft: bool = False
    router_replay: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)
    inference: Mapping[str, Any] = field(default_factory=dict)
    profile: str | None = None


# Requirements that are not vLLM engine arguments, so only the manifest sets them.
MANIFEST_ONLY = ("model_runner", "chat_parser")


def inference_settings(plugin: ModelPlugin, request: JobRequest) -> dict[str, Any]:
    """The sampler settings a job asks for: its profile's ``inference`` defaults, overridden by its own."""
    profile = plugin.manifest.profiles[request.profile] if request.profile is not None else {}
    return {**profile.get("inference", {}), **request.inference}


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
            sublayer_types = {layer_type for layer_type, count in info.sublayers.items() if count > 0}
            declared = plugin.manifest.layer_types | plugin.manifest.sublayer_types
            used = set(info.layer_types) | sublayer_types
            for layer_type in sorted(used - set(declared)):
                reasons.append(f"the config uses layer type {layer_type!r}, which the manifest does not declare")
            for layer_type in sorted(used & set(mixers)):  # "none" layer types have no mixer to validate
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

    if request.profile is not None and request.profile not in plugin.manifest.profiles:
        reasons.append(f"the plugin has no profile {request.profile!r}")
    elif request.job_type == "sampling" and sampler is not None:
        inference = inference_settings(plugin, request)
        if int(inference.get("tensor_parallel_size", 1)) > 1 and not sampler.tensor_parallel:
            reasons.append("the sampler does not declare tensor_parallel")
        if inference.get("speculative_config") is not None and not sampler.speculative_decoding:
            reasons.append("the sampler does not declare speculative_decoding")
        if inference.get("enable_prefix_caching") and not sampler.prefix_caching:
            reasons.append("the sampler does not declare prefix_caching")
        if request.router_replay:
            # vLLM 0.33 records routed experts only with prefix caching on and its V2 model runner.
            if not sampler.prefix_caching or not inference.get("enable_prefix_caching", True):
                reasons.append(
                    "router replay capture needs prefix caching on, and a sampler that declares prefix_caching"
                )
            if sampler.model_runner == "v1":
                reasons.append("router replay capture needs vLLM's V2 model runner; the sampler requires v1")
        for name in MANIFEST_ONLY:
            if name in inference:
                reasons.append(f"{name} is set only by the plugin's manifest")
        for name, required in sampler.requirements().items():
            if name not in MANIFEST_ONLY and name in inference and inference[name] != required:
                reasons.append(f"the job sets {name}={inference[name]!r}, but the plugin requires {required!r}")

    try:
        options = plugin.parse_options(request.options)
    except ValueError as error:
        reasons.append(str(error))
    else:
        if plugin.validate is not None:
            reasons.extend(plugin.validate(request, config, options))

    if len(reasons) > 0:
        raise AdmissionError(reasons)
