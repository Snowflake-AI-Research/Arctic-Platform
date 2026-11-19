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

"""How the platform launches a plugin's sampler: engine settings from the manifest and the job."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from arctic_platform.model_plugin.admission import JobRequest
from arctic_platform.model_plugin.admission import inference_settings
from arctic_platform.model_plugin.plugin import ModelPlugin


@dataclass(frozen=True)
class SamplerLaunch:
    """vLLM engine arguments, the environment the platform sets for them, and the plugin's chat parser."""

    engine_args: dict[str, Any]
    env: dict[str, str] = field(default_factory=dict)
    chat_parser: str | None = None


def sampler_launch(plugin: ModelPlugin, request: JobRequest) -> SamplerLaunch:
    """The sampler's engine settings for an admitted ``request``: its :func:`inference_settings`, with the manifest's
    ``[sampler]`` requirements applied last.

    Prefix caching is on only when the manifest declares the sampler correct with it, because vLLM turns it on by
    default, and for a model with stateful layers that silently switches state caching to a mode those layers may
    not implement. A router-replay job also records routed experts, which vLLM 0.33 does only with prefix caching and
    its V2 model runner (admission checks both are allowed).
    """
    sampler = plugin.manifest.sampler
    if sampler is None:
        raise ValueError(f"{plugin.manifest.name} has no [sampler]")
    args = inference_settings(plugin, request)
    requirements = sampler.requirements()
    model_runner = requirements.pop("model_runner", None)
    chat_parser = requirements.pop("chat_parser", None)
    args.update(requirements)
    args["enable_prefix_caching"] = sampler.prefix_caching and bool(args.get("enable_prefix_caching", True))
    if request.router_replay:  # vLLM 0.33 records routed experts only with prefix caching and its V2 model runner
        args["enable_return_routed_experts"] = True
        model_runner = "v2"
    env = {} if model_runner is None else {"VLLM_USE_V2_MODEL_RUNNER": "1" if model_runner == "v2" else "0"}
    return SamplerLaunch(engine_args=args, env=env, chat_parser=chat_parser)
