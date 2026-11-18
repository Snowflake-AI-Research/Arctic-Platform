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

"""Model plugin contract (draft v1): bring a new model architecture to the training service as a Python package.

The specification is ``docs/model-plugins/spec.md``. This package needs only torch and pydantic, so a plugin can be
developed and its CPU conformance tiers run without the GPU training stack.
"""

from arctic_platform.model_plugin.admission import AdmissionError
from arctic_platform.model_plugin.admission import JobRequest
from arctic_platform.model_plugin.admission import check_job_request
from arctic_platform.model_plugin.attention import AttentionModule
from arctic_platform.model_plugin.attention import varlen_causal_attention
from arctic_platform.model_plugin.experts import ROUTER_REPLAY_FRESH
from arctic_platform.model_plugin.experts import ExpertLayer
from arctic_platform.model_plugin.experts import shard_experts
from arctic_platform.model_plugin.manifest import CONTRACT_VERSION
from arctic_platform.model_plugin.manifest import Capabilities
from arctic_platform.model_plugin.manifest import PluginManifest
from arctic_platform.model_plugin.manifest import SamplerSpec
from arctic_platform.model_plugin.manifest import load_manifest
from arctic_platform.model_plugin.mixers import AttentionMixer
from arctic_platform.model_plugin.mixers import HeadParallelMixer
from arctic_platform.model_plugin.mixers import Mixer
from arctic_platform.model_plugin.parallel import FULL_SHARD
from arctic_platform.model_plugin.parallel import HeadShard
from arctic_platform.model_plugin.parallel import combine_tokens
from arctic_platform.model_plugin.parallel import gather_tokens
from arctic_platform.model_plugin.plugin import ENTRY_POINT_GROUP
from arctic_platform.model_plugin.plugin import ModelPlugin
from arctic_platform.model_plugin.plugin import ParallelismInfo
from arctic_platform.model_plugin.plugin import discover_plugins
from arctic_platform.model_plugin.plugin import load_plugin
from arctic_platform.model_plugin.plugin import plugin_for_model_type
from arctic_platform.model_plugin.weights import skip_weight_sync

__all__ = [
    "AdmissionError",
    "AttentionMixer",
    "AttentionModule",
    "CONTRACT_VERSION",
    "Capabilities",
    "ENTRY_POINT_GROUP",
    "ExpertLayer",
    "FULL_SHARD",
    "HeadParallelMixer",
    "HeadShard",
    "JobRequest",
    "Mixer",
    "ModelPlugin",
    "ParallelismInfo",
    "PluginManifest",
    "ROUTER_REPLAY_FRESH",
    "SamplerSpec",
    "check_job_request",
    "combine_tokens",
    "discover_plugins",
    "gather_tokens",
    "load_manifest",
    "load_plugin",
    "plugin_for_model_type",
    "shard_experts",
    "skip_weight_sync",
    "varlen_causal_attention",
]
