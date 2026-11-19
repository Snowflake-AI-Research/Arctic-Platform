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
developed and its CPU conformance tiers run without the GPU training stack; the ``rl_step`` tier, which uses the
platform's GRPO loss, is skipped unless ``arctic-platform[rl]`` is installed. DeepSpeed is imported only by
:func:`initialize_deepspeed`.
"""

from arctic_platform.model_plugin.admission import AdmissionError
from arctic_platform.model_plugin.admission import JobRequest
from arctic_platform.model_plugin.admission import check_job_request
from arctic_platform.model_plugin.attention import AttentionModule
from arctic_platform.model_plugin.attention import varlen_causal_attention
from arctic_platform.model_plugin.build import build
from arctic_platform.model_plugin.build import init_buffers
from arctic_platform.model_plugin.build import materialize
from arctic_platform.model_plugin.deepspeed_engine import initialize_deepspeed
from arctic_platform.model_plugin.experts import ROUTER_REPLAY_FRESH
from arctic_platform.model_plugin.experts import ExpertLayer
from arctic_platform.model_plugin.experts import ExpertShard
from arctic_platform.model_plugin.experts import StackedExpertLayer
from arctic_platform.model_plugin.experts import expert_parallel
from arctic_platform.model_plugin.experts import shard_experts
from arctic_platform.model_plugin.host import compute_logits
from arctic_platform.model_plugin.host import token_logprobs
from arctic_platform.model_plugin.manifest import CONTRACT_VERSION
from arctic_platform.model_plugin.manifest import Capabilities
from arctic_platform.model_plugin.manifest import PluginManifest
from arctic_platform.model_plugin.manifest import SamplerSpec
from arctic_platform.model_plugin.manifest import load_manifest
from arctic_platform.model_plugin.marks import Marks
from arctic_platform.model_plugin.marks import cast_model
from arctic_platform.model_plugin.marks import lora_target_modules
from arctic_platform.model_plugin.marks import mark
from arctic_platform.model_plugin.marks import optimizer_groups
from arctic_platform.model_plugin.marks import skip_weight_sync
from arctic_platform.model_plugin.mixers import AttentionMixer
from arctic_platform.model_plugin.mixers import HeadParallelMixer
from arctic_platform.model_plugin.mixers import Mixer
from arctic_platform.model_plugin.parallel import FULL_SHARD
from arctic_platform.model_plugin.parallel import FULL_WINDOW
from arctic_platform.model_plugin.parallel import HeadShard
from arctic_platform.model_plugin.parallel import SequenceWindow
from arctic_platform.model_plugin.parallel import combine_tokens
from arctic_platform.model_plugin.parallel import gather_sequence
from arctic_platform.model_plugin.parallel import gather_tokens
from arctic_platform.model_plugin.parallel import previous_tokens
from arctic_platform.model_plugin.parallel import sequence_start
from arctic_platform.model_plugin.plugin import ENTRY_POINT_GROUP
from arctic_platform.model_plugin.plugin import ModelPlugin
from arctic_platform.model_plugin.plugin import ParallelismInfo
from arctic_platform.model_plugin.plugin import discover_plugins
from arctic_platform.model_plugin.plugin import load_plugin
from arctic_platform.model_plugin.plugin import plugin_for_model_type
from arctic_platform.model_plugin.recompute import apply_activation_checkpointing
from arctic_platform.model_plugin.recompute import in_recompute
from arctic_platform.model_plugin.routing import choose_experts
from arctic_platform.model_plugin.sampler import SamplerLaunch
from arctic_platform.model_plugin.sampler import sampler_launch

__all__ = [
    "AdmissionError",
    "apply_activation_checkpointing",
    "AttentionMixer",
    "AttentionModule",
    "build",
    "Capabilities",
    "cast_model",
    "check_job_request",
    "choose_experts",
    "combine_tokens",
    "compute_logits",
    "CONTRACT_VERSION",
    "discover_plugins",
    "ENTRY_POINT_GROUP",
    "expert_parallel",
    "ExpertLayer",
    "ExpertShard",
    "FULL_SHARD",
    "FULL_WINDOW",
    "gather_sequence",
    "gather_tokens",
    "HeadParallelMixer",
    "HeadShard",
    "in_recompute",
    "init_buffers",
    "initialize_deepspeed",
    "JobRequest",
    "load_manifest",
    "load_plugin",
    "lora_target_modules",
    "mark",
    "Marks",
    "materialize",
    "Mixer",
    "ModelPlugin",
    "optimizer_groups",
    "ParallelismInfo",
    "plugin_for_model_type",
    "PluginManifest",
    "previous_tokens",
    "ROUTER_REPLAY_FRESH",
    "sampler_launch",
    "SamplerLaunch",
    "SamplerSpec",
    "sequence_start",
    "SequenceWindow",
    "shard_experts",
    "skip_weight_sync",
    "StackedExpertLayer",
    "token_logprobs",
    "varlen_causal_attention",
]
