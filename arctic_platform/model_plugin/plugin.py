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

"""The plugin object a model package exports, and how the platform finds it."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from importlib.metadata import entry_points
from typing import Any

import torch
from pydantic import BaseModel
from torch import nn

from arctic_platform.model_plugin.experts import ExpertLayer
from arctic_platform.model_plugin.manifest import NO_MIXER
from arctic_platform.model_plugin.manifest import PLATFORM_ATTENTION
from arctic_platform.model_plugin.manifest import PluginManifest
from arctic_platform.model_plugin.mixers import AttentionMixer
from arctic_platform.model_plugin.mixers import Mixer

ENTRY_POINT_GROUP = "arctic_platform.model_plugins"

LayerConverter = Callable[[dict[str, torch.Tensor], int, Any], dict[str, torch.Tensor]]
ReferenceForward = Callable[[nn.Module, torch.Tensor], torch.Tensor]
# validate(request, config, options) -> the reasons this job cannot run; ``request`` is an admission.JobRequest.
Validator = Callable[[Any, Any, Any], Iterable[str]]


@dataclass(frozen=True)
class ParallelismInfo:
    """What the platform needs to know about one model config to plan and validate parallelism.

    ``layer_types`` has one entry per decoder layer, in order. ``sublayers`` counts the sequence-mixing modules
    inside layers, such as a short convolution on attention values, by sub-layer type. ``heads`` maps a layer or
    sub-layer type to the head count of every tensor its core exchanges under head-parallel sequence parallelism
    (attention: query and key/value heads). ``moe_layers`` lists the decoder layers with an expert block, each with
    ``num_experts`` routed experts and ``top_k`` experts per token.
    """

    layer_types: tuple[str, ...]
    sublayers: Mapping[str, int] = field(default_factory=dict)
    heads: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    moe_layers: tuple[int, ...] = ()
    num_experts: int = 0
    top_k: int = 0

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)


@dataclass(frozen=True)
class ModelPlugin:
    """Everything a model package hands the platform. Export one instance and name it in the package's entry point.

    ``build_model(config, options)`` returns the trainer model, an ``nn.Module`` that is a
    :class:`~arctic_platform.model_plugin.interfaces.CausalLM`: it keeps ``config`` as ``model.config``, names decoder
    layer ``i``'s state ``model.layers.<i>.*``, and its forward ``model(input_ids, position_ids, *, cu_seqlens)``
    returns the final hidden states. :mod:`~arctic_platform.model_plugin.interfaces` lists every hook the platform
    calls on the model's modules.
    ``convert_layer_to_hf(state, layer_idx, config)`` / ``convert_layer_to_prime`` translate one decoder layer
    between the trainer's tensors and the Hugging Face checkpoint's; leave them unset when the two already match.
    ``reference_forward(model, input_ids)`` is a slow, independent implementation of one unpacked sequence that
    conformance compares the model against.

    ``options`` is the pydantic model of the job-time options the plugin accepts, such as a kernel choice; options
    never change the checkpoint. ``build_model`` receives the parsed options, or ``None`` when the plugin declares
    none.
    ``validate(request, config, options)`` returns the reasons a job cannot run, for constraints the manifest cannot
    express (for example, an option that rules out sequence parallelism); admission reports them with its own.
    ``passthrough(config)`` returns pass-through patterns that depend on the config, added to the manifest's.
    """

    manifest: PluginManifest
    config_class: Any
    build_model: Callable[[Any, Any], nn.Module]
    parallelism: Callable[[Any], ParallelismInfo]
    mixers: Mapping[str, Mixer] = field(default_factory=dict)
    expert_layer: ExpertLayer | None = None
    convert_layer_to_hf: LayerConverter | None = None
    convert_layer_to_prime: LayerConverter | None = None
    reference_forward: ReferenceForward | None = None
    register: Callable[[], None] | None = None
    options: type[BaseModel] | None = None
    validate: Validator | None = None
    passthrough: Callable[[Any], Iterable[str]] | None = None

    def is_passthrough(self, name: str, config: Any) -> bool:
        """Whether checkpoint tensor ``name`` is passed through: the manifest's patterns, plus any this plugin
        derives from ``config`` (for example MTP layers stored after the last decoder layer)."""
        derived = self.passthrough(config) if self.passthrough is not None else ()
        return self.manifest.checkpoint.is_passthrough(name) or any(re.match(pattern, name) for pattern in derived)

    def parse_options(self, options: Mapping[str, Any] | None = None) -> BaseModel | None:
        """A job's options as the plugin's options model. Unknown options are refused even if the options model would
        ignore them."""
        options = dict(options or {})
        if self.options is None:
            if len(options) > 0:
                raise ValueError(f"{self.manifest.name} takes no options, got {sorted(options)}")
            return None
        unknown = set(options) - set(self.options.model_fields)
        if len(unknown) > 0:
            raise ValueError(f"{self.manifest.name} has no options {sorted(unknown)}")
        return self.options.model_validate(options)

    def layer_mixers(self) -> dict[str, Mixer]:
        """The mixer for every layer and sub-layer type in the manifest: the platform's for attention, the plugin's
        otherwise."""
        resolved: dict[str, Mixer] = {}
        for layer_type, source in (self.manifest.layer_types | self.manifest.sublayer_types).items():
            if source == NO_MIXER:
                continue
            if source == PLATFORM_ATTENTION:
                resolved[layer_type] = AttentionMixer(layer_type)
                continue
            mixer = self.mixers.get(layer_type)
            if mixer is None:
                raise ValueError(f"manifest layer type {layer_type!r} is 'plugin' but the plugin has no mixer for it")
            if mixer.layer_type != layer_type:
                raise ValueError(f"mixer for {layer_type!r} reports layer_type {mixer.layer_type!r}")
            resolved[layer_type] = mixer
        extra = set(self.mixers) - set(resolved)
        if len(extra) > 0:
            raise ValueError(f"plugin supplies mixers for layer types missing from its manifest: {sorted(extra)}")
        return resolved


def load_plugin(target: str) -> ModelPlugin:
    """Import ``'module:attribute'`` and check it is a plugin written against a supported contract."""
    module_name, _, attribute = target.partition(":")
    if len(attribute) == 0:
        raise ValueError(f"plugin target must be 'module:attribute', got {target!r}")
    plugin = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(plugin, ModelPlugin):
        raise TypeError(f"{target} is a {type(plugin).__name__}, not a ModelPlugin")
    plugin.manifest.check_contract()
    plugin.layer_mixers()
    if plugin.register is not None:
        plugin.register()
    return plugin


def discover_plugins() -> dict[str, ModelPlugin]:
    """Every installed plugin, keyed by manifest name."""
    plugins: dict[str, ModelPlugin] = {}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        plugin = load_plugin(entry.value)
        if plugin.manifest.name in plugins:
            raise ValueError(f"two installed plugins are named {plugin.manifest.name}")
        plugins[plugin.manifest.name] = plugin
    return plugins


def plugin_for_model_type(model_type: str, plugins: Mapping[str, ModelPlugin]) -> ModelPlugin:
    matches = [plugin for plugin in plugins.values() if model_type in plugin.manifest.hf_model_types]
    if len(matches) != 1:
        names = sorted(plugin.manifest.name for plugin in matches)
        raise ValueError(f"model_type {model_type!r} must match exactly one plugin, matched {names}")
    return matches[0]
