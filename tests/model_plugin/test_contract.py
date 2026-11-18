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

"""The model plugin contract's own rules: manifest validation, layer binding, admission and discovery (CPU)."""

import dataclasses
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from toy_hybrid_moe import PLUGIN
from toy_hybrid_moe.config import ToyHybridMoEConfig
from toy_hybrid_moe.modeling import ToyHybridMoEForCausalLM

from arctic_platform.model_plugin import AdmissionError
from arctic_platform.model_plugin import JobRequest
from arctic_platform.model_plugin import PluginManifest
from arctic_platform.model_plugin import check_job_request
from arctic_platform.model_plugin import discover_plugins
from arctic_platform.model_plugin import plugin as plugin_module
from arctic_platform.model_plugin import plugin_for_model_type
from arctic_platform.model_plugin.host import bind_layers
from arctic_platform.testing_utils import set_seed


def _manifest(**overrides) -> dict:
    data = PLUGIN.manifest.model_dump()
    data.update(overrides)
    return data


def test_toy_manifest_is_valid_and_on_contract():
    PLUGIN.manifest.check_contract()
    assert PLUGIN.manifest.layer_types == {"full_attention": "platform:attention", "linear_attention": "plugin"}


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(name="toy-hybrid-moe"), "String should match pattern"),
        (dict(layer_types={"full_attention": "vendor:attention"}), "must be 'platform:attention' or 'plugin'"),
        (dict(sampler=None), "requires a [sampler] table"),
        (dict(capabilities=dict(expert_parallel=False, max_expert_parallel=2)), "requires expert_parallel"),
        (dict(capabilities=dict(training=False, sampling=False)), "training, sampling or both"),
        (dict(unknown_key=1), "Extra inputs are not permitted"),
    ],
)
def test_manifest_rejects(overrides, message):
    with pytest.raises(ValidationError, match=message.replace("[", r"\[")):
        PluginManifest.model_validate(_manifest(**overrides))


def test_contract_major_version_mismatch_is_refused():
    manifest = PluginManifest.model_validate(_manifest(contract="2.0"))
    with pytest.raises(ValueError, match="targets contract 2.0"):
        manifest.check_contract()


def test_plugin_layer_type_needs_a_mixer_and_no_extras():
    with pytest.raises(ValueError, match="has no mixer for it"):
        dataclasses.replace(PLUGIN, mixers={}).layer_mixers()
    extra = dict(PLUGIN.mixers, sliding_attention=PLUGIN.mixers["linear_attention"])
    with pytest.raises(ValueError, match="missing from its manifest"):
        dataclasses.replace(PLUGIN, mixers=extra).layer_mixers()


def test_binding_follows_the_config_whatever_the_layer_count():
    set_seed(0)
    for layer_types, mlp_types in [
        (["full_attention"], ["moe"]),
        (["linear_attention"] * 3 + ["full_attention"] * 3, ["dense", "moe"] * 3),
    ]:
        config = ToyHybridMoEConfig(layer_types=layer_types, mlp_types=mlp_types)
        model = ToyHybridMoEForCausalLM(config)
        binding = bind_layers(model, PLUGIN, PLUGIN.parallelism(config))
        assert sorted(layer_type for layer_type, _ in binding.mixers.values()) == sorted(layer_types)
        assert len(binding.experts) == mlp_types.count("moe")


def test_binding_rejects_modules_the_config_does_not_account_for():
    config = ToyHybridMoEConfig()
    model = ToyHybridMoEForCausalLM(config)
    del model.model.layers[0]
    with pytest.raises(ValueError, match="but its config declares"):
        bind_layers(model, PLUGIN, PLUGIN.parallelism(config))

    model = ToyHybridMoEForCausalLM(config)
    model.model.layers[1].self_attn.ap_layer_type = "sliding_attention"
    with pytest.raises(ValueError, match="which the manifest does not declare"):
        bind_layers(model, PLUGIN, PLUGIN.parallelism(config))


def test_admission_accepts_what_the_manifest_and_config_support():
    config = ToyHybridMoEConfig()
    check_job_request(PLUGIN, config, JobRequest(sp_size=2, ep_size=4, router_replay=True))
    check_job_request(PLUGIN, config, JobRequest(job_type="sampling", sampler_tensor_parallel=2))


@pytest.mark.parametrize(
    "request_, reason",
    [
        (JobRequest(sp_size=3), "has 4 heads, not divisible by sp_size 3"),
        (JobRequest(sp_size=4), "has 2 heads, not divisible by sp_size 4"),
        (JobRequest(ep_size=3), "4 experts are not divisible by ep_size 3"),
        (JobRequest(ep_size=8), "exceeds max_expert_parallel 4"),
        (JobRequest(peft=True), "does not declare peft"),
        (JobRequest(job_type="sampling", prefix_caching=True), "does not declare prefix_caching"),
    ],
)
def test_admission_rejects(request_, reason):
    with pytest.raises(AdmissionError, match=reason):
        check_job_request(PLUGIN, ToyHybridMoEConfig(), request_)


def test_admission_checks_the_config_not_just_the_manifest():
    dense = ToyHybridMoEConfig(mlp_types=["dense"] * 4)
    with pytest.raises(AdmissionError) as error:
        check_job_request(PLUGIN, dense, JobRequest(ep_size=2, router_replay=True))
    assert error.value.reasons == [
        "ep_size > 1 needs a model config with expert layers",
        "router replay needs capabilities.router_replay and a model config with expert layers",
    ]


def test_discovery_by_entry_point(monkeypatch):
    entry = SimpleNamespace(value="toy_hybrid_moe:PLUGIN")
    monkeypatch.setattr(
        plugin_module, "entry_points", lambda group: [entry] if group == "arctic_platform.model_plugins" else []
    )
    plugins = discover_plugins()
    assert list(plugins) == ["custom/example/toy-hybrid-moe"]
    assert plugin_for_model_type("toy_hybrid_moe", plugins) is PLUGIN
    with pytest.raises(ValueError, match="must match exactly one plugin"):
        plugin_for_model_type("qwen3", plugins)

    monkeypatch.setattr(plugin_module, "entry_points", lambda group: [entry, entry])
    with pytest.raises(ValueError, match="two installed plugins"):
        discover_plugins()
