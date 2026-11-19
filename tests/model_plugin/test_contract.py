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
from arctic_platform.model_plugin import CausalLM
from arctic_platform.model_plugin import ComputesLogits
from arctic_platform.model_plugin import InitializesBuffers
from arctic_platform.model_plugin import JobRequest
from arctic_platform.model_plugin import PluginManifest
from arctic_platform.model_plugin import SequenceMixingModule
from arctic_platform.model_plugin import StackedMoEModule
from arctic_platform.model_plugin import check_job_request
from arctic_platform.model_plugin import conformance
from arctic_platform.model_plugin import discover_plugins
from arctic_platform.model_plugin import plugin as plugin_module
from arctic_platform.model_plugin import plugin_for_model_type
from arctic_platform.model_plugin.host import bind_layers
from arctic_platform.testing_utils import set_seed

from .conftest import TOY_PLUGIN_ROOT


def _manifest(**overrides) -> dict:
    data = PLUGIN.manifest.model_dump()
    data.update(overrides)
    return data


def test_toy_manifest_is_valid_and_on_contract():
    PLUGIN.manifest.check_contract()
    assert PLUGIN.manifest.layer_types == {
        "full_attention": "platform:attention",
        "linear_attention": "plugin",
        "conv": "plugin",
        "moe": "none",
    }
    assert PLUGIN.manifest.sublayer_types == {"value_conv": "plugin"}


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(name="toy-hybrid-moe"), "String should match pattern"),
        (dict(layer_types={"full_attention": "vendor:attention"}), "must be one of"),
        (dict(sampler=None), "requires a [sampler] table"),
        (dict(capabilities=dict(expert_parallel=False, max_expert_parallel=2)), "requires expert_parallel"),
        (dict(capabilities=dict(training=False, sampling=False)), "training, sampling or both"),
        (dict(unknown_key=1), "Extra inputs are not permitted"),
        (dict(sublayer_types={"conv": "plugin"}), "in both layer_types and sublayer_types"),
        (dict(sublayer_types={"k_conv": "none"}), "must be one of"),
        (
            dict(checkpoint=dict(layer_pattern=r"^model\.layers\."), sampler=PLUGIN.manifest.sampler.model_dump()),
            "exactly one group",
        ),
        (dict(profiles={"p": {"inference": {"tokenizer_mode": "slow"}}}), "sets sampler requirements"),
        (dict(sampler=dict(vllm_architecture="A", env={"X": "1"})), "Extra inputs are not permitted"),
    ],
)
def test_manifest_rejects(overrides, message):
    with pytest.raises(ValidationError, match=message.replace("[", r"\[")):
        PluginManifest.model_validate(_manifest(**overrides))


def test_conformance_spec_needs_a_row_across_every_shard_boundary():
    from arctic_platform.model_plugin.conformance import ConformanceSpec

    with pytest.raises(ValidationError, match="must cross a shard boundary"):
        ConformanceSpec(fixture={}, sequence_lengths=[5, 3, 8], sp_sizes=[2])
    ConformanceSpec(fixture={}, sequence_lengths=[5, 6, 5], sp_sizes=[2])


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
        (["conv", "conv", "full_attention", "linear_attention", "conv"], ["moe"] * 5),
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


def test_toy_modules_implement_the_interfaces_they_declare():
    config = ToyHybridMoEConfig()
    model = ToyHybridMoEForCausalLM(config)
    assert isinstance(model, CausalLM) and isinstance(model, ComputesLogits)
    roles = {
        SequenceMixingModule: sum(isinstance(module, SequenceMixingModule) for module in model.modules()),
        StackedMoEModule: sum(isinstance(module, StackedMoEModule) for module in model.modules()),
        InitializesBuffers: sum(isinstance(module, InitializesBuffers) for module in model.modules()),
    }
    info = PLUGIN.parallelism(config)
    mixed = [layer_type for layer_type in info.layer_types if layer_type in PLUGIN.layer_mixers()]
    assert roles[SequenceMixingModule] == len(mixed) + sum(info.sublayers.values())
    assert roles[StackedMoEModule] == len(info.moe_layers)
    assert roles[InitializesBuffers] > 0


def test_binding_rejects_a_layer_type_on_a_module_without_mix():
    config = ToyHybridMoEConfig()
    model = ToyHybridMoEForCausalLM(config)
    model.model.layers[0].input_norm.ap_layer_type = config.layer_types[0]
    with pytest.raises(ValueError, match=r"input_norm declares layer type .* but has no mix\(\)"):
        bind_layers(model, PLUGIN, PLUGIN.parallelism(config))


def test_admission_accepts_what_the_manifest_and_config_support():
    config = ToyHybridMoEConfig()
    check_job_request(PLUGIN, config, JobRequest(sp_size=2, ep_size=4, router_replay=True))
    check_job_request(PLUGIN, config, JobRequest(job_type="sampling", inference=dict(tensor_parallel_size=2)))
    check_job_request(PLUGIN, config, JobRequest(sp_size=2, options=dict(short_conv_kernel="loop")))
    # A job may restate a requirement, and tunes everything else.
    tunables = dict(enforce_eager=True, max_num_seqs=8, gpu_memory_utilization=0.5, max_model_len=4096)
    check_job_request(PLUGIN, config, JobRequest(job_type="sampling", inference=tunables))


@pytest.mark.parametrize(
    "request_, reason",
    [
        (JobRequest(sp_size=3), "has 4 heads, not divisible by sp_size 3"),
        (JobRequest(sp_size=4), "has 2 heads, not divisible by sp_size 4"),
        (JobRequest(ep_size=3), "4 experts are not divisible by ep_size 3"),
        (JobRequest(ep_size=8), "exceeds max_expert_parallel 4"),
        (JobRequest(peft=True), "does not declare peft"),
        (
            JobRequest(job_type="sampling", router_replay=True, inference=dict(enable_prefix_caching=0)),
            "needs prefix caching on",
        ),
        (JobRequest(job_type="sampling", inference=dict(enforce_eager=False)), "the plugin requires True"),
        (JobRequest(job_type="sampling", inference=dict(model_runner="v2")), "set only by the plugin's manifest"),
        (JobRequest(profile="h100"), "has no profile 'h100'"),
        (JobRequest(options=dict(kernel="fast")), "has no options \\['kernel'\\]"),
        (JobRequest(options=dict(short_conv_kernel="tiled")), "Input should be 'loop' or 'fused'"),
        (JobRequest(sp_size=2, options=dict(short_conv_kernel="fused")), "cannot run with sequence parallelism"),
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


def test_sublayers_are_bound_and_counted_like_layers():
    config = ToyHybridMoEConfig(attention_value_conv=True)
    model = ToyHybridMoEForCausalLM(config)
    binding = bind_layers(model, PLUGIN, PLUGIN.parallelism(config))
    assert sorted(layer_type for layer_type, _ in binding.mixers.values()).count("value_conv") == 2
    model.model.layers[1].self_attn.v_conv = None
    with pytest.raises(ValueError, match="but its config declares"):
        bind_layers(model, PLUGIN, PLUGIN.parallelism(config))


def test_admission_reads_config_dependent_constraints_from_the_plugin():
    with pytest.raises(AdmissionError, match="does not implement attention_value_conv"):
        check_job_request(PLUGIN, ToyHybridMoEConfig(attention_value_conv=True), JobRequest(job_type="sampling"))


def test_sampler_launch_applies_requirements_and_the_prefix_caching_rule():
    from arctic_platform.model_plugin import sampler_launch

    manifest = PLUGIN.manifest.model_copy(
        update=dict(
            sampler=PLUGIN.manifest.sampler.model_copy(
                update=dict(model_runner="v2", reasoning_parser="toy", prefix_caching=False)
            ),
            profiles={"small": {"inference": {"max_num_seqs": 4, "tensor_parallel_size": 2}}},
        )
    )
    plugin = dataclasses.replace(PLUGIN, manifest=manifest)
    request = JobRequest(job_type="sampling", inference=dict(max_num_seqs=16), profile="small")
    check_job_request(plugin, ToyHybridMoEConfig(), request)
    launch = sampler_launch(plugin, request)
    assert launch.engine_args == dict(
        max_num_seqs=16,
        tensor_parallel_size=2,
        enforce_eager=True,
        reasoning_parser="toy",
        enable_prefix_caching=False,
    )
    assert launch.env == {"VLLM_USE_V2_MODEL_RUNNER": "1"}
    # vLLM turns prefix caching on by default; a sampler that does not declare it is launched with it off.
    replay = sampler_launch(PLUGIN, JobRequest(job_type="sampling", router_replay=True))
    assert replay.engine_args["enable_prefix_caching"] is True and replay.engine_args["enable_return_routed_experts"]
    assert replay.env == {"VLLM_USE_V2_MODEL_RUNNER": "1"}


def test_router_replay_capture_needs_a_sampler_that_declares_prefix_caching():
    sampler = PLUGIN.manifest.sampler.model_copy(update=dict(prefix_caching=False))
    plugin = dataclasses.replace(PLUGIN, manifest=PLUGIN.manifest.model_copy(update=dict(sampler=sampler)))
    config = ToyHybridMoEConfig()
    with pytest.raises(AdmissionError, match="router replay capture needs prefix caching"):
        check_job_request(plugin, config, JobRequest(job_type="sampling", router_replay=True))
    with pytest.raises(AdmissionError, match="does not declare prefix_caching"):
        check_job_request(plugin, config, JobRequest(job_type="sampling", inference=dict(enable_prefix_caching=True)))


def test_marks_resolve_outer_to_inner_and_feed_the_platform_attributes():
    import torch
    from torch import nn

    from arctic_platform.model.implementations.fp8 import KEEP_FP32_ATTR
    from arctic_platform.model_plugin import cast_model
    from arctic_platform.model_plugin import lora_target_modules
    from arctic_platform.model_plugin import mark
    from arctic_platform.model_plugin import marks
    from arctic_platform.model_plugin import optimizer_groups
    from arctic_platform.model_plugin import skip_weight_sync

    assert marks.KEEP_FP32_ATTR == KEEP_FP32_ATTR
    model = nn.Sequential(nn.Linear(4, 4), mark(nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)), dtype="fp32"))
    mark(model[1][1], weight_decay=True)
    mark(model[0], lora_target=True)
    skip_weight_sync(model[1][1].weight)
    mark(model[0].bias, frozen=True)
    cast_model(model, torch.bfloat16)
    marks.apply_marks(model)  # what build, materialize and prepare_model do
    assert model[0].weight.dtype == torch.bfloat16 and model[1][0].weight.dtype == torch.float32
    assert getattr(model[1][0].weight, KEEP_FP32_ATTR) and not model[0].bias.requires_grad
    resolved = marks.tensor_marks(model)
    assert not resolved["1.1.weight"].weight_sync and resolved["1.1.weight"].dtype == "fp32"
    groups = {group["name"]: group for group in optimizer_groups(model)}
    # fp32-marked parameters get groups of their own: DeepSpeed flattens a group into one dtype.
    assert {name: len(group["params"]) for name, group in groups.items()} == {
        "decay": 1,
        "decay_fp32": 3,
        "no_decay_fp32": 1,
    }
    assert groups["no_decay_fp32"]["weight_decay"] == 0.0
    assert "weight_decay" not in groups["decay"]  # the optimizer's own setting applies
    assert lora_target_modules(model) == ["0"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        mark(model, dtype_="fp32")
    with pytest.raises(ValueError, match="marks a module, not a tensor"):
        mark(model[0].weight, lora_target=True)


def test_marks_follow_tied_weights_and_survive_the_meta_build():
    import torch
    from torch import nn

    from arctic_platform.model_plugin import cast_model
    from arctic_platform.model_plugin import mark
    from arctic_platform.model_plugin import materialize
    from arctic_platform.model_plugin.marks import tensor_marks

    tied = nn.ModuleDict(dict(embed=nn.Embedding(8, 4), head=nn.Linear(4, 8, bias=False)))
    tied.head.weight = tied.embed.weight
    cast_model(tied, torch.bfloat16)  # one tensor under two names: no KeyError
    mark(tied.head, dtype="fp32")
    with pytest.raises(ValueError, match="one tensor with different marks"):
        tensor_marks(tied)

    with torch.device("meta"):
        model = nn.ModuleDict(dict(embed=nn.Embedding(8, 4), head=nn.Linear(4, 8, bias=False), out=nn.Linear(4, 4)))
    model.head.weight = model.embed.weight
    mark(model.out.weight, dtype="fp32")
    materialize(model, "cpu", torch.bfloat16)
    assert model.head.weight is model.embed.weight  # still tied after to_empty
    assert model.out.weight.dtype == torch.float32 and model.embed.weight.dtype == torch.bfloat16


def test_admission_checks_a_profiles_defaults_too():
    manifest = PLUGIN.manifest.model_copy(
        update=dict(
            sampler=PLUGIN.manifest.sampler.model_copy(update=dict(tensor_parallel=False)),
            profiles={"wide": {"inference": {"tensor_parallel_size": 4}}},
        )
    )
    with pytest.raises(AdmissionError, match="does not declare tensor_parallel"):
        check_job_request(
            dataclasses.replace(PLUGIN, manifest=manifest),
            ToyHybridMoEConfig(),
            JobRequest(job_type="sampling", profile="wide"),
        )


def test_static_tier_refuses_platform_imports_outside_the_public_api(tmp_path, monkeypatch):
    package = tmp_path / "private_import_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("from .modeling import build\n")
    (package / "modeling.py").write_text(
        "from arctic_platform.model_plugin import ModelPlugin\n"
        "from arctic_platform.model_plugin.attention import segment_ids\n"
        "from toy_hybrid_moe.modeling import ToyHybridMoEForCausalLM\n"
        "def build(config, options):\n"
        "    return ToyHybridMoEForCausalLM(config, options)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    from private_import_plugin import build

    plugin = dataclasses.replace(PLUGIN, build_model=build)
    with pytest.raises(AssertionError, match=r"modeling\.py:2: arctic_platform\.model_plugin\.attention"):
        conformance.check_static(plugin, conformance.load_conformance_spec(TOY_PLUGIN_ROOT / "conformance.toml"))
