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

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""New synthetic collective/packing controls, never native GLM admission."""

from __future__ import annotations

import sys
from copy import deepcopy
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

from arctic_platform.debug import ep_weight_sync_evidence as ep
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import tensor_manifest
from arctic_platform.model.implementations.glm53.vllm_weights import convert_glm5_next_layer_to_vllm
from arctic_platform.model.implementations.moe import vllm_weights as native
from arctic_platform.model.weight_export import WeightExportContract
from arctic_platform.model.weight_export import register_weight_export
from arctic_platform.model.weight_export import weight_export_contract


def _identity(layer_sd, layer_idx, *, layer_prefix=None):
    return layer_sd


def _tensor(shape, value=1, dtype=torch.bfloat16):
    return torch.full(shape, value, dtype=dtype)


def _routed_layer():
    prefix = "model.layers.0"
    result = {
        f"{prefix}.mlp.router.gate.weight": _tensor((2, 4)),
        f"{prefix}.mlp.experts.w1": _tensor((2, 3, 4), 2),
        f"{prefix}.mlp.experts.w3": _tensor((2, 3, 4), 3),
        f"{prefix}.mlp.experts.w2": _tensor((2, 4, 3), 4),
        f"{prefix}.mlp.expert_bias": _tensor((2,), 5, torch.float32),
        f"{prefix}.mlp.shared_expert.w1": _tensor((3, 4), 6),
        f"{prefix}.mlp.shared_expert.w3": _tensor((3, 4), 7),
        f"{prefix}.mlp.shared_expert.w2": _tensor((4, 3), 8),
    }
    return prefix, result


def test_ep_evidence_native_glm_routed_shared_bias_packing_preserves_every_byte():
    prefix, inputs = _routed_layer()
    before = dict(inputs)
    proof = ep.pack_layer_with_evidence(convert_glm5_next_layer_to_vllm, inputs, 0, layer_prefix=prefix)
    assert proof["inputs"] == {name: tensor_manifest(value) for name, value in before.items()}
    assert inputs[f"{prefix}.mlp.experts.w13_weight"].shape == (2, 6, 4)
    assert inputs[f"{prefix}.mlp.shared_experts.gate_up_proj.weight"].shape == (6, 4)
    assert inputs[f"{prefix}.mlp.gate.e_score_correction_bias"].dtype == torch.float32
    assert {piece["source"] for entry in proof["wire"].values() for piece in entry["pieces"]} == set(before)


def test_ep_evidence_native_glm_sparse_mla_and_hyper_connection_bytes():
    prefix = "model.layers.0"
    inputs = {
        f"{prefix}.self_attn.q_a_proj.weight": _tensor((2, 4), 1),
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": _tensor((3, 4), 2),
        f"{prefix}.self_attn.indexer.wk.weight": _tensor((2, 4), 3),
        f"{prefix}.self_attn.indexer.weights_proj.weight": _tensor((1, 4), 4),
        f"{prefix}.attn_hc.fn": _tensor((2,), 5),
        f"{prefix}.ffn_hc.scale": _tensor((1,), 6),
    }
    proof = ep.pack_layer_with_evidence(convert_glm5_next_layer_to_vllm, inputs, 0, layer_prefix=prefix)
    assert inputs[f"{prefix}.self_attn.fused_qkv_a_proj.weight"].shape == (5, 4)
    assert inputs[f"{prefix}.self_attn.indexer.wk_weights_proj.weight"].shape == (3, 4)
    assert f"{prefix}.hc_attn_fn" in proof["wire"]
    assert f"{prefix}.hc_ffn_scale" in proof["wire"]


def test_ep_evidence_native_glm_kda_three_complete_conv_regions():
    prefix = "model.layers.0"
    attn = f"{prefix}.self_attn"
    inputs = {
        f"{attn}.{leaf}.weight": _tensor((2, 4), index + 1)
        for index, leaf in enumerate(("q_proj", "k_proj", "v_proj", "b_proj", "forget_gate.f_a_proj", "g_a_proj"))
    }
    inputs[f"{attn}.conv1d.weight"] = torch.arange(18, dtype=torch.float32).reshape(6, 1, 3)
    inputs[f"{attn}.forget_gate.A_log"] = _tensor((2,), 1, torch.float32)
    inputs[f"{attn}.forget_gate.dt_bias"] = _tensor((2,), 2, torch.float32)
    proof = ep.pack_layer_with_evidence(convert_glm5_next_layer_to_vllm, inputs, 0, layer_prefix=prefix)
    pieces = [proof["wire"][f"{attn}.{name}_conv1d.weight"]["pieces"][0] for name in ("q", "k", "v")]
    assert [(piece["dim0_start"], piece["dim0_stop"]) for piece in pieces] == [(0, 2), (2, 4), (4, 6)]
    assert inputs[f"{attn}.in_proj_qkvbfg_a.weight"].shape == (12, 4)


@pytest.mark.parametrize("defect", ["drop", "duplicate", "add", "cast", "split_hole"])
def test_ep_evidence_converter_loss_or_unsupported_arithmetic_rejected(defect):
    def convert(layer_sd, layer_idx, *, layer_prefix=None):
        if defect == "drop":
            layer_sd.pop("value")
        elif defect == "duplicate":
            layer_sd["duplicate"] = layer_sd["value"]
        elif defect == "add":
            layer_sd["value"] = layer_sd["value"] + 1
        elif defect == "cast":
            layer_sd["value"] = layer_sd["value"].to(torch.float64)
        else:
            layer_sd["value"] = layer_sd["value"].chunk(2, dim=0)[0]
        return layer_sd

    with pytest.raises(ValueError):
        ep.pack_layer_with_evidence(convert, {"value": _tensor((4, 2))}, 0)


def test_ep_evidence_native_selected_tokens_buffer_is_not_silently_dropped():
    prefix, inputs = _routed_layer()
    inputs[f"{prefix}.mlp.tokens_per_expert"] = _tensor((2,), 9, torch.float32)
    with pytest.raises(ValueError, match="every actual input"):
        ep.pack_layer_with_evidence(convert_glm5_next_layer_to_vllm, inputs, 0, layer_prefix=prefix)


def test_ep_evidence_native_unknown_router_key_not_relabelled():
    prefix, inputs = _routed_layer()
    inputs[f"{prefix}.mlp.router.unknown"] = _tensor((2,), 1)
    with pytest.raises(RuntimeError, match="unmapped"):
        ep.pack_layer_with_evidence(convert_glm5_next_layer_to_vllm, inputs, 0, layer_prefix=prefix)


def _records():
    return [
        {
            "rank": rank,
            "world_size": 8,
            "parameters": {
                "expert": {
                    **tensor_manifest(_tensor((1, 2), rank + 1)),
                    "requires_grad": True,
                    "ep_ranks": list(range(8)),
                },
                "normal": {**tensor_manifest(_tensor((2,), 1)), "requires_grad": True, "ep_ranks": None},
            },
            "buffers": {
                "bias": {**tensor_manifest(_tensor((2,), 2)), "requires_grad": False, "ep_ranks": None},
            },
        }
        for rank in range(8)
    ]


def test_ep_evidence_global_inventory_counts_ep_shards_once_and_buffers_separately():
    inventory = ep.global_inventory(_records())
    assert inventory["parameters"]["expert"]["global_numel"] == 16
    assert inventory["parameters"]["normal"]["global_numel"] == 2
    assert inventory["buffers"]["bias"]["global_numel"] == 2
    assert inventory["parameters"]["expert"]["ep_groups"] == [list(range(8))]


@pytest.mark.parametrize("defect", ["duplicate_rank", "missing_tensor", "reordered_group", "geometry", "hash"])
def test_ep_evidence_rank_inventory_omissions_or_inconsistent_groups_rejected(defect):
    records = _records()
    if defect == "duplicate_rank":
        records[-1]["rank"] = 0
    elif defect == "missing_tensor":
        records[-1]["parameters"].pop("normal")
    elif defect == "reordered_group":
        records[-1]["parameters"]["expert"]["ep_ranks"].reverse()
    elif defect == "geometry":
        records[-1]["parameters"]["expert"]["numel"] = 3
    else:
        records[-1]["parameters"]["expert"]["sha256"] = "x" * 64
    with pytest.raises(ValueError):
        ep.global_inventory(records)


def test_ep_evidence_actual_gathered_shard_bytes_bind_group_order():
    records = _records()
    order = list(reversed(range(8)))
    shards = [{"rank": rank, **records[rank]["parameters"]["expert"]} for rank in order]
    value = torch.cat([_tensor((1, 2), rank + 1) for rank in order], dim=0)
    sources = {"expert": {"name": "expert", "kind": "parameters", "ordered_shards": shards}}
    assert len(ep.verify_gathered_layer({"expert": value}, sources)["expert"]) == 8
    with pytest.raises(ValueError, match="shard bytes"):
        ep.verify_gathered_layer({"expert": value.flip(0)}, sources)


class Glm5NextCollectiveFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        mlp = torch.nn.Module()
        self.model.layers[0].mlp = mlp
        mlp.experts = torch.nn.Module()
        for index, name in enumerate(("w1", "w2", "w3")):
            value = torch.nn.Parameter(_tensor((1, 2, 2), index + 1))
            value.allreduce = False
            value.group_name = "synthetic-ep8"
            setattr(mlp.experts, name, value)
        mlp.router = torch.nn.Module()
        mlp.router.gate = torch.nn.Linear(2, 8, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            mlp.router.gate.weight.fill_(1)
        mlp.register_buffer("expert_bias", _tensor((8,), 2, torch.float32))
        mlp.register_buffer("counter", _tensor((8,), 0, torch.float32))
        register_weight_export(self, WeightExportContract(vllm=native.build_iter_full_vllm_weights))


class _Sender:
    def __init__(self):
        self.calls = []

    def send(self, weights):
        self.calls.append(weights)
        return {"status": "done", "actual_fixture_receipt": 17}


def _collective(monkeypatch, *, rank=0):
    model = Glm5NextCollectiveFixture()
    for _, value in model.named_parameters():
        if getattr(value, "allreduce", True) is False:
            with torch.no_grad():
                value.add_(rank)
    group = object()
    ds = ModuleType("deepspeed")
    utils = ModuleType("deepspeed.utils")
    groups = ModuleType("deepspeed.utils.groups")
    groups._get_expert_parallel_group = lambda name: group
    ds.utils, utils.groups = utils, groups
    for name, module in (("deepspeed", ds), ("deepspeed.utils", utils), ("deepspeed.utils.groups", groups)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: list(range(8)))
    gathers = []

    def all_gather(outputs, local, group=None):
        gathers.append(group)
        for other, output in enumerate(outputs):
            output.copy_(local + other - rank)

    def all_gather_object(outputs, local):
        parameters = dict(model.named_parameters())
        for other in range(8):
            record = deepcopy(local)
            record["rank"] = other
            for name, entry in record["parameters"].items():
                if entry["ep_ranks"] is not None:
                    entry.update(tensor_manifest(parameters[name].detach() + other - rank))
            outputs[other] = record

    monkeypatch.setattr(dist, "all_gather", all_gather)
    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    sources = source_files(
        type(model),
        type(model).named_parameters,
        type(model).named_buffers,
        type(model).get_submodule,
        native.build_iter_full_vllm_weights,
        convert_glm5_next_layer_to_vllm,
        native.named_weight_sync_tensors,
        native.vllm_layer_converter,
        weight_export_contract,
        ep.prepare_collective_weights,
        TorchDispatchMode,
        dist.all_gather,
        dist.all_gather_object,
        dist.get_process_group_ranks,
        groups._get_expert_parallel_group,
        _Sender.send,
    )
    return model, sources, gathers


def _prepare(model, sources):
    return ep.prepare_collective_weights(model, weight_format="vllm", max_weight_bytes=4096, admitted_sources=sources)


def test_ep_evidence_actual_native_iterator_prepares_once_then_sends_exact_retained_list(monkeypatch):
    model, sources, gathers = _collective(monkeypatch)
    prepared = _prepare(model, sources)
    assert len(gathers) == 3
    assert prepared.inventory["parameters"]["model.layers.0.mlp.experts.w1"]["global_numel"] == 32
    assert prepared.packing[0]["gathered_shards"]
    assert prepared.receipt()["iterator_binding"]["unchanged_native_iterator"] is False
    sender = _Sender()
    result = ep.send_prepared_weights(sender, prepared)
    assert sender.calls == [prepared.weights]
    assert sender.calls[0] is prepared.weights
    assert result["actual_fixture_receipt"] == 17
    assert result["trainer_parameters"]["numel"] == 112
    assert result["trainer_buffers"] == {"numel": 16, "count": 2, "wire_numel": 8}
    assert sum(entry["numel"] for entry in result["sent_manifest"].values()) == 120
    assert len(gathers) == 3


def test_ep_evidence_nonzero_rank_gathers_but_cannot_send(monkeypatch):
    model, sources, gathers = _collective(monkeypatch, rank=7)
    prepared = _prepare(model, sources)
    assert len(gathers) == 3
    assert prepared.weights == [] and prepared.packing == []
    sender = _Sender()
    with pytest.raises(ValueError, match="rank-zero"):
        ep.send_prepared_weights(sender, prepared)
    assert sender.calls == []


def test_ep_evidence_actual_peft_recipe_not_substituted_by_full_weight_export(monkeypatch):
    model, sources, gathers = _collective(monkeypatch)
    model.peft_config = {"default": object()}
    with pytest.raises(ValueError, match="LoRA"):
        _prepare(model, sources)
    assert gathers == []


def test_ep_evidence_actual_wrong_ep_group_fails_before_gather(monkeypatch):
    model, sources, gathers = _collective(monkeypatch)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: list(range(4)))
    with pytest.raises(ValueError, match="EP8"):
        _prepare(model, sources)
    assert gathers == []


def test_ep_evidence_actual_replicated_wire_value_disagreement_not_erased(monkeypatch):
    model, sources, gathers = _collective(monkeypatch)
    collect = dist.all_gather_object

    def collect_mismatched(outputs, local):
        collect(outputs, local)
        outputs[-1]["parameters"]["model.layers.0.mlp.router.gate.weight"]["sha256"] = "0" * 64

    monkeypatch.setattr(dist, "all_gather_object", collect_mismatched)
    with pytest.raises(ValueError, match="replicated"):
        _prepare(model, sources)
    assert gathers == []


def test_ep_evidence_parameter_replacement_after_native_gather_not_admitted(monkeypatch):
    model, sources, _ = _collective(monkeypatch)
    gather = dist.all_gather
    replaced = []

    def gather_and_replace(outputs, local, group=None):
        gather(outputs, local, group=group)
        if not replaced:
            old = model.model.layers[0].mlp.experts.w1
            new = torch.nn.Parameter(old.detach().clone())
            new.allreduce, new.group_name = old.allreduce, old.group_name
            model.model.layers[0].mlp.experts.w1 = new
            replaced.append(True)

    monkeypatch.setattr(dist, "all_gather", gather_and_replace)
    with pytest.raises(RuntimeError, match="identity"):
        _prepare(model, sources)
    assert replaced == [True]


def test_ep_evidence_packing_dtype_promotion_is_not_unchanged_bytes():
    def promote(layer_sd, layer_idx, *, layer_prefix=None):
        layer_sd["fused"] = torch.cat([layer_sd.pop("a"), layer_sd.pop("b")], dim=0)
        return layer_sd

    with pytest.raises(RuntimeError, match="byte region"):
        ep.pack_layer_with_evidence(promote, {"a": _tensor((2,), 1), "b": _tensor((2,), 1, torch.float32)}, 0)


def test_ep_evidence_actual_source_pin_mismatch_fails_before_gathers(monkeypatch):
    model, sources, gathers = _collective(monkeypatch)
    for path in source_files(ep.prepare_collective_weights):
        sources[path] = "0" * 64
    with pytest.raises(ValueError, match="source origins"):
        _prepare(model, sources)
    assert gathers == []


@pytest.mark.parametrize("defect", ["format", "zero3", "world", "byte_budget", "source"])
def test_ep_evidence_preparation_rejects_unsupported_route_before_native_gathers(monkeypatch, defect):
    model, sources, gathers = _collective(monkeypatch)
    kwargs = {"weight_format": "vllm", "max_weight_bytes": 4096, "admitted_sources": sources}
    if defect == "format":
        kwargs["weight_format"] = "hf"
    elif defect == "zero3":
        kwargs["is_zero3"] = True
    elif defect == "world":
        kwargs["expected_world_size"] = 16
    elif defect == "byte_budget":
        kwargs["max_weight_bytes"] = 1
    else:
        kwargs["admitted_sources"] = {}
    with pytest.raises(ValueError):
        ep.prepare_collective_weights(model, **kwargs)
    assert gathers == []


@pytest.mark.parametrize("defect", ["bytes", "metadata", "name"])
def test_ep_evidence_prepared_drift_rejected_before_sender(monkeypatch, defect):
    model, sources, _ = _collective(monkeypatch)
    prepared = _prepare(model, sources)
    if defect == "bytes":
        prepared.weights[0][1].add_(1)
    elif defect == "metadata":
        prepared.inventory["parameters"].clear()
    else:
        prepared.weights.append(prepared.weights[0])
    sender = _Sender()
    with pytest.raises(RuntimeError):
        ep.send_prepared_weights(sender, prepared)
    assert sender.calls == []


def _receiver(prepared):
    received = {name: {**entry, "destinations": [f"native.{name}"]} for name, entry in prepared.manifest.items()}
    return {
        "status": "done",
        "weight_sync_verification": {
            "status": "ok",
            "received": received,
            "destinations": {f"native.{name}": {"uncovered": 0, "mismatched": 0} for name in received},
            "untouched_parameters": {},
            "unverifiable_destinations": [],
            "non_parameter_destinations": [],
            "evidence_sources": source_files(_receiver),
        },
    }


def test_ep_evidence_receiver_binding_consumes_exact_two_rank_results_without_loading(monkeypatch):
    model, sources, _ = _collective(monkeypatch)
    prepared = _prepare(model, sources)
    results = [_receiver(prepared), _receiver(prepared)]
    result = ep.bind_receiver_results(prepared, results, expected_sampler_ranks=2, admitted_sources=sources)
    assert result["wire_bytes_and_shadow_coverage_bound"] is True
    assert result["actual_rank_results"][0]["actual_result"] is results[0]
    assert result["overall_admission"] is False
    assert result["native_rank_identity_admitted"] is False
    assert result["capacity_admitted"] is False


@pytest.mark.parametrize("defect", ["missing_rank", "wire_hash", "drop", "uncovered", "untouched", "source"])
def test_ep_evidence_receiver_missing_or_unproved_evidence_rejected(monkeypatch, defect):
    model, sources, _ = _collective(monkeypatch)
    prepared = _prepare(model, sources)
    results = [_receiver(prepared), _receiver(prepared)]
    record = results[1]["weight_sync_verification"]
    name = next(iter(record["received"]))
    if defect == "missing_rank":
        results.pop()
    elif defect == "wire_hash":
        record["received"][name]["sha256"] = "0" * 64
    elif defect == "drop":
        record["received"][name]["destinations"] = []
    elif defect == "uncovered":
        record["destinations"][f"native.{name}"]["uncovered"] = 1
    elif defect == "untouched":
        record["untouched_parameters"] = {"extra": 1}
    else:
        record["evidence_sources"] = {}
    with pytest.raises(ValueError):
        ep.bind_receiver_results(prepared, results, expected_sampler_ranks=2, admitted_sources=sources)
