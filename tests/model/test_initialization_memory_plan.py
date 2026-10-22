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

"""Capacity arithmetic and native tensor metadata; no model weights or admission."""

import importlib.util
import json
import sys
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def memory(monkeypatch):
    # Load just the two pure modules: model.__init__ registers native loaders.
    root = Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations/debug"
    modules = (
        ("arctic_platform.model.implementations.debug.config", "config.py"),
        ("_initialization_memory_test_domain", "gpu_memory.py"),
    )
    for name, filename in modules:
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape,itemsize,expected", [((), 2, 2), ((0, 16), 2, 0), ((3, 4), 1, 12)])
def test_complete_storage_shape_arithmetic(memory, shape, itemsize, expected):
    assert memory.tensor_storage_bytes(shape, itemsize) == expected


@pytest.mark.parametrize("shape,itemsize", [((True,), 2), ((-1,), 2), ((1.5,), 2), ((2,), 0), ((2,), True)])
def test_invalid_storage_metadata_is_rejected(memory, shape, itemsize):
    with pytest.raises(ValueError):
        memory.tensor_storage_bytes(shape, itemsize)


def reads(memory):
    header = memory.CheckpointTensor
    request = memory.DCPReadRequest
    return (
        {
            "a": header("file1", (100,), 2),
            "b": header("file1", (80,), 2),
            "c": header("file2", (50,), 4),
            "d": header("file3", (20,), 1),
        },
        [
            request("a", (0,), (100,)),
            request("b", (0,), (80,)),
            request("c", (0,), (50,)),
            request("d", (0,), (20,)),
        ],
    )


def test_single_reader_uses_one_largest_slice_not_full_checkpoint(memory):
    headers, requests = reads(memory)
    row = memory.hf_read_slice_envelope(headers, requests, thread_count=1)
    assert row["per_file_slice_bytes"] == {"file1": 200, "file2": 200, "file3": 20}
    assert row["concurrent_files"] == 1
    assert row["slice_payload_bytes"] == 200
    assert row["installed_reader_bound"] is False
    assert row["complete_dcp_peak_bytes"] is None
    assert len(row["unresolved_terms"]) == 3


def test_concurrent_files_sum_file_maxima_not_requests(memory):
    headers, requests = reads(memory)
    row = memory.hf_read_slice_envelope(headers, requests, thread_count=2)
    assert row["slice_payload_bytes"] == 400
    assert row["concurrent_files"] == 2


def test_concurrency_cannot_exceed_actual_file_count(memory):
    headers, requests = reads(memory)
    row = memory.hf_read_slice_envelope(headers, requests, thread_count=100)
    assert row["slice_payload_bytes"] == 420
    assert row["concurrent_files"] == 3


def test_actual_slice_lengths_and_dtype_width_drive_bytes(memory):
    headers = {"w": memory.CheckpointTensor("file", (32, 64), 1)}
    requests = [memory.DCPReadRequest("w", (8, 32), (8, 16))]
    assert memory.hf_read_slice_envelope(headers, requests, thread_count=1)["slice_payload_bytes"] == 128


@pytest.mark.parametrize("offsets,lengths", [((32, 0), (1, 1)), ((-1, 0), (1, 1)), ((0,), (1, 1))])
def test_bad_source_slice_fails_before_arithmetic(memory, offsets, lengths):
    headers = {"w": memory.CheckpointTensor("file", (32, 64), 2)}
    requests = [memory.DCPReadRequest("w", offsets, lengths)]
    with pytest.raises(ValueError):
        memory.hf_read_slice_envelope(headers, requests, thread_count=1)


def test_missing_original_header_is_not_reduced_geometry(memory):
    headers = {"a": memory.CheckpointTensor("file", (32, 64), 2)}
    with pytest.raises(ValueError, match="missing original header"):
        memory.hf_read_slice_envelope(
            headers, [memory.DCPReadRequest("original_full_layer44", (0,), (16,))], thread_count=1
        )


@pytest.mark.parametrize("thread_count", [0, -1, True])
def test_invalid_reader_concurrency_is_rejected(memory, thread_count):
    headers, requests = reads(memory)
    with pytest.raises(ValueError):
        memory.hf_read_slice_envelope(headers, requests, thread_count=thread_count)


def plan(memory, terms, *, limits=None, unresolved_inputs=()):
    return memory.initialization_phase_plan(
        ("load", "ready", "serve"),
        terms,
        limits={"host": 1000} if limits is None else limits,
        unresolved_inputs=unresolved_inputs,
    )


def test_unknown_release_stays_live_with_later_allocations(memory):
    term = memory.MemoryLifetime
    row = plan(memory, [term("source", "host", 300, "load"), term("factors", "host", 200, "serve")])
    assert [phase["known_bytes"] for phase in row["phases"]] == [300, 300, 500]
    assert row["resources"]["host"]["peak_bound_bytes"] == 500
    assert row["capacity_admitted"] is row["runtime_admitted"] is False


def test_explicit_release_allows_phase_max_instead_of_sum(memory):
    term = memory.MemoryLifetime
    row = plan(
        memory,
        [term("source", "host", 300, "load", "load"), term("factors", "host", 200, "serve")],
    )
    assert [phase["known_bytes"] for phase in row["phases"]] == [300, 0, 200]
    assert row["resources"]["host"]["peak_bound_bytes"] == 300
    assert row["all_named_terms_finite"] is True


def test_unknown_native_term_remains_none_not_zero(memory):
    term = memory.MemoryLifetime
    row = plan(memory, [term("base", "host", 300, "load"), term("native", "host", None, "ready")])
    assert row["resources"]["host"]["known_peak_bytes"] == 300
    assert row["resources"]["host"]["peak_bound_bytes"] is None
    assert row["resources"]["host"]["unresolved_terms"] == ["native"]
    assert row["phases"][0]["peak_bound_bytes"] == 300
    assert row["phases"][1]["peak_bound_bytes"] is None
    assert row["all_named_terms_finite"] is False


def test_installed_reader_absence_is_preserved_beside_finite_subtotals(memory):
    row = plan(
        memory,
        [memory.MemoryLifetime("base", "host", 300, "load")],
        unresolved_inputs=("installed_reader_callable", "complete_original_tensor_layout"),
    )
    assert row["unresolved_inputs"] == ["installed_reader_callable", "complete_original_tensor_layout"]
    assert row["all_named_terms_finite"] is True
    assert row["capacity_admitted"] is row["runtime_admitted"] is False


def test_aliases_are_deduplicated_only_on_same_resource(memory):
    term = memory.MemoryLifetime
    row = plan(
        memory,
        [
            term("base", "gpu0", 100, "load", storage_id="weight"),
            term("base_view", "gpu0", 100, "load", storage_id="weight"),
            term("replica", "gpu1", 100, "load", storage_id="weight"),
        ],
        limits={"gpu0": 200, "gpu1": 200},
    )
    assert row["resources"]["gpu0"]["peak_bound_bytes"] == 100
    assert row["resources"]["gpu1"]["peak_bound_bytes"] == 100


def test_inconsistent_complete_alias_sizes_reject(memory):
    term = memory.MemoryLifetime
    with pytest.raises(ValueError, match="aliased complete storage sizes differ"):
        plan(
            memory,
            [term("a", "host", 100, "load", storage_id="x"), term("b", "host", 90, "load", storage_id="x")],
        )


def test_train8_and_sample8_share_host_ledger_without_sp_or_tp_division(memory):
    term = memory.MemoryLifetime
    terms = []
    limits = {"host": 1000}
    for rank in range(8):
        terms.extend(
            [
                term(f"train{rank}.cpu", "host", 10, "load"),
                term(f"sample{rank}.receive", "host", 16, "ready"),
                term(f"sample{rank}.pin_copy", "host", 16, "ready"),
                term(f"train{rank}.base", f"train_gpu{rank}", 100, "load"),
                term(f"sample{rank}.base", f"sample_gpu{rank}", 200, "load"),
            ]
        )
        limits[f"train_gpu{rank}"] = 100
        limits[f"sample_gpu{rank}"] = 200
    row = plan(memory, terms, limits=limits)
    assert row["resources"]["host"]["peak_bound_bytes"] == 8 * (10 + 16 + 16)
    for rank in range(8):
        assert row["resources"][f"train_gpu{rank}"]["peak_bound_bytes"] == 100
        assert row["resources"][f"sample_gpu{rank}"]["peak_bound_bytes"] == 200


def test_known_terms_over_limit_remain_failure_even_with_unknowns(memory):
    term = memory.MemoryLifetime
    row = plan(
        memory,
        [term("base", "host", 1001, "load"), term("native", "host", None, "load")],
    )
    assert row["resources"]["host"]["known_terms_exceed_limit"] is True
    assert row["resources"]["host"]["peak_bound_bytes"] is None


@pytest.mark.parametrize(
    "term_args,limits,match",
    [
        (("a", "host", -1, "load"), {"host": 100}, "term size"),
        (("a", "host", True, "load"), {"host": 100}, "term size"),
        (("a", "host", 1, "absent"), {"host": 100}, "unknown phase"),
        (("a", "host", 1, "serve", "load"), {"host": 100}, "ends before"),
        (("a", "other_host", 1, "load"), {"host": 100}, "resource limit missing"),
        (("a", "host", 1, "load"), {"host": True}, "resource limit"),
    ],
)
def test_invalid_lifetime_or_limit_rejects(memory, term_args, limits, match):
    with pytest.raises(ValueError, match=match):
        plan(memory, [memory.MemoryLifetime(*term_args)], limits=limits)


def test_unused_resource_is_not_implicitly_zero_inventory(memory):
    with pytest.raises(ValueError, match="no inventory terms"):
        plan(memory, [memory.MemoryLifetime("base", "host", 10, "load")], limits={"host": 100, "gpu": 100})


def test_duplicate_terms_and_phases_reject(memory):
    term = memory.MemoryLifetime("base", "host", 10, "load")
    with pytest.raises(ValueError, match="unique named"):
        plan(memory, [term, term])
    with pytest.raises(ValueError, match="duplicate phase"):
        memory.initialization_phase_plan(("load", "load"), [term], limits={"host": 100}, unresolved_inputs=())


def test_finite_gate_rejects_missing_installed_reader_binding(memory):
    with pytest.raises(ValueError, match="unresolved initialization inputs"):
        memory.require_finite_initialization_plan(
            ("load",),
            [memory.MemoryLifetime("base", "host", 10, "load")],
            limits={"host": 100},
            unresolved_inputs=("installed_reader_callable",),
        )


def test_finite_gate_rejects_unknown_native_workspace(memory):
    with pytest.raises(ValueError, match="unresolved memory terms"):
        memory.require_finite_initialization_plan(
            ("load",),
            [memory.MemoryLifetime("native_workspace", "gpu", None, "load")],
            limits={"gpu": 100},
            unresolved_inputs=(),
        )


def test_finite_gate_rejects_real_limit_overrun(memory):
    with pytest.raises(ValueError, match="exceeds limit"):
        memory.require_finite_initialization_plan(
            ("load",),
            [memory.MemoryLifetime("base", "gpu", 101, "load")],
            limits={"gpu": 100},
            unresolved_inputs=(),
        )


def test_finite_gate_does_not_turn_finite_arithmetic_into_native_admission(memory):
    row = memory.require_finite_initialization_plan(
        ("load",),
        [memory.MemoryLifetime("base", "gpu", 100, "load")],
        limits={"gpu": 100},
        unresolved_inputs=(),
    )
    assert row["resources"]["gpu"]["peak_bound_bytes"] == 100
    assert row["capacity_admitted"] is row["runtime_admitted"] is False


def header_fixture():
    return (
        {
            "original1": {
                "header_size_bytes": 16,
                "file_size_bytes": 44,
                "tensors": {
                    "weight": {"shape": [2, 3], "dtype": "BF16", "data_offsets": [0, 12]},
                    "scale": {"shape": [2], "dtype": "F32", "data_offsets": [12, 20]},
                },
            },
            "original2": {
                "header_size_bytes": 16,
                "file_size_bytes": 28,
                "tensors": {"fp8": {"shape": [4], "dtype": "F8_E4M3", "data_offsets": [0, 4]}},
            },
        },
        {"weight": "original1", "scale": "original1", "fp8": "original2"},
        {"BF16": 2, "F32": 4, "F8_E4M3": 1},
    )


def test_header_inventory_counts_tensor_payload_not_file_or_header_bytes(memory):
    shards, index, widths = header_fixture()
    row = memory.checkpoint_header_inventory(shards, weight_map=index, dtype_widths=widths)
    assert row["weight"] == memory.CheckpointTensor("original1", (2, 3), 2)
    assert sum(memory.tensor_storage_bytes(t.shape, t.itemsize) for t in row.values()) == 24


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_index", "original index weight_map"),
        ("missing_dtype", "unresolved dtype width"),
        ("wrong_shard", "shard set differs"),
        ("missing_header_key", "tensor key set differs"),
        ("bad_span_size", "data span"),
        ("overlap", "overlap or leave a payload gap"),
        ("gap", "overlap or leave a payload gap"),
        ("uncovered_tail", "do not cover shard payload"),
        ("bool_shape", "dimension"),
    ],
)
def test_header_inventory_rejects_unbound_or_inconsistent_metadata(memory, change, match):
    shards, index, widths = header_fixture()
    if change == "missing_index":
        index = {}
    elif change == "missing_dtype":
        widths.pop("F32")
    elif change == "wrong_shard":
        index["fp8"] = "missing"
    elif change == "missing_header_key":
        index["missing"] = "original1"
    elif change == "bad_span_size":
        shards["original1"]["tensors"]["weight"]["data_offsets"][1] = 11
    elif change == "overlap":
        shards["original1"]["tensors"]["scale"]["data_offsets"] = [11, 19]
    elif change == "gap":
        shards["original1"]["tensors"]["scale"]["data_offsets"] = [13, 21]
        shards["original1"]["file_size_bytes"] = 45
    elif change == "uncovered_tail":
        shards["original1"]["file_size_bytes"] += 1
    elif change == "bool_shape":
        shards["original1"]["tensors"]["weight"]["shape"][0] = True
    with pytest.raises(ValueError, match=match):
        memory.checkpoint_header_inventory(shards, weight_map=index, dtype_widths=widths)


def load_inputs(memory):
    return {
        "source_headers": {"global_experts": memory.CheckpointTensor("original", (32, 16), 2)},
        "load_headers": {"packed_experts": memory.CheckpointTensor("cache", (8, 16), 2)},
        "requests": [memory.DCPReadRequest("packed_experts", (0, 0), (2, 16))],
        "destination_layouts": {"local_experts": ((2, 16), 2)},
        "name": "rank0",
        "cpu_resource": "host",
        "gpu_resource": "gpu0",
        "destination_resource": "gpu0",
        "first_phase": "load",
        "reader_threads": 1,
        "reader_binding": {"path": "/selected/reader.py", "sha256": "0" * 64},
        "cache_miss": True,
        "conversion_writer": True,
        "packed_layouts": {"global_pack": ((32, 16), 2)},
        "saved_layouts": {"global_save": ((32, 16), 2)},
        "cache_release_phase": None,
        "read_release_phase": None,
        "limits": {"host": 10000, "gpu0": 10000},
    }


def sizes(row):
    return {term.name: term.size_bytes for term in row["terms"]}


def test_load_phase_producer_separates_original_cache_and_local_destination(memory):
    row = memory.dcp_load_phase_terms(**load_inputs(memory))
    terms = sizes(row)
    assert terms["rank0.conversion_source"] == 1024
    assert terms["rank0.conversion_pack"] == 1024
    assert terms["rank0.save_contiguous_copies"] == 1024
    assert terms["rank0.destination"] == 64
    assert terms["rank0.read_slice_payload"] == 64
    assert terms["rank0.planner_cpu_staging"] is terms["rank0.planner_gpu_staging"] is None
    assert row["reader_recipe_is_installed_binding"] is False
    assert row["capacity_admitted"] is row["runtime_admitted"] is False
    with pytest.raises(ValueError, match="unresolved initialization inputs"):
        memory.require_finite_initialization_plan(
            ("load",), row["terms"], limits=row["limits"], unresolved_inputs=row["unresolved_inputs"]
        )


@pytest.mark.parametrize("selector", ["cache_hit", "nonwriter"])
def test_load_phase_producer_omits_only_explicitly_inactive_conversion(memory, selector):
    args = load_inputs(memory)
    args["cache_miss" if selector == "cache_hit" else "conversion_writer"] = False
    row = memory.dcp_load_phase_terms(**args)
    terms = sizes(row)
    assert terms["rank0.conversion_source"] == terms["rank0.conversion_pack"] == 0
    assert terms["rank0.save_contiguous_copies"] == terms["rank0.save_native"] == 0
    assert terms["rank0.mapped_backend_residency"] is None


def test_load_phase_producer_does_not_zero_unknown_cache_or_writer_selection(memory):
    args = load_inputs(memory)
    args["cache_miss"] = args["conversion_writer"] = None
    terms = sizes(memory.dcp_load_phase_terms(**args))
    assert terms["rank0.conversion_source"] is terms["rank0.conversion_pack"] is None
    assert terms["rank0.save_contiguous_copies"] is terms["rank0.save_native"] is None


def test_load_phase_producer_preserves_all_missing_actual_inputs(memory):
    args = load_inputs(memory)
    for key in ("source_headers", "load_headers", "requests", "destination_layouts", "reader_binding", "limits"):
        args[key] = None
    row = memory.dcp_load_phase_terms(**args)
    terms = sizes(row)
    assert terms["rank0.conversion_source"] is terms["rank0.destination"] is terms["rank0.read_slice_payload"] is None
    assert set(row["unresolved_inputs"]) == {
        "installed_reader_planner_native_binding",
        "original_and_cache_payload_identity",
        "complete_original_headers",
        "actual_dcp_source_headers",
        "actual_rank_local_destination_layouts",
        "actual_selected_readplan",
        "selected_reader_source_recipe",
        "actual_usable_resource_limits",
    }


def test_load_phase_producer_keeps_unknown_release_overlapping_serve(memory):
    row = memory.dcp_load_phase_terms(**load_inputs(memory))
    terms = list(row["terms"]) + [memory.MemoryLifetime("sampler.receive", "host", 100, "serve")]
    result = memory.initialization_phase_plan(
        ("load", "serve"), terms, limits=row["limits"], unresolved_inputs=row["unresolved_inputs"]
    )
    host = [phase for phase in result["phases"] if phase["resource"] == "host"]
    assert [phase["known_bytes"] for phase in host] == [3136, 3236]
    assert result["resources"]["host"]["peak_bound_bytes"] is None


def test_load_phase_producer_never_reuses_original_keys_for_converted_readplan(memory):
    args = load_inputs(memory)
    args["load_headers"] = deepcopy(args["source_headers"])
    with pytest.raises(ValueError, match="missing original header"):
        memory.dcp_load_phase_terms(**args)


@pytest.mark.parametrize("change", ["same_resources", "bad_writer_flag", "missing_gpu_limit", "negative_threads"])
def test_load_phase_producer_rejects_invalid_actual_inputs(memory, change):
    args = load_inputs(memory)
    if change == "same_resources":
        args["gpu_resource"] = "host"
    elif change == "bad_writer_flag":
        args["conversion_writer"] = 1
    elif change == "missing_gpu_limit":
        args["limits"].pop("gpu0")
    elif change == "negative_threads":
        args["reader_threads"] = -1
    with pytest.raises(ValueError):
        memory.dcp_load_phase_terms(**args)


def saved_identity():
    return {"model_id": "saved-original-BF16", "revision": "original-revision", "receipt_sha256": "a" * 64}


def test_saved_header_ledger_counts_payload_without_claiming_index_provenance(memory):
    shards, _, widths = header_fixture()
    before = deepcopy(shards)
    identity = saved_identity()
    row = memory.saved_header_ledger(shards, dtype_widths=widths, source_identity=identity)
    assert row["tensor_count"] == 3
    assert row["shard_count"] == 2
    assert row["tensor_payload_bytes"] == 24
    assert row["file_bytes"] == 72
    assert row["header_and_prefix_bytes"] == 48
    assert row["dtype_totals"]["F32"] == {"tensor_count": 1, "payload_bytes": 8}
    assert row["derived_shard_map"] == {"weight": "original1", "scale": "original1", "fp8": "original2"}
    assert row["headers"]["weight"] == memory.CheckpointTensor("original1", (2, 3), 2)
    assert row["source_identity"] == identity
    identity["revision"] = "changed"
    assert row["source_identity"]["revision"] == "original-revision"
    assert row["source_identity_bound"] is row["original_index_bound"] is row["payload_identity_bound"] is False
    assert row["capacity_admitted"] is row["runtime_admitted"] is False
    assert shards == before


@pytest.mark.parametrize(
    "change,match",
    [
        ("duplicate", "unique named"),
        ("missing_dtype", "unresolved saved dtype"),
        ("wrong_width", "data span"),
        ("overlap", "overlap or leave"),
        ("gap", "overlap or leave"),
        ("tail", "do not cover"),
        ("bad_identity", "receipt SHA256"),
        ("bool_shape", "dimension"),
        ("bool_offset", "data offset"),
        ("empty_shards", "saved headers"),
        ("empty_tensors", "invalid saved shard"),
        ("empty_name", "unique named"),
    ],
)
def test_saved_header_ledger_rejects_invalid_saved_metadata(memory, change, match):
    shards, _, widths = header_fixture()
    identity = saved_identity()
    if change == "duplicate":
        shards["original2"]["tensors"] = {"weight": shards["original2"]["tensors"]["fp8"]}
    elif change == "missing_dtype":
        widths.pop("F32")
    elif change == "wrong_width":
        widths["BF16"] = 4
    elif change == "overlap":
        shards["original1"]["tensors"]["scale"]["data_offsets"] = [11, 19]
    elif change == "gap":
        shards["original1"]["tensors"]["scale"]["data_offsets"] = [13, 21]
        shards["original1"]["file_size_bytes"] += 1
    elif change == "tail":
        shards["original1"]["file_size_bytes"] += 1
    elif change == "bad_identity":
        identity["receipt_sha256"] = "not-a-digest"
    elif change == "bool_shape":
        shards["original1"]["tensors"]["weight"]["shape"][0] = True
    elif change == "bool_offset":
        shards["original1"]["tensors"]["weight"]["data_offsets"][0] = False
    elif change == "empty_shards":
        shards = {}
    elif change == "empty_tensors":
        shards["original1"]["tensors"] = {}
    elif change == "empty_name":
        shards["original1"]["tensors"][""] = shards["original1"]["tensors"].pop("weight")
    with pytest.raises(ValueError, match=match):
        memory.saved_header_ledger(shards, dtype_widths=widths, source_identity=identity)


def test_saved_header_ledger_does_not_replace_strict_original_index_requirement(memory):
    shards, _, widths = header_fixture()
    ledger = memory.saved_header_ledger(shards, dtype_widths=widths, source_identity=saved_identity())
    assert ledger["original_index_bound"] is False
    with pytest.raises(ValueError, match="original index weight_map"):
        memory.checkpoint_header_inventory(shards, weight_map={}, dtype_widths=widths)


def glm_metadata_specs(*, experts=32, hidden=4, intermediate=2):
    prefix = "model.language_model.layers.3"
    specs = {
        "lm_head.weight": ((16, hidden), "BF16"),
        f"{prefix}.hc_attn_fn": ((4, hidden), "BF16"),
        f"{prefix}.hc_attn_base": ((4,), "F32"),
        f"{prefix}.hc_ffn_scale": ((3,), "F32"),
        f"{prefix}.self_attn.A_log": ((4,), "F32"),
        f"{prefix}.self_attn.q_conv1d.weight": ((4, 1, 3), "BF16"),
        f"{prefix}.self_attn.k_conv1d.weight": ((2, 1, 3), "BF16"),
        f"{prefix}.self_attn.v_conv1d.weight": ((2, 1, 3), "BF16"),
        f"{prefix}.mlp.gate.weight": ((experts, hidden), "BF16"),
        f"{prefix}.mlp.gate.e_score_correction_bias": ((experts,), "F32"),
    }
    for projection in ("gate_proj", "up_proj", "down_proj"):
        shape = (hidden, intermediate) if projection == "down_proj" else (intermediate, hidden)
        specs[f"{prefix}.mlp.shared_experts.{projection}.weight"] = (shape, "BF16")
        for expert in range(experts):
            specs[f"{prefix}.mlp.experts.{expert}.{projection}.weight"] = (shape, "BF16")
    return specs


def glm_metadata_ledger(memory, specs):
    widths = {"BF16": 2, "F32": 4, "F8_E4M3": 1}
    shard = {"header_size_bytes": 16, "tensors": {}}
    position = 0
    for name, (shape, dtype) in specs.items():
        end = position + memory.tensor_storage_bytes(shape, widths[dtype])
        shard["tensors"][name] = {"shape": list(shape), "dtype": dtype, "data_offsets": [position, end]}
        position = end
    shard["file_size_bytes"] = position + 24
    return memory.saved_header_ledger({"saved-original": shard}, dtype_widths=widths, source_identity=saved_identity())


def test_glm_metadata_conversion_preserves_payload_lineage_and_distinct_namespaces(memory):
    ledger = glm_metadata_ledger(memory, glm_metadata_specs())
    before = deepcopy(ledger)
    row = memory.glm_bf16_conversion_projection(ledger)
    prefix = "model.language_model.layers.3"
    assert row["layouts"][f"{prefix}.mlp.experts.w1"] == ((32, 2, 4), 2)
    assert row["layouts"][f"{prefix}.mlp.experts.w2"] == ((32, 4, 2), 2)
    assert row["layouts"][f"{prefix}.mlp.experts.w3"] == ((32, 2, 4), 2)
    assert row["layouts"][f"{prefix}.self_attn.conv1d.weight"] == ((8, 1, 3), 2)
    assert row["layouts"][f"{prefix}.mlp.shared_expert.w1"] == ((2, 4), 2)
    assert row["layouts"][f"{prefix}.attn_hc.fn"] == ((4, 4), 2)
    assert f"{prefix}.ffn_hc.scale" in row["layouts"]
    assert f"{prefix}.self_attn.forget_gate.A_log" in row["layouts"]
    assert f"{prefix}.mlp.router.gate.weight" in row["layouts"]
    assert f"{prefix}.mlp.expert_bias" in row["layouts"]
    assert row["checkpoint_expert_counts"] == {3: 32}
    assert len(row["source_names"][f"{prefix}.mlp.experts.w1"]) == 32
    assert row["source_names"][f"{prefix}.attn_hc.fn"] == (f"{prefix}.hc_attn_fn",)
    assert row["packed_payload_bytes"] == 3 * 32 * 2 * 4 * 2 + 8 * 1 * 3 * 2
    assert row["converted_payload_bytes"] == row["source_payload_bytes"] == ledger["tensor_payload_bytes"]
    assert f"{prefix}.mlp.experts.0.gate_proj.weight" not in row["layouts"]
    assert "converted_cache_headers" not in row
    assert row["converted_cache_bound"] is row["native_inventory_complete"] is False
    assert row["capacity_admitted"] is row["runtime_admitted"] is False
    assert ledger == before


@pytest.mark.parametrize(
    "target",
    [
        "attn_hc.fn",
        "self_attn.forget_gate.A_log",
        "self_attn.conv1d.weight",
        "mlp.experts.w1",
        "mlp.router.gate.weight",
        "mlp.shared_expert.w1",
    ],
)
def test_glm_metadata_conversion_rejects_target_collisions_without_mutating_source(memory, target):
    specs = glm_metadata_specs()
    specs[f"model.language_model.layers.3.{target}"] = ((1,), "BF16")
    ledger = glm_metadata_ledger(memory, specs)
    before = deepcopy(ledger)
    with pytest.raises(ValueError, match="converted key collision"):
        memory.glm_bf16_conversion_projection(ledger)
    assert ledger == before


@pytest.mark.parametrize(
    "change,match",
    [
        ("expert_gap", "non-contiguous"),
        ("missing_projection", "missing GLM projection"),
        ("expert_shape", "projection shapes differ"),
        ("projection_transpose", "matrix shapes are inconsistent"),
        ("partial_conv", "incomplete GLM convolution"),
        ("conv_shape", "trailing shapes differ"),
        ("fp8", "BF16/F32 widths"),
        ("bf16_scale", "scaled GLM checkpoint"),
        ("expert_f32", "identical BF16"),
        ("shared_f32", "shared expert projection"),
    ],
)
def test_glm_metadata_conversion_rejects_incomplete_or_unsupported_projection(memory, change, match):
    specs = glm_metadata_specs()
    prefix = "model.language_model.layers.3"
    if change == "expert_gap":
        for name in list(specs):
            if ".experts.1." in name:
                del specs[name]
    elif change == "missing_projection":
        del specs[f"{prefix}.mlp.experts.0.gate_proj.weight"]
    elif change == "expert_shape":
        specs[f"{prefix}.mlp.experts.0.gate_proj.weight"] = ((2, 5), "BF16")
    elif change == "projection_transpose":
        for name in list(specs):
            if ".experts." in name and ".down_proj." in name:
                specs[name] = ((2, 4), "BF16")
    elif change == "partial_conv":
        del specs[f"{prefix}.self_attn.q_conv1d.weight"]
    elif change == "conv_shape":
        specs[f"{prefix}.self_attn.k_conv1d.weight"] = ((2, 1, 4), "BF16")
    elif change == "fp8":
        specs[f"{prefix}.mlp.experts.0.gate_proj.weight"] = ((2, 4), "F8_E4M3")
    elif change == "bf16_scale":
        specs[f"{prefix}.mlp.experts.0.gate_proj.weight_scale_inv"] = ((1, 1), "F32")
    elif change == "expert_f32":
        specs[f"{prefix}.mlp.experts.0.gate_proj.weight"] = ((2, 4), "F32")
    elif change == "shared_f32":
        specs[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = ((2, 4), "F32")
    with pytest.raises(ValueError, match=match):
        memory.glm_bf16_conversion_projection(glm_metadata_ledger(memory, specs))


def test_glm_metadata_ep16_rows_use_expert_coordinates_and_do_not_divide_replicas(memory):
    ledger = glm_metadata_ledger(memory, glm_metadata_specs(experts=288, hidden=4096, intermediate=2048))
    conversion = memory.glm_bf16_conversion_projection(ledger)
    prefix = "model.language_model.layers.3"
    routed_bytes = 3 * 288 * 2048 * 4096 * 2
    other_bytes = conversion["converted_payload_bytes"] - routed_bytes
    for rank in range(16):
        row = memory.glm_ep_dcp_projection(conversion, ep_size=16, ep_rank=rank)
        requests = {request.tensor_name: request for request in row["requests"]}
        w1 = requests[f"{prefix}.mlp.experts.w1"]
        assert w1.offsets == (18 * rank, 0, 0)
        assert w1.lengths == (18, 2048, 4096)
        assert memory.tensor_storage_bytes(w1.lengths, 2) == 301989888
        assert requests["lm_head.weight"].offsets == (0, 0)
        assert requests["lm_head.weight"].lengths == (16, 4096)
        assert requests[f"{prefix}.mlp.shared_expert.w1"].lengths == (2048, 4096)
        assert row["destination_layouts"][f"{prefix}.mlp.experts.w1"] == ((18, 2048, 4096), 2)
        assert row["named_destination_bytes"] == routed_bytes // 16 + other_bytes
        assert row["native_buffers_bytes"] is row["native_workspace_bytes"] is row["usable_limits"] is None
        assert row["actual_readplan_bound"] is row["native_inventory_complete"] is False
        assert row["capacity_admitted"] is row["runtime_admitted"] is False
        assert row["conditional_only"] is True
        assert "actual_selected_dcp_readplan" in row["unresolved_inputs"]
        assert f"{prefix}.mlp.experts.0.gate_proj.weight" not in requests


@pytest.mark.parametrize(
    "ep_size,ep_rank,match",
    [(0, 0, "positive"), (True, 0, "ep_size"), (16, -1, "ep_rank"), (16, 16, "exceeds"), (3, 0, "divide equally")],
)
def test_glm_metadata_ep_rows_reject_invalid_coordinates_or_unequal_partitions(memory, ep_size, ep_rank, match):
    conversion = memory.glm_bf16_conversion_projection(glm_metadata_ledger(memory, glm_metadata_specs()))
    with pytest.raises(ValueError, match=match):
        memory.glm_ep_dcp_projection(conversion, ep_size=ep_size, ep_rank=ep_rank)


def test_glm_metadata_conditional_rows_keep_finite_gate_blocked_on_native_inputs(memory):
    conversion = memory.glm_bf16_conversion_projection(glm_metadata_ledger(memory, glm_metadata_specs()))
    row = memory.glm_ep_dcp_projection(conversion, ep_size=16, ep_rank=0)
    terms = [
        memory.MemoryLifetime("conditional_weights", "gpu", row["named_destination_bytes"], "load"),
        memory.MemoryLifetime("native_buffers", "gpu", row["native_buffers_bytes"], "load"),
    ]
    with pytest.raises(ValueError, match="unresolved initialization inputs"):
        memory.require_finite_initialization_plan(
            ("load",), terms, limits={"gpu": 10**12}, unresolved_inputs=row["unresolved_inputs"]
        )


def test_native_model_inventory_keeps_duplicate_paths_buffers_and_complete_view_storage(memory):
    import torch

    model = torch.nn.Module()
    base = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    model.weight = torch.nn.Parameter(base)
    model.tied = model.weight
    model.view = torch.nn.Parameter(base[1:3, 1::2], requires_grad=False)
    model.register_buffer("running", torch.zeros(3, dtype=torch.float64))
    model.register_buffer("scratch", torch.zeros(2, dtype=torch.int64), persistent=False)
    model.register_buffer("missing", None, persistent=False)
    model.register_parameter("missing_parameter", None)
    child = torch.nn.Module()
    child.shared = model.weight
    model.left = child
    model.right = child
    before = base.clone()

    observed = memory.observe_native_model_inventory(model)
    parameters = {row["name"]: row["tensor"] for row in observed["parameters"]}
    assert set(parameters) == {"weight", "tied", "view", "left.shared", "right.shared"}
    assert all(
        parameters[name]["object_id"] == parameters["weight"]["object_id"]
        for name in ("tied", "left.shared", "right.shared")
    )
    assert parameters["view"]["object_id"] != parameters["weight"]["object_id"]
    assert parameters["view"]["shape"] == [2, 2] and parameters["view"]["stride"] == [5, 2]
    assert parameters["view"]["requires_grad"] is False
    assert parameters["view"]["logical_bytes"] == 16
    assert parameters["view"]["storage"] == {
        "storage_id": parameters["weight"]["storage"]["storage_id"],
        "offset_elements": 6,
        "offset_bytes": 24,
        "nbytes": 80,
    }
    buffers = {row["name"]: row for row in observed["buffers"]}
    assert buffers["running"]["persistent"] is True
    assert buffers["scratch"]["persistent"] is False
    assert buffers["scratch"]["tensor"]["dtype"] == "torch.int64"
    assert buffers["scratch"]["tensor"]["device"] == "cpu"
    assert sorted(row["nbytes"] for row in observed["storages"]) == [16, 24, 80]
    assert {row["name"] for row in observed["empty_registrations"]} == {"missing", "missing_parameter"}
    assert json.loads(json.dumps(observed, allow_nan=False)) == observed
    assert observed["capacity_admitted"] is observed["runtime_admitted"] is False
    assert observed["snapshot_only"] is True
    assert torch.equal(base, before) and model.left is model.right


def test_native_model_inventory_meta_has_only_object_aliases_not_pointer_storage(memory, monkeypatch):
    import torch

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty((8, 4), device="meta", dtype=torch.bfloat16))
    model.tied = model.weight
    model.other = torch.nn.Parameter(torch.empty((8, 4), device="meta", dtype=torch.bfloat16))
    model.register_buffer("scratch", torch.empty((3,), device="meta"), persistent=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("meta storage/pointer/materialization was inspected")

    monkeypatch.setattr(torch.Tensor, "data_ptr", forbidden)
    monkeypatch.setattr(torch.Tensor, "untyped_storage", forbidden)
    monkeypatch.setattr(torch.nn.Module, "to_empty", forbidden)
    observed = memory.observe_native_model_inventory(model)
    rows = {row["name"]: row["tensor"] for row in observed["parameters"]}
    assert rows["weight"]["object_id"] == rows["tied"]["object_id"]
    assert rows["weight"]["object_id"] != rows["other"]["object_id"]
    assert rows["weight"]["logical_bytes"] == 64 and rows["weight"]["device"] == "meta"
    assert all(row["storage"] is None for row in rows.values())
    assert observed["buffers"][0]["tensor"]["storage"] is None
    assert observed["storages"] == []


def test_native_model_inventory_zero_storage_does_not_use_equal_null_data_pointers(memory, monkeypatch):
    import torch

    model = torch.nn.Module()
    model.register_buffer("first", torch.empty(0))
    model.register_buffer("second", torch.empty(0))

    def forbidden(*args, **kwargs):
        raise AssertionError("payload pointer access is not needed")

    monkeypatch.setattr(torch.Tensor, "data_ptr", forbidden)
    observed = memory.observe_native_model_inventory(model)
    rows = [row["tensor"] for row in observed["buffers"]]
    assert rows[0]["storage"]["storage_id"] != rows[1]["storage"]["storage_id"]
    assert [row["nbytes"] for row in observed["storages"]] == [0, 0]


def test_native_model_inventory_scalar_reports_actual_dtype_class_and_stride(memory):
    import torch

    model = torch.nn.Module()
    model.register_buffer("scalar", torch.tensor(1 + 2j, dtype=torch.complex128))
    row = memory.observe_native_model_inventory(model)["buffers"][0]["tensor"]
    assert row["shape"] == row["stride"] == []
    assert row["dtype"] == "torch.complex128" and row["element_size"] == 16
    assert row["class"] == {"module": "torch", "qualname": "Tensor"}
    assert row["storage"]["nbytes"] == row["logical_bytes"] == 16


def test_native_model_inventory_registration_walk_cannot_be_hidden_by_custom_enumerators(memory):
    import torch

    class Hidden(torch.nn.Module):
        def named_parameters(self, *args, **kwargs):
            raise AssertionError("custom enumeration cannot replace actual registrations")

        def named_buffers(self, *args, **kwargs):
            raise AssertionError("custom enumeration cannot replace actual registrations")

    model = Hidden()
    model.weight = torch.nn.Parameter(torch.zeros(1))
    model.register_buffer("scratch", torch.zeros(1), persistent=False)
    observed = memory.observe_native_model_inventory(model)
    assert observed["model_class"]["qualname"].endswith(".Hidden")
    assert observed["parameters"][0]["name"] == "weight"
    assert observed["buffers"][0]["name"] == "scratch" and observed["buffers"][0]["persistent"] is False


@pytest.mark.parametrize("kind", ["subclass", "sparse", "quantized"])
def test_native_model_inventory_rejects_unsupported_tensors_before_reporting(memory, kind):
    import torch

    model = torch.nn.Module()
    if kind == "subclass":

        class Unsupported(torch.Tensor):
            pass

        tensor = torch.Tensor._make_subclass(Unsupported, torch.zeros(2), False)
    elif kind == "sparse":
        tensor = torch.sparse_coo_tensor([[0]], [1.0], (2,))
    else:
        tensor = torch.quantize_per_tensor(torch.zeros(2), scale=1.0, zero_point=0, dtype=torch.qint8)
    model.register_buffer("unsupported", tensor)
    with pytest.raises(ValueError, match="unsupported native tensor"):
        memory.observe_native_model_inventory(model)


@pytest.mark.parametrize("kind", ["not_module", "module_cycle"])
def test_native_model_inventory_rejects_invalid_registration_graph(memory, kind):
    import torch

    model = object() if kind == "not_module" else torch.nn.Module()
    if kind == "module_cycle":
        model.loop = model
    with pytest.raises(ValueError, match="module"):
        memory.observe_native_model_inventory(model)


def _native_inventory_dtensor(torch, placement, *, meta=False):
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    from torch.distributed.tensor._dtensor_spec import TensorMeta

    # Actual Torch wrapper/spec with synthetic membership, not a process group.
    mesh = object.__new__(DeviceMesh)
    mesh.device_type = "cpu"
    mesh.mesh = torch.tensor([4, 9], dtype=torch.int64)
    mesh.mesh_dim_names = ("ep",)
    local = torch.empty((2, 3), device="meta" if meta else "cpu")
    spec = DTensorSpec(mesh, (placement,), tensor_meta=TensorMeta(torch.Size((4, 3)), (3, 1), torch.float32))
    return DTensor(local, spec, requires_grad=False)


@pytest.mark.parametrize("meta", [False, True])
def test_native_model_inventory_dtensor_preserves_global_local_and_actual_mesh_without_collectives(
    memory, monkeypatch, meta
):
    import torch
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor import Shard

    tensor = _native_inventory_dtensor(torch, Shard(0), meta=meta)
    model = torch.nn.Module()
    model.register_buffer("distributed", tensor, persistent=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("native inventory must not distribute, gather or query global rank")

    for name in ("all_gather", "get_rank", "init_process_group"):
        monkeypatch.setattr(torch.distributed, name, forbidden)
    for name in ("full_tensor", "redistribute", "to_local"):
        monkeypatch.setattr(DTensor, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "data_ptr", forbidden)
    observed = memory.observe_native_model_inventory(model)
    row = observed["buffers"][0]["tensor"]
    assert row["shape"] == [4, 3] and row["stride"] == [3, 1] and row["logical_bytes"] == 48
    assert row["storage"] is None
    actual = row["dtensor"]
    assert actual["mesh"] == {
        "object_id": f"mesh:{id(tensor.device_mesh):x}",
        "members_object_id": f"tensor:{id(tensor.device_mesh.mesh):x}",
        "device_type": "cpu",
        "shape": [2],
        "members": [4, 9],
        "dim_names": ["ep"],
    }
    assert actual["placements"] == [
        {"object_id": f"placement:{id(tensor.placements[0]):x}", "kind": "Shard", "dim": 0}
    ]
    assert actual["local"]["shape"] == [2, 3] and actual["local"]["logical_bytes"] == 24
    if meta:
        assert actual["local"]["storage"] is None and observed["storages"] == []
    else:
        assert actual["local"]["storage"]["nbytes"] == 24
    assert json.loads(json.dumps(observed, allow_nan=False)) == observed


def test_native_model_inventory_rejects_dtensor_placement_subclasses(memory):
    import torch
    from torch.distributed.tensor import Shard

    class UnsupportedShard(Shard):
        pass

    model = torch.nn.Module()
    model.register_buffer("distributed", _native_inventory_dtensor(torch, UnsupportedShard(0)))
    with pytest.raises(ValueError, match="unsupported native DTensor placement"):
        memory.observe_native_model_inventory(model)


def test_native_model_inventory_stable_snapshots_retain_real_module_tensor_and_storage_identities(memory):
    import torch

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(3))
    model.empty = torch.nn.Module()
    retained = []
    first = memory.observe_native_model_inventory(model, retained_objects=retained)
    count = len(retained)
    assert memory.observe_native_model_inventory(model, retained_objects=retained) == first
    assert len(retained) == count
    assert first["modules"] == [
        {"name": "", "object_id": f"module:{id(model):x}", "class": first["model_class"]},
        {"name": "empty", "object_id": f"module:{id(model.empty):x}", "class": first["model_class"]},
    ]
    parameter = first["parameters"][0]
    assert parameter["owner_id"] == f"module:{id(model):x}"
    assert parameter["tensor"]["object_id"] == f"tensor:{id(model.weight):x}"
    assert parameter["tensor"]["storage"]["storage_id"] == f"storage:{model.weight.untyped_storage()._cdata:x}"
    assert json.loads(json.dumps(first, allow_nan=False)) == first


@pytest.mark.parametrize("kind", ["tensor", "backing_storage", "empty_module"])
def test_native_model_inventory_detects_same_metadata_replacements_across_snapshots(memory, kind):
    import torch

    model = torch.nn.Module()
    model.register_buffer("value", torch.zeros(3))
    model.empty = torch.nn.Module()
    retained = []
    first = memory.observe_native_model_inventory(model, retained_objects=retained)
    if kind == "tensor":
        model.value = torch.zeros(3)
    elif kind == "backing_storage":
        model.value.set_(torch.zeros(3))
    else:
        model.empty = torch.nn.Module()
    second = memory.observe_native_model_inventory(model, retained_objects=retained)
    assert first != second
    left, right = (snapshot["buffers"][0]["tensor"] for snapshot in (first, second))
    assert all(left[key] == right[key] for key in ("dtype", "shape", "stride", "device", "logical_bytes"))
    if kind == "tensor":
        assert left["object_id"] != right["object_id"]
    elif kind == "backing_storage":
        assert left["object_id"] == right["object_id"]
        assert left["storage"]["storage_id"] != right["storage"]["storage_id"]
    else:
        assert left == right
        assert first["modules"][1]["object_id"] != second["modules"][1]["object_id"]


def test_native_model_inventory_caller_retention_prevents_deleted_object_identity_reuse(memory):
    import gc
    import weakref

    import torch

    model = torch.nn.Module()
    model.register_buffer("value", torch.zeros(3))
    model.empty = torch.nn.Module()
    old_tensor = weakref.ref(model.value)
    storage = model.value.untyped_storage()
    old_storage = weakref.ref(storage)
    old_module = weakref.ref(model.empty)
    retained = []
    memory.observe_native_model_inventory(model, retained_objects=retained)
    del storage
    model.value = torch.zeros(3)
    model.empty = torch.nn.Module()
    gc.collect()
    assert all(reference() is not None for reference in (old_tensor, old_storage, old_module))
    retained.clear()
    gc.collect()
    assert all(reference() is None for reference in (old_tensor, old_storage, old_module))


def test_native_model_inventory_rejects_non_list_retention_container(memory):
    import torch

    with pytest.raises(ValueError, match="ordinary retained_objects list"):
        memory.observe_native_model_inventory(torch.nn.Module(), retained_objects=())


@pytest.mark.parametrize("kind", ["local_tensor", "mesh"])
def test_native_model_inventory_detects_dtensor_dependency_replacement(memory, kind):
    import torch
    from torch.distributed.tensor import Shard

    tensor = _native_inventory_dtensor(torch, Shard(0))
    model = torch.nn.Module()
    model.register_buffer("distributed", tensor)
    retained = []
    first = memory.observe_native_model_inventory(model, retained_objects=retained)
    if kind == "local_tensor":
        tensor._local_tensor = torch.empty((2, 3))
    else:
        replacement = _native_inventory_dtensor(torch, Shard(0))
        tensor._spec.mesh = replacement.device_mesh
    second = memory.observe_native_model_inventory(model, retained_objects=retained)
    assert first != second
    left, right = (snapshot["buffers"][0]["tensor"] for snapshot in (first, second))
    assert left["object_id"] == right["object_id"]
    assert left["shape"] == right["shape"]
    if kind == "local_tensor":
        assert left["dtensor"]["local"]["object_id"] != right["dtensor"]["local"]["object_id"]
    else:
        assert left["dtensor"]["mesh"]["object_id"] != right["dtensor"]["mesh"]["object_id"]
        assert left["dtensor"]["mesh"]["members"] == right["dtensor"]["mesh"]["members"]


def test_native_inventory_checkpoint_destinations_join_detached_views_to_full_model_storage(memory):
    import torch

    model = torch.nn.Module()
    base = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    model.weight = torch.nn.Parameter(base)
    model.register_buffer("scratch", base[1:3, 1::2], persistent=False)
    selected = {"renamed.weight": model.weight.detach(), "scratch": model.scratch.detach()}
    retained = []
    observed = memory.observe_native_model_inventory(
        model, retained_objects=retained, checkpoint_destinations=selected
    )
    weight = observed["parameters"][0]["tensor"]
    scratch = observed["buffers"][0]["tensor"]
    destinations = observed["checkpoint_destinations"]
    assert destinations["renamed.weight"]["object_id"] != weight["object_id"]
    assert destinations["renamed.weight"]["requires_grad"] is False
    assert destinations["renamed.weight"]["storage"] == weight["storage"]
    assert destinations["scratch"]["object_id"] != scratch["object_id"]
    assert destinations["scratch"]["storage"] == scratch["storage"]
    assert destinations["scratch"]["shape"] == [2, 2]
    assert destinations["scratch"]["stride"] == [5, 2]
    assert destinations["scratch"]["storage"]["offset_elements"] == 6
    assert len(observed["storages"]) == 1 and observed["storages"][0]["nbytes"] == 80
    assert any(value is selected for value in retained)
    assert all(any(value is tensor for value in retained) for tensor in selected.values())
    assert (
        memory.observe_native_model_inventory(model, retained_objects=retained, checkpoint_destinations=selected)
        == observed
    )
    assert json.loads(json.dumps(observed, allow_nan=False)) == observed
    assert torch.equal(base, torch.arange(20, dtype=torch.float32).reshape(4, 5))
    assert observed["capacity_admitted"] is observed["runtime_admitted"] is False


def test_native_inventory_checkpoint_destinations_do_not_join_unrelated_equal_shape_storage(memory):
    import torch

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(3))
    selected = {"weight": torch.zeros(3)}
    observed = memory.observe_native_model_inventory(model, checkpoint_destinations=selected)
    weight = observed["parameters"][0]["tensor"]
    destination = observed["checkpoint_destinations"]["weight"]
    assert weight["shape"] == destination["shape"]
    assert weight["storage"]["storage_id"] != destination["storage"]["storage_id"]
    assert len(observed["storages"]) == 2 and sum(row["nbytes"] for row in observed["storages"]) == 24


def test_native_inventory_checkpoint_destinations_dtensor_detached_local_joins_actual_storage(memory, monkeypatch):
    import torch
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor import Shard

    tensor = _native_inventory_dtensor(torch, Shard(0))
    # A real wrapper over a detached local view, not a native mesh dispatch proof.
    detached = DTensor(tensor._local_tensor.detach(), tensor._spec, requires_grad=False)
    model = torch.nn.Module()
    model.register_buffer("distributed", tensor)

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint destination census must not gather or materialize")

    for name in ("full_tensor", "redistribute", "to_local"):
        monkeypatch.setattr(DTensor, name, forbidden)
    for name in ("all_gather", "get_rank", "init_process_group"):
        monkeypatch.setattr(torch.distributed, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "data_ptr", forbidden)
    observed = memory.observe_native_model_inventory(model, checkpoint_destinations={"distributed": detached})
    registered = observed["buffers"][0]["tensor"]
    destination = observed["checkpoint_destinations"]["distributed"]
    assert registered["object_id"] != destination["object_id"]
    assert registered["shape"] == destination["shape"] == [4, 3]
    assert registered["dtensor"]["local"]["storage"] == destination["dtensor"]["local"]["storage"]
    assert registered["dtensor"]["mesh"] == destination["dtensor"]["mesh"]
    assert len(observed["storages"]) == 1 and observed["storages"][0]["nbytes"] == 24


@pytest.mark.parametrize("kind", ["not_dict", "invalid_name", "not_tensor"])
def test_native_inventory_checkpoint_destinations_reject_unsupported_input(memory, kind):
    import torch

    selected = () if kind == "not_dict" else {0: torch.zeros(1)} if kind == "invalid_name" else {"x": None}
    with pytest.raises(ValueError, match="checkpoint_destinations|checkpoint destination|unsupported native tensor"):
        memory.observe_native_model_inventory(torch.nn.Module(), checkpoint_destinations=selected)


def test_native_inventory_checkpoint_destinations_reject_mapping_drift_during_observation(memory, monkeypatch):
    import torch

    model = torch.nn.Module()
    destination = torch.zeros(3)
    selected = {"weight": destination}
    original = torch.Tensor.untyped_storage

    def changed(tensor):
        if tensor is destination:
            selected["weight"] = torch.zeros(3)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "untyped_storage", changed)
    with pytest.raises(ValueError, match="checkpoint destinations changed during inventory"):
        memory.observe_native_model_inventory(model, checkpoint_destinations=selected)


def test_native_inventory_checkpoint_destinations_absent_and_empty_preserve_original_model_result(memory):
    import torch

    model = torch.nn.Module()
    model.register_buffer("scratch", torch.zeros(1), persistent=False)
    retained = []
    before = memory.observe_native_model_inventory(model, retained_objects=retained)
    assert (
        memory.observe_native_model_inventory(model, retained_objects=retained, checkpoint_destinations=None) == before
    )
    empty = memory.observe_native_model_inventory(model, retained_objects=retained, checkpoint_destinations={})
    assert empty.pop("checkpoint_destinations") == {}
    assert empty == before


def _reader_callback_inputs(memory):
    def tensor(shape, stride, width, device, storage_id, nbytes):
        return {
            "shape": shape,
            "stride": stride,
            "element_size": width,
            "logical_bytes": memory.tensor_storage_bytes(shape, width),
            "device": device,
            "dtensor": None,
            "storage": {
                "storage_id": storage_id,
                "nbytes": nbytes,
                "offset_elements": 0,
                "offset_bytes": 0,
            },
        }

    weight = tensor([2, 16], [16, 1], 2, "cuda:0", "storage:1", 128)
    scratch = tensor([10], [1], 4, "cpu", "storage:2", 40)
    inventory = {
        "alias_identity_scope": "process-local",
        "snapshot_only": True,
        "parameters": [{"name": "weight", "tensor": weight}, {"name": "alias", "tensor": deepcopy(weight)}],
        "buffers": [{"name": "scratch", "persistent": False, "tensor": scratch}],
        "storages": [
            {"storage_id": "storage:1", "device": "cuda:0", "nbytes": 128},
            {"storage_id": "storage:2", "device": "cpu", "nbytes": 40},
        ],
        "checkpoint_destinations": {"selected.weight": deepcopy(weight)},
    }
    index = namedtuple("ReaderCallbackIndex", "fqn offset index")
    source = index("packed_experts", (4, 0), 3)
    item = SimpleNamespace(
        type=SimpleNamespace(name="TENSOR"),
        storage_index=source,
        dest_index=index("selected.weight", (0, 0), None),
        storage_offsets=(3, 0),
        dest_offsets=(0, 0),
        lengths=(2, 16),
    )
    args = load_inputs(memory)
    args.update(
        requests=None,
        destination_layouts=None,
        native_inventory=inventory,
        storage_scope="host-boot:pid-123:startticks-456",
        storage_resources={"cpu": "host", "cuda:0": "gpu0"},
        read_plan=SimpleNamespace(items=[item]),
        reader_storage={source: SimpleNamespace(relative_path="/cache/cache")},
        cache_path="/cache",
    )
    return args


def test_reader_callback_arithmetic_keeps_full_backing_and_separate_source_slice(memory):
    args = _reader_callback_inputs(memory)
    before = deepcopy(args["native_inventory"])
    row = memory.dcp_load_phase_terms(**args)
    terms = sizes(row)
    assert "rank0.destination" not in terms
    assert terms["rank0.backing_storage.0"] == 128
    assert terms["rank0.backing_storage.1"] == 40
    assert terms["rank0.read_slice_payload"] == 64
    request = row["destination_request_rows"][0]
    assert request["source_index_offset"] == (4, 0)
    assert request["source_index_hint"] == 3
    assert request["source_offsets"] == (3, 0)
    assert request["destination_index_offset"] == request["destination_offsets"] == (0, 0)
    assert request["destination_storage"]["nbytes"] == 128
    assert request["destination_resolution_bound"] is False
    assert row["storage_scope_is_owner_binding"] is row["capacity_admitted"] is row["runtime_admitted"] is False
    assert args["native_inventory"] == before
    backing = row["terms"][:2]
    assert all(term.last_phase is None for term in backing)
    assert backing[0].storage_id == "host-boot:pid-123:startticks-456:cuda:0:storage:1"
    assert {
        "actual_storage_owner_and_device_resource_binding",
        "complete_constructor_and_conversion_lifetimes",
        "selected_planner_destination_resolution",
    } <= set(row["unresolved_inputs"])
    assert terms["rank0.planner_gpu_staging"] is terms["rank0.mapped_backend_residency"] is None


def test_reader_callback_arithmetic_uses_dtensor_local_without_index_offset_addition(memory):
    args = _reader_callback_inputs(memory)
    local = args["native_inventory"]["checkpoint_destinations"]["selected.weight"]
    outer = deepcopy(local)
    outer.update(
        shape=[8, 16],
        logical_bytes=256,
        storage=None,
        dtensor={"local": local, "placements": [{"kind": "Shard", "dim": 0}], "mesh": {"members": [0, 1, 2, 3]}},
    )
    args["native_inventory"]["checkpoint_destinations"]["selected.weight"] = outer
    item = args["read_plan"].items[0]
    item.dest_index = item.dest_index._replace(offset=(6, 0), index=2)
    row = memory.dcp_load_phase_terms(**args)
    request = row["destination_request_rows"][0]
    assert request["destination_shape"] == (2, 16)
    assert request["destination_index_offset"] == (6, 0)
    assert request["destination_offsets"] == (0, 0)
    assert request["destination_dtensor"]["mesh"]["members"] == [0, 1, 2, 3]
    assert sizes(row)["rank0.backing_storage.0"] == 128


def test_reader_callback_arithmetic_scopes_process_local_aliases_separately(memory):
    args = _reader_callback_inputs(memory)
    first = memory.dcp_load_phase_terms(**args)
    args.update(name="rank1", storage_scope="host-boot:pid-124:startticks-456")
    second = memory.dcp_load_phase_terms(**args)
    terms = [term for row in (first, second) for term in row["terms"] if ".backing_storage." in term.name]
    row = memory.initialization_phase_plan(
        ("load", "serve"), terms, limits=args["limits"], unresolved_inputs=first["unresolved_inputs"]
    )
    assert row["resources"]["host"]["known_peak_bytes"] == 80
    assert row["resources"]["gpu0"]["known_peak_bytes"] == 256
    assert row["capacity_admitted"] is False


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_destination",
        "duplicate_requests",
        "missing_scope",
        "missing_resource",
        "meta_storage",
        "duplicate_storage",
        "unreferenced_storage",
        "storage_size_drift",
        "view_outside_storage",
        "logical_size_drift",
        "destination_outside_local",
        "ordinary_nonzero_index",
        "non_tensor_read",
        "wrong_file",
        "outside_cache",
        "parent_path",
        "missing_source_header",
        "missing_destination",
    ],
)
def test_reader_callback_arithmetic_rejects_inconsistent_metadata(memory, change):
    args = _reader_callback_inputs(memory)
    inventory, item = args["native_inventory"], args["read_plan"].items[0]
    if change == "duplicate_destination":
        args["destination_layouts"] = {"weight": ((2, 16), 2)}
    elif change == "duplicate_requests":
        args["requests"] = [memory.DCPReadRequest("packed_experts", (3, 0), (2, 16))]
    elif change == "missing_scope":
        args["storage_scope"] = None
    elif change == "missing_resource":
        del args["storage_resources"]["cpu"]
    elif change == "meta_storage":
        inventory["parameters"][0]["tensor"]["storage"] = None
    elif change == "duplicate_storage":
        inventory["storages"].append(deepcopy(inventory["storages"][0]))
    elif change == "unreferenced_storage":
        inventory["storages"].append({"storage_id": "storage:3", "device": "cpu", "nbytes": 40})
    elif change == "storage_size_drift":
        inventory["storages"][0]["nbytes"] += 2
    elif change == "view_outside_storage":
        inventory["parameters"][0]["tensor"]["stride"][0] = 100
    elif change == "logical_size_drift":
        inventory["parameters"][0]["tensor"]["logical_bytes"] += 2
    elif change == "destination_outside_local":
        item.dest_offsets = (1, 0)
    elif change == "ordinary_nonzero_index":
        item.dest_index = item.dest_index._replace(offset=(1, 0))
    elif change == "non_tensor_read":
        item.type.name = "BYTE_IO"
    elif change == "wrong_file":
        args["reader_storage"][item.storage_index].relative_path = "/cache/other"
    elif change == "outside_cache":
        args["reader_storage"][item.storage_index].relative_path = "/other/cache"
    elif change == "parent_path":
        args["reader_storage"][item.storage_index].relative_path = "../cache"
    elif change == "missing_source_header":
        args["load_headers"] = {"other": args["load_headers"]["packed_experts"]}
    elif change == "missing_destination":
        item.dest_index = item.dest_index._replace(fqn="missing")
    with pytest.raises(ValueError):
        memory.dcp_load_phase_terms(**args)


def test_reader_callback_arithmetic_preserves_unknown_constructor_cache_limits(memory):
    args = _reader_callback_inputs(memory)
    args.update(cache_miss=None, conversion_writer=None, limits=None)
    row = memory.dcp_load_phase_terms(**args)
    terms = sizes(row)
    assert terms["rank0.conversion_source"] is terms["rank0.conversion_pack"] is None
    assert terms["rank0.save_contiguous_copies"] is terms["rank0.save_native"] is None
    assert "actual_usable_resource_limits" in row["unresolved_inputs"]
    assert "complete_constructor_and_conversion_lifetimes" in row["unresolved_inputs"]
