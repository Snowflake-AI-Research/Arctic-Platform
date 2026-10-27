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

"""CPU controls for opt-in sampler evidence; these are not GPU comparisons."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.debug import broadcast_weight_sync_evidence as broadcast
from arctic_platform.debug import rl_sampler_evidence as sampler
from arctic_platform.debug import weight_sync_evidence as evidence


class _PackedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.packed = torch.nn.Parameter(torch.zeros(4))
        self.other = torch.nn.Parameter(torch.zeros(1))

    def load_weights(self, weights):
        for name, value in weights:
            if name == "left":
                self.packed.data[:2].copy_(value)
            elif name == "right":
                self.packed.data[2:].copy_(value)
            elif name == "other":
                self.other.data.copy_(value)
        return None


class _Sender:
    def __init__(self):
        self.seen = None

    def send(self, weights):
        self.seen = weights
        return {"status": "done", "native_receipt": 17}


class _Worker:
    def __init__(self):
        self.engine = SimpleNamespace(module=torch.nn.Linear(2, 1, bias=False))
        self._weight_sender = _Sender()

    def get_weights(self):
        return [(name, value.data) for name, value in self.engine.module.named_parameters()]


def _verifier(model):
    return evidence.ShadowVerifier(model, max_shadow_bytes=1024)


def test_native_broadcast_training_configuration_disabled_is_unchanged(monkeypatch):
    monkeypatch.setenv(evidence.VERIFY_ENV, "previous")
    ds_config = {"bf16": {"enabled": True}}
    assert broadcast.maybe_configure_broadcast_training(ds_config) is ds_config
    assert broadcast.os.environ[evidence.VERIFY_ENV] == "previous"


@pytest.mark.parametrize("option", [True, False])
def test_native_broadcast_training_configuration_consumes_only_its_option(monkeypatch, option):
    monkeypatch.setenv(evidence.VERIFY_ENV, "previous")
    ds_config = {"bf16": {"enabled": True}, broadcast.TRAINING_OPTION: option}
    before = copy.deepcopy(ds_config)
    result = broadcast.maybe_configure_broadcast_training(ds_config)
    assert result == {"bf16": {"enabled": True}}
    assert ds_config == before
    assert broadcast.os.environ[evidence.VERIFY_ENV] == ("1" if option else "previous")


@pytest.mark.parametrize("option", ["1", 1, None, {}])
def test_native_broadcast_training_configuration_rejects_nonboolean_opt_in(monkeypatch, option):
    monkeypatch.delenv(evidence.VERIFY_ENV, raising=False)
    with pytest.raises(ValueError, match="must be a boolean"):
        broadcast.maybe_configure_broadcast_training({broadcast.TRAINING_OPTION: option})
    assert evidence.VERIFY_ENV not in broadcast.os.environ


def test_native_broadcast_sampling_disabled_does_not_inspect_settings():
    sampling = object()
    assert broadcast.maybe_configure_broadcast_sampling(sampling, verify=False) is sampling


@pytest.mark.parametrize("extra_env", [None, {}, {evidence.VERIFY_ENV: "1", "OTHER_SETTING": "kept"}])
def test_native_broadcast_sampling_configuration_preserves_original(extra_env):
    from arctic_platform.client.config import SamplingConfig

    sampling = SamplingConfig(vllm={"extra_env": extra_env})
    before = sampling.model_dump()
    result = broadcast.maybe_configure_broadcast_sampling(sampling, verify=True)
    assert result.vllm["extra_env"] == {**(extra_env or {}), evidence.VERIFY_ENV: "1"}
    assert sampling.model_dump() == before
    assert result is not sampling


@pytest.mark.parametrize("extra_env", ["invalid", [], {evidence.VERIFY_ENV: "0"}])
def test_native_broadcast_sampling_rejects_invalid_or_conflicting_settings(extra_env):
    sampling = SimpleNamespace(vllm={"extra_env": extra_env})
    with pytest.raises(ValueError, match="extra_env must be an object|conflicting"):
        broadcast.maybe_configure_broadcast_sampling(sampling, verify=True)


@pytest.mark.parametrize(
    ("configured", "inherited", "expected"),
    [
        (None, None, None),
        (None, str(16 << 30), str(16 << 30)),
        (str(8 << 30), None, str(8 << 30)),
        ("01024", "1024", "1024"),
    ],
)
def test_native_broadcast_sampling_propagates_explicit_shadow_limit(monkeypatch, configured, inherited, expected):
    from arctic_platform.client.config import SamplingConfig

    key = broadcast.SHADOW_MAX_BYTES_ENV
    if inherited is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, inherited)
    extra_env = {"OTHER_SETTING": "kept"}
    if configured is not None:
        extra_env[key] = configured
    sampling = SamplingConfig(vllm={"extra_env": extra_env})
    before = sampling.model_dump()
    result = broadcast.maybe_configure_broadcast_sampling(sampling, verify=True)
    assert result.vllm["extra_env"].get(key) == expected
    assert result.vllm["extra_env"]["OTHER_SETTING"] == "kept"
    assert sampling.model_dump() == before


@pytest.mark.parametrize("invalid", ["0", "-1", "", "unlimited", "1.5", 1024, True])
def test_native_broadcast_sampling_rejects_invalid_shadow_limit(monkeypatch, invalid):
    key = broadcast.SHADOW_MAX_BYTES_ENV
    monkeypatch.delenv(key, raising=False)
    sampling = SimpleNamespace(vllm={"extra_env": {key: invalid}})
    with pytest.raises(ValueError, match="positive integer byte limit"):
        broadcast.maybe_configure_broadcast_sampling(sampling, verify=True)


def test_native_broadcast_sampling_rejects_conflicting_shadow_limits(monkeypatch):
    key = broadcast.SHADOW_MAX_BYTES_ENV
    monkeypatch.setenv(key, str(16 << 30))
    sampling = SimpleNamespace(vllm={"extra_env": {key: str(8 << 30)}})
    with pytest.raises(ValueError, match="conflicting weight-sync shadow byte limits"):
        broadcast.maybe_configure_broadcast_sampling(sampling, verify=True)


def test_native_broadcast_sampling_rejects_invalid_parent_shadow_limit(monkeypatch):
    monkeypatch.setenv(broadcast.SHADOW_MAX_BYTES_ENV, "unlimited")
    sampling = SimpleNamespace(vllm={})
    with pytest.raises(ValueError, match="positive integer byte limit"):
        broadcast.maybe_configure_broadcast_sampling(sampling, verify=True)


@pytest.mark.parametrize("option", [None, False, True])
def test_native_broadcast_training_engine_consumes_opt_in_before_deepspeed(monkeypatch, option):
    from arctic_platform.common import deepspeed_engine

    monkeypatch.delenv(evidence.VERIFY_ENV, raising=False)
    ds_config = {"bf16": {"enabled": True}}
    if option is not None:
        ds_config[broadcast.TRAINING_OPTION] = option
    before = copy.deepcopy(ds_config)
    seen = []

    def initialize(**kwargs):
        seen.append((kwargs["config"], evidence.enabled()))
        return object(), None, None, None

    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(initialize=initialize))
    monkeypatch.setattr(deepspeed_engine, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))
    engine, optimizer, choice = deepspeed_engine.initialize_training_engine(
        object(), ds_config, optimizer=None, ep_size=1, mpu=None, model_path="unused", rank=0
    )
    assert seen == [({"bf16": {"enabled": True}}, option is True)]
    assert engine is not None and optimizer is None and choice is None
    assert ds_config == before


@pytest.mark.parametrize("previous", [None, "original", "1"])
@pytest.mark.parametrize("fail", [False, True])
def test_native_broadcast_parent_environment_is_restored(monkeypatch, previous, fail):
    if previous is None:
        monkeypatch.delenv(evidence.VERIFY_ENV, raising=False)
    else:
        monkeypatch.setenv(evidence.VERIFY_ENV, previous)
    try:
        with broadcast.weight_sync_verification_environment():
            assert evidence.enabled()
            if fail:
                raise RuntimeError("synthetic sync failure")
    except RuntimeError:
        assert fail
    assert broadcast.os.environ.get(evidence.VERIFY_ENV) == previous


def test_sampler_evidence_manifest_preserves_bf16_bytes_and_geometry():
    value = torch.tensor([1.0, -0.0], dtype=torch.bfloat16)
    record = evidence.tensor_manifest(value)
    assert record["shape"] == [2]
    assert record["dtype"] == "bfloat16"
    assert record["numel"] == 2
    assert len(record["sha256"]) == 64
    assert record["sha256"] != evidence.tensor_manifest(torch.tensor([1.0, 0.0], dtype=torch.bfloat16))["sha256"]


def test_sampler_evidence_sender_passes_exact_existing_list_once(monkeypatch):
    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    worker = _Worker()
    weights = worker.get_weights()
    result = evidence.maybe_send_weights(worker, weights)
    assert worker._weight_sender.seen is weights
    assert result["native_receipt"] == 17
    assert result["trainer_parameters"] == {"numel": 2, "count": 1}
    assert result["sent_manifest"] == {"weight": evidence.tensor_manifest(weights[0][1])}
    assert result["evidence_sources"]


def test_sampler_evidence_disabled_sender_and_pool_are_unchanged(monkeypatch):
    monkeypatch.delenv(evidence.VERIFY_ENV, raising=False)
    worker = _Worker()
    weights = worker.get_weights()
    assert evidence.maybe_send_weights(worker, weights) == {"status": "done", "native_receipt": 17}
    pool_class = object()
    assert sampler.maybe_replica_pool(pool_class) is pool_class


@pytest.mark.parametrize("unsupported", ["zero3", "expert"])
def test_sampler_evidence_incomplete_sender_rejected_before_send(monkeypatch, unsupported):
    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    worker = _Worker()
    parameter = worker.engine.module.weight
    if unsupported == "zero3":
        parameter.ds_id = 0
    else:
        parameter.allreduce = False
    with pytest.raises(ValueError):
        evidence.maybe_send_weights(worker, worker.get_weights())
    assert worker._weight_sender.seen is None


def test_sampler_evidence_shadow_complete_fused_slices_keep_live_objects():
    model = _PackedModel()
    identities = {name: id(value) for name, value in model.named_parameters()}
    verifier = _verifier(model)
    verifier.load("left", torch.tensor([1.0, 2.0]))
    verifier.load("right", torch.tensor([3.0, 4.0]))
    verifier.load("other", torch.tensor([5.0]))
    result = verifier.finish()
    assert result["received"]["left"]["destinations"] == ["packed"]
    assert result["received"]["right"]["destinations"] == ["packed"]
    assert result["destinations"] == {
        "packed": {"uncovered": 0, "mismatched": 0},
        "other": {"uncovered": 0, "mismatched": 0},
    }
    assert result["untouched_parameters"] == {}
    assert model.packed.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert {name: id(value) for name, value in model.named_parameters()} == identities


def test_sampler_evidence_shadow_view_narrow_writes_have_actual_destinations():
    class ViewModel(_PackedModel):
        def load_weights(self, weights):
            for name, value in weights:
                self.packed.data.view(2, 2).narrow(0, 1, 1).copy_(value.reshape(1, 2))

    model = ViewModel()
    verifier = _verifier(model)
    verifier.load("tail", torch.tensor([3.0, 4.0]))
    result = verifier.finish()
    assert result["received"]["tail"]["destinations"] == ["packed"]
    assert result["destinations"]["packed"] == {"uncovered": 2, "mismatched": 0}
    assert model.packed.tolist() == [0.0, 0.0, 3.0, 4.0]


def test_sampler_evidence_shadow_partial_dropped_and_untouched_are_not_erased():
    verifier = _verifier(_PackedModel())
    verifier.load("left", torch.tensor([1.0, 2.0]))
    verifier.load("ignored", torch.tensor([9.0]))
    result = verifier.finish()
    assert result["destinations"]["packed"] == {"uncovered": 2, "mismatched": 0}
    assert result["received"]["ignored"]["destinations"] == []
    assert result["untouched_parameters"] == {"other": 1}


def test_sampler_evidence_shadow_byte_mismatch_is_reported():
    model = _PackedModel()
    verifier = _verifier(model)
    verifier.load("left", torch.tensor([1.0, 2.0]))
    with torch.no_grad():
        model.packed[0] = 7.0
    result = verifier.finish()
    assert result["destinations"]["packed"]["mismatched"] == 1


def test_sampler_evidence_shadow_duplicate_wire_names_fail():
    verifier = _verifier(_PackedModel())
    verifier.load("left", torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="unique"):
        verifier.load("left", torch.tensor([3.0, 4.0]))


def test_sampler_evidence_shadow_budget_checked_before_allocation():
    with pytest.raises(ValueError, match="explicit"):
        evidence.ShadowVerifier(_PackedModel(), max_shadow_bytes=1)


def test_sampler_evidence_shadow_shared_storage_not_inferred():
    model = _PackedModel()
    model.extra = torch.nn.Parameter(model.packed.data)
    with pytest.raises(ValueError, match="shared"):
        _verifier(model)


def test_sampler_evidence_shadow_exception_never_exposes_nan_to_live_model():
    class FailingModel(_PackedModel):
        def load_weights(self, weights):
            super().load_weights(weights)
            raise RuntimeError("fixture loader failed")

    model = FailingModel()
    verifier = _verifier(model)
    with pytest.raises(RuntimeError, match="fixture loader"):
        verifier.load("left", torch.tensor([1.0, 2.0]))
    assert model.packed.tolist() == [1.0, 2.0, 0.0, 0.0]
    assert not torch.isnan(model.packed).any()
    assert verifier.received == {}


def test_sampler_evidence_shadow_buffer_mutation_rejected_before_write():
    class BufferModel(_PackedModel):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.zeros(1))

        def load_weights(self, weights):
            self.buffer.copy_(weights[0][1][:1])

    model = BufferModel()
    with pytest.raises(ValueError, match="buffer"):
        _verifier(model).load("left", torch.tensor([1.0, 2.0]))
    assert model.buffer.item() == 0


def test_sampler_evidence_shadow_selected_source_drift_prevents_load(monkeypatch):
    model = _PackedModel()
    verifier = _verifier(model)
    verifier.sources = {__file__: "0" * 64}
    with pytest.raises(RuntimeError, match="source changed"):
        verifier.load("left", torch.tensor([1.0, 2.0]))
    assert model.packed.tolist() == [0.0] * 4


def test_sampler_evidence_selected_callback_drift_is_not_overwritten():
    model = _PackedModel()
    verifier = _verifier(model)
    verifier.sources[__file__] = "0" * 64
    with pytest.raises(RuntimeError, match="source changed"):
        verifier.load("left", torch.ones(2), loader=lambda name, value: model.load_weights([(name, value)]))
    assert model.packed.tolist() == [0.0] * 4


def test_sampler_evidence_shadow_parameter_replacement_rejected():
    model = _PackedModel()
    verifier = _verifier(model)
    model.packed = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(RuntimeError, match="identities"):
        verifier.finish()


def test_sampler_evidence_shadow_custom_native_operation_fails_before_mutation():
    library = torch.library.Library("sampler_evidence_fixture", "DEF")
    library.define("mutate(Tensor(a!) destination, Tensor value) -> ()")
    library.impl("mutate", lambda destination, value: destination.copy_(value), "CPU")

    class CustomModel(_PackedModel):
        def load_weights(self, weights):
            torch.ops.sampler_evidence_fixture.mutate(self.packed.data[:2], weights[0][1])

    model = CustomModel()
    with pytest.raises(ValueError, match="unobservable native"):
        _verifier(model).load("left", torch.tensor([1.0, 2.0]))
    assert model.packed.tolist() == [0.0] * 4


class _LLM:
    def __init__(self, ranks):
        self.ranks = ranks
        self.calls = []

    async def collective_rpc(self, method, *, args=(), kwargs=None):
        self.calls.append((method, args, kwargs))
        return self.ranks


def _sampler(ranks):
    worker = sampler.SamplerEvidenceWorker()
    worker.state = "ready"
    worker.llm = _LLM(ranks)
    worker._tp = 2
    worker._origins = evidence.source_files(type(worker))
    return worker


def test_sampler_evidence_collective_preserves_every_actual_rank():
    ranks = [{"status": "done", "weight_sync_verification": {"status": "ok", "rank": rank}} for rank in range(2)]
    worker = _sampler(ranks)
    result = asyncio.run(worker.sync_weights("same-allocation-host", 30000, 1, 2))
    assert result["rank_results"] is ranks
    assert result["all_rank_weight_sync_verification"] == [row["weight_sync_verification"] for row in ranks]
    assert len(worker.llm.calls) == 1


@pytest.mark.parametrize(
    "ranks",
    [
        [{"status": "done", "weight_sync_verification": {"status": "ok"}}],
        [{"status": "done"}, {"status": "done"}],
        [{"status": "done", "weight_sync_verification": {}}, {"status": "error"}],
    ],
)
def test_sampler_evidence_collective_missing_rank_or_proof_fails(ranks):
    with pytest.raises(RuntimeError):
        asyncio.run(_sampler(ranks).sync_weights("same-allocation-host", 30000, 1, 2))


# Native broadcast observes the selected production writers, not a replacement loader.


def test_native_broadcast_shadow_observes_real_fused_expert_callback(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "inference/server/weight_sync/utils.py"
    spec = importlib.util.spec_from_file_location("_native_broadcast_writer_control", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(torch.zeros(1, 4, 2))
            self.w2_weight = torch.nn.Parameter(torch.zeros(1, 2, 2))

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return expert_id

        def weight_loader(self, parameter, value, weight_name, shard, expert_id, **kwargs):
            target = parameter.data[expert_id]
            if shard == "w1":
                target = target[:2]
            elif shard == "w3":
                target = target[2:]
            target.copy_(value)
            return True

    class FusedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()

        def load_weights(self, weights):
            raise AssertionError("the fused path must not substitute model.load_weights")

    model = FusedModel()
    verifier = _verifier(model)
    writer = module._ShardAwareFusedWriter(model, torch.device("cpu"))
    broadcast.maybe_bind_broadcast_loader(verifier, writer)
    w13 = torch.arange(8.0).reshape(1, 4, 2)
    w2 = torch.arange(4.0).reshape(1, 2, 2)
    assert broadcast.maybe_load_broadcast_tensor(verifier, writer.feed, "experts.w13_weight", w13) is True
    assert broadcast.maybe_load_broadcast_tensor(verifier, writer.feed, "experts.w2_weight", w2) is True
    result = broadcast.maybe_broadcast_receiver_result({"status": "done"}, verifier, rank=3)
    assert result["rank"] == 3
    assert result["weight_sync_verification"]["destinations"] == {
        "experts.w13_weight": {"uncovered": 0, "mismatched": 0},
        "experts.w2_weight": {"uncovered": 0, "mismatched": 0},
    }
    assert result["weight_sync_verification"]["untouched_parameters"] == {}
    assert torch.equal(model.experts.w13_weight, w13)
    assert torch.equal(model.experts.w2_weight, w2)


def test_native_broadcast_shadow_observes_actual_direct_view_copy():
    model = _PackedModel()
    verifier = _verifier(model)
    view = model.packed.data.view(2, 2)[1]
    result = broadcast.maybe_load_broadcast_tensor(
        verifier, lambda name, value: view.copy_(value), "actual-view", torch.tensor([3.0, 4.0])
    )
    assert result is view
    assert verifier.finish()["destinations"]["packed"] == {"uncovered": 2, "mismatched": 0}
    assert model.packed.tolist() == [0.0, 0.0, 3.0, 4.0]


def _collect_broadcast(rows, *, world=2, offset=1, engine_only=False):
    llm = SimpleNamespace(vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(world_size=world)))
    return broadcast.maybe_collect_broadcast_verification(rows, engine_only=engine_only, llm=llm, rank_offset=offset)


def test_native_broadcast_collective_preserves_every_actual_rank(monkeypatch):
    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    rows = [
        {"status": "done", "rank": rank, "weight_sync_verification": {"status": "ok", "rank": rank}} for rank in (2, 1)
    ]
    result = _collect_broadcast(rows)
    assert result[0]["all_rank_weight_sync_verification"] == [row["weight_sync_verification"] for row in rows]
    assert result[1] is rows[1]
    assert "all_rank_weight_sync_verification" not in rows[0]


@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong-rank", "no-proof", "wrong-proof-rank"])
def test_native_broadcast_collective_rejects_incomplete_identity(monkeypatch, case):
    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    rows = [{"rank": rank, "weight_sync_verification": {"status": "ok", "rank": rank}} for rank in (1, 2)]
    if case == "missing":
        rows.pop()
    elif case == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif case == "wrong-rank":
        rows[1]["rank"] = 8
    elif case == "no-proof":
        rows[1].pop("weight_sync_verification")
    else:
        rows[1]["weight_sync_verification"]["rank"] = 7
    with pytest.raises(RuntimeError, match="every actual sampler rank"):
        _collect_broadcast(rows)


def test_native_broadcast_disabled_and_engine_only_do_not_read_missing_configuration(monkeypatch):
    rows = [{"status": "done"}]
    monkeypatch.delenv(evidence.VERIFY_ENV, raising=False)
    assert broadcast.maybe_collect_broadcast_verification(rows, engine_only=False, llm=object(), rank_offset=1) is rows
    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    assert broadcast.maybe_collect_broadcast_verification(rows, engine_only=True, llm=object(), rank_offset=1) is rows


def test_native_broadcast_disabled_preserves_stream_callback_and_result(monkeypatch):
    monkeypatch.delenv(evidence.VERIFY_ENV, raising=False)
    weights = iter([("weight", torch.ones(2))])
    result = {"status": "done"}

    def send(actual):
        assert actual is weights
        return result

    assert broadcast.maybe_send_broadcast_weights(object(), weights, send) is result
    assert broadcast.maybe_begin_broadcast_load(object(), object(), object()) is None
    assert broadcast.maybe_broadcast_receiver_result(result, None, rank=1) is result
    broadcast.maybe_bind_broadcast_trainer_inventory(object(), is_master=True, is_zero3=True)
    broadcast.maybe_admit_broadcast_receiver(object(), engine_only=False)
    assert broadcast.maybe_load_broadcast_strategy(lambda model, engine: result, object(), object(), None) is result


@pytest.mark.parametrize("failure", [None, "mutation", "missing-inventory", "duplicate-name", "transport"])
def test_native_broadcast_sender_observes_once_and_always_resets_context(monkeypatch, failure):
    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    value = torch.ones(2)
    weights = iter([("weight", value)] * (2 if failure == "duplicate-name" else 1))
    seen = []

    def send(actual):
        state = broadcast._SEND.get()
        if failure != "missing-inventory":
            state.update(
                trainer_parameters={"numel": 2, "count": 1},
                trainer_inventory={"parameters": {"weight": {"global_numel": 2}}},
                evidence_sources=evidence.source_files(send),
            )
        for name, tensor in actual:
            seen.append((name, tensor))
            if failure == "mutation":
                tensor.add_(1)
        if failure == "transport":
            raise RuntimeError("transport failed")
        return {"status": "done", "actual_receipt": 7}

    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            broadcast.maybe_send_broadcast_weights(_Sender(), weights, send)
    else:
        result = broadcast.maybe_send_broadcast_weights(_Sender(), weights, send)
        assert result["actual_receipt"] == 7
        assert result["sent_manifest"] == {"weight": evidence.tensor_manifest(value)}
        assert result["trainer_parameters"] == {"numel": 2, "count": 1}
        assert len(seen) == 1 and seen[0][1] is value
    assert broadcast._SEND.get() is None


def test_native_broadcast_inventory_collects_actual_rank_before_binding(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    model = torch.nn.Linear(2, 1, bias=False)
    calls = []

    def gather(records, local):
        calls.append(local)
        records[:] = [local, {**copy.deepcopy(local), "rank": 1}]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    token = broadcast._SEND.set({"trainer_parameters": None, "trainer_inventory": None})
    try:
        broadcast.maybe_bind_broadcast_trainer_inventory(model, is_master=True, is_zero3=False)
        state = broadcast._SEND.get()
        assert state["trainer_parameters"] == {"numel": 2, "count": 1}
        assert state["trainer_inventory"]["parameters"]["weight"]["global_numel"] == 2
        assert len(calls) == 1
    finally:
        broadcast._SEND.reset(token)


def test_native_broadcast_trainer_replica_disagreement_is_rejected():
    records = [
        {"rank": rank, "parameters": {"shared": {"sha256": digest}, "expert": {"sha256": str(rank % 2)}}}
        for rank, digest in enumerate(["same"] * 4)
    ]
    inventory = {"parameters": {"shared": {"ep_groups": []}, "expert": {"ep_groups": [[0, 1], [2, 3]]}}}
    broadcast._verify_trainer_replicas(records, inventory)
    records[2]["parameters"]["expert"]["sha256"] = "different"
    with pytest.raises(ValueError, match="replica bytes disagree"):
        broadcast._verify_trainer_replicas(records, inventory)


def test_native_broadcast_quantized_receive_rejected_before_transport(monkeypatch):
    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    receiver = SimpleNamespace(model_config=SimpleNamespace(quantization="fp8"))
    broadcast.maybe_admit_broadcast_receiver(receiver, engine_only=True)
    with pytest.raises(ValueError, match="quantized"):
        broadcast.maybe_admit_broadcast_receiver(receiver, engine_only=False)


def test_sampler_evidence_close_resources_even_when_source_guard_would_fail():
    worker = _sampler([])
    worker._origins = {__file__: "0" * 64}
    result = asyncio.run(worker.close_weight_sync())
    assert worker.llm.calls[0][0] == "close_weight_sync"
    assert result["rank_results"] == []


def test_sampler_evidence_external_pool_worker_injection_uses_public_argument(monkeypatch):
    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    actor = object()
    monkeypatch.setattr(sampler, "_worker_actor", lambda: actor)

    def pool_class(*, worker_cls):
        return worker_cls

    assert sampler.maybe_replica_pool(pool_class)() is actor


def test_sampler_evidence_worker_optin_explicitly_propagates_to_ray_actor(monkeypatch):
    calls = []

    def remote(**kwargs):
        calls.append(kwargs)
        return lambda cls: cls

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(remote=remote))
    monkeypatch.setenv("ARCTIC_PLATFORM_WEIGHT_SYNC_SHADOW_MAX_BYTES", "2048")
    assert sampler._worker_actor() is sampler.SamplerEvidenceWorker
    assert calls == [
        {
            "runtime_env": {
                "env_vars": {
                    evidence.VERIFY_ENV: "1",
                    "ARCTIC_PLATFORM_WEIGHT_SYNC_SHADOW_MAX_BYTES": "2048",
                }
            }
        }
    ]


def test_sampler_evidence_canonical_sync_keeps_actual_receipts(monkeypatch):
    monkeypatch.setenv(evidence.VERIFY_ENV, "1")
    received = {"workers": [{"all_rank_weight_sync_verification": [{"status": "ok"}]}]}
    sent = {"sent_manifest": {"wire": {"numel": 1}}}

    async def ready(value):
        return value

    async def run():
        return await sampler.maybe_sync_weights(ready(received), [ready(sent)])

    result = asyncio.run(run())
    assert result["targets"][0]["recv"] is received
    assert result["targets"][0]["send"][0] is sent


def test_sampler_evidence_disabled_sync_retains_original_status_only(monkeypatch):
    monkeypatch.delenv(evidence.VERIFY_ENV, raising=False)

    async def ready():
        return {"native": "actual"}

    async def run():
        return await sampler.maybe_sync_weights(ready(), [ready()])

    assert asyncio.run(run()) == {"status": "ok"}


def test_sampler_evidence_no_fabricated_router_trace_or_generation():
    worker = _sampler([])
    with pytest.raises(ValueError, match="not implemented"):
        asyncio.run(worker.generate([1], {"dss_return_back_router_info": True}))
    assert worker.llm.calls == []
