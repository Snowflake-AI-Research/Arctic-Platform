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

"""New CPU fixtures for opt-in external sampler evidence; no native admission."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest
import torch

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
