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

"""Small CPU score-tensor tests; no model execution or GPU parity claim."""

import asyncio
import copy
import datetime
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from arctic_platform import wire
from arctic_platform.common.utils.batch import _split_batch
from arctic_platform.common.utils.batch import flatten_batch_rows
from arctic_platform.rl.processors import DPOLoss
from arctic_platform.rl.processors import prepare_request_loss
from arctic_platform.rl.processors import resolve_loss
from arctic_platform.rl.processors import resolve_packed_loss_reduction
from arctic_platform.rl.processors import run_pipeline
from arctic_platform.rl.processors.base_loss import USES_TOKEN_BUDGET_PIPELINE
from arctic_platform.rl.processors.base_loss import _attach_loss_object_to_shards
from arctic_platform.rl.processors.microbatch import MicroBatchSpec
from arctic_platform.rl.processors.microbatch import split_padded_tensor_dict_into_mb_list
from arctic_platform.rl.processors.packing import pack_sequences
from arctic_platform.rl.processors.packing import unpack_sequences
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import reserve_free_port
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal


def request(pairs=3, dtype=torch.float32):
    lengths = torch.tensor([5, 3, 4, 2, 3, 5, 2, 4, 5, 2][: pairs * 2])
    attn = torch.arange(6)[None, :] < lengths[:, None]
    mask = attn.clone()
    mask[:, :2] = False
    labels = torch.arange(pairs * 12).reshape(pairs * 2, 6).masked_fill(~mask, -100)
    return {
        "kwargs": {"input_ids": labels.clamp(min=0), "attention_mask": attn},
        "context": {
            "labels": labels,
            "loss_mask": mask,
            "dpo_pair_ids": torch.arange(pairs).repeat_interleave(2),
            "dpo_is_chosen": torch.tensor([True, False] * pairs),
            "dpo_reference_logps": torch.linspace(-1.5, 0.2, pairs * 2, dtype=dtype),
        },
        "processing": {"post": ["compute_logprobs"], "loss_fn": "dpo", "config": {"beta": 0.3}},
    }


def oracle(scores, refs, mask, beta=0.3):
    # Independent full equation: no native loss or reduction helpers.
    mask = mask.reshape(scores.shape)
    refs = refs.reshape(-1)
    sums = torch.where(mask, scores, torch.zeros_like(scores)).sum(-1)
    margin = sums[::2] - sums[1::2] - (refs[::2].detach() - refs[1::2].detach())
    return F.softplus(-beta * margin).mean()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("pairs", [1, 2, 3])
def test_objective_and_derivatives(dtype, pairs):
    req = request(pairs, dtype)
    refs = req["context"]["dpo_reference_logps"].requires_grad_()
    scores = torch.linspace(-2.0, 0.5, pairs * 12, dtype=dtype).reshape(pairs * 2, 6).requires_grad_()
    original_mask = req["context"]["loss_mask"].clone()
    expected_scores = scores.detach().double().requires_grad_()
    expected = oracle(expected_scores, refs.double(), original_mask)
    expected_grad = torch.autograd.grad(expected, expected_scores)[0]
    loss_object = prepare_request_loss(req)
    actual, _ = loss_object.loss({"logprobs": scores}, req["batch"], req["meta"], req["processing"]["config"], "cpu")
    grad = torch.autograd.grad(actual, scores)[0]
    # Absolute tolerance covers FP32 score arithmetic versus the FP64 oracle.
    atol = 1e-10 if dtype == torch.float64 else 1e-6
    torch_assert_close(actual.double(), expected, atol=atol, rtol=0)
    torch_assert_close(grad.double(), expected_grad, atol=atol, rtol=0)
    assert grad[~original_mask].count_nonzero() == 0
    assert refs.grad is None and not req["batch"]["dpo_reference_logps"].requires_grad


def test_boundary_behavior_and_reference_only_zero_pair():
    req = request(2)
    req["context"]["loss_mask"][2:] = False
    refs = req["context"]["dpo_reference_logps"]
    refs[:2] = 0
    scores = torch.zeros(4, 6, requires_grad=True)
    loss = prepare_request_loss(req)
    actual, _ = loss.loss({"logprobs": scores}, req["batch"], {}, {}, "cpu")
    expected = oracle(scores, refs, req["batch"]["loss_mask"], beta=0.1)
    torch_assert_close(actual, expected, atol=1e-6, rtol=0)
    grad = torch.autograd.grad(actual, scores)[0]
    assert grad[2:].count_nonzero() == 0
    req = request(1)
    req["context"]["dpo_reference_logps"].zero_()
    loss = prepare_request_loss(req)
    a, _ = loss.loss({"logprobs": torch.zeros(2, 6)}, req["batch"], {}, {}, "cpu")
    assert abs(a.item() - math.log(2)) < 1e-6
    improved = torch.zeros(2, 6)
    improved[0] = 0.5
    b, _ = loss.loss({"logprobs": improved}, req["batch"], {}, {}, "cpu")
    assert b < a


@pytest.mark.parametrize("workers,pairs", [(1, 3), (2, 3), (4, 5)])
def test_wire_registry_pair_shards_packed_windows_global_mean(workers, pairs):
    req = request(pairs)
    req["kwargs"]["row_tags"] = list(range(pairs * 2))
    req["kwargs"]["scalar"] = torch.tensor(7)
    original = copy.deepcopy(req)
    scores = torch.linspace(-1.3, 0.7, pairs * 12).reshape(pairs * 2, 6).requires_grad_()
    reference = scores.detach().double().requires_grad_()
    expected = oracle(reference, original["context"]["dpo_reference_logps"].double(), original["context"]["loss_mask"])
    expected_grad = torch.autograd.grad(expected, reference)[0]
    # Exercise the production binary codec, name selection, DP cutter and packer.
    decoded = wire.loads(wire.dumps(req))
    obj = prepare_request_loss(decoded)
    assert isinstance(obj, DPOLoss) and isinstance(resolve_loss("dpo"), DPOLoss)
    assert obj.has_capability(USES_TOKEN_BUDGET_PIPELINE)
    shards, _ = _split_batch(decoded, workers)
    total = scores.sum() * 0
    worker_metrics = []
    seen = []
    for shard in shards:
        batch = shard["batch"]
        ids = batch["dpo_pair_ids"]
        assert batch["input_ids"].shape == (ids.shape[0], 2, 6)
        row_index = ids * 2 + (~batch["dpo_is_chosen"]).long()
        assert batch["row_tags"] == row_index.tolist()
        assert batch["scalar"].item() == 7
        batch["test_scores"] = scores[row_index]
        mb_list = split_padded_tensor_dict_into_mb_list(
            {**shard["meta"], **batch}, MicroBatchSpec(max_tokens_per_mb=8)
        )
        reduction = resolve_packed_loss_reduction(shard["processing"], mb_list.mbs, loss_object=obj)
        metrics = {"dpo_loss_sum": 0.0}
        for mb, scale in zip(mb_list.mbs, reduction.loss_scales):
            assert mb["row_tags"] == (mb["dpo_pair_ids"] * 2 + (~mb["dpo_is_chosen"]).long()).tolist()
            rows = flatten_batch_rows(mb)
            packed = pack_sequences(rows)
            pack_meta = packed.pop("_pack_meta")
            torch_assert_equal(unpack_sequences(packed["input_ids"], pack_meta), rows["input_ids"])
            actual, metric = obj.loss(
                {"logprobs": packed.pop("test_scores")}, {}, packed, req["processing"]["config"], "cpu"
            )
            total = total + actual * scale / workers
            metrics["dpo_loss_sum"] += metric["dpo_loss_sum"]
            seen.extend(mb["dpo_pair_ids"][:, 0].tolist())
        worker_metrics.append(metrics)
    torch_assert_close(total.double(), expected, atol=1e-6, rtol=0)
    torch_assert_close(torch.autograd.grad(total, scores)[0].double(), expected_grad, atol=1e-6, rtol=0)
    assert sorted(seen) == list(range(pairs))
    metrics = {}
    obj.metrics_callback(worker_metrics, metrics)
    assert abs(metrics["dpo_loss"] - expected.item()) < 1e-6
    assert obj.reporting_callback(worker_metrics, metrics, -1) == metrics["dpo_loss"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("dpo_pair_ids", torch.tensor([0, 1, 1, 1])),
        ("dpo_pair_ids", torch.tensor([0, 0, 0, 0])),
        ("dpo_pair_ids", torch.tensor([0.0, 0.0, 1.0, 1.0])),
        ("dpo_is_chosen", torch.tensor([False, True, True, False])),
        ("dpo_reference_logps", torch.tensor([0.0, float("nan"), 0.0, 0.0])),
        ("dpo_reference_logps", torch.tensor([0, 0, 0, 0])),
        ("dpo_reference_logps", torch.zeros(3)),
        ("loss_mask", torch.ones(4, 6)),
        ("labels", torch.full((4, 6), -100)),
    ],
)
def test_malformed_fields_fail_before_dispatch(field, value):
    req = request(2)
    req["context"][field] = value
    with pytest.raises(ValueError):
        prepare_request_loss(wire.loads(wire.dumps(req)))


@pytest.mark.parametrize("beta", [True, 0, -0.1, float("nan"), float("inf"), "0.1", 1e100, 1e-100])
def test_invalid_beta(beta):
    req = request(1)
    req["processing"]["config"]["beta"] = beta
    with pytest.raises(ValueError, match="beta"):
        prepare_request_loss(req)


def test_missing_reference_padding_budget_and_incomplete_shards_rejected():
    req = request(1)
    del req["context"]["dpo_reference_logps"]
    with pytest.raises(ValueError, match="dpo_reference_logps"):
        prepare_request_loss(req)
    req = request(1)
    req["context"]["loss_mask"][0, -1] = True
    req["context"]["labels"][0, -1] = 0
    with pytest.raises(ValueError, match="padding"):
        prepare_request_loss(req)
    req = request(1)
    req["context"]["max_tokens_per_mb"] = 7
    with pytest.raises(ValueError, match="exceeds"):
        prepare_request_loss(req)
    req = request(1)
    prepare_request_loss(req)
    with pytest.raises(ValueError, match="at least one sample"):
        _split_batch(req, 2)


def test_flat_hosted_request_keeps_context():
    req = request(1)
    req["processing"]["config"] = None
    flat = {**req["kwargs"], "context": req["context"], "processing": req["processing"]}
    loss = resolve_loss("dpo")
    loss.batching_callback(flat)
    assert loss.global_pairs == 1 and "batch" not in flat
    assert flat["processing"]["config"] == {}
    assert flat["dpo_pair_ids"].tolist() == [[0, 0]]
    assert flat["input_ids"].shape == (1, 2, 6)
    assert "labels" in flat and "dpo_pair_ids" not in flat["context"]


def test_nonfinite_response_scores_fail_and_masked_nan_has_zero_derivative():
    req = request(1)
    obj = prepare_request_loss(req)
    scores = torch.zeros(2, 6, requires_grad=True)
    masked_scores = scores.masked_fill(~req["batch"]["loss_mask"].reshape(scores.shape), float("nan"))
    value, _ = obj.loss({"logprobs": masked_scores}, req["batch"], {}, {}, "cpu")
    assert torch.isfinite(value)
    assert torch.autograd.grad(value, scores)[0][~req["batch"]["loss_mask"].reshape(scores.shape)].count_nonzero() == 0
    with pytest.raises(ValueError, match="finite"):
        obj.loss({"logprobs": torch.full((2, 6), float("inf"))}, req["batch"], {}, {}, "cpu")


class _ScoreTensorEngine:
    """Route precomputed score leaves through packing; never executes a model."""

    global_rank = 0

    def __init__(self, scores):
        self.scores = scores
        self.module = SimpleNamespace(config=SimpleNamespace(_attn_implementation="flash_attention_2"))
        self.calls = 0
        self.backward_calls = 0
        self.boundaries = []

    def train(self):
        pass

    def eval(self):
        pass

    def gradient_accumulation_steps(self):
        return 2

    def __call__(self, input_ids, **kwargs):
        self.calls += 1
        return {"logprobs": self.scores.flatten()[input_ids]}

    def backward(self, loss, scale_wrt_gas):
        self.backward_calls += 1
        assert scale_wrt_gas is False
        loss.backward()

    def set_gradient_accumulation_boundary(self, value):
        self.boundaries.append(value)


def _native_worker(engine, world_size=1):
    from arctic_platform.common.deepspeed_worker import DeepSpeedWorker

    worker = object.__new__(DeepSpeedWorker.__ray_metadata__.modified_class)
    worker.rank = engine.global_rank
    worker.world_size = world_size
    worker.sp_size = 1
    worker.engine = engine
    worker._device = worker.cpu_device = torch.device("cpu")
    return worker


def _distributed_tensor_check(rank, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=datetime.timedelta(seconds=45)
    )
    try:
        req = request(3)
        req["kwargs"]["input_ids"] = torch.arange(36).reshape(6, 6)
        scores = torch.linspace(-1.3, 0.7, 36).reshape(6, 6).requires_grad_()
        expected = oracle(scores, req["context"]["dpo_reference_logps"], req["context"]["loss_mask"])
        expected_grad = torch.autograd.grad(expected, scores)[0]
        req["context"]["max_tokens_per_mb"] = 8
        req = wire.loads(wire.dumps(req))
        obj = prepare_request_loss(req)
        shards, _ = _split_batch(req, 2)
        _attach_loss_object_to_shards(shards, obj)
        shard = shards[rank]
        engine = _ScoreTensorEngine(scores)
        engine.global_rank = rank
        result = _native_worker(engine, world_size=2).forward_backward(shard)
        assert engine.calls == 2 and engine.boundaries == [True]
        dist.all_reduce(scores.grad)
        scores.grad /= 2
        torch_assert_close(scores.grad, expected_grad, atol=1e-6, rtol=0)
        contribution = torch.tensor(result["metrics"]["dpo_loss_sum"])
        dist.all_reduce(contribution)
        torch_assert_close(contribution, expected.detach(), atol=1e-6, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure,match", [("reference", "dpo_reference_logps"), ("token", "non-negative")])
def test_invalid_last_pair_causes_no_worker_rpc(failure, match):
    from arctic_platform.common.ray_server import ArcticRLRayServer

    class Remote:
        def remote(self, shard):
            raise AssertionError("malformed requests must never reach a worker")

    server = object.__new__(ArcticRLRayServer)
    server.jobs = {1: {"job_type": "training", "sp_size": 1}}
    server.training_workers = [type("Worker", (), {"forward_backward": Remote()})() for _ in range(2)]
    req = request(3)
    if failure == "reference":
        req["context"]["dpo_reference_logps"][-1] = float("nan")
    else:
        req["kwargs"]["input_ids"][-1, 0] = -1
    with pytest.raises(ValueError, match=match):
        asyncio.run(server.forward_backward(1, wire.loads(wire.dumps(req))))


@pytest.mark.parametrize("bag", ["kwargs", "context"])
@pytest.mark.parametrize(
    "key", ["cu_seqlens", "cu_seq_lens_q", "cu_seq_lens_k", "seq_idx", "max_length_q", "max_length_k"]
)
def test_caller_packing_boundaries_fail_before_dispatch(bag, key):
    req = request(2)
    req[bag][key] = torch.tensor([0, 1, 2, 3, 24], dtype=torch.int32)
    with pytest.raises(ValueError, match="without caller cu_seqlens"):
        prepare_request_loss(wire.loads(wire.dumps(req)))


@pytest.mark.parametrize("kind", ["offset", "float", "shape", "restart"])
@pytest.mark.parametrize("bag", ["kwargs", "context"])
def test_invalid_position_ids_fail_before_dispatch(kind, bag):
    req = request(2)
    positions = torch.arange(6).expand(4, -1).clone()
    if kind == "offset":
        positions += torch.arange(4)[:, None] * 6
    elif kind == "float":
        positions = positions.float()
    elif kind == "shape":
        positions = positions.flatten()
    else:
        positions[:, 1] = 0
    req[bag]["position_ids"] = positions
    with pytest.raises(ValueError, match="dpo position_ids"):
        prepare_request_loss(wire.loads(wire.dumps(req)))


@pytest.mark.parametrize("bag", ["kwargs", "context"])
def test_canonical_valid_position_ids_allow_arbitrary_padding(bag):
    req = request(2)
    positions = torch.arange(6).expand(4, -1).clone()
    positions[~req["kwargs"]["attention_mask"].bool()] = 99
    req[bag]["position_ids"] = positions
    assert prepare_request_loss(wire.loads(wire.dumps(req))).global_pairs == 2


def test_large_equal_fp64_reference_scores_cancel_before_narrowing():
    req = request(1)
    req["context"]["dpo_reference_logps"] = torch.tensor([-1e100, -1e100], dtype=torch.float64)
    scores = torch.linspace(-2.0, 0.5, 12).reshape(2, 6).requires_grad_()
    expected = oracle(scores, req["context"]["dpo_reference_logps"], req["context"]["loss_mask"])
    expected_grad = torch.autograd.grad(expected, scores)[0]
    obj = prepare_request_loss(req)
    actual, _ = obj.loss({"logprobs": scores}, req["batch"], {}, req["processing"]["config"], "cpu")
    torch_assert_close(actual.double(), expected, atol=1e-6, rtol=0)
    torch_assert_close(torch.autograd.grad(actual, scores)[0], expected_grad, atol=1e-6, rtol=0)
    req = request(1)
    req["context"]["dpo_reference_logps"] = torch.tensor([-1e100, 1e100], dtype=torch.float64)
    with pytest.raises(ValueError, match="reference margins"):
        prepare_request_loss(req)


def test_dpo_filters_post_outputs_and_ignores_auxiliary_model_loss():
    req = request(1)
    req["kwargs"]["input_ids"] = torch.arange(12).reshape(2, 6)
    obj = prepare_request_loss(req)

    class AuxiliaryLossEngine(_ScoreTensorEngine):
        def __call__(self, input_ids, **kwargs):
            assert "labels" not in kwargs
            result = super().__call__(input_ids, **kwargs)
            result["loss"] = torch.tensor(0.0)
            return result

    result = run_pipeline(
        AuxiliaryLossEngine(torch.zeros(2, 6, requires_grad=True)),
        (),
        req["batch"],
        req["meta"],
        req["processing"],
        "cpu",
        backward="loss_only",
        pack=True,
        loss_object=obj,
    )
    assert not result.get("batch")
    assert result["loss_tensor"].requires_grad


@pytest.mark.parametrize("method", ["forward", "forward_backward"])
def test_coordinator_returns_empty_filtered_dpo_batch(monkeypatch, method):
    import arctic_platform.common.ray_server as ray_server

    req = request(3)
    req["kwargs"]["input_ids"] = torch.arange(36).reshape(6, 6)
    req["context"].update(max_tokens_per_mb=8, pad_token_id=0, return_fwd_batch=True, drop_position_ids=True)
    scores = torch.linspace(-1.3, 0.7, 36).reshape(6, 6).requires_grad_()
    expected = oracle(scores, req["context"]["dpo_reference_logps"], req["context"]["loss_mask"])
    engine = _ScoreTensorEngine(scores)
    worker = _native_worker(engine)

    class Compute:
        def remote(self, shard):
            return worker.forward_backward(shard) if method == "forward_backward" else worker.forward_no_grad(shard)

    server = object.__new__(ray_server.ArcticRLRayServer)
    server.jobs = {1: {"job_type": "training", "sp_size": 1}}
    server.training_workers = [type("Worker", (), {"forward_backward": Compute(), "forward_no_grad": Compute()})()]
    monkeypatch.setattr(ray_server.ray, "get", lambda refs: refs)
    response = asyncio.run(getattr(server, method)(1, wire.loads(wire.dumps(req))))
    assert response["batch"] == {}
    assert abs(response["avg_loss"] - expected.item()) < 1e-6
    if method == "forward_backward":
        assert engine.calls == 3 and engine.boundaries == [True]
        torch_assert_close(scores.grad, torch.autograd.grad(expected, scores)[0], atol=1e-6, rtol=0)


def test_chunked_score_forward_keeps_logit_aligned_labels():
    req = request(1)
    req["kwargs"]["input_ids"] = torch.arange(12).reshape(2, 6)
    req["kwargs"]["labels"] = req["context"].pop("labels")
    req["kwargs"]["dss_compute_logprobs"] = True
    obj = prepare_request_loss(req)

    class ChunkedScoreEngine(_ScoreTensorEngine):
        def __call__(self, input_ids, labels, dss_compute_logprobs, **kwargs):
            assert dss_compute_logprobs is True
            assert labels.shape == input_ids.shape
            return super().__call__(input_ids, **kwargs)

    scores = torch.zeros(2, 6, requires_grad=True)
    result = run_pipeline(
        ChunkedScoreEngine(scores),
        (),
        req["batch"],
        req["meta"],
        req["processing"],
        "cpu",
        backward="loss_only",
        pack=True,
        loss_object=obj,
    )
    assert result["batch"] == {}
    assert torch.isfinite(torch.autograd.grad(result["loss_tensor"], scores)[0]).all()


@pytest.mark.parametrize("overflow", ["sums", "scaled_margin", "scaled_loss"])
def test_finite_scores_that_overflow_loss_are_rejected(overflow):
    req = request(1)
    scores = torch.zeros(2, 6)
    req["context"]["dpo_reference_logps"].zero_()
    if overflow == "sums":
        scores[0] = 3e38
    elif overflow == "scaled_margin":
        req["processing"]["config"]["beta"] = 3e38
        scores[1] = 10
    else:
        req["context"]["dp_size"] = 4
        req["processing"]["config"]["beta"] = 1e38
        scores[1, 2] = 2
    obj = prepare_request_loss(req)
    with pytest.raises(ValueError, match="finite on every rank"):
        obj.loss({"logprobs": scores}, req["batch"], req["meta"], req["processing"]["config"], "cpu")


def _distributed_overflow_check(rank, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=datetime.timedelta(seconds=45)
    )
    try:
        req = request(2)
        req["kwargs"]["input_ids"] = torch.arange(24).reshape(4, 6)
        req["processing"]["config"]["beta"] = 3e38
        scores = torch.zeros(4, 6, requires_grad=True)
        with torch.no_grad():
            scores[0] = -10  # Only rank zero overflows.
        obj = prepare_request_loss(req)
        shards, _ = _split_batch(req, 2)
        shard = shards[rank]
        engine = _ScoreTensorEngine(scores)
        with pytest.raises(ValueError, match="finite on every rank"):
            run_pipeline(
                engine,
                (),
                shard["batch"],
                shard["meta"],
                shard["processing"],
                "cpu",
                backward=True,
                pack=True,
                loss_object=obj,
            )
        assert scores.grad is None
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backward", ["loss_only", "summed"])
def test_combined_packed_loss_overflow_is_rejected_before_backward(backward):
    req = request(3)
    req["kwargs"]["input_ids"] = torch.arange(36).reshape(6, 6)
    req["context"]["dpo_reference_logps"].zero_()
    req["context"]["dp_size"] = 4
    req["processing"]["config"]["beta"] = 1e38
    scores = torch.zeros(6, 6, requires_grad=True)
    with torch.no_grad():
        scores[0::2] = -0.6
    obj = prepare_request_loss(req)
    with pytest.raises(ValueError, match="finite on every rank"):
        run_pipeline(
            _ScoreTensorEngine(scores),
            (),
            req["batch"],
            req["meta"],
            req["processing"],
            "cpu",
            backward=backward,
            pack=True,
            max_tokens_per_mb=8,
            loss_object=obj,
        )
    assert scores.grad is None


@pytest.mark.parametrize("dtype,beta", [(torch.float16, 1e6), (torch.bfloat16, 3.4e38)])
def test_derivative_scale_must_fit_original_score_dtype(dtype, beta):
    req = request(1)
    req["context"]["dpo_reference_logps"].zero_()
    req["processing"]["config"]["beta"] = beta
    scores = torch.zeros(2, 6, dtype=dtype, requires_grad=True)
    if dtype == torch.bfloat16:
        with torch.no_grad():
            scores[1, 2] = 1e-37
    obj = prepare_request_loss(req)
    with pytest.raises(ValueError, match="finite on every rank"):
        obj.loss({"logprobs": scores}, req["batch"], req["meta"], req["processing"]["config"], "cpu")
    assert scores.grad is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moderate_beta_preserves_finite_half_precision_score_gradients(dtype):
    req = request(1)
    req["context"]["dpo_reference_logps"].zero_()
    scores = torch.zeros(2, 6, dtype=dtype, requires_grad=True)
    reference_scores = scores.detach().clone().requires_grad_()
    expected = oracle(reference_scores.float(), req["context"]["dpo_reference_logps"], req["context"]["loss_mask"])
    obj = prepare_request_loss(req)
    actual, _ = obj.loss({"logprobs": scores}, req["batch"], req["meta"], req["processing"]["config"], "cpu")
    grad = torch.autograd.grad(actual, scores)[0]
    assert torch.isfinite(grad).all()
    torch_assert_close(actual, expected, atol=1e-6, rtol=0)
    torch_assert_close(grad, torch.autograd.grad(expected, reference_scores)[0], atol=0, rtol=0)


def _distributed_heuristic_schedule_check(rank, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=datetime.timedelta(seconds=45)
    )
    try:
        # FFD initially needs 10 vs 11 windows. Rerunning it with min_groups=11
        # on rank zero instead produced 12, making collective schedules diverge.
        pair_lengths = (
            [7, 4, 6, 7, 10, 20, 19, 10, 8, 16, 12, 11, 12, 2, 11, 12, 8, 8, 14, 8, 2, 6, 8, 13] + [24] * 9 + [2] * 15
        )
        row_lengths = torch.tensor([length for total in pair_lengths for length in (total // 2, total - total // 2)])
        input_ids = torch.arange(row_lengths.numel() * 12).reshape(-1, 12)
        attention_mask = torch.arange(12)[None, :] < row_lengths[:, None]
        mask = attention_mask.clone()
        mask[:, 0] = False
        refs = torch.linspace(-1.5, 0.2, row_lengths.numel())
        req = {
            "kwargs": {"input_ids": input_ids, "attention_mask": attention_mask},
            "context": {
                "labels": input_ids.masked_fill(~mask, -100),
                "loss_mask": mask,
                "dpo_pair_ids": torch.arange(len(pair_lengths)).repeat_interleave(2),
                "dpo_is_chosen": torch.tensor([True, False] * len(pair_lengths)),
                "dpo_reference_logps": refs,
            },
            "processing": {"post": ["compute_logprobs"], "loss_fn": "dpo", "config": {"beta": 0.3}},
        }
        scores = torch.linspace(-1.3, 0.7, input_ids.numel()).reshape(input_ids.shape).requires_grad_()
        expected = oracle(scores, refs, mask)
        expected_grad = torch.autograd.grad(expected, scores)[0]
        req = wire.loads(wire.dumps(req))
        obj = prepare_request_loss(req)
        shards, _ = _split_batch(req, 2)
        shard = shards[rank]
        planned = split_padded_tensor_dict_into_mb_list(
            {**shard["meta"], **shard["batch"]}, MicroBatchSpec(max_tokens_per_mb=24)
        )
        assert len(planned.mbs) == (10 if rank == 0 else 11)
        assert planned.padding_count == (1 if rank == 0 else 0)
        engine = _ScoreTensorEngine(scores)
        result = run_pipeline(
            engine,
            (),
            shard["batch"],
            shard["meta"],
            shard["processing"],
            "cpu",
            backward=True,
            pack=True,
            max_tokens_per_mb=24,
            loss_object=obj,
        )
        assert engine.calls == 11 and engine.backward_calls == 1 and engine.boundaries == [True]
        dist.all_reduce(scores.grad)
        scores.grad /= 2
        torch_assert_close(scores.grad, expected_grad, atol=1e-6, rtol=0)
        contribution = torch.tensor(result["metrics"]["dpo_loss_sum"])
        dist.all_reduce(contribution)
        torch_assert_close(contribution, expected.detach(), atol=1e-6, rtol=0)
    finally:
        dist.destroy_process_group()


class TestDPOSynchronization(TestCasePlus):
    def _run_distributed_check(self, target):
        # A fresh process owns the group: pytest initializes its own group.
        port = reserve_free_port(get_unique_port_number() + 1, span=7)
        env = self.get_env()
        env["PYTHONPATH"] += os.pathsep + self.test_file_dir_str
        code = (
            f"import torch.multiprocessing as mp; from test_dpo import {target}; "
            f"mp.spawn({target}, args=({port},), nprocs=2); print('{target}: PASS')"
        )
        execute_subprocess_async([sys.executable, "-c", code], env=env, timeout=90)

    def test_concurrent_packing_pads_calls_without_repeating_pair_contributions(self):
        self._run_distributed_check("_distributed_tensor_check")

    def test_one_rank_overflow_rejects_all_ranks_before_backward(self):
        self._run_distributed_check("_distributed_overflow_check")

    def test_padding_preserves_original_atomic_windows_when_ffd_would_overshoot(self):
        self._run_distributed_check("_distributed_heuristic_schedule_check")
