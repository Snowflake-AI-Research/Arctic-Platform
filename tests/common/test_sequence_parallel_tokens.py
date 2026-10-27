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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""New synthetic token-domain controls, not native SP/runtime admission."""

import math
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common.utils import sequence_parallel as sp
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal


def _info(rank=2, ranks=(2, 0), world=4):
    return {
        "rank": rank,
        "world_size": world,
        "sp_size": len(ranks),
        "sp_rank": ranks.index(rank),
        "sp_ranks": list(ranks),
    }


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
        "loss_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32),
        "advantages": torch.ones(2, 5),
        "routed_experts": torch.arange(2 * 5 * 3 * 2).reshape(2, 5, 3, 2),
    }


def test_token_partition_aligns_cross_slice_target_before_packing():
    source = _batch()
    original = deepcopy(source)
    local, packed, mask, cu, part = sp.prepare_token_partition(source, sp_info=_info())
    assert (part.start, part.stop, part.rank, part.ranks) == (0, 4, 2, (2, 0))
    torch_assert_equal(local["input_ids"], torch.tensor([[1, 2, 3, 4]]))
    # Token 4 predicts 5, which belongs to the next rank, not token 1.
    torch_assert_equal(local["labels"], torch.tensor([[2, 3, -100, 5]]))
    torch_assert_equal(packed["labels"], torch.tensor([[2, 3, -100, 5, 6, 7, 8, -100]]))
    torch_assert_equal(cu, torch.tensor([0, 3, 8], dtype=torch.int32))
    assert int(mask.sum()) == 8
    assert set(source) == set(original)
    for name in source:
        torch_assert_equal(source[name], original[name])


def test_token_partition_slices_router_layers_without_flattening_or_duplicate_rows():
    source = _batch()
    local, _, mask, _, part = sp.prepare_token_partition(source, sp_info=_info(rank=0))
    assert (part.start, part.stop) == (4, 8)
    torch_assert_equal(local["routed_experts"], source["routed_experts"][mask][4:].unsqueeze(0))
    assert local["routed_experts"].shape == (1, 4, 3, 2)
    torch_assert_equal(local["position_ids"], torch.tensor([[1, 2, 3, 4]]))


def test_token_partition_preserves_explicit_prediction_targets_and_packed_document_resets():
    source = _batch()
    source["labels"] = torch.tensor([[9, 10, -100, -100, -100], [11, 12, 13, 14, -100]])
    source["position_ids"] = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 0, 1, 2]])
    local, packed, _, cu, _ = sp.prepare_token_partition(source, sp_info=_info(rank=0))
    torch_assert_equal(local["labels"], torch.tensor([[12, 13, 14, -100]]))
    torch_assert_equal(packed["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]]))
    torch_assert_equal(cu, torch.tensor([0, 3, 5, 8], dtype=torch.int32))


@pytest.mark.parametrize(
    "positions,masked_target,expected_labels,expected_cu",
    [
        (
            [[0, 1, 0, 0, 0], [0, 1, 2, 3, 4]],
            (0, 1),
            [2, -100, -100, 5, 6, 7, 8, -100],
            [0, 2, 3, 8],
        ),
        (
            [[0, 1, 2, 0, 0], [0, 0, 1, 2, 3]],
            (1, 0),
            [2, 3, -100, -100, 6, 7, 8, -100],
            [0, 3, 4, 8],
        ),
        (
            [[0, 1, 2, 0, 0], [0, 1, 0, 1, 2]],
            (1, 1),
            [2, 3, -100, 5, -100, 7, 8, -100],
            [0, 3, 5, 8],
        ),
    ],
    ids=["inside-first-slice", "at-slice-boundary", "inside-second-slice"],
)
@pytest.mark.parametrize("rank", [2, 0])
def test_token_generated_labels_stop_at_document_reset(rank, positions, masked_target, expected_labels, expected_cu):
    source = _batch()
    source["position_ids"] = torch.tensor(positions)
    source["loss_mask"][masked_target] = 0
    original = deepcopy(source)
    local, packed, mask, cu, part = sp.prepare_token_partition(source, sp_info=_info(rank=rank))
    labels = torch.tensor([expected_labels])
    torch_assert_equal(packed["labels"], labels)
    torch_assert_equal(local["labels"], labels[:, part.start : part.stop])
    torch_assert_equal(cu, torch.tensor(expected_cu, dtype=torch.int32))
    torch_assert_equal(packed["input_ids"], source["input_ids"][mask].unsqueeze(0))
    torch_assert_equal(packed["routed_experts"], source["routed_experts"][mask].unsqueeze(0))
    assert set(source) == set(original)
    for name in original:
        torch_assert_equal(source[name], original[name])


@pytest.mark.parametrize("reset_column", [1, 2])
def test_token_generated_labels_reject_active_loss_across_document_reset(reset_column):
    source = _batch()
    source["position_ids"] = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 3, 4]])
    source["position_ids"][1, reset_column:] -= reset_column
    original = deepcopy(source)
    with pytest.raises(ValueError, match="active loss positions need valid next-token targets"):
        sp.prepare_token_partition(source, sp_info=_info())
    assert "labels" not in source
    for name in original:
        torch_assert_equal(source[name], original[name])


@pytest.mark.parametrize("fault", ["shape", "dtype", "negative"])
def test_token_generated_labels_validate_reset_positions_before_target_generation(fault):
    source = _batch()
    source["position_ids"] = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 0, 1, 2]])
    if fault == "shape":
        source["position_ids"] = source["position_ids"][:, :1]
    elif fault == "dtype":
        source["position_ids"] = source["position_ids"].float()
    else:
        source["position_ids"][1, 2] = -1
    original = deepcopy(source)
    with pytest.raises(ValueError):
        sp.prepare_token_partition(source, sp_info=_info())
    assert "labels" not in source
    for name in original:
        torch_assert_equal(source[name], original[name])


@pytest.mark.parametrize(
    "fault",
    ["unknown", "left_padding", "empty_row", "nonbinary", "remainder", "shape", "dtype", "active_end", "active_pad"],
)
def test_token_partition_rejects_unsupported_domains_without_input_mutation(fault):
    source = _batch()
    if fault == "unknown":
        source["undeclared"] = torch.ones(2, 5)
    elif fault == "left_padding":
        source["attention_mask"][0] = torch.tensor([0, 1, 1, 1, 0])
    elif fault == "empty_row":
        source["attention_mask"][0].zero_()
    elif fault == "nonbinary":
        source["attention_mask"][0, 0] = 2
    elif fault == "remainder":
        source["attention_mask"][0, 3] = 1
    elif fault == "shape":
        source["advantages"] = torch.ones(2)
    elif fault == "dtype":
        source["position_ids"] = torch.zeros(2, 5)
    elif fault == "active_end":
        source["loss_mask"][0, 2] = 1
    else:
        source["loss_mask"][0, 4] = 1
    original = deepcopy(source)
    with pytest.raises(ValueError):
        sp.prepare_token_partition(source, sp_info=_info())
    for name in source:
        torch_assert_equal(source[name], original[name])


@pytest.mark.parametrize("fault", ["bool_rank", "local", "order", "duplicate", "world"])
def test_token_partition_rejects_false_rank_or_membership_binding(fault):
    info = _info()
    if fault == "bool_rank":
        info["rank"] = True
    elif fault == "local":
        info["sp_rank"] = 1
    elif fault == "order":
        info["sp_ranks"].reverse()
    elif fault == "duplicate":
        info["sp_ranks"] = [2, 2]
    else:
        info["world_size"] = 3
    with pytest.raises(ValueError):
        sp.prepare_token_partition(_batch(), sp_info=info)


def _partitions():
    return [sp.prepare_token_partition(_batch(), sp_info=_info(rank=rank))[-1] for rank in (0, 2)]


def test_token_assembly_uses_actual_group_order_and_restores_padding_with_gradient():
    parts = _partitions()
    values = [
        torch.tensor([[[4.0], [5.0], [6.0], [7.0]]], requires_grad=True),
        torch.tensor([[[0.0], [1.0], [2.0], [3.0]]], requires_grad=True),
    ]
    assembled = sp.assemble_token_tensor(values, parts)
    torch_assert_equal(assembled, torch.arange(8.0).reshape(1, 8, 1))
    _, _, mask, _, _ = sp.prepare_token_partition(_batch(), sp_info=_info())
    restored = sp.restore_token_rows(assembled, mask)
    torch_assert_equal(restored[:, :, 0], torch.tensor([[0.0, 1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 5.0, 6.0, 7.0]]))
    restored.sum().backward()
    for value in values:
        torch_assert_equal(value.grad, torch.ones_like(value))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "slice", "length", "dtype", "tail"])
def test_token_assembly_rejects_partial_or_inconsistent_rank_tensor_domain(fault):
    parts = _partitions()
    values = [torch.ones(1, 4), torch.ones(1, 4)]
    if fault == "missing":
        parts, values = parts[:1], values[:1]
    elif fault == "duplicate":
        parts[1] = parts[0]
    elif fault == "slice":
        parts[1] = replace(parts[1], token_count=10)
    elif fault == "length":
        values[1] = torch.ones(1, 3)
    elif fault == "dtype":
        values[1] = values[1].double()
    else:
        values[1] = torch.ones(1, 4, 2)
    with pytest.raises(ValueError):
        sp.assemble_token_tensor(values, parts)


class _Engine:
    global_rank = 2

    def __init__(self):
        self.calls = []
        self.backward_calls = []
        self.weight = torch.arange(9 * 4, dtype=torch.float32).reshape(9, 4) / 40

    def train(self):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["input_ids"]
        return {"logits": torch.nn.functional.one_hot(ids % 4, 4).float() @ self.weight.T}

    def backward(self, loss, *, scale_wrt_gas):
        self.backward_calls.append((loss.detach().clone(), scale_wrt_gas))


def _live_group(monkeypatch, part):
    group = object()
    monkeypatch.setattr(sp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sp.dist, "get_rank", lambda group=None: part.rank if group is None else part.local_rank)
    monkeypatch.setattr(sp.dist, "get_world_size", lambda group=None: part.world_size)
    monkeypatch.setattr(sp.dist, "get_process_group_ranks", lambda group: list(part.ranks))
    return group


def test_token_local_pipeline_calls_model_once_and_preserves_full_logical_loss(monkeypatch):
    from arctic_platform.rl.processors import functional
    from arctic_platform.rl.processors import run_pipeline
    from arctic_platform.rl.processors.compute_logprobs import compute_logprobs_post

    # The loss sees complete logical rows; this fixture only simulates the
    # model's token gather, not a native DeepSpeed loss-reduction group.
    monkeypatch.setattr(functional.groups, "_get_sequence_parallel_world_size", lambda: 1)
    source = _batch()
    _, packed, mask, cu, part = sp.prepare_token_partition(source, sp_info=_info())
    group = _live_group(monkeypatch, part)
    baseline_engine = _Engine()
    baseline = run_pipeline(
        baseline_engine,
        (),
        packed,
        {"cu_seqlens": cu, "dp_size": 2},
        {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": {"loss_agg_mode": "token-mean"}},
        "cpu",
        pack=False,
        backward=False,
        return_tensors=True,
    )
    peer_logits = _Engine()(**{"input_ids": packed["input_ids"][:, 4:]})
    peer_batch = {"labels": packed["labels"][:, 4:]}
    peer_logprobs = compute_logprobs_post(peer_logits, peer_batch, {}, "cpu")["logprobs"]
    gather_calls = []

    def gather(value, *, group):
        gather_calls.append((value.clone(), group))
        return (value, peer_logprobs)

    monkeypatch.setattr(sp.dist_nn, "all_gather", gather)
    engine = _Engine()
    result = sp.run_sequence_parallel_grpo(
        engine,
        source,
        {"dp_size": 2},
        {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": {"loss_agg_mode": "token-mean"}},
        "cpu",
        sp_info=_info(),
        sp_group=group,
    )
    assert len(engine.calls) == len(gather_calls) == len(engine.backward_calls) == 1
    torch_assert_equal(engine.calls[0]["input_ids"], packed["input_ids"][:, :4])
    torch_assert_equal(engine.calls[0]["labels"], packed["labels"][:, :4])
    torch_assert_equal(engine.calls[0]["routed_experts"], packed["routed_experts"][:, :4])
    assert gather_calls[0][0].shape == (1, 4)  # No full-vocabulary gather.
    assert result["avg_loss"] == baseline["avg_loss"]
    assert result["metrics"] == baseline["metrics"]
    torch_assert_close(engine.backward_calls[0][0], torch.tensor(result["avg_loss"] / 2), rtol=0, atol=0)
    assert engine.backward_calls[0][1] is False
    torch_assert_equal(result["batch"]["logprobs"], sp.restore_token_rows(baseline["batch"]["logprobs"], mask))


@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0])
def test_sp_execution_token_loss_scale_changes_only_backward(monkeypatch, backward, scale):
    from arctic_platform.rl.processors import functional
    from arctic_platform.rl.processors.compute_logprobs import compute_logprobs_post

    monkeypatch.setattr(functional.groups, "_get_sequence_parallel_world_size", lambda: 1)
    source = _batch()
    _, packed, _, _, part = sp.prepare_token_partition(source, sp_info=_info())
    group = _live_group(monkeypatch, part)
    peer = compute_logprobs_post(
        _Engine()(input_ids=packed["input_ids"][:, 4:]), {"labels": packed["labels"][:, 4:]}, {}, "cpu"
    )["logprobs"]
    monkeypatch.setattr(sp.dist_nn, "all_gather", lambda value, *, group: (value, peer))
    engine = _Engine()
    result = sp.run_sequence_parallel_grpo(
        engine,
        source,
        {"dp_size": 2},
        {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": {"loss_agg_mode": "token-mean"}},
        "cpu",
        sp_info=_info(),
        sp_group=group,
        backward=backward,
        loss_scale=scale,
    )
    assert len(engine.calls) == 1
    assert "loss_tensor" not in result
    assert math.isfinite(result["avg_loss"])
    assert result["avg_loss"] != 0
    assert result["batch"]["logprobs"].shape == source["input_ids"].shape
    if backward:
        assert len(engine.backward_calls) == 1
        loss, gas_scale = engine.backward_calls[0]
        torch_assert_equal(loss, torch.tensor(result["avg_loss"] * scale / 2))
        assert gas_scale is False
    else:
        assert not engine.backward_calls


@pytest.mark.parametrize("scale", [True, -1.0, float("inf"), float("nan")])
def test_sp_execution_rejects_invalid_loss_scale_before_model(monkeypatch, scale):
    part = sp.prepare_token_partition(_batch(), sp_info=_info())[-1]
    group = _live_group(monkeypatch, part)
    engine = _Engine()
    with pytest.raises(ValueError, match="packed loss scale"):
        sp.run_sequence_parallel_grpo(
            engine,
            _batch(),
            {"dp_size": 2},
            {"post": ["compute_logprobs"], "loss_fn": "grpo"},
            "cpu",
            sp_info=_info(),
            sp_group=group,
            loss_scale=scale,
        )
    assert not engine.calls
    assert not engine.backward_calls


def test_sp_execution_rejects_configured_dp_scale_mismatch_before_model(monkeypatch):
    part = sp.prepare_token_partition(_batch(), sp_info=_info())[-1]
    group = _live_group(monkeypatch, part)
    engine = _Engine()
    with pytest.raises(ValueError, match="logical-DP membership"):
        sp.run_sequence_parallel_grpo(
            engine,
            _batch(),
            {"dp_size": 2},
            {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": {"dp_size": 4}},
            "cpu",
            sp_info=_info(),
            sp_group=group,
        )
    assert not engine.calls


def _selected_sp_reducer(monkeypatch):
    from deepspeed.runtime.zero import stage_1_and_2

    class SelectedZero:
        pass

    monkeypatch.setattr(stage_1_and_2, "DeepSpeedZeroOptimizer", SelectedZero)
    info = _info()
    group, dp_group = object(), object()
    sp_group = group
    monkeypatch.setattr(sp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sp.dist, "get_rank", lambda group=None: 2 if group in (None, dp_group) else 0)
    monkeypatch.setattr(sp.dist, "get_world_size", lambda group=None: 2 if group is sp_group else 4)
    monkeypatch.setattr(
        sp.dist,
        "get_process_group_ranks",
        lambda group: list(info["sp_ranks"]) if group is sp_group else list(range(4)),
    )
    optimizer = SelectedZero()
    optimizer.sequence_parallel_size = 2
    optimizer.partition_gradients = True
    optimizer.reduce_scatter = True
    optimizer.postscale_gradients = True
    optimizer.gradient_predivide_factor = 1.0
    optimizer.dp_process_group = dp_group
    engine = SimpleNamespace(
        optimizer=optimizer,
        sequence_parallel_size=2,
        seq_data_parallel_group=dp_group,
        zero_optimization_stage=lambda: 2,
    )
    return engine, info, group


def test_sp_execution_selected_reducer_binds_dense_sequence_data_group(monkeypatch):
    engine, info, group = _selected_sp_reducer(monkeypatch)
    sp.verify_sequence_parallel_reduction(engine, sp_info=info, sp_group=group)


@pytest.mark.parametrize(
    "fault",
    [
        "class",
        "stage",
        "engine_sp",
        "optimizer_sp",
        "partition",
        "scatter",
        "postscale",
        "predivide",
        "group",
        "members",
    ],
)
def test_sp_execution_rejects_unselected_reducer_or_group_settings(monkeypatch, fault):
    engine, info, group = _selected_sp_reducer(monkeypatch)
    optimizer = engine.optimizer
    if fault == "class":
        engine.optimizer = SimpleNamespace(**vars(optimizer))
    elif fault == "stage":
        engine.zero_optimization_stage = lambda: 1
    elif fault == "engine_sp":
        engine.sequence_parallel_size = 1
    elif fault == "optimizer_sp":
        optimizer.sequence_parallel_size = 1
    elif fault == "partition":
        optimizer.partition_gradients = False
    elif fault == "scatter":
        optimizer.reduce_scatter = False
    elif fault == "postscale":
        optimizer.postscale_gradients = False
    elif fault == "predivide":
        optimizer.gradient_predivide_factor = 2.0
    elif fault == "group":
        optimizer.dp_process_group = object()
    else:
        monkeypatch.setattr(
            sp.dist,
            "get_process_group_ranks",
            lambda selected: list(info["sp_ranks"]) if selected is group else [0, 2],
        )
    with pytest.raises(ValueError, match="native ZeRO2"):
        sp.verify_sequence_parallel_reduction(engine, sp_info=info, sp_group=group)


@pytest.mark.parametrize("fault", ["post", "loss", "aggregation", "entropy", "meta_token", "dp", "group"])
def test_token_local_pipeline_rejects_unsupported_route_before_model(monkeypatch, fault):
    part = sp.prepare_token_partition(_batch(), sp_info=_info())[-1]
    group = _live_group(monkeypatch, part)
    processing = {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": {"loss_agg_mode": "token-mean"}}
    meta = {"dp_size": 2}
    if fault == "post":
        processing["post"] = ["ap_compute_logprobs"]
    elif fault == "loss":
        processing["loss_fn"] = "sft_ce"
    elif fault == "aggregation":
        processing["config"]["loss_agg_mode"] = "seq-mean-token-mean"
    elif fault == "entropy":
        meta["calculate_entropy"] = True
    elif fault == "meta_token":
        meta["advantages"] = torch.ones(2, 5)
    elif fault == "dp":
        meta["dp_size"] = 4
    else:
        monkeypatch.setattr(sp.dist, "get_process_group_ranks", lambda group: [0, 2])
    engine = _Engine()
    with pytest.raises(ValueError):
        sp.run_sequence_parallel_grpo(
            engine,
            _batch(),
            meta,
            processing,
            "cpu",
            sp_info=_info(),
            sp_group=group,
        )
    assert engine.calls == []
