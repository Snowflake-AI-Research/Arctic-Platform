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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from arctic_platform.model.implementations.qwen38 import host_table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gather_from_pinned_host_matches_embedding():
    torch.manual_seed(0)
    dim = 160
    weight = torch.randn(10_000, dim, dtype=torch.bfloat16, device="cuda")
    table = host_table.move_to_host(weight)
    ids = torch.randint(0, weight.shape[0], (3, 4096, 16), device="cuda")
    got = host_table.gather_rows(table, ids)
    assert got.device == ids.device and got.shape == (3, 4096, 16, dim)
    assert torch.equal(got, F.embedding(ids, weight))


@pytest.fixture
def table_on_host(monkeypatch):
    """ngram_table_on_host on the CPU: the host copy is a plain one (pinning needs a GPU) that lookups index."""
    pytest.importorskip("transformers.models.qwen4_exp")
    from arctic_platform.model.implementations.qwen38 import modeling_qwen4_exp

    monkeypatch.setattr(host_table, "_configured", True)
    monkeypatch.setattr(host_table, "move_to_host", lambda weight: weight.clone())
    monkeypatch.setattr(host_table, "gather_rows", lambda table, ids: F.embedding(ids, table))

    def exchange(output, tensor, **_kwargs):  # every lookup is this rank's: each exchange hands the tensor back
        return output.copy_(tensor)

    monkeypatch.setattr(modeling_qwen4_exp.dist, "all_to_all_single", exchange)
    monkeypatch.setattr(modeling_qwen4_exp.dist_nn, "all_to_all_single", exchange)
    return modeling_qwen4_exp.EPShardedEmbedding


def _model(embedding_cls, seed: int) -> nn.Module:
    """A model holding EP rank 1's shard (rows 4..6) of a 7-row n-gram table over two ranks, its host table held."""
    model = nn.Module()
    model.table = embedding_cls(7, 4)
    weight = torch.randn(3, 4, generator=torch.Generator().manual_seed(seed)).bfloat16()
    model.table.weight = nn.Parameter(weight, requires_grad=False)
    model.table.weight.allreduce, model.table.weight.group_name = False, "ep_size_2"  # an expert param to dss
    model.table._ep_rank, model.table._ep_world_size, model.table._ep_group = 1, 2, object()
    host_table.hold_host_tables(model, [model.table])
    return model


def test_a_checkpoint_saves_and_restores_the_rows_of_a_table_on_the_host(table_on_host):
    """dss's expert-parallel checkpointer, inside full_state_dict: DeepSpeed's module state and the expert side file
    (each EP-tagged parameter's data) hold the host rows, and a restore puts them in the host copy."""
    from arctic_platform.model import full_state_dict
    from arctic_platform.model.weight_export import materialized_parameters

    saved = _model(table_on_host, seed=0)
    rows = saved.table.weight.detach().clone()
    ids = torch.tensor([[4, 6, 5, 4]])
    want = saved.table(ids)  # the first forward moves the shard to the host
    assert saved.table.weight.numel() == 0 and torch.equal(saved.table._host_weight, rows)
    with pytest.raises(RuntimeError, match="whole_frozen_weights"):
        saved.state_dict()  # it would hold an empty table
    with full_state_dict(saved):
        module_state = {name: t.clone() for name, t in saved.state_dict().items()}
        expert_shard = {name: p.data.clone() for name, p in saved.named_parameters() if p.allreduce is False}
    assert torch.equal(module_state["table.weight"], rows) and torch.equal(expert_shard["table.weight"], rows)
    assert saved.table.weight.numel() == 0 and torch.equal(saved.table._host_weight, rows)
    with materialized_parameters(saved, [saved.table.weight]):  # the weights-only (HF) export
        assert torch.equal(saved.table.weight, rows)
    assert saved.table.weight.numel() == 0

    # Into a fresh job (its table still on the GPU: it moves at the first forward) and into one whose table moved.
    for moved in (False, True):
        restored = _model(table_on_host, seed=1)
        if moved:
            restored.table(ids)
        with full_state_dict(restored):
            restored.load_state_dict(module_state)
            for name, p in restored.named_parameters():
                if p.allreduce is False:  # as dss's _load_expert_shards writes them
                    p.data.copy_(expert_shard[name].to(device=p.data.device, dtype=p.data.dtype))
        got = restored.table(ids)
        assert restored.table.weight.numel() == 0, moved
        assert torch.equal(restored.table._host_weight, rows) and torch.equal(got, want), moved


def test_the_qwen38_loader_holds_the_host_tables_only_with_the_option(monkeypatch):
    pytest.importorskip("transformers.models.qwen4_exp")
    from arctic_platform.model.implementations.kernels import flashinfer_top_k
    from arctic_platform.model.implementations.moe.frozen_held import _holdings_under
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen_ds
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import load_qwen4_exp_model
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import EPShardedEmbedding

    monkeypatch.setattr(host_table, "_configured", None)
    monkeypatch.setattr(flashinfer_top_k, "prepare_flashinfer_top_k", lambda: None)
    monkeypatch.setattr(qwen_ds, "maybe_apply_row_invariant_projections", lambda model, options: None)
    for enabled in (None, True):
        model = nn.Module()
        model.table = EPShardedEmbedding(7, 4)
        monkeypatch.setattr(qwen_ds, "_load_moe_model", lambda *args, model=model, **kwargs: model)
        options = SimpleNamespace(ngram_table_on_host=enabled, model_dump=dict)
        load_qwen4_exp_model(
            model_name="m", optimization_dtype="bfloat16", attn_implementation="sdpa", ep_size=1, options=options
        )
        holdings = _holdings_under(model)
        if enabled:
            assert len(holdings) == 1 and holdings[0].embeddings == [model.table]
        else:
            assert holdings == []
