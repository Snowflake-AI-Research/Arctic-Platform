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
"""FlashInfer's top-k module is built at model load and never inside a step (kernels/flashinfer_top_k.py).

CPU: FlashInfer's build is mocked. The load path builds it once and a failed build raises; the step path never builds
it, raising on a node where it is not built, and selects with torch.topk without FlashInfer or on CPU tensors; the
indexer models' loaders prepare it before anything else. GPU: with the module prepared, the selection is the k
largest scores, ties to the smaller column.
"""

from __future__ import annotations

import types

import pytest
import torch

from arctic_platform.model.implementations.kernels import flashinfer_top_k


class _Spec:
    def __init__(self, compiled: bool):
        self.is_aot = False
        self.is_compiled = compiled


def _fake_flashinfer(monkeypatch, *, compiled: bool, build=None) -> dict:
    """FlashInfer with a mocked top-k module: ``calls`` counts builds (get_topk_module) and selections."""
    calls = {"build": 0, "select": 0}

    def get_topk_module():
        calls["build"] += 1
        if build is not None:
            build()
        return object()

    def top_k(scores, k, **kwargs):
        calls["select"] += 1
        assert kwargs == {"sorted": False, "deterministic": True, "tie_break": "small"}
        return None, torch.topk(scores, k, dim=-1).indices

    tie_break = types.SimpleNamespace(SMALL="small")
    monkeypatch.setattr(flashinfer_top_k, "_flashinfer", lambda: (top_k, tie_break, lambda: _Spec(compiled)))
    monkeypatch.setattr(flashinfer_top_k, "_ready", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    import flashinfer.topk

    monkeypatch.setattr(flashinfer.topk, "get_topk_module", get_topk_module)
    return calls


def test_the_load_path_builds_the_module_once(monkeypatch):
    pytest.importorskip("flashinfer")
    calls = _fake_flashinfer(monkeypatch, compiled=False)
    flashinfer_top_k.prepare_flashinfer_top_k()
    flashinfer_top_k.prepare_flashinfer_top_k()
    assert calls["build"] == 1 and flashinfer_top_k._ready
    assert flashinfer_top_k._ready_top_k() is not None, "the step path reads the prepared module"
    assert calls["build"] == 1


def test_a_failed_build_raises_at_load(monkeypatch):
    pytest.importorskip("flashinfer")

    def fail():
        raise RuntimeError("nvcc: not found")

    _fake_flashinfer(monkeypatch, compiled=False, build=fail)
    with pytest.raises(RuntimeError, match="failed to build on .*nvcc: not found.*fi_prebuild.sh"):
        flashinfer_top_k.prepare_flashinfer_top_k()
    assert not flashinfer_top_k._ready


def test_the_step_path_never_builds_the_module(monkeypatch):
    pytest.importorskip("flashinfer")
    calls = _fake_flashinfer(monkeypatch, compiled=False)
    with pytest.raises(RuntimeError, match="not built on .*inside a step"):
        flashinfer_top_k._ready_top_k()
    assert calls["build"] == 0
    # Built ahead of time (the node's JIT cache, or AOT kernels): used as it is, still without a build here.
    calls = _fake_flashinfer(monkeypatch, compiled=True)
    assert flashinfer_top_k._ready_top_k() is not None and calls["build"] == 0


def test_without_flashinfer_or_on_cpu_tensors_torch_topk_selects(monkeypatch):
    scores = torch.randn(3, 64)
    want = torch.topk(scores, 5, dim=-1).indices
    monkeypatch.setattr(flashinfer_top_k, "_flashinfer", lambda: None)
    flashinfer_top_k.prepare_flashinfer_top_k()  # nothing to build
    assert torch.equal(flashinfer_top_k.topk_indices(scores, 5), want)
    pytest.importorskip("flashinfer")
    calls = _fake_flashinfer(monkeypatch, compiled=False)
    assert torch.equal(flashinfer_top_k.topk_indices(scores, 5), want), "a CPU tensor never reaches FlashInfer"
    assert calls == {"build": 0, "select": 0}


def test_the_glm_dsa_loader_prepares_the_module_before_it_loads(monkeypatch):
    from arctic_platform.model.implementations.glm52 import deepspeed_integration

    order = []
    monkeypatch.setattr(flashinfer_top_k, "prepare_flashinfer_top_k", lambda: order.append("prepare"))

    def load(*args, **kwargs):
        order.append("load")
        raise _Stop

    monkeypatch.setattr(deepspeed_integration, "_load_moe_model", load)
    options = types.SimpleNamespace(shard_dense=None, tiled_mlp_token_chunk_size=None)
    with pytest.raises(_Stop):
        deepspeed_integration.load_glm_moe_dsa_model(
            model_name="m", optimization_dtype="bfloat16", attn_implementation="sparse_mla", ep_size=1, options=options
        )
    assert order == ["prepare", "load"]


class _Stop(Exception):
    pass


def _barrier_worker(rank: int, rendezvous: str, results) -> None:
    import logging
    import time

    import torch.distributed as dist

    class _Ready(logging.Handler):
        def emit(self, record):
            self.args = record.args  # (host, own build seconds, seconds until every rank is ready)

    ready = _Ready()
    flashinfer_top_k.logger.addHandler(ready)
    flashinfer_top_k.logger.setLevel(logging.INFO)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    calls = _fake_flashinfer(pytest.MonkeyPatch(), compiled=False, build=(lambda: time.sleep(3)) if rank else None)
    dist.barrier()  # both ranks start together, after their imports
    start = time.monotonic()
    flashinfer_top_k.prepare_flashinfer_top_k()
    results[rank] = (calls["build"], time.monotonic() - start, ready.args[1])
    dist.destroy_process_group()


def test_the_ranks_of_warm_nodes_wait_for_a_cold_build_on_the_cpu(tmp_path):
    """Rank 1's build takes 3 s; rank 0 (warm) returns from the load step only after it, from a gloo barrier. Each
    rank logs its own build time, before the wait."""
    pytest.importorskip("flashinfer")
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(_barrier_worker, args=(f"file://{tmp_path}/rendezvous", results), nprocs=2, join=True)
    assert results[0][0] == results[1][0] == 1
    assert results[0][1] >= 2.5, dict(results)
    assert results[0][2] < 1.0 <= 2.5 <= results[1][2], dict(results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_prepared_module_selects_the_largest_scores():
    pytest.importorskip("flashinfer")
    flashinfer_top_k.prepare_flashinfer_top_k()
    scores = torch.randn(8, 131072, device="cuda")
    scores[0] = -10.0
    scores[0, :4096] = 0.0  # a tie across the boundary goes to the smaller columns
    got = flashinfer_top_k.topk_indices(scores, 2048)
    want = torch.topk(scores, 2048, dim=-1).indices
    assert torch.equal(got[1:].sort(-1).values, want[1:].sort(-1).values)
    assert torch.equal(got[0].sort().values, torch.arange(2048, device="cuda"))
