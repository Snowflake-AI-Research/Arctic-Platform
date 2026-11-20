"""The worker returns the checksums each rank computed before loading its weights."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from test_fp32_lm_head import _load_worker_module


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(remote=lambda cls: cls))
    monkeypatch.setenv("ARCTIC_INFERENCE_DUMP_PARAM_L2", "0")
    monkeypatch.delenv("ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY", raising=False)
    return _load_worker_module(monkeypatch).InferenceWorker()


def _worker_load(worker, method, rank_results, *args, **kwargs):
    async def collective_rpc(name, args=()):
        assert name == method
        return rank_results

    worker.llm = SimpleNamespace(collective_rpc=collective_rpc)
    return asyncio.run(getattr(worker, method)(*args, **kwargs))


@pytest.mark.parametrize(
    "method, args",
    [
        ("load_weights_cuda_ipc", ({},)),
        ("load_weights_cuda_ipc_chunk", ({}, None)),
        ("load_weights_from_shm_path", ("/dev/shm/x",)),
        ("sync_weights_broadcast", ("localhost", 12345, 1, 3)),
    ],
)
def test_worker_reports_every_tp_ranks_received_checksum(worker, method, args):
    ranks = [{"recv_l2_sq": 32.5, "loaded": 3}, {"recv_l2_sq": 30.0, "loaded": 3}]
    result = _worker_load(worker, method, ranks, *args)
    assert result == {"recv_l2_sq": 32.5, "loaded": 3, "rank_recv_l2_sq": [32.5, 30.0]}


def test_worker_marks_a_rank_without_a_checksum(worker):
    result = _worker_load(worker, "load_weights_cuda_ipc", [{"recv_l2_sq": 32.5}, {"status": "done"}], {})
    assert result["rank_recv_l2_sq"] == [32.5, None]


def test_broadcast_keeps_checksums_with_parameter_traces(worker, monkeypatch):
    monkeypatch.setenv("ARCTIC_INFERENCE_DUMP_PARAM_L2", "1")
    ranks = [{"recv_l2_sq": 32.5, "param_l2": {"w": 2.5}}, {"param_l2": {"w": 4.0}}]
    result = _worker_load(worker, "sync_weights_broadcast", ranks, "localhost", 12345, 1, 3)
    assert result == {
        "recv_l2_sq": 32.5,
        "rank_recv_l2_sq": [32.5, None],
        "all_rank_param_l2": [{"w": 2.5}, {"w": 4.0}],
    }


def test_broadcast_rendezvous_reports_engine_state(worker):
    ranks = [{"status": "engine_ready", "rank": i, "world_size": 3} for i in (1, 2)]
    result = _worker_load(worker, "sync_weights_broadcast", ranks, "localhost", 12345, 1, 3, engine_only=True)
    assert result == ranks[0]
