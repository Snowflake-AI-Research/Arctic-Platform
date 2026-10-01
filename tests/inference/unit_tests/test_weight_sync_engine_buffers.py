from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

if "ray" not in sys.modules:
    ray_module = types.ModuleType("ray")
    ray_module.actor = SimpleNamespace(ActorHandle=object)

    def remote(*args, **kwargs):
        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(obj):
            return obj

        return decorator

    ray_module.remote = remote
    ray_module.init = lambda *args, **kwargs: None
    ray_module.nodes = lambda: []
    ray_module.get = lambda *args, **kwargs: None
    ray_module.kill = lambda *args, **kwargs: None
    sys.modules["ray"] = ray_module

if "vllm" not in sys.modules:
    vllm_module = types.ModuleType("vllm")
    config_module = types.ModuleType("vllm.config")
    loggers_module = types.ModuleType("vllm.v1.metrics.loggers")
    stats_module = types.ModuleType("vllm.v1.metrics.stats")

    class VllmConfig:
        pass

    class StatLoggerBase:
        pass

    class IterationStats:
        pass

    class SchedulerStats:
        pass

    config_module.VllmConfig = VllmConfig
    loggers_module.StatLoggerBase = StatLoggerBase
    stats_module.IterationStats = IterationStats
    stats_module.SchedulerStats = SchedulerStats

    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.config"] = config_module
    sys.modules["vllm.v1"] = types.ModuleType("vllm.v1")
    sys.modules["vllm.v1.metrics"] = types.ModuleType("vllm.v1.metrics")
    sys.modules["vllm.v1.metrics.loggers"] = loggers_module
    sys.modules["vllm.v1.metrics.stats"] = stats_module

if "arctic_inference.server" not in sys.modules:
    package_root = Path(__file__).parents[3] / "inference" / "arctic_inference" / "server"
    server_module = types.ModuleType("arctic_inference.server")
    server_module.__path__ = [str(package_root)]
    weight_sync_module = types.ModuleType("arctic_inference.server.weight_sync")
    weight_sync_module.__path__ = [str(package_root / "weight_sync")]
    sys.modules["arctic_inference.server"] = server_module
    sys.modules["arctic_inference.server.weight_sync"] = weight_sync_module

from arctic_inference.server.weight_sync.engine import NCCLEngine


class _FakeStream:
    def __init__(self) -> None:
        self.ops: list[tuple] = []

    def wait_event(self, event) -> None:
        self.ops.append(("wait_event", event.name))

    def wait_stream(self, stream) -> None:
        self.ops.append(("wait_stream", stream))

    def synchronize(self) -> None:
        self.ops.append(("synchronize",))


class _FakeEvent:
    _next = 0

    def __init__(self) -> None:
        self.name = f"event-{_FakeEvent._next}"
        _FakeEvent._next += 1
        self.recorded_on = None

    def record(self, stream) -> None:
        self.recorded_on = stream
        stream.ops.append(("record_event", self.name))


def test_receive_weights_waits_for_consumer_before_reusing_bucket(monkeypatch):
    default_stream = _FakeStream()
    nccl_stream = _FakeStream()

    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: default_stream)

    engine = NCCLEngine.__new__(NCCLEngine)
    engine.is_sender = False
    engine.device = torch.device("cpu")
    engine._nccl_stream = nccl_stream
    engine._data_bufs = [
        torch.zeros(8, dtype=torch.uint8),
        torch.zeros(8, dtype=torch.uint8),
    ]
    engine._meta_by_buf = {}

    recv_metas = [
        (
            0,
            {
                "is_last": False,
                "tensors": [
                    {
                        "name": "a",
                        "shape": [1],
                        "dtype": "torch.uint8",
                        "offset": 0,
                        "nbytes": 1,
                    }
                ],
            },
        ),
        (
            1,
            {
                "is_last": False,
                "tensors": [
                    {
                        "name": "b",
                        "shape": [1],
                        "dtype": "torch.uint8",
                        "offset": 0,
                        "nbytes": 1,
                    }
                ],
            },
        ),
        (
            0,
            {
                "is_last": True,
                "tensors": [
                    {
                        "name": "c",
                        "shape": [1],
                        "dtype": "torch.uint8",
                        "offset": 0,
                        "nbytes": 1,
                    }
                ],
            },
        ),
    ]

    def recv_pair(idx, stream):
        expected_idx, metadata = recv_metas.pop(0)
        assert idx == expected_idx
        stream.ops.append(("recv_pair", idx))
        engine._meta_by_buf[idx] = metadata

    def parse_meta(idx):
        return engine._meta_by_buf[idx]

    engine._recv_pair = recv_pair
    engine._parse_meta = parse_meta

    assert [name for name, _ in engine.receive_weights()] == ["a", "b", "c"]

    assert nccl_stream.ops[:3] == [
        ("wait_stream", default_stream),
        ("recv_pair", 0),
        ("synchronize",),
    ]
    reuse_idx = nccl_stream.ops.index(("recv_pair", 0), 4)
    assert ("wait_event", "event-0") in nccl_stream.ops[:reuse_idx]
    assert ("record_event", "event-0") in default_stream.ops


def test_bucket_transfer_chunks_and_reassembles_oversized_tensor(monkeypatch):
    default_stream = _FakeStream()
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: default_stream)

    sender = NCCLEngine.__new__(NCCLEngine)
    sender.is_sender = True
    sender.device = torch.device("cpu")
    sender.bucket_size = 4
    sender._port = 1234
    sender._nccl_stream = _FakeStream()
    sender._data_bufs = [
        torch.zeros(4, dtype=torch.uint8),
        torch.zeros(4, dtype=torch.uint8),
    ]
    sender.ensure_staging_buffers = lambda: None

    buckets = []

    def flush(idx, meta, *, is_last, prev_idx):
        del prev_idx
        buckets.append(
            (
                idx,
                {"tensors": [dict(entry) for entry in meta], "is_last": is_last},
                sender._data_bufs[idx].clone(),
            )
        )

    sender._flush_bucket_send = flush
    result = sender.send_weights(
        [
            ("large", torch.arange(10, dtype=torch.uint8)),
            ("tail", torch.tensor([10, 11], dtype=torch.uint8)),
        ]
    )

    assert result["params_sent"] == 2
    assert result["buckets"] == 3
    assert [metadata["is_last"] for _, metadata, _ in buckets] == [
        False,
        False,
        True,
    ]
    assert [
        metadata["tensors"][0].get("tensor_offset")
        for _, metadata, _ in buckets
    ] == [0, 4, 8]
    assert buckets[-1][1]["tensors"][1]["name"] == "tail"
    assert [data.tolist() for _, _, data in buckets] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]

    receiver = NCCLEngine.__new__(NCCLEngine)
    receiver.is_sender = False
    receiver.device = torch.device("cpu")
    receiver._nccl_stream = _FakeStream()
    receiver._data_bufs = [
        torch.zeros(4, dtype=torch.uint8),
        torch.zeros(4, dtype=torch.uint8),
    ]
    receiver.ensure_staging_buffers = lambda: None
    receiver._meta_by_buf = {}
    pending = list(buckets)

    def recv_pair(idx, stream):
        expected_idx, metadata, data = pending.pop(0)
        assert idx == expected_idx
        receiver._data_bufs[idx].copy_(data)
        receiver._meta_by_buf[idx] = metadata
        stream.ops.append(("recv_pair", idx))

    receiver._recv_pair = recv_pair
    receiver._parse_meta = lambda idx: receiver._meta_by_buf[idx]

    received = {
        name: tensor.clone() for name, tensor in receiver.receive_weights()
    }
    torch.testing.assert_close(received["large"], torch.arange(10, dtype=torch.uint8))
    torch.testing.assert_close(
        received["tail"], torch.tensor([10, 11], dtype=torch.uint8)
    )
