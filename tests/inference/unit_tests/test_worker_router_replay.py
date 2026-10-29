import asyncio
from types import SimpleNamespace

import pytest
import torch

import arctic_platform.inference.server.worker as worker_module
from arctic_platform.inference.server import router_replay
from arctic_platform.inference.server.router_replay import RouterReplayCacheTX
from arctic_platform.inference.server.worker import (
    InferenceWorker,
    _has_string_stop,
    _normalize_stop_token_sequences,
    _pop_router_replay_engine_kwargs,
    _result_from_output,
)


class _Cache:
    def __init__(self):
        self.puts = []
        self.put_news = []

    def put(self, sample_id, tensor):
        self.puts.append((sample_id, tensor.clone()))

    def put_new(self, sample_id, tensor):
        self.put_news.append((sample_id, tensor.clone()))


def test_result_from_output_caches_routed_experts():
    routed = torch.tensor(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dtype=torch.int64,
    )
    output = SimpleNamespace(
        prompt_token_ids=[10, 11],
        num_cached_tokens=1,
        prompt_logprobs=None,
        routed_experts=routed,
        outputs=[
            SimpleNamespace(
                text="ok",
                token_ids=[12],
                finish_reason="stop",
                logprobs=None,
            )
        ],
    )
    cache = _Cache()

    result = _result_from_output(
        output,
        return_sampled_logprobs_only=False,
        cache_tx=cache,
        sample_id="sample-1",
        replay_id="rr1:attempt-1",
        request_id="request-1",
        replica_label="replica-a",
        return_back_router_info=True,
    )

    assert cache.puts == []
    assert cache.put_news[0][0] == "rr1:attempt-1"
    assert torch.equal(cache.put_news[0][1], routed)
    assert result["router_replay"]["sample_id"] == "rr1:attempt-1"
    assert result["router_replay"]["replay_id"] == "rr1:attempt-1"
    assert result["router_replay"]["trajectory_id"] == "sample-1"
    assert result["router_replay"]["request_id"] == "request-1"
    assert result["router_replay"]["replica"] == "replica-a"
    assert result["router_replay"]["prompt_len"] == 2
    assert result["router_replay"]["generation_len"] == 1
    assert result["router_replay"]["capture_len"] == 2
    assert result["router_replay"]["routed_experts"] == routed.tolist()


def test_result_from_late_exact_output_honors_discard_tombstone():
    routed = torch.tensor(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dtype=torch.int64,
    )
    output = SimpleNamespace(
        prompt_token_ids=[10, 11],
        num_cached_tokens=1,
        prompt_logprobs=None,
        routed_experts=routed,
        outputs=[
            SimpleNamespace(
                text="late",
                token_ids=[12],
                finish_reason="stop",
                logprobs=None,
            )
        ],
    )
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    cache.discard(["rr1:late-attempt"])

    result = _result_from_output(
        output,
        return_sampled_logprobs_only=False,
        cache_tx=cache,
        sample_id="sample-1",
        replay_id="rr1:late-attempt",
        request_id="request-1",
        replica_label="replica-a",
    )

    assert "rr1:late-attempt" not in cache
    assert result["router_replay"]["replay_id"] == "rr1:late-attempt"


def test_result_from_output_rejects_capture_length_mismatch_before_cache_write():
    routed = torch.zeros((3, 2, 2), dtype=torch.int64)
    output = SimpleNamespace(
        prompt_token_ids=[10, 11],
        num_cached_tokens=0,
        prompt_logprobs=None,
        routed_experts=routed,
        outputs=[
            SimpleNamespace(
                text="ok",
                token_ids=[12],
                finish_reason="stop",
                logprobs=None,
            )
        ],
    )
    cache = _Cache()

    with pytest.raises(
        ValueError,
        match=r"prompt_len=2 generation_len=1 capture_len=3 expected_capture_len=2",
    ):
        _result_from_output(
            output,
            return_sampled_logprobs_only=False,
            cache_tx=cache,
            sample_id="sample-1",
            replay_id="rr1:attempt-1",
            request_id="request-1",
            replica_label="replica-a",
        )

    assert cache.puts == []
    assert cache.put_news == []


def test_router_replay_cache_device_is_popped_and_validated_before_engine_start():
    kwargs = {"model": "m", "router_replay_max_cache_bytes": 1234, "router_replay_cache_device": "cpu"}

    assert _pop_router_replay_engine_kwargs(kwargs) == {
        "router_replay_max_cache_bytes": 1234,
        "router_replay_cache_device": "cpu",
    }
    assert kwargs == {"model": "m"}
    with pytest.raises(ValueError, match="router_replay_cache_device must be one of"):
        _pop_router_replay_engine_kwargs({"router_replay_cache_device": "gpu"})


@pytest.mark.parametrize("staging_bytes", [0, -1, True, "8MiB"])
def test_router_replay_staging_bytes_must_be_a_positive_integer(staging_bytes):
    assert _pop_router_replay_engine_kwargs({"router_replay_staging_bytes": 1 << 23}) == {
        "router_replay_staging_bytes": 1 << 23,
    }
    with pytest.raises(ValueError, match="router_replay_staging_bytes must be a positive integer"):
        _pop_router_replay_engine_kwargs({"router_replay_staging_bytes": staging_bytes})


@pytest.mark.parametrize("cache_device,expected", [(None, "cuda:0"), ("cuda", "cuda:0"), ("cpu", "cpu")])
def test_tx_cache_is_placed_on_the_configured_device(monkeypatch, cache_device, expected):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    worker = InferenceWorker.__ray_metadata__.modified_class()
    router_kwargs = {"router_replay_max_cache_bytes": 1 << 20}
    if cache_device is not None:
        router_kwargs["router_replay_cache_device"] = cache_device
        router_kwargs["router_replay_staging_bytes"] = 1 << 23

    worker._maybe_init_router_replay_tx({"enable_return_routed_experts": True}, router_kwargs)

    assert worker._router_replay_tx.device == torch.device(expected)
    assert worker._router_replay_tx.max_bytes == 1 << 20
    assert worker._router_replay_tx_on_host is (cache_device == "cpu")
    assert worker._router_replay_staging_bytes == (None if cache_device is None else 1 << 23)


def test_nccl_init_keeps_a_host_tx_cache_and_its_entries(monkeypatch):
    groups = []
    monkeypatch.setattr(worker_module, "_clear_ray_blanked_cuda_visible_devices", lambda: False)
    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", lambda: 1, raising=False)
    monkeypatch.setattr(router_replay, "init_router_replay_group", lambda **kw: groups.append(kw) or "group")
    tx = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    tx.put("rr1:a", torch.ones(4, 2, 2, dtype=torch.uint8))
    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker._router_replay_tx = tx
    worker._router_replay_tx_on_host = True
    worker._router_replay_staging_bytes = 1 << 23
    worker._router_replay_group = None

    result = asyncio.run(worker.init_router_replay("127.0.0.1", 1, rank=2, world_size=3))

    assert result["status"] == "ok"
    assert worker._router_replay_tx is tx and "rr1:a" in tx
    assert groups[0]["device"] == torch.device("cuda:0")
    assert groups[0]["staging_bytes"] == 1 << 23
    assert worker._router_replay_group == "group"


def test_router_replay_stop_token_sequence_normalization():
    assert _has_string_stop({"stop": ["</answer>"]}) is True
    assert _has_string_stop({"stop": [1, 2]}) is False

    assert _normalize_stop_token_sequences([[1, 2], {"token_ids": [3], "include_in_output": True}]) == [
        {"token_ids": [1, 2], "include_in_output": False},
        {"token_ids": [3], "include_in_output": True},
    ]
