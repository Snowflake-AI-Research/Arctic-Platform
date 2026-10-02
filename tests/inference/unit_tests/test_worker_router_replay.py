from types import SimpleNamespace

import pytest
import torch

from arctic_platform.inference.server.router_replay import RouterReplayCacheTX
from arctic_platform.inference.server.worker import (
    _has_string_stop,
    _normalize_stop_token_sequences,
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


def test_router_replay_stop_token_sequence_normalization():
    assert _has_string_stop({"stop": ["</answer>"]}) is True
    assert _has_string_stop({"stop": [1, 2]}) is False

    assert _normalize_stop_token_sequences([[1, 2], {"token_ids": [3], "include_in_output": True}]) == [
        {"token_ids": [1, 2], "include_in_output": False},
        {"token_ids": [3], "include_in_output": True},
    ]
