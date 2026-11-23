import asyncio
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import arctic_platform.inference.server.worker as worker_module
from arctic_platform.inference.server import router_replay
from arctic_platform.inference.server.router_replay import RouterReplayCacheTX
from arctic_platform.inference.server.worker import (
    InferenceWorker,
    WorkerLifecycleState,
    _has_string_stop,
    _normalize_stop_token_sequences,
    _pop_router_replay_engine_kwargs,
    _result_from_output,
)
from arctic_platform.inference.vllm.forced_routes import FORCED_ROUTES_KEY
from vllm.v1.metrics.stats import RequestStateStats


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


def test_a_request_naming_its_previous_capture_keeps_those_routes_on_the_positions_it_covers():
    # Turn 1: prompt 3 ids, decodes 2 (capture rows 0-3, decode routes 12, 13 at 2-3). Turn 2 re-prefills rows 2-3,
    # whose block turn 1 never completed, so its engine capture holds its own prefill routes 22, 23 there.
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker.state, worker._router_replay_tx, worker._replica_label = WorkerLifecycleState.READY, cache, "replica-a"
    # Hits stop on 2-token blocks, one block lower under speculative decoding.
    worker._forced_route_blocks = (2, 2)
    engine_params = []

    def generate(prompt, tokens, routes, replay_id, *, metrics=None, **extra_args):
        async def fake_generate_once(_prompt, params, _request_id, **_kwargs):
            engine_params.append(params)
            # vLLM returns one numpy row per captured position.
            routed = np.repeat(np.asarray(routes, dtype=np.uint8).reshape(-1, 1, 1), 2, axis=2)
            choice = SimpleNamespace(text="ok", token_ids=tokens, finish_reason="stop", logprobs=None)
            return SimpleNamespace(
                outputs=[choice],
                prompt_token_ids=prompt,
                num_cached_tokens=0,
                prompt_logprobs=None,
                routed_experts=routed,
                metrics=metrics,
            )

        worker._generate_once = fake_generate_once
        params = dict(max_tokens=4, dss_sample_id="trajectory", dss_router_replay_id=replay_id, extra_args=extra_args)
        return asyncio.run(worker.generate(prompt, params))["router_replay"]

    def cached(replay_id):
        return cache.get(replay_id)[:, 0, 0].tolist()

    turn1 = generate([1, 2, 3], [4, 5], [10, 11, 12, 13], "rr1:turn-1")
    turn2 = generate(
        [1, 2, 3, 4, 5, 6, 7], [8], [10, 11, 22, 23, 24, 25, 26], "rr1:turn-2", dss_router_replay_prefix_id="rr1:turn-1"
    )
    assert cached("rr1:turn-2") == [10, 11, 12, 13, 24, 25, 26]
    assert (turn1["prefix_len"], turn2["prefix_len"], turn2["capture_len"]) == (0, 4, 7)
    # Nothing was cached, so rows 0-1 ran on the engine's own routes under stored ones; no stats, no preemption count.
    assert (turn2["rerouted_ranges"], turn2["preemptions"]) == ([[0, 2]], None)
    assert cached("rr1:turn-1") == [10, 11, 12, 13]  # read, never released: the caller owns its captures
    # The prefix id never reaches vLLM; the stored rows from the lowest block boundary the hit can stop at do.
    assert engine_params[-1].extra_args == {
        FORCED_ROUTES_KEY: {"start": 2, "dtype": "|u1", "rows": bytes([12, 12, 13, 13])}
    }
    # A capture ending on a block boundary, without the drop, is a hit the continuation can take whole: no rows.
    worker._forced_route_blocks = (2, 1)
    aligned = generate(
        [1, 2, 3, 4, 5], [6], [20, 21, 22, 23, 24], "rr1:aligned", dss_router_replay_prefix_id="rr1:turn-1"
    )
    assert engine_params[-1].extra_args is None and aligned["prefix_len"] == 4
    assert aligned["rerouted_ranges"] == [[0, 4]]
    # After the engine's prefix cache is reset, no KV of the capture is left to hit: every stored row is sent.
    async def reset_prefix_cache():
        return True

    worker.llm = SimpleNamespace(reset_prefix_cache=reset_prefix_cache)
    asyncio.run(worker.reset_prefix_cache())
    after_reset = generate(
        [1, 2, 3, 4, 5], [6], [20, 21, 22, 23, 24], "rr1:after-reset", dss_router_replay_prefix_id="rr1:turn-1",
        metrics=RequestStateStats(num_preemptions=1),
    )
    assert engine_params[-1].extra_args == {
        FORCED_ROUTES_KEY: {"start": 0, "dtype": "|u1", "rows": bytes([10, 10, 11, 11, 12, 12, 13, 13])}
    }
    assert (after_reset["rerouted_ranges"], after_reset["preemptions"]) == ([], 1)

    # Another replica served the previous request: the capture stays as the engine computed it.
    elsewhere = generate(
        [1, 2, 3, 4, 5], [6], [10, 11, 32, 33, 34], "rr1:turn-3", dss_router_replay_prefix_id="rr1:other"
    )
    assert elsewhere["prefix_len"] == 0 and cached("rr1:turn-3") == [10, 11, 32, 33, 34]
    assert engine_params[-1].extra_args is None

    # A prefix capture must end before the prompt does; the request is refused before anything is cached.
    with pytest.raises(ValueError, match="must end inside the prompt"):
        generate([1, 2, 3, 4, 5, 6], [7], [0] * 6, "rr1:turn-4", dss_router_replay_prefix_id="rr1:turn-2")
    assert "rr1:turn-4" not in cache and "rr1:turn-2" in cache
    # Only the server sets the stored rows a request runs under.
    with pytest.raises(ValueError, match="is set by the server"):
        generate([1, 2], [3], [0, 0], "rr1:turn-5", **{FORCED_ROUTES_KEY: {}})
    prefix_only = dict(max_tokens=1, extra_args=dict(dss_router_replay_prefix_id="rr1:turn-2"))
    with pytest.raises(ValueError, match="dss_router_replay_prefix_id requires dss_router_replay_id"):
        asyncio.run(worker.generate([1], prefix_only))


def _continuation_worker():
    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker.state = WorkerLifecycleState.READY
    worker._router_replay_tx = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    worker._forced_route_blocks = (2, 2)
    forwards = []

    async def generate(prefix=None, before_reply=None, **sampling):
        prompt = list(range(4 if prefix is None else 5))

        async def engine_reply(_prompt, params, _id, **kwargs):
            rows = np.arange(len(prompt), dtype=np.uint8).reshape(-1, 1, 1)
            if prefix is not None:
                rows += 10  # Cold prefill chooses different experts from the parent.
                forced = (params.extra_args or {}).get(FORCED_ROUTES_KEY)
                if forced:
                    stored = np.frombuffer(forced["rows"], dtype=forced["dtype"]).reshape(-1, 1, 1)
                    rows[forced["start"] : forced["start"] + len(stored)] = stored
            forwards.append(rows)
            choice = SimpleNamespace(text="", token_ids=[5], finish_reason="length", logprobs=None)
            if before_reply:
                await before_reply()
            return SimpleNamespace(outputs=[choice], prompt_token_ids=prompt, num_cached_tokens=0,
                                   prompt_logprobs=None, routed_experts=rows)

        worker._generate_once = engine_reply
        key = "parent" if prefix is None else "rr1:child"
        params = dict(max_tokens=1, dss_sample_id=key, dss_router_replay_id=key,
                      extra_args={} if prefix is None else {"dss_router_replay_prefix_id": prefix}, **sampling)
        return (await worker.generate(prompt, params))["router_replay"]

    return worker, generate, forwards


def test_prompt_logprobs_that_require_a_cold_prefill_use_all_parent_rows():
    async def run():
        _, generate, forwards = _continuation_worker()
        await generate()
        marker = await generate("parent", prompt_logprobs=0)
        assert np.array_equal(forwards[0], forwards[1][:4])
        assert marker["rerouted_ranges"] == []

    asyncio.run(run())


@pytest.mark.parametrize("late_parent", [False, True], ids=["during-reset-rpc", "late-parent-reply"])
def test_reset_overlapping_generation_replays_every_stored_row(late_parent):
    async def run():
        worker, generate, forwards = _continuation_worker()
        cleared, reply = asyncio.Event(), asyncio.Event()

        async def reset():
            cleared.set()
            await reply.wait()
            return True

        worker.llm = SimpleNamespace(reset_prefix_cache=reset)
        if late_parent:
            reply.set()
            await generate(before_reply=worker.reset_prefix_cache)
            marker = await generate("parent")
        else:
            await generate()
            resetting = asyncio.create_task(worker.reset_prefix_cache())
            await cleared.wait()
            child = asyncio.create_task(generate("parent"))
            await asyncio.sleep(0)  # Admit the continuation before acknowledging the cache clear.
            reply.set()
            await resetting
            marker = await child
        assert np.array_equal(forwards[0], forwards[1][:4])
        return marker

    marker = asyncio.run(run())
    assert marker["prefix_len"] == 4 and marker["rerouted_ranges"] == []


def test_a_parent_discarded_during_its_continuation_is_not_held_and_not_spliced():
    async def run():
        worker, generate, forwards = _continuation_worker()
        await generate()
        cache = worker._router_replay_tx
        parent = weakref.ref(cache.get("parent"))

        async def change_parent():
            cache.discard(["parent"])
            assert parent() is None  # nothing but the cache held the parent while the child ran

        marker = await generate("parent", before_reply=change_parent)
        assert marker["prefix_len"] == 0
        assert np.array_equal(cache.get("rr1:child").numpy(), forwards[1])

    asyncio.run(run())


def test_router_replay_cache_device_is_popped_and_validated_before_engine_start():
    kwargs = {"model": "m", "router_replay_max_cache_bytes": 1234, "router_replay_cache_device": "cpu"}

    assert _pop_router_replay_engine_kwargs(kwargs) == {
        "router_replay_max_cache_bytes": 1234,
        "router_replay_cache_device": "cpu",
    }
    assert kwargs == {"model": "m"}
    with pytest.raises(ValueError, match="router_replay_cache_device must be one of"):
        _pop_router_replay_engine_kwargs({"router_replay_cache_device": "gpu"})


@pytest.mark.parametrize("staging_bytes", [0, -1, True, "8MiB", 1.5])
def test_router_replay_staging_bytes_must_be_a_positive_integer(staging_bytes):
    assert _pop_router_replay_engine_kwargs({"router_replay_staging_bytes": 1 << 23}) == {
        "router_replay_staging_bytes": 1 << 23,
    }
    whole_float = _pop_router_replay_engine_kwargs({"router_replay_staging_bytes": float(1 << 23)})
    assert whole_float == {"router_replay_staging_bytes": 1 << 23}
    assert type(whole_float["router_replay_staging_bytes"]) is int
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
