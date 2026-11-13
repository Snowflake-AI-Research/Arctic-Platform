import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import torch

from arctic_platform.inference.server import api
from arctic_platform.inference.server.replica_pool import ReplicaPool
from arctic_platform.inference.server.router_replay import RouterReplayCacheTX, RouterReplayPrefixMissingError
from arctic_platform.inference.server.router_replay.cache import RouterReplayCacheFull
from arctic_platform.inference.server.router_replay_request import strict_replay_options
from arctic_platform.inference.server.scheduler import Scheduler
from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState


def _params(replay_id, prefix=None, owner="owner-a"):
    extra = {"dss_router_replay_strict": True}
    if prefix is not None:
        extra.update(dss_router_replay_prefix_id=prefix, dss_router_replay_prefix_replica=owner)
    return dict(max_tokens=1, dss_sample_id="trajectory", dss_router_replay_id=replay_id, extra_args=extra)


def _worker(max_bytes=1024, dtype=torch.uint8):
    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker.state = WorkerLifecycleState.READY
    worker._replica_label = "owner-a"
    worker._router_replay_tx = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=max_bytes)

    async def generate(prompt, params, request_id, **kwargs):
        await asyncio.sleep(0)
        assert not params.extra_args
        tokens = prompt["prompt_token_ids"]
        return SimpleNamespace(
            prompt_token_ids=tokens,
            num_cached_tokens=0,
            prompt_logprobs=None,
            routed_experts=torch.full((len(tokens), 2, 2), 9, dtype=dtype),
            outputs=[SimpleNamespace(text="", token_ids=[8], finish_reason="length", logprobs=None)],
        )

    worker._generate_once = AsyncMock(side_effect=generate)
    return worker


@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16])
def test_strict_capture_retention_pressure_retry_and_release(dtype):
    async def run():
        # A 16-byte prefix and 32-byte continuation cannot coexist in a 40-byte cache.
        width = torch.empty((), dtype=dtype).element_size()
        worker = _worker(max_bytes=40 * width, dtype=dtype)
        cache = worker._router_replay_tx
        a = await worker.generate([1, 2, 3, 4], _params("rr1:a"))
        original = cache.get("rr1:a").clone()
        cache.get("rr1:a").fill_(3)
        prefix = cache.get("rr1:a").clone()
        assert a["router_replay"]["strict"] is True
        with pytest.raises(RouterReplayCacheFull):
            await worker.generate(list(range(8)), _params("rr1:failed", "rr1:a"))
        assert "rr1:failed" not in cache
        assert torch.equal(cache.get("rr1:a"), prefix)
        assert not cache._borrowers
        assert cache.stats()["bytes_in_use"] == 16 * width

        cache.max_bytes = 128 * width
        b = await worker.generate(list(range(8)), _params("rr1:b", "rr1:a"))
        retry = await worker.generate(list(range(8)), _params("rr1:retry", "rr1:a"))
        assert b["router_replay"]["prefix_len"] == retry["router_replay"]["prefix_len"] == 4
        assert torch.equal(cache.get("rr1:b")[:4], prefix)
        assert not torch.equal(cache.get("rr1:b")[:4], original)
        assert torch.equal(cache.get("rr1:b"), cache.get("rr1:retry"))
        assert cache.get("rr1:b")[4:].tolist() == torch.full((4, 2, 2), 9).tolist()
        cache.discard(["rr1:a", "rr1:retry"])
        c = await worker.generate(list(range(12)), _params("rr1:c", "rr1:b"))
        assert c["router_replay"]["prefix_len"] == 8
        assert torch.equal(cache.get("rr1:c")[:8], cache.get("rr1:b"))

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["missing", "expired", "wrong_owner", "disabled"])
def test_strict_unavailable_prefix_fails_before_generation(reason):
    async def run():
        worker = _worker()
        cache = worker._router_replay_tx
        cache.put_new("rr1:a", torch.zeros((4, 2, 2), dtype=torch.uint8), retain=True)
        owner = "owner-a"
        if reason == "missing":
            cache.discard(["rr1:a"])
        elif reason == "expired":
            cache._created_at["rr1:a"] -= cache.ttl_s + 1
        elif reason == "wrong_owner":
            owner = "owner-b"
        else:
            worker._router_replay_tx = None
        with pytest.raises(RuntimeError):
            await worker.generate(list(range(8)), _params("rr1:b", "rr1:a", owner))
        worker._generate_once.assert_not_awaited()
        assert "rr1:b" not in cache

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["cancel", "engine", "no_capture", "no_output", "shape"])
def test_prefix_borrow_is_released_after_generation_failure(failure):
    async def run():
        worker = _worker()
        cache = worker._router_replay_tx
        cache.put_new("rr1:a", torch.zeros((4, 2, 2), dtype=torch.uint8), retain=True)
        original = worker._generate_once.side_effect

        async def generate(*args, **kwargs):
            assert cache._borrowers == {"rr1:a": 1}
            if failure == "cancel":
                raise asyncio.CancelledError
            if failure == "engine":
                raise RuntimeError("engine failed")
            if failure == "no_output":
                return None
            output = await original(*args, **kwargs)
            output.routed_experts = None if failure == "no_capture" else torch.zeros((8, 3, 2), dtype=torch.uint8)
            return output

        worker._generate_once.side_effect = generate
        with pytest.raises((asyncio.CancelledError, RuntimeError, ValueError)):
            await worker.generate(list(range(8)), _params("rr1:b", "rr1:a"))
        assert not cache._borrowers
        assert "rr1:a" in cache and "rr1:b" not in cache

    asyncio.run(run())


def test_borrowed_prefix_survives_concurrent_eviction_and_discard():
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=48)
    source = torch.full((4, 2, 2), 3, dtype=torch.uint8)
    cache.put_new("rr1:a", source)
    with cache.borrow_prefix("rr1:a") as first:
        with cache.borrow_prefix("rr1:a") as second:
            cache._created_at["rr1:a"] -= cache.ttl_s + 1
            cache.put_new("rr1:b", torch.zeros((8, 2, 2), dtype=torch.uint8))
            assert "rr1:a" in cache
            assert cache._borrowers == {"rr1:a": 2}
            cache.discard(["rr1:a"])
            assert torch.equal(first, source) and first is second
        assert cache._borrowers == {"rr1:a": 1}
    assert not cache._borrowers


def test_retained_capture_ttl_discard_clear_and_tombstone():
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=16)
    source = torch.zeros((4, 2, 2), dtype=torch.uint8)
    cache.put_new("rr1:a", source, retain=True)
    with pytest.raises(RouterReplayCacheFull):
        cache.put_new("rr1:b", source)
    cache._created_at["rr1:a"] -= cache.ttl_s + 1
    cache.put_new("rr1:b", source, retain=True)
    assert "rr1:a" not in cache._retained
    assert cache.discard(["rr1:b"]) == 1
    assert not cache._retained
    cache.discard(["rr1:late"])
    with pytest.raises(RouterReplayPrefixMissingError):
        cache.put_new("rr1:late", source, retain=True)
    cache.put_new("rr1:c", source, retain=True)
    cache.clear()
    assert not cache._retained


def test_owner_loss_while_queued_does_not_decrement_another_request():
    async def run():
        handle = SimpleNamespace(generate=SimpleNamespace(remote=AsyncMock()))
        scheduler = Scheduler([handle], initial_concurrency=1, poll_interval=100, adjust_interval=100)
        owner = scheduler._workers[0]
        owner.replay_replica = "owner-a"
        owner.active_requests = 1
        try:
            future = scheduler.submit([1, 2], _params("rr1:b", "rr1:a"))
            await asyncio.sleep(0.02)
            assert not future.done()
            scheduler.mark_worker_unavailable(0)
            with pytest.raises(RouterReplayPrefixMissingError):
                await asyncio.wait_for(future, 1)
            assert owner.active_requests == 1
            handle.generate.remote.assert_not_awaited()
        finally:
            owner.active_requests = 0
            await scheduler.shutdown()

    asyncio.run(run())


def test_http_batch_continuations_keep_each_owners_routes(monkeypatch):
    async def run():
        workers = [_worker(), _worker()]
        workers[1]._replica_label = "owner-b"
        handles = [SimpleNamespace(generate=SimpleNamespace(remote=w.generate)) for w in workers]
        pool = ReplicaPool()
        scheduler = Scheduler(handles, initial_concurrency=1, poll_interval=100, adjust_interval=100)
        pool._scheduler = scheduler
        monkeypatch.setattr(api, "backend", pool)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                first = await client.post(
                    "/generate",
                    json={
                        "prompts": [list(range(4)), list(range(4))],
                        "sampling_params": [_params("rr1:a"), _params("rr1:b")],
                    },
                )
                assert first.status_code == 200
                markers = [result["router_replay"] for result in first.json()["results"]]
                assert {marker["replica"] for marker in markers} == {"owner-a", "owner-b"}
                for i, worker in enumerate(workers):
                    worker._router_replay_tx.get(markers[i]["replay_id"]).fill_(i + 1)
                scheduler._next_group_affinity_worker_idx = 1
                second = await client.post(
                    "/generate",
                    json={
                        "prompts": [list(range(8)), list(range(8))],
                        "routing_key": ["changed-a", "changed-b"],
                        "sampling_params": [
                            _params(f"rr1:next-{i}", marker["replay_id"], marker["replica"])
                            for i, marker in enumerate(markers)
                        ],
                    },
                )
                assert second.status_code == 200
                for i, result in enumerate(second.json()["results"]):
                    assert result["router_replay"]["prefix_len"] == 4
                    assert result["router_replay"]["replica"] == markers[i]["replica"]
                    cached = workers[i]._router_replay_tx.get(f"rr1:next-{i}")
                    assert cached[:4].tolist() == torch.full((4, 2, 2), i + 1).tolist()
        finally:
            await scheduler.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    [
        {"dss_router_replay_id": "legacy"},
        {"dss_sample_id": None},
        {"extra_args": {"dss_router_replay_strict": "true"}},
        {"extra_args": {"dss_router_replay_strict": 1}},
        {"extra_args": {"dss_router_replay_strict": True, "dss_router_replay_prefix_id": "rr1:a"}},
        {
            "extra_args": {
                "dss_router_replay_strict": True,
                "dss_router_replay_prefix_id": "legacy",
                "dss_router_replay_prefix_replica": "owner-a",
            }
        },
        {"extra_args": {"dss_router_replay_strict": True, "dss_router_replay_prefix_replica": "owner-a"}},
        {"extra_args": {"dss_router_replay_strict": False, "dss_router_replay_prefix_replica": "owner-a"}},
    ],
)
def test_strict_request_validation(change):
    with pytest.raises(ValueError):
        strict_replay_options(_params("rr1:new") | change)


def test_strict_request_requires_new_attempt_id():
    with pytest.raises(ValueError, match="new replay id"):
        strict_replay_options(_params("rr1:a", "rr1:a"))


@pytest.mark.parametrize("owner_state", ["available", "unavailable", "removed", "replaced", "unknown"])
def test_scheduler_routes_to_capture_owner_or_fails(owner_state):
    async def run():
        workers = [
            SimpleNamespace(
                generate=SimpleNamespace(remote=AsyncMock(return_value={"router_replay": {"replica": owner}}))
            )
            for owner in ("owner-a", "owner-b")
        ]
        scheduler = Scheduler(workers, initial_concurrency=1, poll_interval=100, adjust_interval=100)
        try:
            first = await scheduler.submit([1], _params("rr1:a"))
            assert first["router_replay"]["replica"] == "owner-a"
            scheduler._group_affinity_assignments.clear()
            scheduler._next_group_affinity_worker_idx = 1
            owner = "owner-a"
            if owner_state == "unavailable":
                scheduler.mark_worker_unavailable(0)
            elif owner_state == "removed":
                scheduler._workers.pop(0)
            elif owner_state == "replaced":
                scheduler._workers[0].replay_replica = "new-owner"
            elif owner_state == "unknown":
                owner = "unknown-owner"
            if owner_state == "available":
                # A busy owner must wait, even when another worker is idle.
                scheduler._workers[0].active_requests = 1
                continuation = scheduler.submit([1, 2], _params("rr1:b", "rr1:a", owner), routing_key="different")
                await asyncio.sleep(0.02)
                assert not continuation.done()
                workers[1].generate.remote.assert_not_awaited()
                scheduler._workers[0].active_requests = 0
                await asyncio.wait_for(continuation, 1)
                assert workers[0].generate.remote.await_count == 2
            else:
                with pytest.raises(RouterReplayPrefixMissingError):
                    await asyncio.wait_for(scheduler.submit([1, 2], _params("rr1:b", "rr1:a", owner)), 1)
            workers[1].generate.remote.assert_not_awaited()
        finally:
            await scheduler.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize(
    "error,status",
    [
        (RouterReplayPrefixMissingError("missing prefix"), 410),
        (RouterReplayCacheFull("cache full"), 503),
    ],
)
@pytest.mark.parametrize("ray_wrapped", [False, True])
def test_http_errors_preserve_strict_replay_failure(error, status, ray_wrapped, monkeypatch):
    if ray_wrapped:
        from ray.exceptions import RayTaskError

        error = RayTaskError("generate", "", error).as_instanceof_cause()
    monkeypatch.setattr(api, "backend", SimpleNamespace(generate=AsyncMock(side_effect=error)))

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            response = await client.post("/generate", json={"prompts": [[1, 2]], "sampling_params": _params("rr1:a")})
            assert response.status_code == status
            assert str(error) in response.json()["detail"]

    asyncio.run(run())
