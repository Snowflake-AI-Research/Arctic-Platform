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

"""Training-worker RPCs must await ObjectRefs (not ``ray.get``) and stay single-flight."""

from __future__ import annotations

import asyncio
import tempfile
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from arctic_platform.common.ray_server import ArcticRLRayServer
from arctic_platform.testing_utils import TestCasePlus


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _Worker:
    def __init__(self, log, hold, *, fail_once=False):
        self._log = log
        self._hold = hold
        self._fail_once = fail_once
        self._calls = 0
        self.step = _Remote(self._step)
        self.offload_to_cpu = _Remote(self._offload)

    async def _step(self):
        self._calls += 1
        self._log.append(f"step-{self._calls}")
        if self._fail_once and self._calls == 1:
            raise RuntimeError("worker failed")
        await self._hold.wait()
        self._log.append(f"step-done-{self._calls}")
        return {"metrics": {}, "batch": {}}

    async def _offload(self):
        self._log.append("sleep")
        return {"ok": True}


class _Plain:
    """``.remote()`` returns a value, not an ObjectRef."""

    def remote(self):
        return {"metrics": {}, "batch": {}}


class _State:
    def __init__(self, log):
        self._log = log
        self.reset_prefix_cache = _Remote(self._reset)
        self._empty_training_cache = _Remote(self._empty)

    async def _reset(self, job_id):
        self._log.append(f"reset-{job_id}")
        return {}

    async def _empty(self, workers):
        self._log.append("empty")
        return ["ok"]


def _server(workers):
    # ``__init__`` starts Ray. Tests build the lock themselves, the same object
    # ``ArcticRLRayServer.__init__`` assigns.
    server = object.__new__(ArcticRLRayServer)
    server.jobs = {1: {"job_type": "training"}}
    server.training_workers = workers
    server._training_op_lock = asyncio.Lock()
    server._training_failed = False
    server._training_failure = None
    server.colocate = False
    return server


class TestGatherTrainingRefs(TestCasePlus):
    def test_step_does_not_call_ray_get(self):
        hold = asyncio.Event()
        hold.set()
        server = _server([_Worker([], hold)])

        async def _run():
            with patch("arctic_platform.common.ray_server.ray.get", side_effect=AssertionError("ray.get")):
                return await server.step(1)

        out = asyncio.run(_run())
        self.assertEqual(out["job_id"], 1)

    def test_lock_keeps_training_ops_single_flight(self):
        hold = asyncio.Event()
        log: list[str] = []
        server = _server([_Worker(log, hold), _Worker(log, hold)])

        async def _run():
            first = asyncio.create_task(server.step(1))
            for _ in range(50):
                if log.count("step-1") == 2:
                    break
                await asyncio.sleep(0)
            self.assertEqual(log.count("step-1"), 2)
            second = asyncio.create_task(server.step(1))
            await asyncio.sleep(0)
            self.assertEqual(log.count("step-1"), 2, "second step submitted remotes while the first was in flight")
            hold.set()
            await asyncio.gather(first, second)
            self.assertEqual(log[:4], ["step-1", "step-1", "step-done-1", "step-done-1"])
            self.assertEqual(log.count("step-1"), 2)
            self.assertEqual(log.count("step-2"), 2)
            self.assertEqual(log.count("step-done-1"), 2)
            self.assertEqual(log.count("step-done-2"), 2)

        asyncio.run(_run())

    def test_lock_covers_sleep_and_weight_sync(self):
        hold = asyncio.Event()
        log: list[str] = []
        server = _server([_Worker(log, hold), _Worker(log, hold)])
        server.colocate = False
        server.sampling_pool = object()
        server.jobs[2] = {"job_type": "sampling"}
        server.arctic_rl_ray_server_state = _State(log)

        async def _sync(workers, pool):
            log.append("sync")
            return {"status": "ok"}

        server._sync_weights_nccl = _sync

        async def _run():
            first = asyncio.create_task(server.step(1))
            for _ in range(50):
                if log.count("step-1") == 2:
                    break
                await asyncio.sleep(0)
            self.assertEqual(log.count("step-1"), 2)
            sleep_task = asyncio.create_task(server.sleep_training(1))
            sync_task = asyncio.create_task(server.weight_sync(1, {"source_sub_job_id": 1, "target_sub_job_ids": [2]}))
            empty_task = asyncio.create_task(server.empty_training_cache(1))
            for _ in range(10):
                await asyncio.sleep(0)
            self.assertNotIn("sleep", log)
            self.assertNotIn("sync", log)
            self.assertNotIn("empty", log)
            hold.set()
            await first
            await asyncio.gather(sleep_task, sync_task, empty_task)
            self.assertEqual(log.count("sleep"), 2)
            self.assertIn("sync", log)
            self.assertIn("empty", log)
            self.assertLess(log.index("step-done-1"), log.index("sleep"))
            self.assertLess(log.index("step-done-1"), log.index("sync"))
            self.assertLess(log.index("step-done-1"), log.index("empty"))

        asyncio.run(_run())

    def test_failed_worker_cancels_siblings_and_fails_the_run(self):
        hold = asyncio.Event()
        log: list[str] = []
        server = _server([_Worker(log, hold, fail_once=True), _Worker(log, hold)])

        async def _run():
            first = asyncio.create_task(server.step(1))
            for _ in range(50):
                if log.count("step-1") == 2:
                    break
                await asyncio.sleep(0)
            self.assertEqual(log.count("step-1"), 2)
            with self.assertRaises(RuntimeError):
                await first
            self.assertNotIn("step-done-1", log)
            self.assertTrue(server._training_failed)
            self.assertFalse(server._training_op_lock.locked())
            with self.assertRaisesRegex(RuntimeError, "refusing further training ops"):
                await server.step(1)

        asyncio.run(_run())

    def test_failed_worker_does_not_hold_the_lock_for_a_sibling_that_never_returns(self):
        log: list[str] = []

        class _Fail:
            def __init__(self):
                self.step = _Remote(self._step)

            async def _step(self):
                log.append("fail")
                raise RuntimeError("boom")

        class _Hang:
            def __init__(self):
                self.step = _Remote(self._step)

            async def _step(self):
                log.append("hang")
                await asyncio.Event().wait()

        server = _server([_Fail(), _Hang()])

        async def _run():
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await server.step(1)
            self.assertLess(time.monotonic() - started, 2.0)
            self.assertTrue(server._training_failed)
            self.assertFalse(server._training_op_lock.locked())
            with self.assertRaisesRegex(RuntimeError, "refusing further training ops"):
                await server.step(1)

        asyncio.run(_run())

    def test_sibling_that_ignores_cancel_fails_later_ops_immediately(self):
        log: list[str] = []

        class _Fail:
            def __init__(self):
                self.step = _Remote(self._step)

            async def _step(self):
                raise RuntimeError("boom")

        class _Stubborn:
            def __init__(self):
                self.step = _Remote(self._step)

            async def _step(self):
                log.append("stubborn")
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await asyncio.sleep(1)

        server = _server([_Fail(), _Stubborn()])

        async def _run():
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await asyncio.wait_for(server.step(1), timeout=2.0)
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "refusing further training ops"):
                await server.step(1)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(log.count("stubborn"), 1)

        asyncio.run(_run())

    def test_caller_cancel_cancels_worker_refs_and_marks_failed(self):
        cancelled: list[Any] = []

        class _Ref:
            def __init__(self):
                self._started = asyncio.Event()

            def __await__(self):
                return self._wait().__await__()

            async def _wait(self):
                self._started.set()
                await asyncio.Event().wait()

        class _RemoteRef:
            def __init__(self, ref):
                self._ref = ref

            def remote(self):
                return self._ref

        refs = [_Ref(), _Ref()]
        server = _server([])
        server.training_workers = [type("W", (), {"step": _RemoteRef(ref)})() for ref in refs]

        async def _run():
            with (
                patch("arctic_platform.common.ray_server.ray.ObjectRef", _Ref),
                patch(
                    "arctic_platform.common.ray_server.ray.cancel",
                    side_effect=lambda ref, force=False: cancelled.append(ref),
                ),
            ):
                task = asyncio.create_task(server.step(1))
                for _ in range(50):
                    if all(ref._started.is_set() for ref in refs):
                        break
                    await asyncio.sleep(0)
                self.assertTrue(all(ref._started.is_set() for ref in refs))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(cancelled, refs)
            self.assertTrue(server._training_failed)
            self.assertFalse(server._training_op_lock.locked())
            with self.assertRaisesRegex(RuntimeError, "refusing further training ops"):
                await server.step(1)

        asyncio.run(_run())

    def test_non_awaitable_worker_result_is_rejected(self):
        server = _server([])
        started = []

        class _Slow:
            def remote(self):
                async def _run():
                    started.append("ran")
                    return {"metrics": {}, "batch": {}}

                return _run()

        server.training_workers = [type("W", (), {"step": _Slow()})(), type("W", (), {"step": _Plain()})()]

        async def _run():
            with self.assertRaises(TypeError):
                await server.step(1)

        asyncio.run(_run())
        self.assertEqual(started, [])

    def test_colocated_wake_inference_waits_for_the_training_lock(self):
        log = []
        hold = asyncio.Event()
        server = _server([_Worker(log, hold)])
        server.colocate = True
        server.jobs[2] = {"job_type": "sampling"}

        async def _wake(tags=None, *, empty_training_cache=True):
            self.assertFalse(empty_training_cache)
            log.append("wake")
            return {"sampling": {}}

        server.arctic_rl_ray_server_state = _State(log)
        server.arctic_rl_ray_server_state.wake_inference = _Remote(_wake)

        async def _run():
            async with server._training_op_lock:
                task = asyncio.create_task(server.wake_inference(2))
                for _ in range(50):
                    await asyncio.sleep(0)
                    if "empty" in log:
                        break
                self.assertNotIn("empty", log)
                self.assertFalse(task.done())
            await task
            self.assertEqual(log, ["empty", "wake"])

        asyncio.run(_run())

    def test_colocated_wake_skips_empty_cache_after_training_failure(self):
        log = []
        server = _server([])
        server.colocate = True
        server._training_failed = True
        server.jobs[2] = {"job_type": "sampling"}

        async def _wake(tags=None, *, empty_training_cache=True):
            self.assertFalse(empty_training_cache)
            log.append("wake")
            return {"sampling": {}}

        server.arctic_rl_ray_server_state = _State(log)
        server.arctic_rl_ray_server_state.wake_inference = _Remote(_wake)

        async def _run():
            result = await server.wake_inference(2)
            self.assertEqual(result, {"job_id": 2, "sampling": {}})

        asyncio.run(_run())
        self.assertEqual(log, ["wake"])

    def test_sampling_weight_norm_failure_does_not_block_step(self):
        hold = asyncio.Event()
        hold.set()
        worker = _Worker([], hold)

        async def _norm():
            return {"norm": 1.0, "num_params": 1}

        worker.weight_norm = _Remote(_norm)
        server = _server([worker])
        server.jobs[2] = {"job_type": "sampling"}

        async def _sampler_norm():
            raise RuntimeError("sampler norm failed")

        server.sampling_pool = type("P", (), {})()
        server.sampling_pool._workers = [type("W", (), {"compute_weight_norm": _Remote(_sampler_norm)})()]

        async def _run():
            with self.assertRaisesRegex(RuntimeError, "sampler norm failed"):
                await server.weight_norm(1, 2)
            self.assertFalse(server._training_failed)
            await server.step(1)

        asyncio.run(_run())

    def test_training_hang_latches_and_destroy_still_runs(self):
        log: list[str] = []

        class _Fail:
            def __init__(self):
                self.step = _Remote(self._step)

            async def _step(self):
                log.append("fail")
                raise RuntimeError("boom")

        class _Hang:
            def __init__(self):
                self.step = _Remote(self._step)

            async def _step(self):
                log.append("hang")
                await asyncio.Event().wait()

        server = _server([_Fail(), _Hang()])
        destroyed = []

        async def _destroy(job_id, job_type):
            destroyed.append((job_id, job_type))
            return {"status": "destroyed"}

        server.arctic_rl_ray_server_state = type("S", (), {})()
        server.arctic_rl_ray_server_state.destroy = _Remote(_destroy)

        async def _run():
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await server.step(1)
            self.assertTrue(server._training_failed)
            result = await server.destroy(1, "training")
            self.assertEqual(result, {"status": "destroyed"})
            self.assertEqual(destroyed, [(1, "training")])
            self.assertTrue(server._training_failed)
            with self.assertRaisesRegex(RuntimeError, "refusing further training ops"):
                await server.step(1)

        asyncio.run(_run())

    def test_save_refuses_after_the_training_latch_without_advancing_latest(self):
        saved = []

        async def _save(path, export_hf):
            saved.append(path)
            return {"hf_path": None, "global_step": 1}

        worker = type("W", (), {"save_checkpoint": _Remote(_save)})()
        server = _server([worker])
        server._training_failed = True

        async def _run():
            with tempfile.TemporaryDirectory() as root:
                server.jobs[1]["checkpoint_path"] = root
                latest = f"{root}/latest"
                with open(latest, "w", encoding="utf-8") as f:
                    f.write("7")
                with self.assertRaisesRegex(RuntimeError, "refusing further training ops"):
                    await server.save(1, {"step": 8})
                with open(latest, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "7")
            self.assertEqual(saved, [])

        asyncio.run(_run())


class TestNcclWeightSyncRefs(TestCasePlus):
    async def _sync(self, send, receive, *, ready=True):
        server = _server([])
        server.weight_sync_ready = ready
        server.weight_sync_bucket_size = 4096

        async def _ip():
            return "127.0.0.1"

        async def _max_bytes():
            return 4096

        async def _init(*args):
            return None

        async def _wake():
            return {}

        workers = [
            SimpleNamespace(
                get_ip=_Remote(_ip),
                max_param_bytes=_Remote(_max_bytes),
                init_weight_sender=_Remote(_init),
                send_weights=_Remote(fn),
            )
            for fn in send
        ]
        server.training_workers = workers
        server.arctic_rl_ray_server_state = SimpleNamespace(wake_inference=_Remote(_wake))
        pool = SimpleNamespace(num_replicas=1, tp_size=1, sync_weights=lambda **kwargs: receive())
        schedule = SimpleNamespace(
            groups=[
                SimpleNamespace(group_id=rank, sender_train_rank=rank, world_size=2, replica_ids=[0])
                for rank in range(len(workers))
            ]
        )
        schedule_class = SimpleNamespace(build=lambda **kwargs: schedule)
        with (
            patch("arctic_platform.common.ray_server._transfer_schedule_cls", return_value=schedule_class),
            patch("arctic_platform.debug.lora_sync_hooks.selected_lora_request", return_value=None),
        ):
            async with server._training_op_lock:
                result = await server._sync_weights_nccl(workers, pool)
        return server, result

    def test_unselected_sync_preserves_native_status_and_sizing_awaiter(self):
        async def _done():
            return {}

        async def _run():
            with (
                patch("arctic_platform.debug.rl_sampler_evidence.enabled", return_value=False),
                patch.object(
                    ArcticRLRayServer,
                    "_await_training_refs",
                    autospec=True,
                    side_effect=ArcticRLRayServer._await_training_refs,
                ) as awaiting,
            ):
                server, result = await self._sync([_done], _done, ready=False)
            self.assertEqual(result, {"status": "ok"})
            self.assertEqual(
                awaiting.await_count, 5, "IP, sizing, sender init, send and receive must use native awaiters"
            )
            self.assertTrue(server.weight_sync_ready)
            self.assertFalse(server._training_failed)

        asyncio.run(_run())

    def test_completed_byte_verification_receipts_are_retained(self):
        sent = {"all_rank_weight_sync_verification": {"sender": True}}
        received = {"workers": [{"all_rank_weight_sync_verification": {"receiver": True}}]}

        async def _send():
            return sent

        async def _receive():
            return received

        async def _run():
            with patch("arctic_platform.debug.rl_sampler_evidence.enabled", return_value=True):
                server, result = await self._sync([_send], _receive)
            self.assertEqual(result["targets"], [{"send": [sent], "recv": received}])
            self.assertFalse(server._training_failed)

        asyncio.run(_run())

    def test_sender_failure_cancels_hanging_sender_and_receiver(self):
        cancelled = []
        servers = []

        async def _fail():
            raise RuntimeError("sender failed")

        async def _hang(name):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(name)

        async def _run():
            original = ArcticRLRayServer._await_weight_sync_refs

            async def _collect(server, *args):
                servers.append(server)
                return await original(server, *args)

            with patch.object(ArcticRLRayServer, "_await_weight_sync_refs", _collect):
                with self.assertRaisesRegex(RuntimeError, "sender failed"):
                    await asyncio.wait_for(
                        self._sync([_fail, lambda: _hang("sender")], lambda: _hang("receiver")), timeout=2.0
                    )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertCountEqual(cancelled, ["sender", "receiver"])
            server = servers[0]
            self.assertTrue(server._training_failed)
            self.assertFalse(server._training_op_lock.locked())
            with self.assertRaisesRegex(RuntimeError, "refusing further training ops"):
                await server.step(1)

        asyncio.run(_run())

    def test_receiver_failure_cancels_pending_sender_and_latches_training(self):
        server = _server([])
        cancelled = asyncio.Event()

        async def _send():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def _receive():
            raise RuntimeError("receiver failed")

        async def _run():
            with self.assertRaisesRegex(RuntimeError, "receiver failed"):
                await asyncio.wait_for(server._await_weight_sync_refs(_receive(), [_send()]), timeout=2.0)
            await asyncio.sleep(0)
            self.assertTrue(cancelled.is_set())
            self.assertTrue(server._training_failed)

        asyncio.run(_run())

    def test_receiver_failure_after_send_completion_does_not_latch_training(self):
        server = _server([])
        sent = asyncio.Event()

        async def _send():
            sent.set()
            return {}

        async def _receive():
            await sent.wait()
            await asyncio.sleep(0)
            raise RuntimeError("receiver failed")

        async def _run():
            with self.assertRaisesRegex(RuntimeError, "receiver failed"):
                await server._await_weight_sync_refs(_receive(), [_send()])
            self.assertFalse(server._training_failed)

        asyncio.run(_run())

    def test_caller_cancellation_cancels_both_transfer_sides(self):
        server = _server([])
        started, cancelled = [], []

        async def _hang(name):
            started.append(name)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(name)

        async def _run():
            task = asyncio.create_task(server._await_weight_sync_refs(_hang("receiver"), [_hang("sender")]))
            for _ in range(50):
                if len(started) == 2:
                    break
                await asyncio.sleep(0)
            self.assertEqual(len(started), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertCountEqual(cancelled, ["sender", "receiver"])
            self.assertTrue(server._training_failed)

        asyncio.run(_run())
