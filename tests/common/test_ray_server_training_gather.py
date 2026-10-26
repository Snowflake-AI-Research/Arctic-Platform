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
    server._weight_norm_disabled = None
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

    def test_sampling_weight_norm_failure_returns_error_and_does_not_block_step(self):
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
            result = await server.weight_norm(1, 2)
            self.assertEqual(
                result,
                {
                    "error_code": "weight_norm_unavailable",
                    "error": "RuntimeError: sampler norm failed",
                },
            )
            self.assertFalse(server._training_failed)
            await server.step(1)

        asyncio.run(_run())

    def test_training_weight_norm_failure_returns_error_and_does_not_latch(self):
        hold = asyncio.Event()
        hold.set()
        worker = _Worker([], hold)

        async def _norm():
            raise RuntimeError("training norm failed")

        worker.weight_norm = _Remote(_norm)
        server = _server([worker])
        server.jobs[2] = {"job_type": "sampling"}

        async def _sampler_norm():
            return {"norm": 1.0, "num_params": 1}

        server.sampling_pool = type("P", (), {})()
        server.sampling_pool._workers = [type("W", (), {"compute_weight_norm": _Remote(_sampler_norm)})()]

        async def _run():
            result = await server.weight_norm(1, 2)
            self.assertEqual(
                result,
                {
                    "error_code": "weight_norm_unavailable",
                    "error": "RuntimeError: training norm failed",
                },
            )
            self.assertFalse(server._training_failed)
            await server.step(1)

        asyncio.run(_run())

    def test_cancelled_training_weight_norm_latches(self):
        """Worker 0 may be left inside the ZeRO-3 all-gather, which a later step's collectives could pair with."""
        started = asyncio.Event()

        async def _norm():
            started.set()
            await asyncio.Event().wait()

        worker = type("W", (), {"weight_norm": _Remote(_norm)})()
        server = _server([worker])
        server.jobs[2] = {"job_type": "sampling"}

        async def _sampler_norm():
            return {"norm": 1.0, "num_params": 1}

        server.sampling_pool = type("P", (), {})()
        server.sampling_pool._workers = [type("W", (), {"compute_weight_norm": _Remote(_sampler_norm)})()]

        async def _run():
            task = asyncio.create_task(server.weight_norm(1, 2))
            await asyncio.wait_for(started.wait(), timeout=2.0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(server._training_failed)
            self.assertFalse(server._training_op_lock.locked())
            with self.assertRaisesRegex(RuntimeError, "refusing further training ops"):
                await server.step(1)

        asyncio.run(_run())

    def test_weight_norm_invalid_job_returns_error(self):
        server = _server([])

        async def _run():
            result = await server.weight_norm(1, 2)
            self.assertEqual(
                result,
                {
                    "error_code": "weight_norm_unavailable",
                    "error": "ValueError: Job 2 not found",
                },
            )
            self.assertIsNone(server._weight_norm_disabled)

        asyncio.run(_run())

    def test_weight_norm_structural_refusal_disables_later_calls(self):
        calls = {"training": 0, "sampling": 0}
        hold = asyncio.Event()
        hold.set()
        worker = _Worker([], hold)

        async def _norm():
            calls["training"] += 1
            return {"norm": 1.0, "num_params": 1}

        async def _sampler_norm():
            calls["sampling"] += 1
            return {
                "error_code": "weight_norm_unavailable",
                "error": "exact only at tensor_parallel_size=1",
                "disable": True,
            }

        worker.weight_norm = _Remote(_norm)
        server = _server([worker])
        server.jobs[2] = {"job_type": "sampling"}
        server.sampling_pool = type("P", (), {})()
        server.sampling_pool._workers = [type("W", (), {"compute_weight_norm": _Remote(_sampler_norm)})()]

        async def _run():
            first = await server.weight_norm(1, 2)
            second = await server.weight_norm(1, 2)
            self.assertEqual(
                first,
                {"error_code": "weight_norm_unavailable", "error": "exact only at tensor_parallel_size=1"},
            )
            self.assertEqual(second, first)
            self.assertFalse(server._training_failed)
            await server.step(1)

        asyncio.run(_run())
        self.assertEqual(calls, {"training": 0, "sampling": 1})

    def test_weight_norm_pool_tp_disables_without_calling_workers(self):
        calls = {"n": 0}
        server = _server([])
        server.jobs[2] = {"job_type": "sampling"}

        async def _sampler_norm():
            calls["n"] += 1
            return {"norm": 1.0, "num_params": 1}

        server.sampling_pool = type("P", (), {"tp_size": 2})()
        server.sampling_pool._workers = [type("W", (), {"compute_weight_norm": _Remote(_sampler_norm)})()]

        async def _run():
            result = await server.weight_norm(1, 2)
            self.assertEqual(result["error_code"], "weight_norm_unavailable")
            self.assertIn("tensor_parallel_size=2", result["error"])
            again = await server.weight_norm(1, 2)
            self.assertEqual(again, result)

        asyncio.run(_run())
        self.assertEqual(calls["n"], 0)

    def test_weight_norm_temporary_error_tries_again(self):
        calls = {"training": 0, "sampling": 0}
        hold = asyncio.Event()
        hold.set()
        worker = _Worker([], hold)

        async def _norm():
            calls["training"] += 1
            return {"norm": 1.0, "num_params": 1}

        async def _sampler_norm():
            calls["sampling"] += 1
            raise RuntimeError("sampler norm failed")

        worker.weight_norm = _Remote(_norm)
        server = _server([worker])
        server.jobs[2] = {"job_type": "sampling"}
        server.sampling_pool = type("P", (), {})()
        server.sampling_pool._workers = [type("W", (), {"compute_weight_norm": _Remote(_sampler_norm)})()]

        async def _run():
            first = await server.weight_norm(1, 2)
            second = await server.weight_norm(1, 2)
            self.assertEqual(
                first,
                {"error_code": "weight_norm_unavailable", "error": "RuntimeError: sampler norm failed"},
            )
            self.assertEqual(second, first)
            self.assertIsNone(server._weight_norm_disabled)

        asyncio.run(_run())
        self.assertEqual(calls, {"training": 0, "sampling": 2})

    def test_weight_norm_gathers_every_training_rank(self):
        calls = []
        hold = asyncio.Event()
        hold.set()

        def _worker(rank):
            async def _norm():
                calls.append(rank)
                return {"norm": 1.0, "num_params": 3}

            worker = _Worker([], hold)
            worker.weight_norm = _Remote(_norm)
            return worker

        server = _server([_worker(0), _worker(1)])
        server.jobs[2] = {"job_type": "sampling"}

        async def _sampler_norm():
            return {"norm": 1.0, "num_params": 3}

        server.sampling_pool = type("P", (), {})()
        server.sampling_pool._workers = [type("W", (), {"compute_weight_norm": _Remote(_sampler_norm)})()]

        async def _run():
            result = await server.weight_norm(1, 2)
            self.assertEqual(result["training_norm"], 1.0)
            self.assertEqual(result["sampling_norm"], 1.0)

        asyncio.run(_run())
        self.assertEqual(calls, [0, 1])

    def test_weight_norm_after_training_failure_does_not_disable(self):
        hold = asyncio.Event()
        hold.set()
        server = _server([_Worker([], hold)])
        server._training_failed = True
        server.jobs[2] = {"job_type": "sampling"}

        async def _sampler_norm():
            return {"norm": 1.0, "num_params": 1}

        server.sampling_pool = type("P", (), {})()
        server.sampling_pool._workers = [type("W", (), {"compute_weight_norm": _Remote(_sampler_norm)})()]

        async def _run():
            result = await server.weight_norm(1, 2)
            self.assertEqual(result["error_code"], "weight_norm_unavailable")
            self.assertIn("refusing further training ops", result["error"])
            self.assertIsNone(server._weight_norm_disabled)

        asyncio.run(_run())

    def test_weight_norm_mismatch_disables_the_check(self):
        """A divergence that persists across the one retry is real and latches the check off."""
        calls = {"training": 0, "sampling": 0}
        hold = asyncio.Event()
        hold.set()
        worker = _Worker([], hold)

        async def _norm():
            calls["training"] += 1
            return {"norm": 2.0, "num_params": 1}

        async def _sampler_norm():
            calls["sampling"] += 1
            return {"norm": 1.0, "num_params": 1}

        worker.weight_norm = _Remote(_norm)
        server = _server([worker])
        server.jobs[2] = {"job_type": "sampling"}
        server.sampling_pool = type("P", (), {})()
        server.sampling_pool._workers = [type("W", (), {"compute_weight_norm": _Remote(_sampler_norm)})()]

        async def _run():
            result = await server.weight_norm(1, 2)
            self.assertEqual(result["error_code"], "weight_norm_unavailable")
            self.assertIn("weight norms diverge", result["error"])
            self.assertNotIn("training_norm", result)
            again = await server.weight_norm(1, 2)
            self.assertEqual(again, result)
            self.assertFalse(server._training_failed)

        asyncio.run(_run())
        # One retry inside the first call (the persistent-mismatch path reads both sides
        # twice before disabling); the second call returns the latched result untouched.
        self.assertEqual(calls, {"training": 2, "sampling": 2})

    def test_weight_norm_transient_mismatch_is_not_disabled(self):
        """A one-shot divergence (a step/sync landing between the two non-atomic reads) is retried, not latched."""
        calls = {"training": 0, "sampling": 0}
        hold = asyncio.Event()
        hold.set()
        worker = _Worker([], hold)

        async def _norm():
            calls["training"] += 1
            return {"norm": 1.0, "num_params": 1}

        async def _sampler_norm():
            calls["sampling"] += 1
            # Diverges only on the first read; agrees from the second read onward.
            return {"norm": 2.0 if calls["sampling"] == 1 else 1.0, "num_params": 1}

        worker.weight_norm = _Remote(_norm)
        server = _server([worker])
        server.jobs[2] = {"job_type": "sampling"}
        server.sampling_pool = type("P", (), {})()
        server.sampling_pool._workers = [type("W", (), {"compute_weight_norm": _Remote(_sampler_norm)})()]

        async def _run():
            result = await server.weight_norm(1, 2)
            self.assertEqual(
                result,
                {"training_norm": 1.0, "sampling_norm": 1.0, "training_num_params": 1, "sampling_num_params": 1},
            )
            self.assertIsNone(server._weight_norm_disabled)

        asyncio.run(_run())
        self.assertEqual(calls, {"training": 2, "sampling": 2})

    def test_weight_sync_reports_a_checksum_mismatch_without_raising(self):
        """The sampler is already written when the checksum is read, so raising would not restore it."""
        hold = asyncio.Event()
        hold.set()
        log: list[str] = []
        server = _server([_Worker(log, hold)])
        server.jobs[1]["cuda_ipc"] = True
        server.jobs[2] = {"job_type": "sampling"}
        server.colocate = True
        server.sampling_pool = object()
        server.log_prob_pool = None
        server.arctic_rl_ray_server_state = _State(log)
        mismatch = {
            "status": "ok",
            "sent_l2_sq": 32.5,
            "recv_l2_sq": {"sampling replica 0 rank 0": 27.5},
            "checksum_rtol": 1e-9,
            "checksum_match": False,
            "checksum_error": "received {'sampling replica 0 rank 0': 27.5}, sent 32.5",
        }

        async def _sync(workers, pool, lp_pool):
            return mismatch

        server._sync_weights_cuda_ipc = _sync

        async def _run():
            with self.assertLogs("arctic_platform.common.ray_server", level="WARNING") as logs:
                result = await server.weight_sync(1, {"source_sub_job_id": 1, "target_sub_job_ids": [2]})
            self.assertEqual(result, {"job_id": 1, **mismatch})
            self.assertIn("did not prove delivery: received {'sampling replica 0 rank 0': 27.5}", logs.output[0])
            self.assertFalse(server._training_failed)
            await server.step(1)

        asyncio.run(_run())

    def test_nccl_sync_compares_each_sender_with_every_sampler_rank(self):
        class _Sender:
            def __init__(self, sent):
                self.get_ip = _Remote(self._ip)
                self.send_weights = _Remote(self._send)
                self._sent = sent

            async def _ip(self):
                return "127.0.0.1"

            async def _send(self):
                return {"status": "done", "sent_l2_sq": self._sent}

        class _Pool:
            num_replicas = 2
            tp_size = 2

            async def sync_weights(self, groups, bucket_size):
                return {"workers": [{"rank_recv_l2_sq": [32.5, 32.5]}, {"rank_recv_l2_sq": [32.5, 30.0]}]}

        class _Schedule:
            groups = [SimpleNamespace(group_id=0, sender_train_rank=0, world_size=5, replica_ids=[0, 1])]

            @classmethod
            def build(cls, **kwargs):
                return cls()

        async def _wake():
            return None

        server = _server([_Sender(32.5)])
        server.weight_sync_ready = True
        server.weight_sync_bucket_size = 1024
        server.arctic_rl_ray_server_state = SimpleNamespace(wake_inference=_Remote(_wake))

        async def _run():
            async with server._training_op_lock:
                with patch("arctic_platform.common.ray_server._transfer_schedule_cls", return_value=_Schedule):
                    return await server._sync_weights_nccl(server.training_workers, _Pool())

        result = asyncio.run(_run())
        self.assertEqual(result["sent_l2_sq"], 32.5)
        self.assertEqual(
            result["recv_l2_sq"],
            {
                "sampling replica 0 rank 0": 32.5,
                "sampling replica 0 rank 1": 32.5,
                "sampling replica 1 rank 0": 32.5,
                "sampling replica 1 rank 1": 30.0,
            },
        )
        self.assertFalse(result["checksum_match"])
        self.assertEqual(result["checksum_error"], "received {'sampling replica 1 rank 1': 30.0}, sent 32.5")

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
