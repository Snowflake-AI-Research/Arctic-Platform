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
import os
import sys
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from arctic_platform.common.ray_server import ArcticRLRayServer
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async


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
    server._training_workers_aborted = False
    server._weight_norm_disabled = None
    server.colocate = False
    return server


def _run_native_sp_worker_abort(rendezvous: str, ray_root: str) -> None:
    import ray
    import torch.distributed as dist

    class CollectiveWorker:
        def initialize(self, rank):
            self.rank = rank
            dist.init_process_group(
                "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=120)
            )

        def forward(self):
            dist.barrier()
            if self.rank == 0:
                raise RuntimeError("injected model forward failure")
            dist.all_reduce(torch.ones(1))
            raise AssertionError("blocked peer unexpectedly completed")

    ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False, log_to_driver=False, _temp_dir=ray_root)
    workers = []
    try:
        actor = ray.remote(num_cpus=1)(CollectiveWorker)
        workers = [actor.remote() for _ in range(2)]
        ray.get([worker.initialize.remote(rank) for rank, worker in enumerate(workers)], timeout=60)
        server = _server(workers)
        server.jobs[1]["sp_size"] = 2
        refs = [worker.forward.remote() for worker in workers]
        started = time.monotonic()

        async def run():
            try:
                await asyncio.wait_for(server._gather_training_refs(lambda: refs, abort_on_failure=True), timeout=15)
            except RuntimeError as error:
                assert "injected model forward failure" in str(error), error
                assert server._training_failure is error
            else:
                raise AssertionError("model failure was not propagated")
            assert server._training_failed
            assert server._training_workers_aborted

        asyncio.run(run())
        ready, pending = ray.wait(refs, num_returns=2, timeout=10)
        assert not pending and len(ready) == 2, "failed request left a collective peer running"
        assert time.monotonic() - started < 25
        for worker in workers:
            try:
                ray.get(worker.forward.remote(), timeout=10)
            except ray.exceptions.RayActorError:
                pass
            else:
                raise AssertionError("aborted worker was restarted")
        print("NATIVE_SP_WORKER_ABORT_PASS", flush=True)
    finally:
        for worker in workers:
            ray.kill(worker, no_restart=True)
        ray.shutdown()


class TestGatherTrainingRefs(TestCasePlus):
    def test_native_sp_model_failure_terminates_blocked_gloo_peer(self):
        scratch = Path(self.get_auto_remove_tmp_dir())
        env = self.get_env()
        env["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).parent), env.get("PYTHONPATH", "")))
        env.pop("RAY_ADDRESS", None)
        execute_subprocess_async(
            [
                sys.executable,
                "-c",
                (
                    "from test_ray_server_training_gather import _run_native_sp_worker_abort; "
                    f"_run_native_sp_worker_abort({str(scratch / 'gloo')!r}, {str(scratch / 'ray')!r})"
                ),
            ],
            env=env,
            timeout=120,
        )

    def test_sp_abort_preserves_primary_error_and_kills_each_actor_once(self):
        server = _server([object(), object()])
        server.jobs[1]["sp_size"] = 2
        error = RuntimeError("primary model failure")
        with patch("arctic_platform.common.ray_server.ray.kill", side_effect=RuntimeError("Ray disconnected")) as kill:
            server._abort_sequence_parallel_workers(error)
            server._abort_sequence_parallel_workers(RuntimeError("later failure"))
        self.assertEqual(kill.call_count, 2)
        self.assertIs(server._training_failure, error)
        self.assertTrue(server._training_failed)
        for call in kill.call_args_list:
            self.assertEqual(call.kwargs, {"no_restart": True})

    def test_membership_rpc_yields_to_the_actor_event_loop(self):
        async def run():
            hold = asyncio.Event()
            started = asyncio.Event()

            async def membership():
                started.set()
                await hold.wait()
                return {"sp_size": 2, "sp_ranks": [0, 1]}

            server = _server([])
            workers = [SimpleNamespace(get_sequence_parallel_info=_Remote(membership)) for _ in range(2)]
            payload = dict(batch={}, meta={}, processing=dict(loss_fn="grpo", post=["compute_logprobs"]))
            with (
                patch("arctic_platform.common.ray_server.ray.get", side_effect=AssertionError("ray.get")),
                patch(
                    "arctic_platform.common.ray_server.ray_split_batch", return_value=([payload, payload], None)
                ) as split,
                patch("arctic_platform.model.implementations.gpu.sp.utils.validate_sequence_parallel_microbatches"),
            ):
                task = asyncio.create_task(server._split_model_batch(payload, workers, sp_size=2, backward=True))
                await asyncio.wait_for(started.wait(), timeout=1)
                self.assertFalse(task.done())
                hold.set()
                _, _, info = await task
                self.assertEqual(info, [{"sp_size": 2, "sp_ranks": [0, 1]}] * 2)
                self.assertEqual(split.call_args.kwargs["worker_sp_info"], info)

        asyncio.run(run())

    def test_sp_preflight_rejection_drains_peers_and_preserves_workers(self):
        from arctic_platform.model.implementations.gpu.sp.utils import _PreflightValueError

        async def run():
            server = _server([object(), object()])
            server.jobs[1]["sp_size"] = 2
            drained = []

            async def reject(rank):
                if rank == 1:
                    await asyncio.sleep(0.02)
                drained.append(rank)
                raise _PreflightValueError("SP request: unknown field")

            with patch("arctic_platform.common.ray_server.ray.kill") as kill:
                with self.assertRaisesRegex(ValueError, "unknown field"):
                    await server._gather_training_refs(lambda: [reject(0), reject(1)], abort_on_failure=True)
                self.assertEqual(sorted(drained), [0, 1])
                kill.assert_not_called()
                self.assertFalse(server._training_failed)
                self.assertFalse(server._training_workers_aborted)
                self.assertEqual(await server._gather_training_refs(lambda: [asyncio.sleep(0, result="ok")]), ["ok"])

        asyncio.run(run())

    def test_sp_save_failure_does_not_abort_workers(self):
        async def save(path, export_hf):
            raise OSError("checkpoint disk unavailable")

        async def run():
            server = _server([SimpleNamespace(save_checkpoint=_Remote(save))])
            server.jobs[1].update(sp_size=2, checkpoint_path=self.get_auto_remove_tmp_dir())
            with patch("arctic_platform.common.ray_server.ray.kill") as kill:
                with self.assertRaisesRegex(OSError, "checkpoint disk unavailable"):
                    await server.save(1, {"step": 1})
                kill.assert_not_called()
                self.assertFalse(server._training_failed)
                self.assertFalse(server._training_workers_aborted)

        asyncio.run(run())

    def test_sp_mixed_preflight_and_execution_failure_aborts_blocked_peer(self):
        from arctic_platform.model.implementations.gpu.sp.utils import _PreflightValueError

        async def run():
            server = _server([object(), object(), object()])
            server.jobs[1]["sp_size"] = 2
            rejected = asyncio.Event()
            stopped = asyncio.Event()

            async def reject():
                rejected.set()
                raise _PreflightValueError("SP request: unknown field")

            async def fail():
                await rejected.wait()
                await asyncio.sleep(0.02)
                raise RuntimeError("model execution failed")

            async def blocked():
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

            with patch("arctic_platform.common.ray_server.ray.kill") as kill:
                with self.assertRaisesRegex(RuntimeError, "model execution failed"):
                    await asyncio.wait_for(
                        server._gather_training_refs(lambda: [reject(), fail(), blocked()], abort_on_failure=True),
                        timeout=1,
                    )
                await asyncio.wait_for(stopped.wait(), timeout=1)
                self.assertEqual(kill.call_count, 3)
                self.assertTrue(server._training_failed)
                self.assertTrue(server._training_workers_aborted)

        asyncio.run(run())

    def test_sp_cancelled_weight_transfer_does_not_abort_workers(self):
        async def run():
            server = _server([object(), object()])
            server.jobs[1]["sp_size"] = 2
            started = asyncio.Event()

            async def transfer():
                started.set()
                await asyncio.Event().wait()

            with patch("arctic_platform.common.ray_server.ray.kill") as kill:
                task = asyncio.create_task(server._gather_training_refs(lambda: [transfer()]))
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                kill.assert_not_called()
                self.assertFalse(server._training_workers_aborted)

        asyncio.run(run())

    def test_sp_sampler_failure_during_weight_sync_leaves_training_actors_alive(self):
        async def run():
            started, stopped = asyncio.Event(), asyncio.Event()

            async def send():
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

            class Pool:
                num_replicas = 1
                tp_size = 1

                async def sync_weights(self, **kwargs):
                    await started.wait()
                    raise OSError("sampler transfer failed")

            class Schedule:
                groups = [SimpleNamespace(group_id=0, sender_train_rank=0, world_size=2, replica_ids=[0])]

                @classmethod
                def build(cls, **kwargs):
                    return cls()

            worker = SimpleNamespace(
                get_ip=_Remote(lambda: asyncio.sleep(0, result="127.0.0.1")),
                send_weights=_Remote(send),
            )
            server = _server([worker])
            server.jobs[1]["sp_size"] = 2
            server.jobs[2] = {"job_type": "sampling"}
            server.sampling_pool = Pool()
            server.weight_sync_ready = True
            server.weight_sync_bucket_size = 1024
            server.arctic_rl_ray_server_state = SimpleNamespace(wake_inference=_Remote(lambda: asyncio.sleep(0)))
            with (
                patch("arctic_platform.common.ray_server._transfer_schedule_cls", return_value=Schedule),
                patch("arctic_platform.common.ray_server.ray.kill") as kill,
            ):
                with self.assertRaisesRegex(OSError, "sampler transfer failed"):
                    await server.weight_sync(1, {"source_sub_job_id": 1, "target_sub_job_ids": [2]})
                await asyncio.wait_for(stopped.wait(), timeout=1)
                kill.assert_not_called()
                self.assertFalse(server._training_workers_aborted)

        asyncio.run(run())

    def test_sp_validation_yields_and_runs_once_per_logical_group(self):
        async def run():
            server = _server([])
            payload = dict(batch={}, meta={}, processing=dict(loss_fn="grpo", post=["compute_logprobs"]))
            infos = [dict(sp_size=2, sp_ranks=[0, 1])] * 2 + [dict(sp_size=2, sp_ranks=[2, 3])] * 2
            workers = [
                SimpleNamespace(get_sequence_parallel_info=_Remote(lambda info=info: asyncio.sleep(0, result=info)))
                for info in infos
            ]
            started = threading.Event()
            release = threading.Event()
            loop_thread = threading.get_ident()

            def validate(*args, **kwargs):
                self.assertNotEqual(threading.get_ident(), loop_thread)
                started.set()
                self.assertTrue(release.wait(timeout=5))

            with (
                patch("arctic_platform.common.ray_server.ray_split_batch", return_value=([payload] * 4, None)),
                patch(
                    "arctic_platform.model.implementations.gpu.sp.utils.validate_sequence_parallel_microbatches",
                    side_effect=validate,
                ) as check,
            ):
                task = asyncio.create_task(server._split_model_batch(payload, workers, sp_size=2, backward=True))
                try:
                    for _ in range(100):
                        if started.is_set():
                            break
                        await asyncio.sleep(0.01)
                    self.assertTrue(started.is_set(), "batch validation did not start")
                    self.assertFalse(task.done())
                finally:
                    release.set()
                await task
                self.assertEqual(check.call_count, 2)

        asyncio.run(run())

    def test_forward_accepts_a_legacy_job_without_an_sp_size_key(self):
        async def forward(shard):
            return dict(batch=dict(logprobs=torch.ones(1, 2)), metrics={}, avg_loss=1.0)

        server = _server([SimpleNamespace(forward_no_grad=_Remote(forward))])
        payload = dict(batch=dict(input_ids=torch.tensor([[1, 2]])), meta={}, processing={})
        with patch("arctic_platform.rl.processors.prepare_request_loss", return_value=None):
            result = asyncio.run(server.forward(1, payload))
        self.assertEqual(result["job_id"], 1)
        self.assertEqual(result["avg_loss"], 1.0)

    def test_log_probs_rejects_sp_before_tokenization_or_worker_dispatch(self):
        server = _server([])
        server.jobs[1] = dict(job_type="log_prob", engine="deepspeed", sp_size=2)
        with self.assertRaisesRegex(ValueError, "log_probs does not support sequence-parallel"):
            asyncio.run(server.log_probs(1, dict(prompts=["hello"], top_k=1)))

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
            self.assertFalse(server._training_failed)
            self.assertFalse(server._training_workers_aborted)
            self.assertIsNone(server._training_failure)
            self.assertEqual(server.training_workers, [])
            self.assertNotIn(1, server.jobs)
            # A replacement job uses a new proxy, and must be independently abortable.
            replacement = _server([object(), object()])
            replacement.jobs[1]["sp_size"] = 2
            with patch("arctic_platform.common.ray_server.ray.kill") as kill:
                replacement._abort_sequence_parallel_workers(RuntimeError("new model failure"))
                self.assertEqual(kill.call_count, 2)

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
