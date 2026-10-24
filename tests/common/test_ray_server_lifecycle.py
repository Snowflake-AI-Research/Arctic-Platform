# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only ownership tests for Ray server resource teardown."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

from arctic_platform.common.ray_server import ArcticRLRayServer
from arctic_platform.common.ray_server import ArcticRLRayServerState
from arctic_platform.common.ray_server import create_arctic_rl_ray_server_state
from arctic_platform.common.utils.ray_pg import ColocatePlacement
from arctic_platform.common.utils.ray_pg import create_colocate_placement
from arctic_platform.rl.ray_client import ArcticRLRayClient
from arctic_platform.testing_utils import TestCasePlus


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _Worker:
    def __init__(self, name: str, log: list[str], *, fail: bool = False):
        self.name = name
        self._log = log
        self._fail = fail
        self.destroy = _Remote(self._destroy)

    async def _destroy(self):
        self._log.append(f"graceful-{self.name}")
        if self._fail:
            raise RuntimeError(f"{self.name} failed")
        return True


class _Pool:
    def __init__(self, name: str, log: list[str], *, initialized: bool = True):
        self.name = name
        self._log = log
        self._config = object() if initialized else None

    async def shutdown(self):
        self._log.append(f"pool-{self.name}")
        self._config = None


def _state(log: list[str]) -> ArcticRLRayServerState:
    state = object.__new__(ArcticRLRayServerState)
    state.jobs = {}
    state.training_workers = []
    state.log_prob_workers = []
    state.log_prob_tokenizer = None
    state.sampling_pool = None
    state.log_prob_pool = None
    state.placement = ColocatePlacement()
    state.placement_groups = []
    state.gpus_per_node = 0
    state.n_bundles = 0
    state.placement_group = None
    state._shutdown_lock = asyncio.Lock()
    state._shutdown_complete = False
    return state


class TestRayServerStateLifecycle(TestCasePlus):
    def test_server_state_actor_readiness_failure_preserves_original_error(self):
        log: list[str] = []
        placement = object()
        state = SimpleNamespace(
            __ray_ready__=_Remote(lambda: "ready-ref"),
        )

        def _remote(**_options):
            return lambda _cls: SimpleNamespace(remote=lambda **_kwargs: state)

        def _remove(_):
            log.append("remove-placement")
            raise RuntimeError("cleanup failed")

        with (
            patch(
                "arctic_platform.common.ray_server.placement_group",
                return_value=placement,
            ),
            patch("arctic_platform.common.ray_server.ray.remote", side_effect=_remote),
            patch(
                "arctic_platform.common.ray_server.ray.get",
                side_effect=ValueError("actor init failed"),
            ),
            patch(
                "arctic_platform.common.ray_server.ray.kill",
                side_effect=lambda _: log.append("kill-state"),
            ),
            patch(
                "arctic_platform.common.ray_server.remove_placement_group",
                side_effect=_remove,
            ),
            self.assertRaisesRegex(ValueError, "actor init failed"),
        ):
            create_arctic_rl_ray_server_state(
                training_gpus=1,
                sampling_gpus=1,
                log_prob_gpus=0,
                log_prob_engine="deepspeed",
                colocate=True,
            )
        self.assertEqual(log, ["kill-state", "remove-placement"])

    def test_failed_placement_creation_removes_partial_groups(self):
        groups = [SimpleNamespace(ready=lambda: "ready-0"), SimpleNamespace(ready=lambda: "ready-1")]
        removed = []
        with (
            patch(
                "arctic_platform.common.utils.ray_pg.placement_group",
                side_effect=groups,
            ),
            patch(
                "arctic_platform.common.utils.ray_pg.ray.get",
                side_effect=RuntimeError("not ready"),
            ),
            patch(
                "arctic_platform.common.utils.ray_pg.remove_placement_group",
                side_effect=removed.append,
            ),
            self.assertRaisesRegex(RuntimeError, "not ready"),
        ):
            create_colocate_placement(n_bundles=2, gpus_per_node=1)
        self.assertEqual(removed, groups)

    def test_shutdown_releases_jobs_before_placement_groups(self):
        log: list[str] = []
        training = _Worker("training", log)
        log_prob = _Worker("log-prob", log)
        sampling = _Pool("sampling", log)
        group = object()
        state = _state(log)
        state.jobs = {
            1: {"job_type": "training"},
            2: {"job_type": "sampling"},
            3: {"job_type": "log_prob", "engine": "deepspeed"},
        }
        state.training_workers = [training]
        state.log_prob_workers = [log_prob]
        state.sampling_pool = sampling
        state.placement = ColocatePlacement(
            placement_groups=[group],
            gpus_per_node=2,
            n_bundles=1,
        )
        state.placement_groups = state.placement.placement_groups
        state.gpus_per_node = 2
        state.n_bundles = 1
        state.placement_group = group

        def _kill(worker):
            log.append(f"kill-{worker.name}")

        def _remove(placement_group):
            self.assertIs(placement_group, group)
            log.append("remove-placement")

        async def _run():
            with (
                patch("arctic_platform.common.ray_server.ray.kill", side_effect=_kill),
                patch(
                    "arctic_platform.common.ray_server.remove_placement_group",
                    side_effect=_remove,
                ),
            ):
                result = await state.shutdown()
            self.assertEqual(result, {"shutdown": True, "failures": 0})

        asyncio.run(_run())
        self.assertEqual(
            log,
            [
                "graceful-training",
                "kill-training",
                "pool-sampling",
                "graceful-log-prob",
                "kill-log-prob",
                "remove-placement",
            ],
        )
        self.assertEqual(state.jobs, {})
        self.assertEqual(state.training_workers, [])
        self.assertEqual(state.log_prob_workers, [])
        self.assertEqual(state.placement.placement_groups, [])
        self.assertEqual(state.placement_groups, [])
        self.assertIsNone(state.placement_group)

    def test_partial_initialization_cleanup_is_idempotent(self):
        log: list[str] = []
        worker = _Worker("partial", log, fail=True)
        sampling = _Pool("partial", log)
        group = object()
        state = _state(log)
        # No job record: initialization failed after allocating resources.
        state.training_workers = [worker]
        state.sampling_pool = sampling
        state.placement = ColocatePlacement(
            placement_groups=[group],
            gpus_per_node=2,
            n_bundles=1,
        )
        state.placement_groups = state.placement.placement_groups
        state.placement_group = group

        async def _run():
            with (
                patch(
                    "arctic_platform.common.ray_server.ray.kill",
                    side_effect=lambda owned: log.append(f"kill-{owned.name}"),
                ),
                patch(
                    "arctic_platform.common.ray_server.remove_placement_group",
                    side_effect=lambda _: log.append("remove-placement"),
                ),
            ):
                first = await state.shutdown()
                second = await state.shutdown()
            self.assertEqual(first, {"shutdown": True, "failures": 0})
            self.assertEqual(second, {"shutdown": True, "failures": 0})

        asyncio.run(_run())
        self.assertEqual(
            log,
            [
                "graceful-partial",
                "kill-partial",
                "pool-partial",
                "remove-placement",
            ],
        )

    def test_hung_graceful_destroy_is_bounded_then_killed(self):
        log: list[str] = []
        worker = _Worker("hung", log)

        async def _hang():
            log.append("graceful-hung")
            await asyncio.Event().wait()

        worker.destroy = _Remote(_hang)
        state = _state(log)
        workers = [worker]

        async def _run():
            with (
                patch.dict(
                    os.environ,
                    {"ARCTIC_DEEPSPEED_WORKER_SHUTDOWN_TIMEOUT_S": "0.001"},
                ),
                patch(
                    "arctic_platform.common.ray_server.ray.kill",
                    side_effect=lambda _: log.append("kill-hung"),
                ),
            ):
                await state._destroy_deepspeed_workers(workers)

        asyncio.run(_run())
        self.assertEqual(log, ["graceful-hung", "kill-hung"])
        self.assertEqual(workers, [])

    def test_server_shutdown_terminates_state_then_scheduler_placement(self):
        log: list[str] = []

        async def _shutdown_state():
            log.append("shutdown-state")
            return {"shutdown": True}

        state = SimpleNamespace(shutdown=_Remote(_shutdown_state))
        server = object.__new__(ArcticRLRayServer)
        server.arctic_rl_ray_server_state = state
        server._server_placement_group = "server-pg"
        server._shutdown_complete = False

        async def _run():
            with (
                patch(
                    "arctic_platform.common.ray_server.ray.kill",
                    side_effect=lambda _: log.append("kill-state"),
                ),
                patch(
                    "arctic_platform.common.ray_server.remove_placement_group",
                    side_effect=lambda _: log.append("remove-server-placement"),
                ),
                patch(
                    "arctic_platform.common.ray_server.ray.get",
                    side_effect=AssertionError("shutdown must await the ObjectRef"),
                ),
            ):
                await server.shutdown()
                await server.shutdown()

        asyncio.run(_run())
        self.assertEqual(log, ["shutdown-state", "kill-state", "remove-server-placement"])


class TestRayClientLifecycle(TestCasePlus):
    def test_constructor_preserves_initialize_error_when_cleanup_raises(self):
        config = SimpleNamespace(
            training_job_id=None,
            training_gpus=1,
            sampling_gpus=1,
            log_prob_gpus=0,
            log_prob_engine="deepspeed",
            colocate=True,
        )
        with (
            patch(
                "arctic_platform.rl.ray_client.create_arctic_rl_ray_server_state",
                return_value=object(),
            ),
            patch.object(
                ArcticRLRayClient,
                "_initialize_jobs",
                side_effect=ValueError("initialize root cause"),
            ),
            patch.object(
                ArcticRLRayClient,
                "_cleanup_failed_initialization",
                side_effect=SystemExit("cleanup failure"),
            ),
            self.assertRaisesRegex(ValueError, "initialize root cause"),
        ):
            ArcticRLRayClient(config, None)

    def test_failed_initialize_still_shuts_state_when_pg_lookup_fails(self):
        log: list[str] = []
        state = SimpleNamespace(
            get_server_placement_group=_Remote(lambda: "pg-ref"),
            shutdown=_Remote(lambda: log.append("submit-shutdown") or "shutdown-ref"),
        )
        client = object.__new__(ArcticRLRayClient)
        client._arctic_rl_ray_server_state = state

        def _get(ref):
            if ref == "pg-ref":
                raise RuntimeError("actor init failed")
            self.assertEqual(ref, "shutdown-ref")
            log.append("await-shutdown")
            return {"shutdown": True}

        with (
            patch("arctic_platform.rl.ray_client.ray.get", side_effect=_get),
            patch(
                "arctic_platform.rl.ray_client.ray.kill",
                side_effect=lambda _: log.append("kill-state"),
            ),
        ):
            client._cleanup_failed_initialization()
        self.assertEqual(log, ["submit-shutdown", "await-shutdown", "kill-state"])

    def test_job_destroy_failure_still_runs_state_shutdown(self):
        log: list[str] = []
        client = object.__new__(ArcticRLRayClient)
        client._training_job_id = 1
        client._sampling_job_id = 2
        client._log_prob_job_id = None

        async def _destroy_job(job_id, job_type):
            log.append(f"destroy-{job_type}-{job_id}")
            raise RuntimeError("destroy failed")

        async def _shutdown_server():
            log.append("shutdown-server")

        client._destroy_job = _destroy_job
        client._arctic_rl_ray_server = SimpleNamespace(shutdown=_shutdown_server)

        async def _run():
            with self.assertRaisesRegex(RuntimeError, "destroy failed"):
                await client.shutdown()

        asyncio.run(_run())
        self.assertEqual(log, ["destroy-training-1", "shutdown-server"])
        self.assertIsNone(client._training_job_id)
        self.assertIsNone(client._sampling_job_id)
        self.assertIsNone(client._log_prob_job_id)


class TestRayClientSyncWeightsChecksum(TestCasePlus):
    """``checksum_match`` must be acted on, not only logged (see ``ArcticRLRayServer.weight_sync``)."""

    @staticmethod
    def _client(response: dict) -> ArcticRLRayClient:
        async def _weight_sync(*_args, **_kwargs):
            return response

        client = object.__new__(ArcticRLRayClient)
        client._training_job_id = 1
        client._sampling_job_id = 2
        client._arctic_rl_ray_server = SimpleNamespace(weight_sync=_weight_sync)
        return client

    def test_sync_weights_raises_on_checksum_mismatch(self):
        client = self._client(
            {
                "sent_l2_sq": 32.5,
                "recv_l2_sq": {"sampling replica 0 rank 0": 27.5},
                "checksum_match": False,
                "checksum_error": "received {'sampling replica 0 rank 0': 27.5}, sent 32.5",
            }
        )

        async def _run():
            with self.assertRaisesRegex(RuntimeError, "did not prove delivery"):
                await client.sync_weights()

        asyncio.run(_run())

    def test_sync_weights_strict_checksum_false_returns_the_dict_instead(self):
        mismatch = {"sent_l2_sq": 32.5, "recv_l2_sq": {"r0": 27.5}, "checksum_match": False, "checksum_error": "x"}
        client = self._client(mismatch)

        async def _run():
            out = await client.sync_weights(strict_checksum=False)
            self.assertEqual(out, mismatch)

        asyncio.run(_run())

    def test_sync_weights_missing_checksum_does_not_raise(self):
        client = self._client({"status": "ok"})

        async def _run():
            out = await client.sync_weights()
            self.assertNotIn("checksum_match", out)

        asyncio.run(_run())
