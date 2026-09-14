# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Training-worker RPCs must await ObjectRefs (not ``ray.get``) and stay single-flight."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from arctic_platform.common.ray_server import ArcticRLRayServer
from arctic_platform.testing_utils import TestCasePlus


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _Worker:
    def __init__(self, log, hold):
        self._log = log
        self._hold = hold
        self.step = _Remote(self._step)

    async def _step(self):
        self._log.append("start")
        await self._hold.wait()
        self._log.append("done")
        return {"metrics": {}, "batch": {}}


def _server(workers):
    server = object.__new__(ArcticRLRayServer)
    server.jobs = {1: {"job_type": "training"}}
    server.training_workers = workers
    server._training_op_lock = asyncio.Lock()
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
                if log.count("start") == 2:
                    break
                await asyncio.sleep(0)
            self.assertEqual(log.count("start"), 2)
            second = asyncio.create_task(server.step(1))
            await asyncio.sleep(0)
            self.assertEqual(log.count("start"), 2, "second step submitted remotes while the first was in flight")
            hold.set()
            await asyncio.gather(first, second)
            self.assertEqual(log[:4], ["start", "start", "done", "done"])
            self.assertEqual(log.count("start"), 4)
            self.assertEqual(log.count("done"), 4)

        asyncio.run(_run())
