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
"""Failed training initialize must not leave DeepSpeed workers or a stuck job slot."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from arctic_platform.common.utils.server_models import JobConfig
from arctic_platform.testing_utils import TestCasePlus


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeWorker:
    def __init__(self, fail_initialize=False):
        self.destroy_calls = 0
        self.fail_initialize = fail_initialize
        self.get_ip = _Remote(self._get_ip)
        self.initialize = _Remote(self._initialize)
        self.destroy = _Remote(self._destroy)

    async def _get_ip(self):
        return "127.0.0.1"

    async def _initialize(self, *args, **kwargs):
        if self.fail_initialize:
            raise RuntimeError("worker initialize failed")

    async def _destroy(self):
        self.destroy_calls += 1


class _WorkerFactory:
    def __init__(self, created, fail_initialize=False):
        self.created = created
        self.fail_initialize = fail_initialize

    def remote(self, *args, **kwargs):
        w = FakeWorker(fail_initialize=self.fail_initialize)
        self.created.append(w)
        return w


def _http_state(*, training_gpus=1):
    from arctic_platform.common.http_server import app

    app.state.training_gpus = training_gpus
    app.state.sampling_gpus = 0
    app.state.log_prob_gpus = 0
    app.state.colocate = False
    app.state.placement = None
    app.state.training_workers = []
    app.state.log_prob_workers = []
    app.state.jobs = {}
    app.state.next_job_id = 1
    return app


def _ray_stub(*, training_gpus=1):
    from arctic_platform.common.ray_server import ArcticRLRayServerState

    server = object.__new__(ArcticRLRayServerState)
    server.training_gpus = training_gpus
    server.sampling_gpus = 0
    server.log_prob_gpus = 0
    server.colocate = False
    server.placement = None
    server.ds_master_port = 29500
    server.training_workers = []
    server.jobs = {}
    server.next_job_id = 1
    return server


class TestFailedTrainingInitializeCleanup(TestCasePlus):
    def test_http_missing_checkpoint_path_leaves_no_workers(self):
        from fastapi import HTTPException

        from arctic_platform.common.http_server import initialize

        app = _http_state()
        created = []
        factory = _WorkerFactory(created)
        with patch("arctic_platform.common.deepspeed_worker.DeepSpeedWorker.options", return_value=factory):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(initialize(JobConfig(model_name="m", job_type="training")))
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual(app.state.training_workers, [])
            self.assertEqual(app.state.jobs, {})
            self.assertEqual(created, [])

            out = asyncio.run(
                initialize(
                    JobConfig(
                        model_name="m",
                        job_type="training",
                        checkpoint_path=str(self.get_auto_remove_tmp_dir()),
                    )
                )
            )
        self.assertTrue(out["running"])
        self.assertEqual(len(app.state.training_workers), 1)
        self.assertEqual(len(app.state.jobs), 1)

    def test_http_worker_init_failure_destroys_actors(self):
        from arctic_platform.common.http_server import initialize

        app = _http_state()
        created = []
        factory = _WorkerFactory(created, fail_initialize=True)
        with patch("arctic_platform.common.deepspeed_worker.DeepSpeedWorker.options", return_value=factory):
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    initialize(
                        JobConfig(
                            model_name="m",
                            job_type="training",
                            checkpoint_path=str(self.get_auto_remove_tmp_dir()),
                        )
                    )
                )
        self.assertEqual(app.state.training_workers, [])
        self.assertEqual(app.state.jobs, {})
        self.assertTrue(created)
        self.assertTrue(all(w.destroy_calls == 1 for w in created))

    def test_ray_missing_checkpoint_path_leaves_no_workers(self):
        server = _ray_stub()
        created = []
        factory = _WorkerFactory(created)
        with patch("arctic_platform.common.deepspeed_worker.DeepSpeedWorker.options", return_value=factory):
            with self.assertRaises(ValueError):
                asyncio.run(server.initialize({"model_name": "m", "job_type": "training"}))
            self.assertEqual(server.training_workers, [])
            self.assertEqual(server.jobs, {})
            self.assertEqual(created, [])

            out = asyncio.run(
                server.initialize(
                    {
                        "model_name": "m",
                        "job_type": "training",
                        "checkpoint_path": str(self.get_auto_remove_tmp_dir()),
                    }
                )
            )
        self.assertTrue(out["running"])
        self.assertEqual(len(server.training_workers), 1)
        self.assertEqual(len(server.jobs), 1)

    def test_ray_worker_init_failure_destroys_actors(self):
        server = _ray_stub()
        created = []
        factory = _WorkerFactory(created, fail_initialize=True)
        with patch("arctic_platform.common.deepspeed_worker.DeepSpeedWorker.options", return_value=factory):
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    server.initialize(
                        {
                            "model_name": "m",
                            "job_type": "training",
                            "checkpoint_path": str(self.get_auto_remove_tmp_dir()),
                        }
                    )
                )
        self.assertEqual(server.training_workers, [])
        self.assertEqual(server.jobs, {})
        self.assertTrue(created)
        self.assertTrue(all(w.destroy_calls == 1 for w in created))
