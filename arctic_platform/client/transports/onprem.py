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
"""On-prem transport base for the Arctic-Platform server.

HTTP and in-process Ray share one control plane (`OnPremTransport`:
job creation + ordering); they differ only in the delivery
primitives (`_start`, `call`, `_destroy`, `_wait_running`). The client already
resolved the job id onto each Request, so `call` just delivers it against the
server's uniform `op(job_id, body) -> dict` surface.

Concrete transports live alongside this base: `HttpTransport` in
`onprem_http.py` and `RayTransport` in `onprem_ray.py`.
"""

from __future__ import annotations

import logging
from abc import abstractmethod

from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.transport import JOB_CREATE_ORDER
from arctic_platform.client.transport import JOB_TYPES
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Transport
from arctic_platform.client.transport import unresolved_ops

logger = logging.getLogger(__name__)


class OnPremTransport(Transport):
    def __init__(self, config: ArcticClientConfig) -> None:
        self.config = config
        self.jobs = JobHandles()

    def initialize(self) -> JobHandles:
        reconnect = JobHandles.from_config(self.config)
        if reconnect.any_set:
            self.jobs = reconnect
            return self.jobs
        cfg = self.config
        try:
            for job_type in JOB_CREATE_ORDER:
                if cfg.gpus_for(job_type) > 0:
                    self.jobs.set(job_type, self._start(cfg.to_onprem(job_type)))
            self._wait_running()
        except Exception:
            # Tear down partial jobs / launched server so GPUs and the port are not orphaned.
            try:
                self.shutdown()
            except Exception:
                logger.exception("cleanup after failed initialize() also failed")
            raise
        return self.jobs

    def shutdown(self) -> None:
        # Clear handles after destroy so a second call (transport + client guard) is a no-op.
        for job_type in JOB_TYPES:
            job_id = getattr(self.jobs, job_type)
            if job_id is not None:
                self._destroy(job_id, job_type)
        self.jobs = JobHandles()

    def _check_op_coverage(self, target: object) -> None:
        """Warn if ``target`` is missing any canonical op method (do not fail)."""
        missing = unresolved_ops(target)
        if missing:
            logger.warning("%s cannot resolve ops (they will fail if called): %s", type(self).__name__, missing)

    # delivery primitives — the only things a concrete transport implements
    # (plus `call`/`acall` from the Transport ABC, which deliver one op end to end).
    @abstractmethod
    def _start(self, payload: dict) -> JobId:
        """Create one job and return its id."""

    @abstractmethod
    def _destroy(self, job_id: JobId, job_type: str) -> None:
        """Tear down one job."""

    @abstractmethod
    def _wait_running(self) -> None:
        """Block until all created jobs are ready."""
