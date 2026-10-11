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
"""Cancel Cortex jobs by id so their GPUs are released.

    python -m arctic_platform.integrations.openhands.release JOB_ID

A killed driver never reaches its own cleanup. The launcher calls this with
the id written to ``ARCTIC_CORTEX_JOB_ID_FILE`` at create time.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> None:
    from arctic_platform.client import ArcticClientConfig
    from arctic_platform.client.config import CortexConfig
    from arctic_platform.client.transports.cortex import CortexTransport

    transport = CortexTransport(ArcticClientConfig(model_name="unused", backend=CortexConfig()))
    for job_id in argv:
        transport.cancel_job(job_id)
        print(f"released {job_id}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
