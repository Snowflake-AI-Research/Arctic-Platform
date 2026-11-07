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
"""FP32 parameter groups stay FP32 through a ZeRO-2 engine on several ranks."""

import os
import subprocess
import sys
from pathlib import Path

from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import require_torch_multi_gpu


@require_deepspeed
@require_torch_multi_gpu
def test_fp32_groups_keep_gradients_and_masters_across_ranks():
    driver = Path(__file__).with_name("zero_fp32_groups_driver.py")
    repository = str(Path(__file__).resolve().parents[2])
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [repository, os.environ.get("PYTHONPATH")])),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", str(driver)],
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
