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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""GPU e2e: save → load → resume + eval-loss restore + HF export (A1/A2)."""

from __future__ import annotations

import sys

import pytest

from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import reserve_free_port

_PORT_BASE = get_unique_port_number()


@require_torch_gpu
@pytest.mark.gpu_serial
class TestSftCkptResumeHttpE2EGPU(TestCasePlus):
    def test_save_load_resume_eval_and_hf_export(self):
        ckpt = self.get_auto_remove_tmp_dir()
        env = self.get_env()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["WANDB_DISABLED"] = "true"
        env.setdefault("HF_HOME", "/data-fast/huggingface")
        http_port = reserve_free_port(_PORT_BASE + 1, span=3)
        env["MASTER_PORT"] = str(reserve_free_port(_PORT_BASE + 4, span=4))
        cmd = [
            sys.executable,
            "-m",
            "arctic_platform.sft.examples.run_sft_ckpt_resume_demo",
            "--launch-local-server",
            "--server-cuda-visible-devices",
            "0",
            "--training-gpus",
            "1",
            "--port",
            str(http_port),
            "--checkpoint-dir",
            str(ckpt),
            "--pre-save-steps",
            "2",
            "--post-save-steps",
            "2",
        ]
        result = execute_subprocess_async(cmd, env=env, timeout=900)
        out = "\n".join(result.stdout + result.stderr)
        self.assertIn("A1_OK", out, msg=out[-4000:])
