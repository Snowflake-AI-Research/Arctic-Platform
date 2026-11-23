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
"""fla's two forward norm kernels run one configuration once the GLM-5.3 norms are imported.

fla times each kernel's menu on the first call of a row-count bucket (``NB``) and keeps the fastest entry; the entries
sum a row's squares in different orders. In a subprocess under Triton's CPU interpreter, with Triton's benchmark made
to raise, the test imports the GLM-5.3 norms and launches fla's own wrappers for the output norm and the q/k norm at
NB 1 and NB 2: no launch may time the menu, and each must run BT 16 with 4 warps.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("fla")
pytest.importorskip("triton")

_SCRIPT = """
import torch
from triton.runtime.autotuner import Autotuner

def refuse(self, *args, config, **kwargs):
    raise AssertionError(f"timed {self.base_fn.__name__} with {config}")

Autotuner._bench = refuse
import arctic_platform.model.implementations.glm53.fused_norms  # noqa: F401
from fla.modules.fused_norm_gate import layer_norm_gated_fwd_kernel
from fla.modules.fused_norm_gate import rms_norm_gated
from fla.modules.l2norm import l2norm_fwd
from fla.modules.l2norm import l2norm_fwd_kernel

torch.manual_seed(0)
for rows in (64 * 3, 64 * 1025 + 1):  # NB 1, NB 2
    x, gate = torch.randn(rows, 128, dtype=torch.bfloat16), torch.randn(rows, 128, dtype=torch.bfloat16)
    rms_norm_gated(x, gate, torch.ones(128, dtype=torch.bfloat16), None, activation="sigmoid", eps=1e-5)
    l2norm_fwd(x)
    for tuner in (layer_norm_gated_fwd_kernel.fn, l2norm_fwd_kernel):
        used = tuner.best_config
        assert (used.kwargs, used.num_warps) == ({"BT": 16}, 4), (tuner.base_fn.__name__, rows, used)
print("ok")
"""


def test_fla_forward_norms_run_one_configuration():
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and result.stdout.split()[-1:] == ["ok"], result.stderr[-3000:]
