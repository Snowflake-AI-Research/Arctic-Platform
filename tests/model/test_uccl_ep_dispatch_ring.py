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

"""EP32 EFA dispatch: the default ring cached dispatch picks, with CPU tensors and a stand-in for the uccl runtime."""

import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

INT32_MAX = (1 << 31) - 1


class _Config:
    def __init__(self, num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv):
        self.num_sms = num_sms
        self.slots = rdma_recv
        self.send = rdma_send

    def get_rdma_buffer_size_hint(self, hidden_bytes, num_ranks):
        # Normal-mode uccl::Config hint with eight NVL peers.
        size = (self.num_sms // 2) * (num_ranks // 8) * (144 + self.slots * (2 * hidden_bytes + 4144))
        return (size + 127) // 128 * 128


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_ep32_efa_dispatch_ring_holds_a_channels_stream(monkeypatch, dtype):
    distributed = Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations/moe/distributed"
    monkeypatch.syspath_prepend(str(distributed))
    ep = SimpleNamespace(Config=_Config, EventHandle=object)
    with patch.dict(sys.modules, {"uccl": SimpleNamespace(ep=ep), "uccl.ep": ep}):
        buffer_type = import_module("uccl_ep").Buffer
    monkeypatch.setattr(buffer_type, "_is_efa", staticmethod(lambda: True))
    buffer = object.__new__(buffer_type)
    buffer.group_size = 32
    buffer.proxies = []
    buffer.runtime = Mock()
    buffer.runtime.get_num_rdma_ranks.return_value = 4
    buffer._ll_compute_stream_ptr = lambda device: 0
    buffer.get_comm_stream = lambda: None

    def dispatch(hidden):
        rows = torch.empty((0, hidden), dtype=dtype)
        x = (rows, torch.empty((0, hidden // 128))) if rows.element_size() == 1 else rows
        handle = (rows,) * 7 + (0, 0) + (rows,) * 3
        buffer.dispatch(x, handle=handle)
        config = next(arg for arg in buffer.runtime.internode_dispatch.call_args.args if isinstance(arg, _Config))
        assert config.get_rdma_buffer_size_hint(2 * hidden, 32) <= INT32_MAX
        return config

    narrow = dispatch(4096)
    # 1,024 rows per SM per rank fit without wrapping, and a full ring spans at most 16 EFA tail updates.
    assert narrow.slots * (narrow.num_sms // 2) >= 1024 * narrow.num_sms
    assert narrow.slots % narrow.send == 0 and narrow.slots // narrow.send <= 16
    dispatch(7168)
