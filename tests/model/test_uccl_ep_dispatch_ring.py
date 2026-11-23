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

"""EP32 EFA dispatch ring: no wrap up to 20,480 tokens per rank where the RDMA buffer allows it, without uccl."""

from __future__ import annotations

import ast
import math
from pathlib import Path

BUFFER_PATH = (
    Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations/moe/distributed/uccl_ep/buffer.py"
)
INT32_MAX = (1 << 31) - 1


class _Config:
    """The five sizes of ``uccl.ep.Config`` and its RDMA hint (``ep/include/ep_config.hpp``, eight NVL peers)."""

    def __init__(self, num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv):
        self.num_sms = num_sms
        self.num_max_rdma_chunked_send_tokens = rdma_send
        self.num_max_rdma_chunked_recv_tokens = rdma_recv

    def get_rdma_buffer_size_hint(self, hidden_bytes, num_ranks):
        channels, rdma_ranks = self.num_sms // 2, num_ranks // 8
        per_token = 2 * hidden_bytes + 4144
        num_bytes = channels * rdma_ranks * (144 + self.num_max_rdma_chunked_recv_tokens * per_token)
        return (num_bytes + 127) // 128 * 128


def _efa_buffer():
    tree = ast.parse(BUFFER_PATH.read_text())
    buffer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Buffer")
    buffer.body = [
        node
        for node in buffer.body
        if (isinstance(node, ast.FunctionDef) and node.name == "get_dispatch_config")
        or (isinstance(node, ast.AnnAssign) and node.target.id == "num_sms")
    ]
    namespace = {"Config": _Config}
    exec(compile(ast.Module(body=[buffer], type_ignores=[]), str(BUFFER_PATH), "exec"), namespace)
    namespace["Buffer"]._is_efa = staticmethod(lambda: True)
    return namespace["Buffer"]


def test_ep32_efa_dispatch_ring_fits_the_stream_when_the_buffer_allows():
    buffer = _efa_buffer()
    unwrapped = 0
    for hidden in range(1024, 16385, 1024):
        config = buffer.get_dispatch_config(32, 2 * hidden)
        ring = config.num_max_rdma_chunked_recv_tokens
        assert config.get_rdma_buffer_size_hint(2 * hidden, 32) <= INT32_MAX
        assert ring % config.num_max_rdma_chunked_send_tokens == 0
        deep = _Config(config.num_sms, 32, 288, 32, 2048)
        if deep.get_rdma_buffer_size_hint(2 * hidden, 32) <= INT32_MAX:
            assert math.ceil(20_480 / (config.num_sms // 2)) <= ring
            unwrapped += 1
    assert unwrapped > 0
