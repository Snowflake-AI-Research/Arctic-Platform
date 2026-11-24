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
"""SonicMoE's autotune sweep benches each config for a bounded GPU time, so the sweep's length does not grow with the
rows the calling EP rank received: a rank holding most of the routed rows must not sweep for longer than DeepEP's
~100 s barrier timeout while its peers wait in the next dispatch / combine."""

import pytest
import torch

pytest.importorskip("quack")
pytest.importorskip("cutlass")
pytest.importorskip("cuda.bindings")


def _functional():
    from arctic_platform.model.implementations.kernels.sonicmoe import functional

    return functional


def _require_sm90() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SonicMoE's autotuner benches on SM90+")


def _sleep_cycles_per_ms() -> float:
    torch.cuda._sleep(1000)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    torch.cuda.synchronize()
    return 10_000_000 / start.elapsed_time(end)


def test_a_sweep_over_long_launches_makes_a_few_launches_per_config():
    """QuACK's Autotuner (patched by the import), two configs of a 20 ms launch: six launches per config, not 254."""
    _require_sm90()
    import time

    from quack.autotuner import AutotuneConfig
    from quack.autotuner import Autotuner

    _functional()
    cycles_per_ms = _sleep_cycles_per_ms()
    calls = []

    def fn(x, delay_ms=None):
        calls.append(delay_ms)
        torch.cuda._sleep(int(delay_ms * cycles_per_ms))
        x[:1].add_(1)

    tuner = Autotuner(fn, key=[], configs=[AutotuneConfig(delay_ms=20.0), AutotuneConfig(delay_ms=30.0)])
    # 192 MiB: more than 3 x the 50 MiB L2 of an H100/H200, so the sweep benches one operand set.
    x = torch.zeros(48 << 20, dtype=torch.float32, device="cuda")
    start = time.perf_counter()
    tuner(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert tuner.best_config.kwargs == {"delay_ms": 20.0}
    assert calls.count(30.0) == 6
    assert calls.count(20.0) == 6 + 1
    assert elapsed < 3.0
