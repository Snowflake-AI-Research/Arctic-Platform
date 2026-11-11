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


def test_quack_autotuner_benches_through_the_bounded_helpers():
    import quack.autotuner

    f = _functional()
    assert quack.autotuner._bench_cuda_graph_l2_rotate is f._bench_cuda_graph_l2_rotate_bounded
    assert quack.autotuner._pick_l2_rotate_count is f._pick_l2_rotate_count_bounded


@pytest.mark.parametrize("n_sets", [1, 4, 16])
def test_bench_launch_counts_bound_each_config_by_gpu_time(n_sets):
    f = _functional()
    for est_ms in (0.002, 0.05, 0.3, 1.0, 5.0, 22.0, 100.0, 1000.0):
        n_warmup, rounds = f._bench_launch_counts(est_ms, n_sets)
        timed = rounds * n_sets
        assert 1 <= n_warmup <= f._TUNE_MAX_WARMUP_CALLS
        assert n_warmup * est_ms <= max(f._TUNE_WARMUP_MS, est_ms)
        assert f._TUNE_MIN_TIMED_CALLS <= timed < f._TUNE_MAX_TIMED_CALLS + n_sets
        # Timed GPU work: the budget, or the minimum count of launches, rounded up to whole rounds.
        assert timed * est_ms < max(f._TUNE_TIMED_MS, f._TUNE_MIN_TIMED_CALLS * est_ms) + n_sets * est_ms


def test_bench_launch_counts_keep_quack_statistics_for_short_launches_and_few_launches_for_long_ones():
    f = _functional()
    # A 10 us launch: QuACK's minimum warmup count and its 200 timed launches.
    assert f._bench_launch_counts(0.01, 1) == (f._TUNE_MAX_WARMUP_CALLS, f._TUNE_MAX_TIMED_CALLS)
    # The hot EP rank's ~20 ms launch (about 1.3M routed rows of Nemotron-3-Nano's up projection) on one operand set:
    # 1 priming + 1 probe + 1 warmup + 3 timed launches per config, where QuACK's bench made 3 + 1 + 50 + 200.
    n_warmup, rounds = f._bench_launch_counts(22.0, 1)
    assert 1 + 1 + n_warmup + rounds == 6


def _fake_cuda(monkeypatch, l2_bytes: int, free_bytes: int) -> None:
    class Props:
        L2_cache_size = l2_bytes

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: Props())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free_bytes, 2 * free_bytes))


@pytest.mark.parametrize(
    "operand_mib, free_gib, expected",
    [
        (1, 100, 16),  # small operands: rotate up to 16 sets
        (20, 100, 8),  # 3 x 50 MiB of L2 over 20 MiB sets
        (60, 100, 4),  # two sets would do; QuACK's minimum of 4 stays
        (320, 100, 1),  # one set (e.g. the expert weights alone) is larger than 3 L2 caches
        (10 * 1024, 100, 1),  # a hot rank's receive buffer: not cloned 4 times
        (8, 0.01, 1),  # no room for a second set
    ],
)
def test_l2_rotate_count_clones_one_set_once_it_defeats_l2(monkeypatch, operand_mib, free_gib, expected):
    f = _functional()
    _fake_cuda(monkeypatch, l2_bytes=50 << 20, free_bytes=int(free_gib * (1 << 30)))
    operand = torch.empty(operand_mib << 20, dtype=torch.uint8, device="meta")
    assert f._pick_l2_rotate_count_bounded((operand,), {}) == expected


def _sleep_cycles_per_ms() -> float:
    torch.cuda._sleep(1000)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    torch.cuda.synchronize()
    return 10_000_000 / start.elapsed_time(end)


def test_a_sweep_over_long_launches_makes_a_few_launches_per_config():
    """QuACK's own Autotuner (patched by the import) over two configs of a 20 ms launch on one large operand set: six
    launches per config plus the winner's run, where QuACK's bench made 254 per config (about 10 s here)."""
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
