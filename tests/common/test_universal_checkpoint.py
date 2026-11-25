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
"""CPU tests for the universal-checkpoint DP resize helpers, without torch.distributed (a single rank)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from arctic_platform.common.utils import universal_checkpoint as uc
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_deepspeed

TAG = "global_step3"


def _write_checkpoint(tag_dir: Path, *, dp: int, shards: list[int] | None = None, **state) -> None:
    import torch

    tag_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"dp_world_size": dp, "mp_world_size": 1, **state}, tag_dir / "mp_rank_00_model_states.pt")
    for rank in range(dp) if shards is None else shards:
        (tag_dir / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(b"")
    (tag_dir.parent / "latest").write_text(tag_dir.name, encoding="utf-8")


class _FakeEngine:
    """Records each ``load_checkpoint`` call with the universal flag it saw."""

    def __init__(self, dp: int, stage: int = 2, mpu=None):
        self.seq_dp_world_size = dp
        self.mpu = mpu
        self._stage = stage
        self._config = SimpleNamespace(load_universal_checkpoint=False)
        self.loads: list[tuple[str, str | None, bool, dict]] = []

    def zero_optimization_stage(self) -> int:
        return self._stage

    def load_checkpoint(self, load_dir, tag=None, **kwargs):
        self.loads.append((load_dir, tag, self._config.load_universal_checkpoint, kwargs))
        return os.path.join(load_dir, str(tag)), {}


@require_deepspeed
class TestCheckpointInspection(TestCasePlus):
    def test_shard_ranks_match_bf16_and_fp32_names_and_paths(self):
        names = [
            "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
            "global_step3/zero_pp_rank_2_mp_rank_00_optim_states.pt",
            "mp_rank_00_model_states.pt",
            "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt.tmp",
        ]
        self.assertEqual(uc.optimizer_shard_ranks(names), {0, 2})

    def test_saved_dp_and_latest_tag(self):
        root = Path(self.get_auto_remove_tmp_dir())
        self.assertIsNone(uc.latest_tag(str(root)))
        self.assertIsNone(uc.saved_dp_world_size(str(root / TAG)))
        _write_checkpoint(root / TAG, dp=4)
        self.assertEqual(uc.latest_tag(str(root)), TAG)
        self.assertEqual(uc.saved_dp_world_size(str(root / TAG)), 4)

    def test_partial_checkpoint_is_refused(self):
        root = Path(self.get_auto_remove_tmp_dir())
        _write_checkpoint(root / TAG, dp=4, shards=[0, 1])
        with self.assertRaisesRegex(RuntimeError, r"lacks the optimizer shards of rank\(s\) \[2, 3\]"):
            uc.require_complete_checkpoint(str(root / TAG))
        _write_checkpoint(root / TAG, dp=4)
        self.assertEqual(uc.require_complete_checkpoint(str(root / TAG)), 4)


@require_deepspeed
class TestConversion(TestCasePlus):
    def _run_capturing(self, returncode: int = 0, stderr: str = ""):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, returncode, stdout="out", stderr=stderr)

        original = uc.subprocess.run
        uc.subprocess.run = fake_run
        self.addCleanup(setattr, uc.subprocess, "run", original)
        return calls

    def test_missing_info_injected_only_without_model_parallelism(self):
        from deepspeed.checkpoint.constants import CHECKPOINT_PARALLEL_DIMS

        root = Path(self.get_auto_remove_tmp_dir())
        calls = self._run_capturing()
        _write_checkpoint(root / "dense", dp=2)
        _write_checkpoint(root / "tp", dp=2, mp_world_size=2)
        _write_checkpoint(root / "pp", dp=2, **{CHECKPOINT_PARALLEL_DIMS: {"pp_degree": 2, "tp_degree": 1}})
        for name in ("dense", "tp", "pp"):
            uc.convert_to_universal(str(root / name), str(root / "out" / name))
        injected = ["--inject_missing_state" in cmd for cmd in calls]
        self.assertEqual(injected, [True, False, False])
        self.assertEqual(calls[0][1:3], ["-m", "deepspeed.checkpoint.ds_to_universal"])

    def test_converter_failure_carries_its_output(self):
        root = Path(self.get_auto_remove_tmp_dir())
        self._run_capturing(returncode=3, stderr="AssertionError: boom")
        _write_checkpoint(root / TAG, dp=2)
        with self.assertRaisesRegex(RuntimeError, r"(?s)exit 3.*AssertionError: boom"):
            uc.convert_to_universal(str(root / TAG), str(root / "out"))

    def test_universal_load_flips_the_engine_flag_for_one_load(self):
        engine = _FakeEngine(dp=2)
        with uc.universal_load(engine):
            self.assertTrue(engine._config.load_universal_checkpoint)
        self.assertFalse(engine._config.load_universal_checkpoint)


@require_deepspeed
class TestResizeSupport(TestCasePlus):
    def test_zero3_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "ZeRO stages 1 and 2, not stage 3"):
            uc.require_resize_support(_FakeEngine(dp=2, stage=3))

    def test_sequence_parallel_needs_a_fixed_deepspeed(self):
        import deepspeed

        sp_mpu = SimpleNamespace(initialize_sequence_parallel=lambda size: None)
        original = deepspeed.__version__
        self.addCleanup(setattr, deepspeed, "__version__", original)
        deepspeed.__version__ = "0.20.1"
        with self.assertRaisesRegex(RuntimeError, r"deepspeed>=0\.20\.2, found 0\.20\.1"):
            uc.require_resize_support(_FakeEngine(dp=2, mpu=sp_mpu))
        uc.require_resize_support(_FakeEngine(dp=2))
        deepspeed.__version__ = uc.SEQUENCE_PARALLEL_MIN_DEEPSPEED
        uc.require_resize_support(_FakeEngine(dp=2, mpu=sp_mpu))


@require_deepspeed
class TestLoadWithDpResize(TestCasePlus):
    def setUp(self):
        super().setUp()
        self.root = Path(self.get_auto_remove_tmp_dir())
        self.converted = []

        def fake_convert(src, out):
            self.converted.append((src, out))
            Path(out).mkdir(parents=True)
            (Path(out).parent / "latest_universal").write_text(Path(out).name, encoding="utf-8")

        original = uc.convert_to_universal
        uc.convert_to_universal = fake_convert
        self.addCleanup(setattr, uc, "convert_to_universal", original)

    def test_same_dp_is_a_plain_load(self):
        _write_checkpoint(self.root / TAG, dp=2)
        engine = _FakeEngine(dp=2)
        uc.load_checkpoint_with_dp_resize(engine, str(self.root), load_optimizer_states=False)
        self.assertEqual(engine.loads, [(str(self.root), None, False, {"load_optimizer_states": False})])
        self.assertEqual(self.converted, [])

    def test_other_dp_loads_a_universal_copy_and_removes_it(self):
        _write_checkpoint(self.root / TAG, dp=4)
        engine = _FakeEngine(dp=2)
        load_path, _ = uc.load_checkpoint_with_dp_resize(engine, str(self.root))
        out_tag = TAG + uc.UNIVERSAL_TAG_SUFFIX
        self.assertEqual(self.converted, [(str(self.root / TAG), str(self.root / out_tag))])
        self.assertEqual(engine.loads, [(str(self.root), out_tag, True, {})])
        self.assertEqual(load_path, str(self.root / out_tag))
        self.assertFalse(engine._config.load_universal_checkpoint)
        left = sorted(p.name for p in self.root.iterdir())
        self.assertEqual(left, [TAG, "latest"])

    def test_scratch_dir_receives_the_copy(self):
        _write_checkpoint(self.root / TAG, dp=4)
        scratch = self.root / "scratch"
        engine = _FakeEngine(dp=1)
        uc.load_checkpoint_with_dp_resize(engine, str(self.root), TAG, scratch_dir=str(scratch))
        self.assertEqual(self.converted[0][1], str(scratch / (TAG + uc.UNIVERSAL_TAG_SUFFIX)))
        self.assertEqual(list(scratch.iterdir()), [])

    def test_partial_checkpoint_fails_before_loading(self):
        _write_checkpoint(self.root / TAG, dp=4, shards=[0, 1, 2])
        engine = _FakeEngine(dp=2)
        with self.assertRaisesRegex(RuntimeError, r"rank\(s\) \[3\]"):
            uc.load_checkpoint_with_dp_resize(engine, str(self.root))
        self.assertEqual(engine.loads, [])
