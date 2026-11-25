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


_EXPERTS = "layers.0.mlp.experts"
_EXPERT_SHAPES = {"w1": (2, 3, 4), "w2": (2, 4, 3), "w3": (2, 3, 4)}


def _expert_shard_paths(tag_dir: Path) -> list[str]:
    return [str(tag_dir / f"experts_ep_rank_{rank:03d}.pt") for rank in range(2)]


def _write_expert_parallel_checkpoint(tag_dir: Path) -> dict:
    """A DP 2 x EP 2 checkpoint with two local experts per rank; returns each EP rank's expert tensors."""
    import torch

    param_shapes = [{"dense.weight": (4, 4)}, {f"{_EXPERTS}.{w}": shape for w, shape in _EXPERT_SHAPES.items()}]
    _write_checkpoint(tag_dir, dp=2, param_shapes=param_shapes, module={"dense.weight": torch.zeros(4, 4)})
    shards = {}
    for rank, path in enumerate(_expert_shard_paths(tag_dir)):
        shards[rank] = {
            f"{_EXPERTS}.{w}": torch.arange(torch.Size(shape).numel(), dtype=torch.float32).view(shape) + 100 * rank
            for w, shape in _EXPERT_SHAPES.items()
        }
        torch.save(shards[rank], path)
    return shards


class _FakeEngine:
    """Records each ``load_checkpoint`` call with the universal flag it saw."""

    def __init__(self, dp: int, stage: int = 2, mpu=None, params: dict | None = None, node_local: bool = False):
        self.seq_dp_world_size = dp
        self.mpu = mpu
        self.module = SimpleNamespace(named_parameters=lambda: iter((params or {}).items()))
        self._stage = stage
        self._node_local = node_local
        self._config = SimpleNamespace(load_universal_checkpoint=False)
        self.loads: list[tuple[str, str | None, bool, dict]] = []

    def zero_optimization_stage(self) -> int:
        return self._stage

    def use_node_local_storage(self) -> bool:
        return self._node_local

    def load_checkpoint(self, load_dir, tag=None, **kwargs):
        self.loads.append((load_dir, tag, self._config.load_universal_checkpoint, kwargs))
        return os.path.join(load_dir, str(tag)), {}


_DEFAULT_TIMEOUT_S = 2
_CONVERSION_S = 3 * _DEFAULT_TIMEOUT_S


def _slow_conversion_worker(rank: int, world_size: int, init_method: str, root: str) -> None:
    """One rank of a gloo job whose rank 0 converts for longer than the default process-group timeout."""
    import time
    from datetime import timedelta

    import torch.distributed as dist

    def slow_convert(src, out):
        time.sleep(_CONVERSION_S)
        Path(out).mkdir(parents=True)

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_DEFAULT_TIMEOUT_S),
    )
    uc.convert_to_universal = slow_convert
    try:
        engine = _FakeEngine(dp=world_size)
        uc.load_checkpoint_with_dp_resize(engine, root, conversion_timeout_s=60)
        assert [universal for _dir, _tag, universal, _kwargs in engine.loads] == [True]
    finally:
        dist.destroy_process_group()


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

    def test_expert_shards_are_staged_as_autoep_input_and_removed(self):
        root = Path(self.get_auto_remove_tmp_dir())
        _write_expert_parallel_checkpoint(root / TAG)
        out = root / "out" / TAG
        staged = []

        def fake_run(cmd, **kwargs):
            input_dir = Path(cmd[cmd.index("--input_folder") + 1])
            staged.append((input_dir, sorted(p.name for p in input_dir.iterdir())))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        original = uc.subprocess.run
        uc.subprocess.run = fake_run
        self.addCleanup(setattr, uc.subprocess, "run", original)
        uc.convert_to_universal(str(root / TAG), str(out), expert_shards=_expert_shard_paths(root / TAG))
        ((input_dir, names),) = staged
        self.assertEqual(input_dir, Path(str(out) + ".autoep-input"))
        self.assertIn("layer_0_expert_3_mp_rank_00_model_states.pt", names)
        self.assertIn("bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt", names)
        self.assertFalse(input_dir.exists())

    def test_universal_load_flips_the_engine_flag_for_one_load(self):
        engine = _FakeEngine(dp=2)
        with uc.universal_load(engine):
            self.assertTrue(engine._config.load_universal_checkpoint)
        self.assertFalse(engine._config.load_universal_checkpoint)


@require_deepspeed
class TestAutoEPStaging(TestCasePlus):
    def test_metadata_describes_each_experts_block(self):
        shapes = {f"layers.{i}.mlp.experts.{w}": s for i in (0, 1) for w, s in _EXPERT_SHAPES.items()}
        layers = uc.autoep_layers_metadata({"dense.weight": (4, 4), **shapes}, list(shapes), ep_size=4)
        self.assertEqual(
            layers,
            [
                {
                    "moe_layer_id": i,
                    "expert_key_prefix": f"layers.{i}.mlp.experts",
                    "num_experts": 8,
                    "num_local_experts": 2,
                    "ep_size": 4,
                }
                for i in (0, 1)
            ],
        )

    def test_metadata_refuses_what_the_converter_cannot_consolidate(self):
        shapes = {f"{_EXPERTS}.{w}": s for w, s in _EXPERT_SHAPES.items()}
        engram = "layers.0.engram.embedding.weight"
        with self.assertRaisesRegex(ValueError, "not an experts.w1/w2/w3 weight"):
            uc.autoep_layers_metadata({**shapes, engram: (8, 4)}, [*shapes, engram], ep_size=2)
        with self.assertRaisesRegex(ValueError, "has no optimizer state"):
            uc.autoep_layers_metadata({}, list(shapes), ep_size=2)
        with self.assertRaisesRegex(ValueError, r"expected \['w1', 'w2', 'w3'\]"):
            uc.autoep_layers_metadata(shapes, list(shapes)[:2], ep_size=2)
        uneven = {**shapes, f"{_EXPERTS}.w3": (3, 3, 4)}
        with self.assertRaisesRegex(ValueError, "disagree on the local expert count"):
            uc.autoep_layers_metadata(uneven, list(uneven), ep_size=2)

    def test_staged_input_matches_autoep_layout(self):
        import torch

        root = Path(self.get_auto_remove_tmp_dir())
        src = root / TAG
        shards = _write_expert_parallel_checkpoint(src)
        staging = root / "staging"
        layers = uc.stage_autoep_input(str(src), str(staging), _expert_shard_paths(src))
        state = torch.load(staging / "mp_rank_00_model_states.pt", weights_only=False)
        self.assertEqual(state["ds_autoep_layers"], layers)
        self.assertEqual(layers[0]["num_experts"], 4)
        self.assertTrue(torch.equal(state["module"]["dense.weight"], torch.zeros(4, 4)))
        for name in ("bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt", "experts_ep_rank_001.pt"):
            self.assertEqual(os.readlink(staging / name), str(src / name))
        for expert in range(4):
            rank, local = divmod(expert, 2)
            saved = torch.load(staging / f"layer_0_expert_{expert}_mp_rank_00_model_states.pt", weights_only=True)
            self.assertEqual(sorted(saved), [f"{_EXPERTS}.{w}.{expert}" for w in ("w1", "w2", "w3")])
            for w in ("w1", "w2", "w3"):
                tensor = saved[f"{_EXPERTS}.{w}.{expert}"]
                self.assertTrue(torch.equal(tensor, shards[rank][f"{_EXPERTS}.{w}"][local]))
                self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())

    def test_shards_with_different_parameters_are_refused(self):
        import torch

        root = Path(self.get_auto_remove_tmp_dir())
        _write_expert_parallel_checkpoint(root / TAG)
        torch.save({f"{_EXPERTS}.w1": torch.zeros(2, 3, 4)}, _expert_shard_paths(root / TAG)[1])
        with self.assertRaisesRegex(RuntimeError, r"holds \['layers.0.mlp.experts.w2'"):
            uc.stage_autoep_input(str(root / TAG), str(root / "staging"), _expert_shard_paths(root / TAG))


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
        ((src, out),) = self.converted
        out_tag = Path(out).name
        self.assertRegex(out_tag, rf"^{TAG}{uc.UNIVERSAL_TAG_SUFFIX}_[0-9a-f]{{8}}$")
        self.assertEqual((src, out), (str(self.root / TAG), str(self.root / out_tag)))
        self.assertEqual(engine.loads, [(str(self.root), out_tag, True, {})])
        self.assertEqual(load_path, str(self.root / out_tag))
        self.assertFalse(engine._config.load_universal_checkpoint)
        left = sorted(p.name for p in self.root.iterdir())
        self.assertEqual(left, [TAG, "latest"])

    def test_each_load_converts_into_its_own_copy(self):
        # Jobs resizing the same shared checkpoint at once must not write into, or remove, each other's copy.
        _write_checkpoint(self.root / TAG, dp=4)
        uc.load_checkpoint_with_dp_resize(_FakeEngine(dp=2), str(self.root))
        uc.load_checkpoint_with_dp_resize(_FakeEngine(dp=2), str(self.root))
        first, second = (out for _src, out in self.converted)
        self.assertNotEqual(first, second)

    def test_scratch_dir_receives_the_copy(self):
        _write_checkpoint(self.root / TAG, dp=4)
        scratch = self.root / "scratch"
        engine = _FakeEngine(dp=1)
        uc.load_checkpoint_with_dp_resize(engine, str(self.root), TAG, scratch_dir=str(scratch))
        self.assertEqual(Path(self.converted[0][1]).parent, scratch)
        self.assertEqual(list(scratch.iterdir()), [])

    def test_a_failed_load_still_removes_the_copy(self):
        _write_checkpoint(self.root / TAG, dp=4)
        engine = _FakeEngine(dp=2)

        def failing_load(load_dir, tag=None, **kwargs):
            raise OSError("unreadable universal file")

        engine.load_checkpoint = failing_load
        with self.assertRaisesRegex(OSError, "unreadable universal file"):
            uc.load_checkpoint_with_dp_resize(engine, str(self.root))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [TAG, "latest"])

    def test_a_failed_conversion_removes_its_partial_copy(self):
        _write_checkpoint(self.root / TAG, dp=4)

        def failing_convert(src, out):
            Path(out).mkdir(parents=True)
            raise RuntimeError("ds_to_universal failed")

        uc.convert_to_universal = failing_convert
        engine = _FakeEngine(dp=2)
        with self.assertRaisesRegex(RuntimeError, "ds_to_universal failed"):
            uc.load_checkpoint_with_dp_resize(engine, str(self.root))
        self.assertEqual(engine.loads, [])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [TAG, "latest"])

    def test_ranks_wait_for_a_conversion_longer_than_the_default_timeout(self):
        import torch.multiprocessing as mp

        _write_checkpoint(self.root / TAG, dp=4)
        mp.spawn(_slow_conversion_worker, args=(2, (self.root / "rendezvous").as_uri(), str(self.root)), nprocs=2)

    def test_expert_parallel_engine_is_refused_before_converting(self):
        _write_checkpoint(self.root / TAG, dp=4)
        expert = SimpleNamespace(allreduce=False, group_name="ep_size_2")
        engine = _FakeEngine(dp=2, params={f"{_EXPERTS}.w1": expert, "dense.weight": SimpleNamespace()})
        with self.assertRaisesRegex(RuntimeError, r"convert_to_universal\(expert_shards=\.\.\.\)"):
            uc.load_checkpoint_with_dp_resize(engine, str(self.root))
        self.assertEqual((self.converted, engine.loads), ([], []))

    def test_converter_follows_the_declared_storage(self):
        # Ranks 0-3 on host a, 4-7 on host b; every process sees one GPU, so LOCAL_RANK would be 0 everywhere.
        hosts = ["a"] * 4 + ["b"] * 4
        original_world, original_gather = uc._world, uc._gather
        self.addCleanup(setattr, uc, "_world", original_world)
        self.addCleanup(setattr, uc, "_gather", original_gather)
        uc._gather = lambda value: list(hosts)
        for node_local, converters in ((False, [0]), (True, [0, 4])):
            found = []
            for rank in range(8):
                uc._world = lambda rank=rank: (rank, 8)
                if uc.is_universal_converter(_FakeEngine(dp=8, node_local=node_local)):
                    found.append(rank)
            self.assertEqual(found, converters, node_local)

    def test_partial_checkpoint_fails_before_loading(self):
        _write_checkpoint(self.root / TAG, dp=4, shards=[0, 1, 2])
        engine = _FakeEngine(dp=2)
        with self.assertRaisesRegex(RuntimeError, r"rank\(s\) \[3\]"):
            uc.load_checkpoint_with_dp_resize(engine, str(self.root))
        self.assertEqual(engine.loads, [])
