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
"""CPU tests for ``DeepSpeedWorker.prune_checkpoint_dirs`` (A3)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from arctic_platform.common.utils.checkpoint import load_module_state_skipping_ep_shards
from arctic_platform.common.utils.checkpoint import prune_checkpoint_dirs
from arctic_platform.common.utils.checkpoint import resolve_checkpoint_save_paths
from arctic_platform.common.utils.checkpoint import restore_trainable_ep_shards
from arctic_platform.testing_utils import TestCasePlus


class TestPruneCheckpointDirs(TestCasePlus):
    def _make_ckpts(self, parent: Path, steps: list[int]) -> None:
        for s in steps:
            d = parent / f"checkpoint-{s}"
            d.mkdir(parents=True)
            (d / "marker").write_text(str(s), encoding="utf-8")

    def test_keeps_newest_n(self):
        parent = Path(self.get_auto_remove_tmp_dir())
        self._make_ckpts(parent, [1, 2, 3, 4])
        removed = prune_checkpoint_dirs(str(parent), keep=2)
        self.assertEqual(removed, 2)
        left = sorted(p.name for p in parent.iterdir() if p.is_dir())
        self.assertEqual(left, ["checkpoint-3", "checkpoint-4"])

    def test_noop_when_keep_nonpositive(self):
        parent = Path(self.get_auto_remove_tmp_dir())
        self._make_ckpts(parent, [1, 2])
        self.assertEqual(prune_checkpoint_dirs(str(parent), keep=0), 0)
        self.assertEqual(prune_checkpoint_dirs(str(parent), keep=-1), 0)
        self.assertEqual(len(list(parent.iterdir())), 2)

    def test_ignores_non_checkpoint_dirs(self):
        parent = Path(self.get_auto_remove_tmp_dir())
        self._make_ckpts(parent, [1, 2, 3])
        (parent / "hf").mkdir()
        (parent / "latest").write_text("3", encoding="utf-8")
        removed = prune_checkpoint_dirs(str(parent), keep=1)
        self.assertEqual(removed, 2)
        self.assertTrue((parent / "checkpoint-3").is_dir())
        self.assertTrue((parent / "hf").is_dir())
        self.assertTrue((parent / "latest").is_file())


class TestSaveTotalLimitPruneRoot(TestCasePlus):
    """save_total_limit must prune the job checkpoint dir, not its parent."""

    def test_save_without_step_does_not_delete_shared_root_checkpoints(self):
        tmp = Path(self.get_auto_remove_tmp_dir())
        shared = tmp / "client"
        job = shared / "job-a"
        job.mkdir(parents=True)
        for root, steps in ((shared, [1, 2]), (job, [1, 2, 3])):
            for s in steps:
                d = root / f"checkpoint-{s}"
                d.mkdir()
                (d / "marker").write_text(str(s), encoding="utf-8")

        save_dir, prune_root = resolve_checkpoint_save_paths(str(job), step=None)
        self.assertEqual(save_dir, str(job))
        self.assertEqual(prune_root, str(job))
        prune_checkpoint_dirs(prune_root, keep=1)

        self.assertTrue((shared / "checkpoint-1").is_dir())
        self.assertTrue((shared / "checkpoint-2").is_dir())
        self.assertFalse((job / "checkpoint-1").exists())
        self.assertFalse((job / "checkpoint-2").exists())
        self.assertTrue((job / "checkpoint-3").is_dir())

    def test_save_with_step_still_prunes_the_job_root(self):
        job = Path(self.get_auto_remove_tmp_dir()) / "job-a"
        save_dir, prune_root = resolve_checkpoint_save_paths(str(job), step=7)
        self.assertEqual(save_dir, str(job / "checkpoint-7"))
        self.assertEqual(prune_root, str(job))


class TestLoadModuleStateSkippingEpShards(TestCasePlus):
    def _model(self) -> nn.Module:
        model = nn.Module()
        model.experts = nn.Parameter(torch.zeros(2, 3), requires_grad=False)
        model.experts.group_name = "ep_size_2"
        model.experts.allreduce = False
        model.base = nn.Parameter(torch.zeros(3), requires_grad=False)
        model.adapter = nn.Parameter(torch.zeros(3))
        return model

    def test_skips_rank_zero_experts_and_absent_frozen_weights(self):
        model = self._model()
        load_module_state_skipping_ep_shards({"experts": torch.ones(2, 3), "adapter": torch.ones(3)}, model)
        torch.testing.assert_close(model.experts, torch.zeros(2, 3))
        torch.testing.assert_close(model.adapter, torch.ones(3))

    def test_matches_names_through_activation_checkpoint_wrappers(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        model = nn.Module()
        model.layer = checkpoint_wrapper(self._model())
        self.assertIn("layer._checkpoint_wrapped_module.experts", dict(model.named_parameters()))
        load_module_state_skipping_ep_shards(
            {"layer.experts": torch.ones(2, 3), "layer.adapter": torch.ones(3)}, model
        )
        torch.testing.assert_close(model.layer.experts, torch.zeros(2, 3))
        torch.testing.assert_close(model.layer.adapter, torch.ones(3))
        with self.assertRaisesRegex(RuntimeError, "adapter"):
            load_module_state_skipping_ep_shards({"layer.base": torch.ones(3)}, model)

    def test_missing_trainable_parameter_raises(self):
        with self.assertRaisesRegex(RuntimeError, "adapter"):
            load_module_state_skipping_ep_shards({"base": torch.ones(3)}, self._model())

    def test_unexpected_key_raises(self):
        with self.assertRaisesRegex(RuntimeError, "stale"):
            load_module_state_skipping_ep_shards({"adapter": torch.ones(3), "stale": torch.ones(1)}, self._model())


class TestRestoreTrainableEpShards(TestCasePlus):
    def _engine(self, trainable_experts: bool, optimizer) -> SimpleNamespace:
        module = nn.Module()
        module.experts = nn.Parameter(torch.zeros(2, 3), requires_grad=trainable_experts)
        module.experts.group_name = "ep_size_2"
        module.experts.allreduce = False
        module.adapter = nn.Parameter(torch.zeros(3))
        return SimpleNamespace(module=module, optimizer=optimizer)

    def test_refreshes_parameters_from_the_optimizer_with_trainable_experts(self):
        calls = []
        restore_trainable_ep_shards(self._engine(True, SimpleNamespace(update_lp_params=lambda: calls.append(1))))
        self.assertEqual(calls, [1])

    def test_skips_frozen_experts(self):
        restore_trainable_ep_shards(self._engine(False, SimpleNamespace()))

    def test_unsupported_optimizer_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "SimpleNamespace"):
            restore_trainable_ep_shards(self._engine(True, SimpleNamespace()))
