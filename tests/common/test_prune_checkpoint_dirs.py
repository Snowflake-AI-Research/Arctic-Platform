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

from arctic_platform.common.utils.checkpoint import prune_checkpoint_dirs
from arctic_platform.common.utils.checkpoint import resolve_checkpoint_save_paths
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
