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
