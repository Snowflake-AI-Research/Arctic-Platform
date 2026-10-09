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

"""A checkpoint file written on one node is readable on the node that loads it."""

from __future__ import annotations

from pathlib import Path

from arctic_platform.common.utils.checkpoint import checkpoint_install_command
from arctic_platform.common.utils.checkpoint import merge_checkpoint_tree
from arctic_platform.common.utils.checkpoint import publish_node_checkpoint
from arctic_platform.common.utils.checkpoint import require_checkpoint_files
from arctic_platform.testing_utils import TestCasePlus

_RANK0 = "global_step10/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
_RANK8 = "global_step10/bf16_zero_pp_rank_8_mp_rank_00_optim_states.pt"


class TestCheckpointVisibility(TestCasePlus):
    def _node(self, tmp: Path, name: str, files: dict[str, bytes]) -> Path:
        root = tmp / name / "arctic_rl_job_1"
        for rel, payload in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return root

    def test_missing_rank_file_is_not_readable(self):
        tmp = Path(self.get_auto_remove_tmp_dir())
        node = self._node(tmp, "node-a", {_RANK0: b"rank0", "latest": b"global_step10"})
        with self.assertRaises(FileNotFoundError) as caught:
            require_checkpoint_files(str(node), [_RANK0, _RANK8, "latest"])
        self.assertIn(_RANK8, str(caught.exception))

    def test_rank_file_written_on_one_node_is_readable_on_the_other(self):
        tmp = Path(self.get_auto_remove_tmp_dir())
        node_a = self._node(tmp, "node-a", {_RANK0: b"rank0", "latest": b"global_step10"})
        node_b = self._node(tmp, "node-b", {_RANK8: b"rank8"})
        roots = {"10.0.0.1": node_a, "10.0.0.2": node_b}

        def peer_sees(host, path):
            return False

        def install(root, host):
            merge_checkpoint_tree(root, str(roots[host]))

        files_a = publish_node_checkpoint(str(node_a), ["10.0.0.2"], install=install, peer_sees=peer_sees)
        files_b = publish_node_checkpoint(str(node_b), ["10.0.0.1"], install=install, peer_sees=peer_sees)
        expected = sorted(set(files_a) | set(files_b))
        require_checkpoint_files(str(node_a), expected)
        require_checkpoint_files(str(node_b), expected)
        self.assertEqual((node_b / _RANK8).read_bytes(), b"rank8")
        self.assertEqual((node_a / _RANK8).read_bytes(), b"rank8")
        self.assertEqual((node_b / "latest").read_bytes(), b"global_step10")

    def test_shared_directory_is_not_copied(self):
        tmp = Path(self.get_auto_remove_tmp_dir())
        node = self._node(tmp, "shared", {_RANK8: b"rank8", "latest": b"global_step10"})
        calls = []

        def peer_sees(host, path):
            return Path(path).is_file()

        def install(root, host):
            calls.append(host)

        publish_node_checkpoint(str(node), ["10.0.0.2"], install=install, peer_sees=peer_sees)
        self.assertEqual(calls, [])
        require_checkpoint_files(str(node), [_RANK8, "latest"])

    def test_install_command_extracts_into_the_parent(self):
        argv, remote = checkpoint_install_command("/data-fast/tmp/job/arctic_rl_job_1")
        self.assertEqual(argv, ["tar", "cf", "-", "-C", "/data-fast/tmp/job", "arctic_rl_job_1"])
        self.assertIn("mkdir -p /data-fast/tmp/job", remote)
        self.assertIn("tar xf - -C /data-fast/tmp/job", remote)
        self.assertTrue(remote.endswith("&& sync"))
