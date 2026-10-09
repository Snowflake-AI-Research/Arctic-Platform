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
"""Checkpoint directory helpers (no Ray / DeepSpeed imports)."""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess

logger = logging.getLogger(__name__)


def resolve_checkpoint_save_paths(root: str, step: int | None) -> tuple[str, str]:
    """Return ``(save_dir, prune_root)`` for a training checkpoint save.

    ``prune_root`` is the job checkpoint directory (the parent of ``checkpoint-*``
    children). It is always ``root``, including when ``step`` is omitted.
    """
    if step is not None:
        save_dir = os.path.join(root, f"checkpoint-{int(step)}")
    else:
        save_dir = root
    return save_dir, root


def prune_checkpoint_dirs(parent_dir: str, keep: int) -> int:
    """Keep the newest ``keep`` ``checkpoint-*`` dirs under ``parent_dir``; remove older.

    Returns the number of directories removed. No-op when ``keep <= 0``.
    """
    if keep is None or int(keep) <= 0:
        return 0
    keep = int(keep)
    if not os.path.isdir(parent_dir):
        return 0
    pat = re.compile(r"^checkpoint-(\d+)$")
    found = []
    for name in os.listdir(parent_dir):
        m = pat.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(parent_dir, name)))
    found.sort(key=lambda x: x[0])
    to_remove = found[:-keep] if len(found) > keep else []
    removed = 0
    for _, path in to_remove:
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


_PROBE_NAME = ".dss_ckpt_visibility_probe"


def relative_checkpoint_files(root: str) -> list[str]:
    """Return sorted file paths under ``root``, relative to it.

    The visibility probe is not a checkpoint file and is omitted.
    """
    if not os.path.isdir(root):
        return []
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name == _PROBE_NAME:
                continue
            found.append(os.path.relpath(os.path.join(dirpath, name), root))
    found.sort()
    return found


def require_checkpoint_files(root: str, relative_paths: list[str]) -> None:
    """Open every listed file under ``root`` and read one byte.

    Raises ``FileNotFoundError`` listing every path that cannot be read.
    """
    missing = []
    for rel in relative_paths:
        path = os.path.join(root, rel)
        try:
            with open(path, "rb") as handle:
                handle.read(1)
        except OSError:
            missing.append(path)
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"checkpoint files are not readable: {joined}")


def merge_checkpoint_tree(source_root: str, dest_root: str) -> list[str]:
    """Copy files present under ``source_root`` and absent under ``dest_root``.

    Returns the relative paths copied. A file the destination already has is left untouched.
    """
    copied = []
    for rel in relative_checkpoint_files(source_root):
        src = os.path.join(source_root, rel)
        dst = os.path.join(dest_root, rel)
        if os.path.isfile(dst):
            continue
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    return copied


def checkpoint_install_command(root: str) -> tuple[list[str], str]:
    """Build the local ``tar`` argv and the remote shell that extracts it.

    The archive is the checkpoint directory itself, extracted into that directory's parent so a peer
    receives the same absolute path.
    """
    root = os.path.abspath(root)
    parent, name = os.path.split(root.rstrip(os.sep))
    if not parent or not name:
        raise ValueError(f"checkpoint directory must have a parent, got {root}")
    remote = f"mkdir -p {shlex.quote(parent)} && tar xf - -C {shlex.quote(parent)} && sync"
    return ["tar", "cf", "-", "-C", parent, name], remote


def _fsync_directory(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_visibility_probe(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    probe = os.path.join(root, _PROBE_NAME)
    fd = os.open(probe, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
    try:
        os.write(fd, b"1")
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(root)
    return probe


def peers_share_checkpoint_root(root: str, peers: list[str], peer_sees) -> bool:
    """True when a file created under ``root`` is already visible on every peer."""
    probe = _write_visibility_probe(root)
    try:
        return all(peer_sees(host, probe) for host in peers)
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


def _ssh_peer_sees(host: str, path: str) -> bool:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"test -f {shlex.quote(path)}"],
        capture_output=True,
    )
    return completed.returncode == 0


def _ssh_install_checkpoint_tree(root: str, host: str) -> None:
    argv, remote = checkpoint_install_command(root)
    tar = subprocess.Popen(argv, stdout=subprocess.PIPE)
    if tar.stdout is None:
        raise RuntimeError(f"failed to stream checkpoint {root}")
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, remote],
            stdin=tar.stdout,
            capture_output=True,
        )
    finally:
        tar.stdout.close()
        tar_rc = tar.wait()
    if tar_rc != 0 or completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"copying checkpoint {root} to {host} failed (tar={tar_rc}, ssh={completed.returncode}): {detail}"
        )


def publish_node_checkpoint(root: str, peers: list[str], *, install=None, peer_sees=None) -> list[str]:
    """Copy this node's checkpoint files to ``peers`` when they cannot see them.

    Returns the relative paths present on this node. Each of those paths is opened locally before
    returning. A shared directory is not copied: a probe file written here is already visible on every peer.
    """
    if install is None:
        install = _ssh_install_checkpoint_tree
    if peer_sees is None:
        peer_sees = _ssh_peer_sees
    if peers and not peers_share_checkpoint_root(root, peers, peer_sees):
        for peer in peers:
            install(root, peer)
            logger.info("copied checkpoint %s to %s", root, peer)
    files = relative_checkpoint_files(root)
    require_checkpoint_files(root, files)
    return files
