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

import glob
import os
import shutil
import signal
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from deepspeed.comm import init_distributed

# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(1, git_repo_path)

# Imported after the sys.path insert above so a local checkout shadows any installed copy (multi-checkout dev).
from arctic_platform.testing_utils import get_unique_port_number  # noqa: E402


def _is_xdist_worker(config) -> bool:
    """True for a pytest-xdist worker process (``gw0`` etc.); False otherwise.

    Workers carry a ``workerinput`` attribute; neither the xdist controller nor a
    plain serial run does.
    """
    return hasattr(config, "workerinput")


def _is_xdist_controller(config) -> bool:
    """True for the pytest-xdist controller process (which does not run tests).

    Workers carry a ``workerinput`` attribute; the controller does not. ``-n``
    sets ``numprocesses``. The controller must not set up torch.distributed: it
    would bind the same MASTER_PORT that worker ``gw0`` derives and collide.
    """
    if _is_xdist_worker(config):
        return False
    return bool(getattr(config.option, "numprocesses", None))


def _reap_orphan_ray_clusters() -> None:
    """Kill processes + temp dirs orphaned by a previously-crashed Arctic RL run.

    A SIGKILLed run can leave several kinds of stragglers behind, each squatting ports / GPU memory and breaking
    the *next* run:
      - Ray daemons (``ray start --head`` gcs / raylet / monitor) under temp dirs named ``ray_arctic_*``.
      - The local HTTP server (``python -m arctic_platform.rl.http_server``).
      - Ray-actor workers (e.g. the sampling ``InferenceWorker``) and the vLLM ``EngineCore`` subprocesses they
        spawn.

    The daemons / actors carry the ``ray_arctic_*`` session dir on their command line and the HTTP server carries
    its module name, so we seed on a *cmdline-only* substring match (specific enough to never hit unrelated ray /
    uvicorn processes) and expand to descendant trees -- which catches a still-linked vLLM ``EngineCore`` under the
    matched ``raylet`` -> ``InferenceWorker`` (the common lingering-cluster case). An ``EngineCore`` that has been
    fully reparented to init rewrites its command line to a bare title (``VLLM::EngineCore``) and is no longer a
    descendant of anything we match; for that narrow case we additionally seed any process *titled* exactly
    ``VLLM::EngineCore`` whose inherited environment still references ``ray_arctic_*``. Crucially we only ever read
    ``/proc/<pid>/environ`` for that tiny title-matched candidate set -- never for arbitrary processes -- so we
    cannot kill an unrelated tree merely because it inherited a marker env var, and ``os.kill`` failing on procs we
    don't own keeps this scoped to our own uid.

    Invoked once at session start in the xdist controller / serial main (never in an xdist worker), so it runs
    BEFORE any worker spins up a cluster and every match is necessarily an orphan from an earlier run. Cheap no-op
    for non-RL sessions.
    """
    cmd_markers = ("ray_arctic_", "arctic_platform.rl.http_server")
    engine_title = "VLLM::EngineCore"

    def _read(pid: str, name: str) -> bytes:
        try:
            with open(f"/proc/{pid}/{name}", "rb") as fh:
                return fh.read()
        except OSError:
            return b""

    def _cmdline(pid: str) -> str:
        return _read(pid, "cmdline").replace(b"\x00", b" ").decode("utf-8", "replace").strip()

    def _ppid(pid: str) -> int | None:
        # ppid is field 2 after the ``)`` that closes comm (comm itself may contain spaces / parens).
        data = _read(pid, "stat")
        rparen = data.rfind(b")")
        if rparen == -1:
            return None
        fields = data[rparen + 2 :].split()
        try:
            return int(fields[1])
        except (IndexError, ValueError):
            return None

    my_pid = os.getpid()
    pids = [e for e in os.listdir("/proc") if e.isdigit()]
    children: dict[int, list[int]] = {}
    for pid in pids:
        parent = _ppid(pid)
        if parent is not None:
            children.setdefault(parent, []).append(int(pid))

    seeds: set[int] = set()
    for pid in pids:
        ipid = int(pid)
        if ipid == my_pid:
            continue
        cmdline = _cmdline(pid)
        if any(m in cmdline for m in cmd_markers):
            seeds.add(ipid)
        elif cmdline.startswith(engine_title) and b"ray_arctic_" in _read(pid, "environ"):
            seeds.add(ipid)

    # Expand seeds to their full descendant trees (catches a still-linked EngineCore under raylet/InferenceWorker).
    victims: set[int] = set()
    stack = list(seeds)
    while stack:
        pid = stack.pop()
        if pid in victims or pid == my_pid:
            continue
        victims.add(pid)
        stack.extend(children.get(pid, ()))

    roots = {os.environ.get("TMPDIR") or "/tmp", "/tmp", "/data-fast/tmp"}
    temp_dirs = sorted(d for root in roots for d in glob.glob(os.path.join(root, "ray_arctic_*")))

    if not victims and not temp_dirs:
        return

    print(
        f"[conftest] reaping orphans from a previous run: {len(victims)} process(es), {len(temp_dirs)} temp dir(s)",
        file=sys.stderr,
    )
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for d in temp_dirs:
        shutil.rmtree(d, ignore_errors=True)


def pytest_configure(config):
    # Once per session, in the controller (xdist) or the serial main process --
    # but NOT in an xdist worker, which could kill a sibling's live cluster --
    # reap Ray daemons orphaned by a previously-crashed run before any worker
    # starts a new one.
    if not _is_xdist_worker(config):
        _reap_orphan_ray_clusters()

    # Under pytest-xdist only the workers run tests (each in its own process with a
    # unique MASTER_PORT); the controller must not initialize a dist group.
    if _is_xdist_controller(config):
        return

    # Initialize the driver's single-rank dist group (xdist workers / serial main, never the controller). The
    # accelerator follows the actual hardware, not a marker: GPU-ness is encoded by the require_torch_* skip guards
    # (and the gpu_serial/vllm markers), so a CPU-only box must not be forced onto DS_ACCELERATOR=cuda. This lets a
    # single command run the CPU and GPU tests together -- the GPU integration tests destroy and re-create this group
    # per client session, and the CPU unit tests deliberately avoid collectives on it.
    _setup_dist()


def pytest_unconfigure(config):
    # Symmetric teardown for the dist group set up in pytest_configure. Without it torch/NCCL warns at process exit
    # that destroy_process_group() was never called (leaked resources). The controller never initialized a group.
    if _is_xdist_controller(config):
        return

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _setup_dist():
    os.environ["DS_ACCELERATOR"] = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # Unique per pytest-xdist worker so concurrent workers don't collide on the master port.
    os.environ["MASTER_PORT"] = str(get_unique_port_number())
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_SIZE"] = "1"

    init_distributed(auto_mpi_discovery=False)
