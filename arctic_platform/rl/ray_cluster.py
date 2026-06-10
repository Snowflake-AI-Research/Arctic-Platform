# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multinode Ray cluster bootstrap.

Behaviour:
1. If a Ray cluster is already running, attach to it.
2. Otherwise start a head node locally and, if ``/job/hostfile`` lists remote
   hosts, fan out ``ray start`` via ``pdsh``.
3. On exit, only tear down a cluster this process started.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from arctic_platform.rl.utils.debug import see_memory_usage, pr, pr0

import ray
from ray._private.utils import read_ray_address

logger = logging.getLogger(__name__)

_HOSTFILE = "/job/hostfile"
_DEFAULT_RAY_PORT = 6379
_DEFAULT_RAY_DASHBOARD_PORT = 8265

_spawned_cluster = False
_spawned_temp_dir: str | None = None


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
        except OSError:
            return False
    return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _resolve_port(env_var: str, default: int) -> int:
    """Resolve a Ray port.

    If ``env_var`` is set, the user-supplied port is returned verbatim (no
    fallback: a subsequent bind collision is surfaced as a hard failure, so the
    user's explicit choice is never silently overridden). Otherwise return
    ``default`` if free, falling back to an OS-assigned free port.
    """
    override = os.environ.get(env_var)
    if override is not None:
        return int(override)
    if _is_port_free(default):
        return default
    port = _free_port()
    logger.info(f"Port {default} is in use for {env_var}; falling back to free port {port}")
    return port


def primary_ip() -> str:
    """Return the routable IP of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _peer_hosts() -> list[str]:
    """Return remote host entries from ``/job/hostfile``, excluding this machine."""
    try:
        with open(_HOSTFILE, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    head = primary_ip()
    hosts = [ln.split()[0] for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return [h for h in hosts if h != head]


def _pdsh(hosts: list[str], cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    env = {**os.environ, "PDSH_RCMD_TYPE": "ssh"}
    return subprocess.run(["pdsh", "-w", ",".join(hosts)] + cmd, env=env, **kwargs)


def _ray_bin() -> str:
    return str(Path(sys.executable).resolve().parent / "ray")


def init_ray_cluster(auto_attach: bool = True) -> None:
    """Attach to an existing Ray cluster or start one (with multinode support).

    Args:
        auto_attach: If True (default), attempt to attach to a pre-existing Ray
            cluster before starting a new one. The attach is only honored if the
            existing cluster exposes GPU resources; CPU-only clusters (e.g. one
            started by an unrelated integration that also uses Ray) are ignored
            and a new cluster is started instead. Set to False to always start a
            fresh cluster.
    """
    global _spawned_cluster, _spawned_temp_dir

    # 1. Try attaching to a running cluster (only if it has GPUs).
    if 1: # auto_attach:
        try:
            ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)
            gpus = ray.cluster_resources().get("GPU", 0)
            if gpus > 0:
                logger.info(
                    "Attached to existing Ray cluster at %s (%.0f GPU(s))",
                    ray.get_runtime_context().gcs_address,
                    gpus,
                )
                return
            logger.info(
                "Existing Ray cluster has no GPU resources; ignoring and starting a new cluster."
            )
            ray.shutdown()
        except ConnectionError:
            pass

    # 2. Start a new head node. Pick non-colliding ports so we don't clash with
    # any pre-existing (e.g. CPU-only) cluster on this host. Use a unique
    # ``--temp-dir`` so we can stop only our processes via ``ray stop --temp-dir``.
    r = _ray_bin()
    ray_port = _resolve_port("RAY_PORT", _DEFAULT_RAY_PORT)
    dashboard_port = _resolve_port("RAY_DASHBOARD_PORT", _DEFAULT_RAY_DASHBOARD_PORT)
    _spawned_temp_dir = tempfile.mkdtemp(prefix="ray_arctic_")
    subprocess.run(
        [
            r,
            "start",
            "--head",
            f"--port={ray_port}",
            f"--dashboard-port={dashboard_port}",
            f"--temp-dir={_spawned_temp_dir}",
            "--disable-usage-stats",
        ],
        check=True,
        timeout=300,
        env=os.environ,
    )
    pr0(f"[init_ray_cluster] ray started with port {ray_port} and dashboard port {dashboard_port}")

    # 3. Start workers on peer nodes (if any).
    peers = _peer_hosts()
    pr0(f"[init_ray_cluster] peers: {peers}")
    gcs = read_ray_address(_spawned_temp_dir)
    if peers:
        logger.info("Starting Ray workers on %s (address=%s)", peers, gcs)
        result = _pdsh(
            peers,
            [r, "start", f"--address={gcs}", f"--temp-dir={_spawned_temp_dir}", "--disable-usage-stats"],
            check=False,
            timeout=600,
        )
        if result.returncode != 0:
            logger.warning("pdsh ray start returned exit code %d", result.returncode)

    _spawned_cluster = True
    # Use the explicit GCS address instead of ``auto`` so we don't accidentally
    # reconnect to a different Ray cluster running on this host.
    ray.init(address=gcs, ignore_reinit_error=True, log_to_driver=True)
    pr0(f"[init_ray_cluster] ray initialized with address {gcs}")
    resources = ray.available_resources()
    logger.info(
        "Ray cluster: %.0f GPU(s), %.0f CPU(s), %d node(s)",
        resources.get("GPU", 0),
        resources.get("CPU", 0),
        sum(1 for k in resources if k.startswith("node:")),
    )


def _shutdown() -> None:
    global _spawned_cluster, _spawned_temp_dir
    try:
        ray.shutdown()
    except Exception:
        pass
    if not _spawned_cluster:
        return
    _spawned_cluster = False
    if _spawned_temp_dir is None:
        return

    # Scope ``ray stop`` to our cluster's temp dir so coexisting Ray clusters
    # owned by this user are left untouched.
    r = _ray_bin()
    stop_cmd = [r, "stop", "-f", f"--temp-dir={_spawned_temp_dir}"]
    subprocess.run(stop_cmd, check=False, timeout=120, capture_output=True)
    peers = _peer_hosts()
    if peers:
        _pdsh(peers, stop_cmd, check=False, timeout=180, capture_output=True)
    shutil.rmtree(_spawned_temp_dir, ignore_errors=True)
    _spawned_temp_dir = None


atexit.register(_shutdown)
