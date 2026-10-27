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


"Gateway jobs, explicit request packing, and reinforcement learning calls."

from __future__ import annotations

import base64
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import select
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client.requests import log_probs_request

from .config import ReviewedClientConfig
from .config import ReviewedSamplingConfig
from .config import ReviewedTrainingConfig
from .config import correctness_microbatch_tokens
from .cortex import CorrectnessClient as ArcticClient
from .cortex import CorrectnessClient as RLClient
from .cortex import CortexTrainingConfig
from .storage import _ds_ssh
from .storage import parse_hostfile
from .storage import require_data_fast_path
from .transport import Checkpoint
from .transport import UnsafeCleanupError


@dataclass
class StepResult:
    avg_loss: float
    grad_norms: Dict[str, float]
    model_calls: Optional[int] = None
    packed_rows: Optional[int] = None
    optimizer_state_manifest: Optional[str] = None
    participation: Optional[dict] = None


@dataclass(frozen=True)
class GatewaySession:
    workdir: Path
    slots: int
    backend: OnPremConfig | CortexConfig = field(default_factory=OnPremConfig)


@dataclass
class ArcticJob:
    client: ArcticClient
    job_type: str

    @property
    def job_id(self):
        return getattr(self.client.jobs, self.job_type)


def available_gpus() -> int:
    import torch

    local = torch.cuda.device_count() if torch.cuda.is_available() else 0
    hostfile = os.environ.get("DSS_GATEWAY_HOSTFILE") or os.environ.get("ARCTIC_HOSTFILE")
    if not hostfile or not Path(hostfile).exists():
        return local
    return sum(entry.slots for entry in parse_hostfile(hostfile)) or local


_STANDARD_MASTER_PORT = 29500
_RAY_PROCESS_PATTERN = (
    r"([r]aylet|[g]cs_server|[d]ashboard.py|[m]onitor.py|[l]og_monitor.py|[d]efault_worker.py|[r]ay::)"
)
_GATEWAY_PROC_ROOT = Path("/proc")


def _allocation_hostfile() -> Path | None:
    value = os.environ.get("DSS_GATEWAY_HOSTFILE") or os.environ.get("ARCTIC_HOSTFILE")
    if value:
        path = Path(value)
        if path.is_file():
            return path
    for default in (Path("/data-fast/hostfile"), Path("/job/hostfile")):
        if default.is_file():
            return default
    return None


def _gateway_state_directory() -> Path:
    return Path(tempfile.gettempdir()) / f"arctic-correctness-gateway-{os.getuid()}"


def _gateway_process_identity(pid: int) -> dict | None:
    """Bind a PID to this user's actual live Linux process, including its start time."""
    directory = _GATEWAY_PROC_ROOT / str(pid)
    try:
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        return {"pid": pid, "start_time": fields[19], "uid": directory.stat().st_uid}
    except FileNotFoundError:
        return None


def _stop_stale_gateway(gateway: dict, log_path: str) -> None:
    if _gateway_process_identity(gateway["pid"]) != gateway:
        return
    print(
        f"Stopping stale owned Cortex-training gateway (PID {gateway['pid']}); log: {log_path}",
        flush=True,
    )
    os.kill(gateway["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 120
    while _gateway_process_identity(gateway["pid"]) == gateway:
        if time.monotonic() >= deadline:
            os.killpg(gateway["pid"], signal.SIGKILL)
            raise UnsafeCleanupError(
                f"stale owned gateway required forced shutdown; inspect {log_path} before retrying"
            )
        time.sleep(0.1)
    print("Stale owned Cortex-training gateway stopped.", flush=True)


def _legacy_owned_gateways() -> list[tuple[dict, str]]:
    """Recognize pre-registry orphans only by the harness launch and retained log layout."""
    found = []
    for directory in _GATEWAY_PROC_ROOT.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            command = (directory / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            if (
                len(command) != 11
                or command[1:5] != [b"-m", b"dss.ray_dss.gateway_server", b"--host", b"127.0.0.1"]
                or command[5] != b"--port"
                or not command[6].isdigit()
                or command[7:10] != [b"--backend", b"gpu", b"--hostfile"]
            ):
                continue
            pid = int(directory.name)
            fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
            if fields[1] != "1" or os.getpgid(pid) != pid:
                continue
            checkout = (directory / "cwd").resolve()
            log_path = Path(os.readlink(directory / "fd" / "1"))
            if (
                not (checkout / "skills" / "correctness" / "SKILL.md").is_file()
                or not log_path.is_relative_to(checkout)
                or log_path.name != "gateway.log"
                or os.readlink(directory / "fd" / "2") != str(log_path)
            ):
                continue
            relative = log_path.relative_to(checkout).parts
            onboarding = (
                len(relative) > 3 and relative[:2] == ("work", "logs") and relative[2].startswith("onboarding-")
            )
            regression = "cases" in relative[:-1]
            if onboarding or regression:
                identity = _gateway_process_identity(pid)
                if identity is not None:
                    found.append((identity, str(log_path)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def _recover_owned_gateway(record_path: Path) -> None:
    if not record_path.exists():
        return
    try:
        record = json.loads(record_path.read_text())
        if record["schema"] != "correctness-owned-gateway-v1":
            raise ValueError("unknown gateway ownership record")
        owner, gateway = record["owner"], record["gateway"]
        if owner["uid"] != os.getuid() or gateway["uid"] != os.getuid():
            raise ValueError("gateway ownership record belongs to a different user")
        if _gateway_process_identity(owner["pid"]) == owner:
            raise UnsafeCleanupError("another correctness command still owns the local gateway; stop it first")
        if _gateway_process_identity(gateway["pid"]) != gateway:
            _reap_owned_gateway_descendants(gateway, record["token"])
            record_path.unlink()
            return
        environment = (_GATEWAY_PROC_ROOT / str(gateway["pid"]) / "environ").read_bytes().split(b"\0")
        marker = f"DSS_CORRECTNESS_GATEWAY_OWNER={record['token']}".encode()
        if marker not in environment or os.getpgid(gateway["pid"]) != gateway["pid"]:
            raise ValueError("live process does not match the recorded harness-owned gateway")
        try:
            _stop_stale_gateway(gateway, record["log"])
        finally:
            _reap_owned_gateway_descendants(gateway, record["token"])
        record_path.unlink()
    except UnsafeCleanupError:
        raise
    except ProcessLookupError:
        record_path.unlink(missing_ok=True)
    except Exception as exc:
        raise UnsafeCleanupError(
            f"cannot safely recover the previous local gateway; inspect {record_path} before retrying"
        ) from exc


@contextlib.contextmanager
def _owned_gateway_slot() -> Iterator[Path]:
    directory = _gateway_state_directory()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / "owner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise UnsafeCleanupError(
                "another correctness command still owns the local gateway; stop it first"
            ) from exc
        try:
            record_path = directory / "gateway.json"
            _recover_owned_gateway(record_path)
            for gateway, log_path in _legacy_owned_gateways():
                try:
                    _stop_stale_gateway(gateway, log_path)
                except ProcessLookupError:
                    pass
            yield record_path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def recover_local_cortex_gateway() -> None:
    """Recover an orphan before onboarding starts independent HF model work."""
    with _owned_gateway_slot():
        pass


@contextlib.contextmanager
def local_cortex_gateway(
    directory: Path,
    *,
    gateway_module: str = "dss.ray_dss.gateway_server",
    allocation_members: dict | None = None,
    gateway_command_builder=None,
    gateway_command_source: dict | None = None,
) -> Iterator[str]:
    """Recover an orphan, then start and close this command's owned DSS service."""
    if gateway_module not in (
        "dss.ray_dss.gateway_server",
        "arctic_platform.correctness.harness.gateway",
        "arctic_platform.debug.correctness_participation",
    ):
        raise ValueError("unsupported correctness gateway module")
    if allocation_members is not None and gateway_module != "arctic_platform.correctness.harness.gateway":
        raise ValueError("explicit allocation identities require the correctness gateway")
    options = {} if allocation_members is None else {"allocation_members": allocation_members}
    if gateway_command_builder is not None or gateway_command_source is not None:
        from .cortex_optimizer import verify_native_callback

        if gateway_module != "dss.ray_dss.gateway_server":
            raise ValueError("native gateway command builder requires the upstream DSS gateway")
        verify_native_callback(gateway_command_builder, gateway_command_source)
        options.update(
            gateway_command_builder=gateway_command_builder,
            gateway_command_source=gateway_command_source,
        )
    with _owned_gateway_slot() as record_path:
        with _started_local_cortex_gateway(directory, record_path, gateway_module=gateway_module, **options) as url:
            yield url


@contextlib.contextmanager
def _started_local_cortex_gateway(
    directory: Path,
    record_path: Path,
    *,
    gateway_module: str = "dss.ray_dss.gateway_server",
    allocation_members: dict | None = None,
    gateway_command_builder=None,
    gateway_command_source: dict | None = None,
) -> Iterator[str]:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    hostfile = _allocation_hostfile()
    if hostfile is None:
        hostfile = directory / "hostfile"
        hostfile.write_text(f"{socket.gethostname()} slots={torch.cuda.device_count()}\n")
    previous = os.environ.get("DSS_CORRECTNESS_TMPDIR")
    prepared = None
    if gateway_module == "arctic_platform.correctness.harness.gateway":
        options = {} if allocation_members is None else {"allocation_members": allocation_members}
        prepared = _prepare_gateway_scratch(directory, hostfile, **options)
        os.environ["DSS_CORRECTNESS_TMPDIR"] = prepared["path"]
    command_options = {}
    if gateway_command_builder is not None or gateway_command_source is not None:
        command_options.update(
            gateway_command_builder=gateway_command_builder,
            gateway_command_source=gateway_command_source,
        )
    try:
        with _launch_local_cortex_gateway(
            directory,
            record_path,
            hostfile,
            gateway_module=gateway_module,
            **command_options,
        ) as url:
            yield url
    finally:
        if prepared is not None:
            if previous is None:
                os.environ.pop("DSS_CORRECTNESS_TMPDIR", None)
            else:
                os.environ["DSS_CORRECTNESS_TMPDIR"] = previous


def _gateway_scratch_identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_uid]


def _gateway_scratch_entries(host_bytes: bytes) -> list:
    if not host_bytes or len(host_bytes) > 65536:
        raise ValueError("scratch hostfile snapshot must contain bounded allocation metadata")
    with tempfile.NamedTemporaryFile() as snapshot:
        snapshot.write(host_bytes)
        snapshot.flush()
        entries = parse_hostfile(snapshot.name)
    if not entries or len({entry.host for entry in entries}) != len(entries):
        raise ValueError("scratch hostfile snapshot requires unique allocation members")
    return entries


def _gateway_scratch_members(host_bytes: bytes, members: dict) -> dict:
    """Validate explicit node identities against the exact head hostfile snapshot."""
    entries = _gateway_scratch_entries(host_bytes)
    if not isinstance(members, dict) or members.keys() != {entry.host for entry in entries}:
        raise ValueError("scratch allocation identities must cover exactly the unique hostfile members")
    for entry in entries:
        member = members[entry.host]
        if (
            not isinstance(member, dict)
            or set(member) != {"host", "address", "boot_id", "slots"}
            or any(not isinstance(member[key], str) or not member[key] for key in ("host", "address", "boot_id"))
            or entry.host not in {member["host"], member["address"]}
            or type(member["slots"]) is not int
            or member["slots"] != entry.slots
        ):
            raise ValueError("scratch allocation member, address, boot or slots binding differs")
    if any(len({member[key] for member in members.values()}) != len(entries) for key in ("host", "address")):
        raise ValueError("scratch allocation identities contain duplicate nodes or addresses")
    return members


def _gateway_scratch_node(request: dict) -> dict:
    receipt = {
        "token": request["token"],
        "host": socket.gethostname(),
        "path": request["path"],
        "ok": False,
    }
    try:
        hostfile = Path(request["hostfile"])
        members = request.get("allocation_members")
        transported = "hostfile_base64" in request
        if members is None and not transported:
            host_bytes = hostfile.read_bytes()
            hosts = {row.host for row in parse_hostfile(hostfile)}
        else:
            # Allocation hostfiles are head-local; peers authenticate the transported snapshot.
            host_bytes = base64.b64decode(request["hostfile_base64"], validate=True)
            if members is None:
                hosts = {row.host for row in _gateway_scratch_entries(host_bytes)}
            else:
                members = _gateway_scratch_members(host_bytes, members)
                hosts = set(members)
        host_sha = hashlib.sha256(host_bytes).hexdigest()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        address = socket.gethostbyname(receipt["host"])
        receipt.update(
            boot_id=boot,
            address=address,
            hostfile_sha256=host_sha,
            python=str(Path(sys.executable).resolve()),
        )
        if (
            host_sha != request["hostfile_sha256"]
            or not hosts.intersection({receipt["host"], address})
            or str(Path(sys.executable).resolve()) != request["python"]
        ):
            raise ValueError("scratch allocation member, hostfile or interpreter differs")
        if members is not None:
            selected = [key for key in members if key in {receipt["host"], address}]
            if len(selected) != 1:
                raise ValueError("scratch allocation member selection is ambiguous or missing")
            expected = members[selected[0]]
            if (expected["host"], expected["address"], expected["boot_id"]) != (
                receipt["host"],
                address,
                boot,
            ):
                raise ValueError("scratch allocation node address or boot differs from supplied identity")
            receipt.update(member=selected[0], slots=expected["slots"])
        root = Path(request["path"])
        if not root.is_absolute() or root.resolve() != root or not root.is_relative_to("/data-fast"):
            raise ValueError("scratch parent must be a canonical FAST descendant")
        if root == Path("/data-fast"):
            raise ValueError("scratch parent cannot be the FAST root")
        identities = []
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        current = Path("/")
        created = False
        try:
            for part in root.parts[1:]:
                current /= part
                try:
                    next_descriptor = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                except FileNotFoundError:
                    if not current.is_relative_to("/data-fast") or current == Path("/data-fast"):
                        raise ValueError("existing FAST anchor required")
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    next_descriptor = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                    if current == root:
                        created = True
                os.close(descriptor)
                descriptor = next_descriptor
                identities.append((current, _gateway_scratch_identity(os.fstat(descriptor))))
            info = os.fstat(descriptor)
            if info.st_uid != os.geteuid():
                raise ValueError("scratch parent is not owned by the executing user")
            for path, identity in identities:
                if _gateway_scratch_identity(path.lstat()) != identity:
                    raise ValueError("scratch ancestor identity changed")
            if (
                (not transported and hashlib.sha256(hostfile.read_bytes()).hexdigest() != host_sha)
                or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != boot
                or socket.gethostname() != receipt["host"]
                or socket.gethostbyname(receipt["host"]) != address
            ):
                raise ValueError("scratch allocation identity changed")
            receipt.update(
                ok=True,
                created=created,
                identity=_gateway_scratch_identity(info),
                boot_id=boot,
                address=address,
                hostfile_sha256=host_sha,
                python=str(Path(sys.executable).resolve()),
            )
        finally:
            os.close(descriptor)
    except Exception:
        receipt["error"] = traceback.format_exc()
    return receipt


def _prepare_gateway_scratch(directory: Path, hostfile: Path, *, allocation_members: dict | None = None) -> dict:
    """Prepare the exact same-allocation scratch parent before starting a gateway."""
    token = uuid.uuid4().hex
    root = Path(os.environ.get("DSS_CORRECTNESS_TMPDIR") or f"/data-fast/tmp/correctness-gateway-{token}")
    if not root.is_absolute() or root.resolve() != root or not root.is_relative_to("/data-fast"):
        raise ValueError("gateway scratch parent must be a canonical FAST descendant")
    host_bytes = hostfile.read_bytes()
    entries = _gateway_scratch_entries(host_bytes)
    hosts = {row.host for row in entries}
    if not hosts or len(hosts) != len(entries):
        raise ValueError("gateway scratch preparation requires unique allocation members")
    host_identity = hostfile.stat()
    host_link_identity = hostfile.lstat()
    request = {
        "token": token,
        "path": str(root),
        "hostfile": str(hostfile.absolute()),
        "hostfile_sha256": hashlib.sha256(host_bytes).hexdigest(),
        "hostfile_base64": base64.b64encode(host_bytes).decode("ascii"),
        "python": str(Path(sys.executable).resolve()),
    }
    marker = "CORRECTNESS_GATEWAY_SCRATCH "
    raw_path = directory / f"scratch-preparation-{token}.transport.log"
    receipt_path = directory / f"scratch-preparation-{token}.json"
    result = {
        "path": str(root),
        "request": request,
        "nodes": {},
        "transport_log": str(raw_path),
        "ok": False,
    }
    primary = None
    try:
        if allocation_members is not None:
            members = _gateway_scratch_members(host_bytes, allocation_members)
            if members.keys() != hosts:
                raise ValueError("scratch hostfile changed while reading allocation members")
            request.update(
                allocation_members=copy.deepcopy(members),
            )
        source = Path(__file__).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        import_roots = list(dict.fromkeys([str(source.parents[3]), *sys.path]))
        script = (
            "import hashlib, json, sys\nfrom pathlib import Path\n"
            f"sys.path[:] = {import_roots!r}\n"
            f"source = Path({str(source)!r})\n"
            f"assert hashlib.sha256(source.read_bytes()).hexdigest() == {digest!r}, 'scratch source differs'\n"
            "from arctic_platform.correctness.harness import dss_driver as module\n"
            "assert Path(module.__file__).resolve() == source, 'scratch import differs'\n"
            f"assert hashlib.sha256(source.read_bytes()).hexdigest() == {digest!r}, 'scratch source changed'\n"
            f"receipt = module._gateway_scratch_node(json.loads({json.dumps(request)!r}))\n"
            f"print({marker!r} + json.dumps(receipt), flush=True)\n"
        )
        encoded = base64.b64encode(script.encode()).decode()
        with raw_path.open("xb") as stream:
            process = subprocess.Popen(
                [
                    _ds_ssh(),
                    "-f",
                    str(hostfile),
                    f"echo {encoded} | base64 -d | {shlex.quote(sys.executable)} -",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                result["transport_exit"] = process.wait(timeout=120)
            except BaseException as exc:
                try:
                    _reap_gateway_scratch_transport(process)
                    result["transport_group_empty"] = True
                except Exception as cleanup:
                    result["transport_cleanup_error"] = traceback.format_exc()
                    exc.add_note(f"Scratch transport cleanup remains unverified: {cleanup}")
                raise
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                result["transport_group_empty"] = True
            else:
                _reap_gateway_scratch_transport(process)
                raise ValueError("scratch transport left descendants after its direct child exited")
        with raw_path.open("rb") as stream:
            consumed = 0
            while line := stream.readline(65537):
                consumed += len(line)
                if len(line) > 65536 or consumed > 1024**2:
                    raise ValueError("scratch transport metadata exceeds bounded reader limit; full log retained")
                prefix, separator, raw = line.decode().partition(marker)
                if not separator:
                    continue
                host = prefix.strip().removesuffix(":").strip()
                if host not in hosts or host in result["nodes"]:
                    raise ValueError("unexpected or duplicate scratch allocation receipt")
                row = json.loads(raw)
                result["nodes"][host] = row
                if (
                    not isinstance(row, dict)
                    or row.get("token") != token
                    or row.get("path") != str(root)
                    or row.get("ok") is not True
                    or row.get("hostfile_sha256") != request["hostfile_sha256"]
                    or row.get("python") != request["python"]
                    or not row.get("boot_id")
                    or not isinstance(row.get("identity"), list)
                    or len(row["identity"]) != 4
                    or any(type(value) is not int for value in row["identity"])
                    or row["identity"][3] != os.geteuid()
                    or not stat.S_ISDIR(row["identity"][2])
                    or not isinstance(row.get("host"), str)
                    or not row["host"]
                    or host not in {row["host"], row.get("address")}
                ):
                    raise ValueError(f"scratch allocation preparation rejected {host}: {row}")
                if allocation_members is not None:
                    expected = request["allocation_members"][host]
                    if (
                        row.get("member") != host
                        or (row["host"], row.get("address"), row["boot_id"])
                        != (expected["host"], expected["address"], expected["boot_id"])
                        or type(row.get("slots")) is not int
                        or row["slots"] != expected["slots"]
                    ):
                        raise ValueError(f"scratch allocation receipt identity differs for {host}: {row}")
        if (
            result["transport_exit"]
            or result["nodes"].keys() != hosts
            or len({row["host"] for row in result["nodes"].values()}) != len(hosts)
            or hostfile.read_bytes() != host_bytes
            or any(
                (
                    _gateway_scratch_identity(actual),
                    actual.st_size,
                    actual.st_mtime_ns,
                    actual.st_ctime_ns,
                )
                != (
                    _gateway_scratch_identity(expected),
                    expected.st_size,
                    expected.st_mtime_ns,
                    expected.st_ctime_ns,
                )
                for actual, expected in (
                    (hostfile.stat(), host_identity),
                    (hostfile.lstat(), host_link_identity),
                )
            )
        ):
            raise ValueError("scratch allocation transport, members or hostfile identity differs")
        result["ok"] = True
    except BaseException as exc:
        result["error"] = traceback.format_exc()
        exc.add_note(f"Scratch preparation evidence: {receipt_path}; full transport: {raw_path}")
        primary = exc
        raise
    finally:
        try:
            with receipt_path.open("x") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
        except Exception as exc:
            if primary is None:
                raise
            primary.add_note(f"Could not publish scratch preparation receipt: {exc}")
    return result


def _reap_gateway_scratch_transport(process) -> None:
    """Reap only the transport session created by this scratch preparation."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            process.wait(timeout=10)
            return
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                process.wait(timeout=10)
                return
            time.sleep(0.05)
    process.wait(timeout=10)
    raise RuntimeError("scratch transport process group did not quiesce within its cleanup deadline")


def _owned_gateway_descendants(gateway: dict, token: str) -> list[dict]:
    """Find token-bound children, including workers that detached into another session."""
    leader = _gateway_process_identity(gateway["pid"])
    descendants = []
    marker = f"DSS_CORRECTNESS_GATEWAY_OWNER={token}".encode()
    for directory in _GATEWAY_PROC_ROOT.iterdir():
        if not directory.name.isdecimal() or int(directory.name) == gateway["pid"]:
            continue
        original_session = False
        try:
            fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
            if fields[0] in {"Z", "X"}:
                continue
            original_session = int(fields[3]) == gateway["pid"]
            if not original_session and directory.stat().st_uid != gateway["uid"]:
                continue
            identity = _gateway_process_identity(int(directory.name))
            if identity is None:
                continue
            environment = (directory / "environ").read_bytes().split(b"\0")
            if not original_session and marker not in environment:
                continue
            if (
                (original_session and int(fields[2]) != gateway["pid"])
                or identity["uid"] != gateway["uid"]
                or int(identity["start_time"]) < int(gateway["start_time"])
                or identity["start_time"] != fields[19]
                or marker not in environment
            ):
                if _gateway_process_identity(int(directory.name)) is None:
                    continue
                raise UnsafeCleanupError(
                    f"process {directory.name} in the owned gateway session failed ownership validation"
                )
            descendants.append(identity)
        except FileNotFoundError:
            continue
        except PermissionError:
            if not original_session:
                continue
            # Linux may revoke environ access between a killed task's stat reads.
            if _gateway_process_identity(int(directory.name)) is None:
                continue
            raise
    if descendants and leader is not None and leader != gateway:
        raise UnsafeCleanupError("owned gateway PID was reused; refusing descendant cleanup")
    return descendants


def _open_gateway_pidfd(pid: int) -> int:
    """Pin a Linux process even on Python builds made against older libc headers."""
    if sys.platform != "linux":
        raise UnsafeCleanupError("owned gateway descendant cleanup requires Linux pidfds")
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        open_pidfd = libc.pidfd_open
    except AttributeError as exc:
        raise UnsafeCleanupError("Linux libc does not expose pidfd_open for safe descendant cleanup") from exc
    open_pidfd.argtypes = [ctypes.c_int, ctypes.c_uint]
    open_pidfd.restype = ctypes.c_int
    descriptor = open_pidfd(pid, 0)
    if descriptor < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return descriptor


def _send_gateway_pidfd_signal(descriptor: int, signum: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(descriptor, signum)
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        send = libc.pidfd_send_signal
    except AttributeError as exc:
        raise UnsafeCleanupError("Linux libc does not expose pidfd_send_signal for safe cleanup") from exc
    send.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    send.restype = ctypes.c_int
    if send(descriptor, signum, None, 0) < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _reap_owned_gateway_descendants(gateway: dict, token: str, *, timeout: float = 10) -> None:
    """Reap token-bound local descendants even after the gateway itself has exited."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + timeout
        while True:
            try:
                descendants = _owned_gateway_descendants(gateway, token)
                break
            except UnsafeCleanupError:
                # Exiting Ray agents can clear their environment before becoming zombies.
                # Wait for verified absence; never signal a process that failed ownership.
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        if not descendants:
            return
        print(
            f"Stopping owned gateway descendants {[entry['pid'] for entry in descendants]}"
            f" with {signal.Signals(signum).name}.",
            flush=True,
        )
        for identity in descendants:
            try:
                descriptor = _open_gateway_pidfd(identity["pid"])
            except ProcessLookupError:
                continue
            try:
                # A pidfd pins the recipient; validate its birth again after opening it.
                if _gateway_process_identity(identity["pid"]) != identity:
                    raise UnsafeCleanupError("gateway descendant identity changed before signalling")
                _send_gateway_pidfd_signal(descriptor, signum)
            except ProcessLookupError:
                pass
            finally:
                os.close(descriptor)
        deadline = time.monotonic() + timeout
        # Once pinned and signalled, wait on births without rereading an exiting task's environ.
        while any(_gateway_process_identity(entry["pid"]) == entry for entry in descendants):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    if _owned_gateway_descendants(gateway, token):
        raise UnsafeCleanupError("owned gateway descendants survived bounded teardown")


@contextlib.contextmanager
def _gateway_owner_guard(directory: Path, record_path: Path, token: str) -> Iterator[None]:
    """A pipe lifetime guard survives owner SIGKILL without monitoring unrelated services."""
    reader, writer = os.pipe()
    guard = None
    log_path = directory / "gateway-cleanup.log"
    try:
        with log_path.open("w") as log:
            guard = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("gateway_cleanup.py")),
                    str(reader),
                    str(record_path.resolve()),
                    token,
                ],
                pass_fds=(reader,),
                stdout=subprocess.PIPE,
                stderr=log,
                start_new_session=True,
            )
            os.close(reader)
            reader = None
            if not select.select([guard.stdout], [], [], 30)[0] or guard.stdout.readline() != b"ready\n":
                raise UnsafeCleanupError(f"owned gateway cleanup guard did not start; inspect {log_path}")
            try:
                yield
            finally:
                os.write(writer, b"done\n")
                os.close(writer)
                writer = None
                if guard.wait(timeout=30) != 0:
                    raise UnsafeCleanupError(f"owned gateway cleanup guard failed; inspect {log_path}")
    finally:
        if reader is not None:
            os.close(reader)
        if writer is not None:
            os.close(writer)
        if guard is not None:
            if guard.poll() is None:
                guard.kill()
                guard.wait()
            guard.stdout.close()


@contextlib.contextmanager
def _launch_local_cortex_gateway(
    directory: Path,
    record_path: Path,
    hostfile: Path,
    *,
    gateway_module: str,
    gateway_command_builder=None,
    gateway_command_source: dict | None = None,
) -> Iterator[str]:
    import requests

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = directory / "gateway.log"
    print(
        f"\n*** Target: Starting Cortex Training Gateway ***\nGateway log: {log_path}\n",
        flush=True,
    )
    previous_handlers = {}
    token = uuid.uuid4().hex

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}; stopping owned Cortex-training gateway")

    with _gateway_owner_guard(directory, record_path, token), log_path.open("w") as log:
        process = None
        gateway = None
        try:
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, interrupt)
            command = [
                sys.executable,
                "-m",
                gateway_module,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--backend",
                "gpu",
                "--hostfile",
                str(hostfile),
            ]
            if gateway_command_builder is not None or gateway_command_source is not None:
                from .cortex_optimizer import verify_native_callback

                if gateway_module != "dss.ray_dss.gateway_server":
                    raise ValueError("native gateway command builder requires the upstream DSS gateway")
                verify_native_callback(gateway_command_builder, gateway_command_source)
                options = command[3:]
                command = gateway_command_builder(tuple(command))
                verify_native_callback(gateway_command_builder, gateway_command_source)
                if (
                    type(command) not in (list, tuple)
                    or len(command) <= len(options)
                    or any(type(value) is not str or not value or "\0" in value for value in command)
                    or command[0] != sys.executable
                    or list(command[-len(options) :]) != options
                ):
                    raise ValueError("native gateway command builder must preserve interpreter and runtime options")
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                # Ray retitles its agents; keep the ownership marker readable in /proc/environ.
                env={
                    **os.environ,
                    "DSS_CORRECTNESS_GATEWAY_OWNER": token,
                    "SPT_NOENV": "1",
                },
            )
            owner = _gateway_process_identity(os.getpid())
            gateway = _gateway_process_identity(process.pid)
            if owner is None or gateway is None:
                raise RuntimeError(f"cannot record live Cortex-training gateway ownership; see {log_path}")
            record = {
                "schema": "correctness-owned-gateway-v1",
                "owner": owner,
                "gateway": gateway,
                "token": token,
                "log": str(log_path.resolve()),
            }
            temporary = record_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record) + "\n")
            temporary.replace(record_path)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Cortex-training gateway exited {process.returncode}; see {log_path}")
                try:
                    response = requests.get(
                        f"{url}/health",
                        timeout=1,
                        allow_redirects=False,
                        proxies={"http": "", "https": "", "all": ""},
                    )
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError(f"Cortex-training gateway did not become ready; see {log_path}")
            yield url
        finally:
            # A repeated stop signal must not interrupt the owned child's bounded teardown.
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
            try:
                try:
                    if process is not None:
                        print(
                            f"Stopping owned Cortex-training gateway (PID {process.pid}); log: {log_path}",
                            flush=True,
                        )
                        if process.poll() is None:
                            process.terminate()
                            shutdown_error = None
                            try:
                                process.wait(timeout=120)
                            except subprocess.TimeoutExpired as exc:
                                try:
                                    os.killpg(process.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    # The group can exit between the timeout and signal; still reap the owned child.
                                    pass
                                process.wait()
                                shutdown_error = UnsafeCleanupError(
                                    "owned Cortex-training gateway did not shut down cleanly; inspect"
                                    f" {log_path} before retrying"
                                )
                                shutdown_error.__cause__ = exc
                        else:
                            process.wait()
                            shutdown_error = None
                        if gateway is not None:
                            _reap_owned_gateway_descendants(gateway, token)
                        if shutdown_error is not None:
                            raise shutdown_error
                        record_path.unlink(missing_ok=True)
                        print(
                            "Owned Cortex-training gateway stopped and reaped.",
                            flush=True,
                        )
                except UnsafeCleanupError:
                    raise
                except Exception as exc:
                    raise UnsafeCleanupError(
                        f"owned Cortex-training gateway cleanup failed; inspect {log_path} before retrying"
                    ) from exc
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)


def _run_on_allocation(script: str) -> subprocess.CompletedProcess[str]:
    hostfile = _allocation_hostfile()
    if hostfile is None:
        command = ["bash", "-lc", script]
    else:
        ds_ssh = Path(sys.executable).resolve().parent / "ds_ssh"
        command = [str(ds_ssh), "-f", str(hostfile), script]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if completed.returncode != 0:
        raise RuntimeError(
            f"allocation runtime command failed ({completed.returncode}): {completed.stdout}{completed.stderr}"
        )
    return completed


def _verify_gateway_runtime_idle() -> None:
    """Refuse occupied resources without stopping any allocation processes."""
    port_probe = (
        "import socket; "
        "s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
        f"s.bind(('', {_STANDARD_MASTER_PORT})); s.close()"
    )
    _run_on_allocation(
        f"pgrep -af {shlex.quote(_RAY_PROCESS_PATTERN)}\n"
        'status=$?\nif [ "$status" -ne 1 ]; then exit 1; fi\n'
        f"{shlex.quote(sys.executable)} -c {shlex.quote(port_probe)}"
    )


def _load_gateway_ray_cluster() -> Any:
    from arctic_platform.common import ray_cluster

    return ray_cluster


@contextlib.contextmanager
def _gateway_runtime(slots: int) -> Iterator[None]:
    """Give one correctness job a clean Ray tree sized to its requested GPU topology."""
    import torch

    include_peers = slots > torch.cuda.device_count()
    previous_master_port = os.environ.get("MASTER_PORT")
    os.environ["MASTER_PORT"] = str(_STANDARD_MASTER_PORT)
    ray_cluster = None
    previous_hostfile = None
    start_attempted = False
    try:
        _verify_gateway_runtime_idle()
        ray_cluster = _load_gateway_ray_cluster()
        previous_hostfile = ray_cluster._HOSTFILE
        if not include_peers:
            ray_cluster._HOSTFILE = os.devnull
        start_attempted = True
        ray_cluster.init_ray_cluster(auto_attach=False)
        yield
    finally:
        try:
            if start_attempted:
                try:
                    ray_cluster._shutdown()
                finally:
                    _verify_gateway_runtime_idle()
        except Exception as exc:
            raise UnsafeCleanupError(
                "gateway runtime cleanup failed; inspect owned resources before retrying"
            ) from exc
        finally:
            if ray_cluster is not None and previous_hostfile is not None:
                ray_cluster._HOSTFILE = previous_hostfile
            if previous_master_port is None:
                os.environ.pop("MASTER_PORT", None)
            else:
                os.environ["MASTER_PORT"] = previous_master_port


@contextlib.contextmanager
def gateway(
    tmp_dir: Path, slots: int, *, backend: OnPremConfig | CortexConfig | None = None
) -> Iterator[GatewaySession]:
    workdir = require_data_fast_path(tmp_dir, label="gateway workdir")
    workdir.mkdir(parents=True, exist_ok=True)
    yield GatewaySession(
        workdir=workdir,
        slots=slots,
        backend=backend if backend is not None else OnPremConfig(),
    )


def _ds_optimizer(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    value = deepcopy(value)
    name = value.pop("name", value.pop("type", "AdamW"))
    return {"type": name, "params": value}


def _training_client_config(session: GatewaySession, payload: dict) -> ArcticClientConfig:
    if "local_model_dir" in payload and not isinstance(getattr(session, "backend", None), CortexConfig):
        raise ValueError("initial training model directory requires local Cortex")
    tc = deepcopy(payload["training_config"])
    n_gpus = int(tc.pop("n_gpus"))
    max_seq_len = int(tc.pop("max_seq_len", 8192))
    ds = deepcopy(tc.pop("ds_config", {}))
    train_batch_size = tc.pop("train_batch_size", None)
    gradient_clipping = tc.pop("gradient_clipping", None)
    optimizer = tc.pop("optimizer", None)
    peft = tc.pop("peft_config", None)
    if train_batch_size is not None:
        ds.setdefault("train_batch_size", train_batch_size)
    if gradient_clipping is not None:
        ds.setdefault("gradient_clipping", gradient_clipping)
    if optimizer is not None:
        ds.setdefault("optimizer", _ds_optimizer(optimizer))
    activation_checkpointing = tc.pop("activation_checkpointing", None)
    if activation_checkpointing is not None:
        tc.setdefault("enable_gradient_checkpointing", bool(activation_checkpointing))
    debug = tc.get("debug") if isinstance(tc.get("debug"), dict) else {}
    backend = getattr(session, "backend", OnPremConfig())
    checkpoint_path = None
    if not isinstance(backend, CortexConfig):
        workdir = require_data_fast_path(session.workdir, label="training workdir")
        checkpoint_path = str(require_data_fast_path(workdir / "checkpoints", label="training checkpoint output"))
    config_type = CortexTrainingConfig if isinstance(backend, CortexConfig) else ReviewedClientConfig
    return config_type(
        model_name=payload["model_name"],
        seed=payload.get("seed"),
        dtype=payload.get("dtype"),
        max_seq_len=max_seq_len,
        training_gpus=n_gpus,
        training=ReviewedTrainingConfig(
            weight_format=payload.get("weight_format"),
            full_determinism=bool(debug.get("full_determinism", False)),
            checkpoint_path=checkpoint_path,
            ds_config=ds,
            ds_worker_config=tc,
            peft=peft,
        ),
        backend=backend,
        **({"hardware": payload.get("hardware")} if isinstance(backend, CortexConfig) else {}),
        **(
            {"initial_training_model_dir": payload["local_model_dir"]}
            if "local_model_dir" in payload and isinstance(backend, CortexConfig)
            else {}
        ),
    )


def _inference_client_config(
    payload: dict, *, backend: OnPremConfig | CortexConfig | None = None
) -> ArcticClientConfig:
    inf = deepcopy(payload.get("inference_config") or {})
    n_gpus = int(inf.pop("n_gpus"))
    max_seq_len = int(inf.pop("max_seq_len", 8192))
    vllm = deepcopy(inf.pop("vllm_config", {}))
    vllm.setdefault("max_model_len", max_seq_len)
    job_type = payload.get("job_type")
    config = ReviewedClientConfig(
        model_name=payload["model_name"],
        seed=payload.get("seed"),
        dtype=payload.get("dtype"),
        max_seq_len=max_seq_len,
        sampling_gpus=n_gpus if job_type == "sampling" else 0,
        log_prob_gpus=n_gpus if job_type == "log_prob" else 0,
        sampling=ReviewedSamplingConfig(
            weight_format=payload.get("weight_format"),
            vllm=vllm,
            arctic_inference_config=inf or None,
        ),
        backend=backend if backend is not None else OnPremConfig(),
    )
    return CortexTrainingConfig.from_reviewed(config) if isinstance(config.backend, CortexConfig) else config


@contextlib.contextmanager
def running_job(session: GatewaySession, payload: dict) -> Iterator[ArcticJob]:
    job_type = str(payload.get("job_type", "training"))
    config = (
        _training_client_config(session, payload)
        if job_type == "training"
        else _inference_client_config(payload, backend=session.backend)
    )
    if isinstance(session.backend, OnPremConfig):
        config.to_onprem(job_type)
    runtime = (
        _gateway_runtime(session.slots) if isinstance(session.backend, OnPremConfig) else contextlib.nullcontext()
    )
    shared_adapter = (payload.get("training_config", {}).get("debug") or {}).get("shared_peft_adapter")
    overlay = contextlib.nullcontext()
    if shared_adapter is not None:
        if not isinstance(session.backend, OnPremConfig):
            raise ValueError("supplied PEFT adapters require the on-prem correctness worker")
        from .optimizer_capture import optimizer_capture_worker

        overlay = optimizer_capture_worker()
    with runtime, overlay:
        if shared_adapter is not None:
            import ray

            from .shared_lora_adapter import adapter_files

            files = adapter_files(shared_adapter["path"], shared_adapter["identity"])
            config.training.ds_worker_config["debug"]["shared_peft_adapter"] = {
                "object_ref": ray.put(files),
                "identity": shared_adapter["identity"],
            }
        client = ArcticClient(config)
        try:
            yield ArcticJob(client=client, job_type=job_type)
        finally:
            try:
                client.shutdown()
            except Exception as exc:
                raise UnsafeCleanupError("client shutdown failed; inspect owned resources before retrying") from exc


def cortex_training_body(body: dict) -> dict:
    """Lower the correctness SFT batch into the Cortex zone's tensor RPC contract."""
    import torch

    if "batch" not in body:
        return body
    batches = body["batch"]
    if isinstance(batches, list):
        if not batches or any(set(batch) != set(batches[0]) for batch in batches):
            raise ValueError("Cortex SFT microbatches must be non-empty and carry identical tensor keys")
        # Cortex performs its own packing and GAS split from mb_spec and ds_config.
        tensors = {key: torch.cat([batch[key] for batch in batches], dim=0) for key in batches[0]}
    else:
        tensors = dict(batches)
    processing = deepcopy(body.get("processing") or {})
    if processing.get("loss_fn") == "sft_ce":
        raise ValueError("Cortex SFT does not support request-level memory CE options; use --backend onprem")
    if processing.get("loss_fn") == "sft":
        if set(processing) != {"loss_fn"}:
            raise ValueError("Cortex SFT does not support request-level processing options; use --backend onprem")
        processing = {}
    lowered = {
        "args": (),
        "kwargs": tensors,
        "context": dict(body.get("meta") or {}),
    }
    # No processing dictionary selects supervised dispatch; a dictionary selects the RL pipeline.
    if processing:
        lowered["processing"] = processing
    return lowered


def training_body(job: ArcticJob, body: dict) -> dict:
    if isinstance(getattr(getattr(job.client, "config", None), "backend", None), CortexConfig):
        return cortex_training_body(body)
    return body


def copy_checkpoint_peft_adapter(checkpoint_root: Path, destination: Path) -> Path:
    """Copy the PEFT pack from a weights-only checkpoint when the exporter retained one."""
    destination = require_data_fast_path(destination, label="PEFT adapter output")
    from safetensors import safe_open

    candidates = [
        checkpoint_root / "hf",
        checkpoint_root / "default",
        *sorted(checkpoint_root.glob("global_step*/default"), reverse=True),
    ]
    source = next(
        (path for path in candidates if (path / "adapter_model.safetensors").is_file()),
        None,
    )
    if source is None or not (source / "adapter_config.json").is_file():
        raise RuntimeError(f"initial adapter checkpoint produced no PEFT pack under {checkpoint_root}")
    if not json.loads((source / "adapter_config.json").read_text()).get("peft_type"):
        raise RuntimeError(f"initial adapter checkpoint produced no PEFT type under {source}")
    with safe_open(source / "adapter_model.safetensors", framework="pt") as adapter:
        if not adapter.keys():
            raise RuntimeError(f"initial adapter checkpoint produced no adapter tensors under {source}")
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination)
    return destination


def save_weights_only_checkpoint(session: GatewaySession, job: ArcticJob, path: str | Path) -> str:
    path = require_data_fast_path(path, label="weights checkpoint output")
    saved = job.client.save_checkpoint(path=str(path), export_hf=True)
    directory = Path(saved.get("hf_path") or saved["path"])
    if not (directory / "config.json").exists():
        raise RuntimeError(f"weights-only save reported {directory}, which holds no model config")
    return str(directory)


def pin_suite_determinism(debug: dict) -> dict:
    debug["full_determinism"] = True
    debug["full_determinism_must_comply"] = False
    return debug


def _max_tokens_per_mb_of(training_config: dict) -> int:
    from arctic_platform.rl.processors.microbatch import DEFAULT_MAX_TOKENS_PER_MB

    return int((training_config.get("mb_spec") or {}).get("max_tokens_per_mb") or DEFAULT_MAX_TOKENS_PER_MB)


def build_payload(
    training_config: dict,
    model_path: str,
    seed: int,
    *,
    attn_implementation: Optional[str] = None,
    optimizer_state_output_dir: Optional[Path] = None,
    lm_head_token_chunk_size: Optional[int] = None,
    determinism: str = "best-effort",
    gradient_norms_per_param: bool = True,
    weight_format: Optional[str] = None,
) -> dict:
    tc = json.loads(json.dumps(training_config))
    if lm_head_token_chunk_size is not None:
        tc.setdefault("fused_lm_head_token_chunk_size", lm_head_token_chunk_size)
    if attn_implementation is not None:
        tc["attn_implementation"] = attn_implementation
    mb_spec = dict(tc.get("mb_spec") or {})
    mb_spec["max_tokens_per_mb"] = correctness_microbatch_tokens(_max_tokens_per_mb_of(tc))
    tc["mb_spec"] = mb_spec
    debug = dict(tc.get("debug") or {})
    debug.update(
        {
            "gradient_norms_per_param": gradient_norms_per_param,
            "gradient_sample_max_numel": 0,
        }
    )
    if determinism == "off":
        debug["full_determinism"] = False
        debug["full_determinism_must_comply"] = False
    elif determinism == "best-effort":
        pin_suite_determinism(debug)
    else:
        raise ValueError(f"determinism must be 'best-effort' or 'off', got {determinism!r}")
    debug.setdefault("fp32_precision", False)
    if optimizer_state_output_dir is not None:
        debug["optimizer_state_output_dir"] = str(
            require_data_fast_path(optimizer_state_output_dir, label="optimizer artifact output")
        )
    tc["debug"] = debug
    payload = {
        "model_name": model_path,
        "job_type": "training",
        "seed": seed,
        "dtype": tc.get("dtype", "bfloat16"),
        "training_config": tc,
    }
    if weight_format is not None:
        payload["weight_format"] = weight_format
    return payload


def _labels_for_provider(batch, model_provider: str):
    return batch.shifted_labels() if model_provider == "prime_rl" else batch.labels


def _attention_mask_from_position_ids(position_ids):
    import torch

    offsets = torch.arange(position_ids.shape[1], device=position_ids.device).unsqueeze(0)
    expected_positions = position_ids[:, :1] + offsets
    return position_ids.eq(expected_positions).to(torch.long).cumprod(dim=1)


def _batch_dict(input_ids, position_ids, labels) -> dict:
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": _attention_mask_from_position_ids(position_ids),
        "labels": labels,
    }


def _batch_body(
    batch,
    labels,
    *,
    labels_are_shifted: bool = False,
) -> dict:
    return {
        "batch": _batch_dict(batch.input_ids, batch.position_ids, labels),
        "meta": {"labels_are_shifted": True} if labels_are_shifted else {},
        "processing": {"loss_fn": "sft"},
    }


def pack(
    batch,
    *,
    model_provider: str = "huggingface",
) -> dict:
    labels_are_shifted = model_provider == "prime_rl"
    return _batch_body(
        batch,
        _labels_for_provider(batch, model_provider),
        labels_are_shifted=labels_are_shifted,
    )


def pack_logit_aligned(batch) -> dict:
    body = _batch_body(batch, batch.shifted_labels(), labels_are_shifted=True)
    body["processing"]["return_logprobs"] = True
    return body


def _scalar(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def _rank_owned_mapping(value: Any) -> Dict[str, Any]:
    """Return the rank-owned mapping from a step-response field.

    ``merge_dict_shards`` appends each rank's non-list value, so a mapping that only rank zero returns arrives as a
    one-element list when that rank is not the first worker. A mapping is returned unchanged. An empty list or any
    other shape contributes no entries.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item:
                return item
    return {}


def _rank_owned_path(value: Any) -> Optional[str]:
    """Return a rank-owned path metric after Ray response merging."""
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            path = _rank_owned_path(item)
            if path is not None:
                return path
        return None
    raise TypeError(f"rank-owned path metric must be a string or path-like object, got {type(value).__name__}")


def _globalize_expert_norms(per_parameter: Dict[str, float], per_expert: Dict[str, list[float]]) -> Dict[str, float]:
    result = {key: float(value) for key, value in per_parameter.items()}
    for name, values in per_expert.items():
        result[name] = math.sqrt(sum(float(value) ** 2 for value in values))
    return result


def fwd_bwd_step(session: GatewaySession, job: ArcticJob, body: dict, learning_rate: float = 0.0) -> StepResult:
    response = job.client.fwd_bwd(training_body(job, body))
    stepped = job.client.step(learning_rate)
    step_metrics = stepped.get("metrics") or {}
    response_metrics = response.get("metrics") or {}
    norms = _globalize_expert_norms(
        _rank_owned_mapping(
            stepped.get("gradient_norms_per_param") or step_metrics.get("gradient_norms_per_param") or {}
        ),
        _rank_owned_mapping(
            stepped.get("gradient_norms_per_expert") or step_metrics.get("gradient_norms_per_expert") or {}
        ),
    )
    loss = response.get("avg_loss", response_metrics.get("loss"))
    if loss is None:
        raise RuntimeError(f"forward-backward response carries no loss: {sorted(response)}")
    calls = response_metrics.get("sft_model_calls")
    rows = response_metrics.get("sft_packed_rows")
    return StepResult(
        avg_loss=_scalar(loss),
        grad_norms=norms,
        model_calls=int(calls) if calls is not None else None,
        packed_rows=int(rows) if rows is not None else None,
        optimizer_state_manifest=_rank_owned_path(
            stepped.get("optimizer_state_manifest") or step_metrics.get("optimizer_state_manifest")
        ),
        participation=stepped.get("participation") or step_metrics.get("participation"),
    )


def sampling_payload(model_dir: str | Path, seed: int, *, dtype: str, n_gpus: int, max_seq_len: int) -> dict:
    return {
        "model_name": str(model_dir),
        "job_type": "log_prob",
        "dtype": dtype,
        "seed": seed,
        "inference_config": {
            "n_gpus": n_gpus,
            "max_seq_len": max_seq_len,
            "vllm_config": {
                "tensor_parallel_size": n_gpus,
                "max_model_len": max_seq_len,
                "dtype": dtype,
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.85,
                "disable_custom_all_reduce": True,
                "enforce_eager": True,
            },
        },
    }


def log_probs(
    session: GatewaySession,
    job: ArcticJob,
    prompts: List[str],
    completions: List[str],
    *,
    top_k: int,
) -> List[dict]:
    response = job.client._call(log_probs_request(job.client.jobs, prompts, completions, top_k))
    results = response["results"]
    if len(results) != len(prompts):
        raise RuntimeError(f"log-probs returned {len(results)} results for {len(prompts)} prompts")
    return list(results)


def forward_loss(session: GatewaySession, job: ArcticJob, body: dict) -> float:
    """Return the loss from Arctic Platform's forward-only response."""
    response = job.client.fwd_no_grad(training_body(job, body))
    metrics = response.get("metrics") or {}
    loss = response.get("avg_loss", metrics.get("loss"))
    if loss is None:
        raise RuntimeError(f"forward response carries no loss: {sorted(response)}")
    return _scalar(loss)


def forward_logprobs(session: GatewaySession, job: ArcticJob, body: dict):
    import torch

    response = job.client.fwd_no_grad(training_body(job, body))
    if "logprobs" not in response:
        metrics = response.get("metrics") or {}
        if "logprobs" in metrics:
            response = metrics
        else:
            raise RuntimeError(f"forward response carries no logprobs: {sorted(response)}")
    return torch.as_tensor(response["logprobs"], dtype=torch.float32)


@dataclass
class GatewayTrainingJob:
    session: GatewaySession
    job: ArcticJob

    def fwd_bwd_step(self, body: dict, *, learning_rate: float) -> float:
        return fwd_bwd_step(self.session, self.job, body, learning_rate=learning_rate).avg_loss

    def save_resumable(self) -> Checkpoint:
        saved = self.job.client.save_checkpoint()
        path = str(saved["path"])
        return Checkpoint(reference=path, source_job=path)


class GatewayTransport:
    def __init__(
        self,
        workdir: Path,
        training_config: dict,
        model_path: str,
        seed: int,
        *,
        slots: int,
        attn_implementation: Optional[str] = None,
        determinism: str = "best-effort",
    ) -> None:
        self.session = GatewaySession(require_data_fast_path(workdir, label="gateway transport workdir"), slots)
        self.payload = build_payload(
            training_config,
            model_path,
            seed,
            attn_implementation=attn_implementation,
            determinism=determinism,
        )

    def __enter__(self) -> "GatewayTransport":
        require_data_fast_path(self.session.workdir, label="gateway transport workdir").mkdir(
            parents=True, exist_ok=True
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    @contextlib.contextmanager
    def job(self, *, resume_from: Optional[Checkpoint] = None) -> Iterator[GatewayTrainingJob]:
        with running_job(self.session, self.payload) as job:
            if resume_from is not None:
                job.client.load_checkpoint(path=resume_from.reference)
            yield GatewayTrainingJob(self.session, job)


def sub_job_of_type(cfg, job_type: str) -> dict:
    matches = [sub for sub in cfg.sub_jobs if sub.get("job_type") == job_type]
    if len(matches) != 1:
        raise ValueError(f"{cfg.config_id}: expected one {job_type} sub-job, found {len(matches)}")
    return matches[0]


def rl_sampling_payload(cfg, model_path: str) -> dict:
    """The config's sampling sub-job as written, serving the explicitly selected model."""
    payload = copy.deepcopy(sub_job_of_type(cfg, "sampling"))
    payload["model_name"] = str(model_path)
    return payload


def rl_gpu_count(cfg) -> int:
    """GPUs the two sub-jobs need at once: each takes its own devices, so the count is the sum."""
    sampling = int(sub_job_of_type(cfg, "sampling")["inference_config"]["n_gpus"])
    return int(cfg.n_gpus) + sampling


@contextlib.contextmanager
def running_rl_client(
    cfg,
    training_payload: dict,
    model_path: str,
    workdir: Path,
    *,
    backend: OnPremConfig | CortexConfig | None = None,
    weight_sync_verification: bool = False,
    router_replay_verification: bool = False,
) -> Iterator[ArcticClient]:
    """Create one native client whose training and sampling jobs share the reviewed config."""
    if cfg.native is None:
        raise ValueError(f"{cfg.config_id}: native ArcticClientConfig is required")
    session = GatewaySession(
        Path(workdir),
        rl_gpu_count(cfg),
        backend if backend is not None else OnPremConfig(),
    )
    payload = dict(training_payload)
    payload.setdefault("weight_format", getattr(cfg.native.training, "weight_format", None))
    training_only = _training_client_config(session, payload)
    updates = {
        "model_name": str(model_path),
        "training": training_only.training,
        "backend": training_only.backend,
    }
    from arctic_platform.debug.broadcast_weight_sync_evidence import maybe_configure_broadcast_sampling
    from arctic_platform.debug.router_replay_trace import maybe_configure_router_replay_sampling

    updates["sampling"] = maybe_configure_broadcast_sampling(cfg.native.sampling, verify=weight_sync_verification)
    updates["sampling"] = maybe_configure_router_replay_sampling(
        updates["sampling"], verify=router_replay_verification
    )
    if isinstance(session.backend, CortexConfig):
        native = CortexTrainingConfig.from_reviewed(cfg.native, **updates)
    else:
        native = ReviewedClientConfig.model_validate(
            {
                **cfg.native.model_dump(),
                **updates,
                "training": training_only.training.model_dump(),
            }
        )
        native.to_onprem("training")
        native.to_onprem("sampling")
    client = RLClient(native)
    try:
        training_replay = (native.training.ds_worker_config or {}).get("router_replay") or {}
        sampling_replay = (native.sampling.arctic_inference_config or {}).get("router_replay") or {}
        if training_replay.get("enabled") and sampling_replay.get("enabled"):
            client.bootstrap_router_replay(max_cache_bytes=training_replay.get("max_cache_bytes"))
        yield client
    finally:
        try:
            client.shutdown()
        except Exception as exc:
            raise UnsafeCleanupError("RL client shutdown failed; inspect owned resources before retrying") from exc


def generate(client: RLClient, prompts: Sequence[Sequence[int]], sampling_params: Sequence[dict]) -> List[dict]:
    """Generate one result per prompt, preserving each row's sampling parameters."""
    results = client.generate(
        [list(prompt) for prompt in prompts],
        [dict(params) for params in sampling_params],
    )
    if not isinstance(results, list) or len(results) != len(prompts):
        raise RuntimeError(f"generate returned {type(results).__name__} for {len(prompts)} prompts")
    return results


def sync_weights(client: RLClient) -> dict:
    """Sync the native client's training job into its sampling job."""
    return client.sync_weights()


def rl_fwd_bwd(client: RLClient, batch: dict, *, replay_sampling: bool = False) -> dict:
    """Run one RL forward-backward, optionally carrying the sampler identity for router replay."""
    replay = None
    if replay_sampling:
        replay = {"sampling_job_id": str(client.jobs.sampling), "identity_version": 1}
    return client.fwd_bwd(batch, router_replay=replay)


def step(client: RLClient, learning_rate: float) -> dict:
    return client.step(learning_rate)


def rl_batch(rows: Sequence[Sequence[int]], loss_starts: Sequence[int], pad_id: int) -> Dict:
    """Right-padded RL batch in the format ``tests/sp/sp_gateway_harness.py:padded_rl_batch`` sends.

    ``loss_mask`` is logit aligned: the entry at ``s`` weights the prediction of the token at ``s + 1``, so a
    row whose scored tokens begin at ``p`` carries ones over ``p - 1`` through ``len(row) - 2``.
    """
    import torch

    width = max(len(row) for row in rows)
    count = len(rows)
    input_ids = torch.full((count, width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros(count, width, dtype=torch.long)
    loss_mask = torch.zeros(count, width, dtype=torch.float32)
    for index, (row, start) in enumerate(zip(rows, loss_starts, strict=True)):
        input_ids[index, : len(row)] = torch.tensor(list(row), dtype=torch.long)
        attention_mask[index, : len(row)] = 1
        loss_mask[index, start - 1 : len(row) - 1] = 1.0
    return {
        "args": (),
        "kwargs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        "context": {
            "input_ids": input_ids.clone(),
            "advantages": torch.ones(count, width, dtype=torch.float32),
            "loss_mask": loss_mask,
            "temperature": 1.0,
        },
        "processing": {
            "post": ["compute_logprobs"],
            "loss_fn": "grpo",
            "config": {"loss_agg_mode": "token-mean"},
        },
    }


def find_key(value, key: str) -> List:
    """Every value stored under ``key`` anywhere in a nested response, in traversal order."""
    found: List = []
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key:
                found.append(item)
            else:
                found.extend(find_key(item, key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(find_key(item, key))
    return found
