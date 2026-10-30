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


"""Allocation storage, durable checkpoints, and reference records."""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import traceback
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

# Storage

CORRECTNESS_TMPDIR_ENV = "DSS_CORRECTNESS_TMPDIR"
CORRECTNESS_LOGDIR_ENV = "DSS_CORRECTNESS_LOGDIR"
_DATA_FAST_ROOT = Path("/data-fast")
_DATA_FAST_TMPDIR = Path("/data-fast/tmp")


def require_data_fast_path(path: str | Path, *, label: str = "correctness storage path") -> Path:
    """Return a canonical fast-storage descendant without creating any directories."""
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        raise ValueError(f"{label} must resolve under /data-fast") from None
    if resolved == _DATA_FAST_ROOT or not resolved.is_relative_to(_DATA_FAST_ROOT):
        raise ValueError(f"{label} must be a resolved descendant of /data-fast")
    if not _DATA_FAST_ROOT.is_dir():
        raise ValueError(f"{label} requires an existing /data-fast directory")
    return resolved


def correctness_workdir(prefix: str) -> Path:
    """Create a heavyweight correctness workdir exclusively on fast storage."""
    if not isinstance(prefix, str) or os.sep in prefix or (os.altsep and os.altsep in prefix):
        raise ValueError("correctness workdir prefix must not contain a directory path")
    override = os.environ.get(CORRECTNESS_TMPDIR_ENV)
    root = require_data_fast_path(
        override if override else _DATA_FAST_TMPDIR,
        label=CORRECTNESS_TMPDIR_ENV if override else "correctness workdir",
    )
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def correctness_logdir(prefix: str) -> Path:
    """Keep small, shareable diagnostics in the operator's current workspace."""
    root = Path(os.environ.get(CORRECTNESS_LOGDIR_ENV) or Path.cwd() / "work" / "logs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


@dataclass(frozen=True)
class HostfileEntry:
    host: str
    slots: int


def parse_hostfile(path: str | Path) -> list[HostfileEntry]:
    entries: list[HostfileEntry] = []
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        fields = line.split()
        host = fields[0]
        slots = 1
        for field in fields[1:]:
            if field.startswith("slots="):
                slots = int(field.partition("=")[2])
        if slots < 1:
            raise ValueError(f"{path}:{number}: slots must be positive, got {slots}")
        entries.append(HostfileEntry(host=host, slots=slots))
    return entries


MARKER = "INTERRUPTED_EXTENSION_BUILD"


def extensions_root() -> str:
    """The torch extensions cache the DeepSpeed workers build into, resolved the way torch resolves it."""
    from torch.utils.cpp_extension import get_default_build_root

    return os.environ.get("TORCH_EXTENSIONS_DIR") or get_default_build_root()


def node_script(root: str) -> str:
    """Shell that prints one marker line per build directory holding a lock and no shared object.

    A lock beside a compiler that is still running belongs to a build in progress, not to an interrupted one,
    so nothing is reported while ``ninja``, ``nvcc`` or ``cicc`` runs on the node.
    """
    return (
        f"root={shlex.quote(root)}\n"
        "pgrep -x 'ninja|nvcc|cicc' >/dev/null && exit 0\n"
        'for lock in "$root"/*/*/lock; do\n'
        '  [ -e "$lock" ] || continue\n'
        '  d=$(dirname "$lock")\n'
        '  ls "$d"/*.so >/dev/null 2>&1 && continue\n'
        f'  echo "{MARKER} $(hostname) $d"\n'
        "done\n"
    )


def parse(output: str) -> List[Tuple[str, str]]:
    """``(host, directory)`` for every marker line, whatever prefix the remote shell put in front of it."""
    found = []
    for line in output.splitlines():
        _, sep, rest = line.partition(MARKER + " ")
        if sep:
            host, _, directory = rest.strip().partition(" ")
            found.append((host, directory))
    return found


def _ds_ssh() -> str:
    beside = Path(sys.executable).with_name("ds_ssh")
    return str(beside) if beside.exists() else (shutil.which("ds_ssh") or "ds_ssh")


def onboarding_node_prerequisites(request: dict) -> dict:
    """Inspect one executing environment without loading weights or executing a model."""
    import socket

    errors = []
    modules = {}
    versions = {}
    for name in request["modules"]:
        try:
            modules[name] = importlib.import_module(name)
            version = getattr(modules[name], "__version__", None)
            if version is not None:
                versions[name] = str(version)
        except Exception as exc:
            if name == "dss" or name.startswith("dss."):
                installation = (
                    "Install the DSS checkout into this interpreter with python -m pip install -e "
                    "'/path/to/dss-platform' and complete its gateway/worker setup; "
                    "Arctic Platform's .[correctness] extra does not include DSS."
                )
            elif name == "arctic_platform.inference" or name.startswith("arctic_platform.inference."):
                installation = "Install python -m pip install -e '.[inference]' for native sampling."
            elif name == "arctic_inference" or name.startswith("arctic_inference."):
                installation = "Install python -m pip install -e '.[rl]' for legacy OnPrem sampling."
            elif name == "vllm" or name.startswith("vllm."):
                installation = (
                    "Install python -m pip install -e '.[inference]' for native sampling, "
                    "or '.[rl]' for legacy OnPrem sampling."
                )
            else:
                installation = "Install python -m pip install -e '.[correctness]' in this interpreter."
            errors.append(f"cannot import {name}: {type(exc).__name__}: {exc}. {installation}")
    native_inference = modules.get("arctic_platform.inference.utils")
    if native_inference is not None:
        try:
            native_inference.require_supported_vllm_version(feature="DSS correctness sampling")
        except Exception as exc:
            errors.append(f"native sampling runtime: {type(exc).__name__}: {exc}")
    torch = modules.get("torch")
    gpu_count = 0
    if torch is not None:
        try:
            if request["min_gpus"] and not torch.cuda.is_available():
                errors.append("CUDA is unavailable; use a CUDA-enabled PyTorch build and expose the allocated GPUs")
            elif request["min_gpus"]:
                gpu_count = torch.cuda.device_count()
                if gpu_count < request["min_gpus"]:
                    errors.append(f"requires {request['min_gpus']} visible GPUs, found {gpu_count}")
                for index in range(gpu_count):
                    device = torch.cuda.get_device_properties(index)
                    wanted = request.get("gpu_type")
                    if wanted and wanted.lower() not in device.name.lower():
                        errors.append(f"GPU {index} is {device.name}, but this config requires {wanted}")
                    capability = (device.major, device.minor)
                    if "flash_attention_3" in request["attention"] and capability != (9, 0):
                        errors.append(
                            f"GPU {index} has compute capability {capability}; FlashAttention3 requires Hopper"
                        )
                    if "flash_attention_2" in request["attention"] and capability < (8, 0):
                        errors.append(
                            f"GPU {index} has compute capability {capability}; FlashAttention2 requires SM80+"
                        )
                    if "flash_attention_4" in request["attention"] and capability < (9, 0):
                        errors.append(
                            f"GPU {index} has compute capability {capability}; FlashAttention4 requires SM90+"
                        )
        except Exception as exc:
            errors.append(f"cannot inspect CUDA devices: {type(exc).__name__}: {exc}")
    attention_imports = {
        "eager": ("torch.nn.functional", None),
        "sdpa": ("torch.nn.functional", "scaled_dot_product_attention"),
        "flash_attention_2": ("flash_attn", "flash_attn_varlen_func"),
        "flash_attention_3": ("flash_attn_interface", "flash_attn_varlen_func"),
        "flash_attention_4": ("flash_attn.cute", "flash_attn_varlen_func"),
        "qsa_flex": ("torch.nn.attention.flex_attention", "flex_attention"),
        "sparse_mla": ("torch.nn.functional", "scaled_dot_product_attention"),
        "flashinfer": ("flashinfer.mla", "BatchMLAPagedAttentionWrapper"),
        "flashmla": ("flash_mla", "flash_mla_sparse_fwd"),
    }
    for backend in request["attention"]:
        try:
            module_name, attribute = attention_imports[backend]
            module = importlib.import_module(module_name)
            if attribute and not callable(getattr(module, attribute, None)):
                raise ImportError(f"{module_name}.{attribute} is unavailable")
            if backend in request.get("hf_attention", []) and backend in {"flash_attention_2", "flash_attention_3"}:
                availability = importlib.import_module("transformers.utils")
                checker = getattr(availability, f"is_flash_attn_{backend[-1]}_available")
                if not checker():
                    raise ImportError(f"Transformers does not recognize an installed {backend} package")
        except Exception as exc:
            errors.append(
                f"attention backend {backend}: {type(exc).__name__}: {exc}. "
                f"Install the matching backend kernels for Python {sys.version.split()[0]}, "
                f"PyTorch {versions.get('torch', 'unavailable')} and its CUDA build in {sys.executable}; "
                "another FlashAttention version is not a substitute. The configured backend is unchanged."
            )
    writable = dict(request["writable"])
    source = request.get("source")
    if source:
        try:
            path = Path(source)
            with (path / "config.json").open("rb") as stream:
                stream.read(1)
            if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
                raise FileNotFoundError("no model weight files found")
        except OSError as exc:
            errors.append(f"source checkpoint {source} is not readable on this node: {exc}")
    elif "huggingface_hub" in modules and request.get("model_id"):
        try:
            from huggingface_hub import constants
            from huggingface_hub import try_to_load_from_cache

            writable["Hugging Face cache"] = constants.HF_HUB_CACHE
            if constants.HF_HUB_OFFLINE:
                cached = try_to_load_from_cache(request["model_id"], "config.json")
                if not isinstance(cached, str):
                    errors.append("HF_HUB_OFFLINE is set but this model's config.json is not in the local Hub cache")
        except Exception as exc:
            errors.append(f"cannot inspect Hugging Face cache: {exc}")
    for label, value in writable.items():
        try:
            path = Path(value).expanduser().resolve()
            while not path.exists() and path != path.parent:
                path = path.parent
            if not path.is_dir():
                raise NotADirectoryError(str(path))
            with tempfile.TemporaryFile(dir=path) as stream:
                stream.write(b"preflight")
        except (OSError, RuntimeError) as exc:
            errors.append(f"{label} {value} is not writable on this node: {exc}")
    return {
        "host": socket.gethostname(),
        "python": sys.executable,
        "gpu_count": gpu_count,
        "versions": versions,
        "errors": errors,
    }


def allocation_onboarding_prerequisites(request: dict, hostfile: Path) -> dict[str, dict]:
    """Run the same lightweight check through the allocation's existing ds_ssh transport."""
    entries = parse_hostfile(hostfile)
    hosts = {entry.host for entry in entries}
    if not entries or len(hosts) != len(entries):
        raise ValueError("onboarding preflight requires a nonempty hostfile with unique hosts")
    marker = "CORRECTNESS_PREREQUISITES "
    script = (
        "import json\n"
        "from arctic_platform.correctness.harness.storage import onboarding_node_prerequisites\n"
        f"result = onboarding_node_prerequisites(json.loads({json.dumps(request)!r}))\n"
        f"print({marker!r} + json.dumps(result), flush=True)\n"
    )
    encoded = base64.b64encode(script.encode()).decode()
    try:
        done = subprocess.run(
            [_ds_ssh(), "-f", str(hostfile), f"echo {encoded} | base64 -d | {shlex.quote(sys.executable)} -"],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"onboarding preflight could not run {sys.executable} on {hostfile}: {exc}") from exc
    if done.returncode:
        raise ValueError(
            f"onboarding preflight could not run {sys.executable} on {hostfile}: "
            f"{(done.stdout + done.stderr).strip()[-2000:]}"
        )
    results = {}
    for line in done.stdout.splitlines():
        prefix, separator, raw = line.partition(marker)
        if not separator:
            continue
        host = prefix.strip().removesuffix(":").strip()
        if host not in hosts or host in results:
            raise ValueError(f"unexpected or duplicate onboarding preflight response from {host!r}")
        result = json.loads(raw)
        if not isinstance(result, dict) or not isinstance(result.get("errors"), list):
            raise ValueError(f"invalid onboarding preflight response from {host}")
        results[host] = result
    missing = hosts - results.keys()
    if missing:
        raise ValueError(f"onboarding preflight received no response from: {', '.join(sorted(missing))}")
    for entry in entries:
        if results[entry.host]["gpu_count"] < entry.slots:
            results[entry.host]["errors"].append(
                f"hostfile requests {entry.slots} GPUs but only {results[entry.host]['gpu_count']} are visible"
            )
    return results


def ensure_no_interrupted_builds(hostfile: Optional[str] = None) -> None:
    """Stop the run when a node of the named multi-node allocation holds an interrupted extension build.

    Applies only when ``DSS_GATEWAY_URL`` and ``DSS_GATEWAY_HOSTFILE`` name a gateway over more than one node;
    a gateway the harness starts itself serves this node alone. The script is sent base64-encoded because
    ``ds_ssh`` passes its arguments to ``pdsh`` unquoted, which would expand the globs on this node.
    """
    if hostfile is None:
        if not os.environ.get("DSS_GATEWAY_URL"):
            return
        hostfile = os.environ.get("DSS_GATEWAY_HOSTFILE")
    if not hostfile or not Path(hostfile).exists():
        return

    if len(parse_hostfile(hostfile)) < 2:
        return
    encoded = base64.b64encode(node_script(extensions_root()).encode()).decode()
    done = subprocess.run(
        [_ds_ssh(), "-f", hostfile, f"echo {encoded} | base64 -d | bash"], capture_output=True, text=True, timeout=120
    )
    if done.returncode != 0:
        raise SystemExit(
            f"could not inspect the torch extensions cache on the nodes of {hostfile} "
            f"(ds_ssh exit {done.returncode}): {(done.stdout + done.stderr).strip()[-2000:]}"
        )
    found = parse(done.stdout)
    if found:
        listed = "\n".join(f"  {host}: {directory}" for host, directory in found)
        raise SystemExit(
            "interrupted torch extension build(s): each directory below holds a lock and no shared object, and "
            "every DeepSpeed worker on that node waits on the lock indefinitely. Remove the directory, or rerun "
            "the provisioning script on that node, which rebuilds fused_adam:\n"
            + listed
        )


# Checkpoint Store

_FAST_ROOT = Path("/data-fast")
_MARKER = ".correctness-checkpoint-owner.json"
_RECEIPT = "NATIVE_CHECKPOINT_STORE "


class CheckpointStoreError(RuntimeError):
    def __init__(
        self,
        message: str,
        receipts: dict | None = None,
        *,
        transport_stdout: str | bytes | None = None,
        transport_stderr: str | bytes | None = None,
    ) -> None:
        super().__init__(message)
        self.receipts = receipts or {}
        self.transport_stdout = (
            transport_stdout.decode(errors="replace")
            if isinstance(transport_stdout, bytes)
            else transport_stdout or ""
        )
        self.transport_stderr = (
            transport_stderr.decode(errors="replace")
            if isinstance(transport_stderr, bytes)
            else transport_stderr or ""
        )
        for name, output in (("stdout", self.transport_stdout), ("stderr", self.transport_stderr)):
            if output:
                self.add_note(f"Checkpoint launcher {name}:\n{output}")


def _identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_uid]


def _canonical_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute() or root.resolve() != root or not root.is_relative_to(_FAST_ROOT):
        raise ValueError("checkpoint root must be a canonical FAST descendant")
    if root == _FAST_ROOT:
        raise ValueError("checkpoint root cannot be the FAST root")
    return root


def _native_path(source_job_id: str) -> Path:
    from dss.ray_dss.jobs.checkpointing import _job_ckpt_dir

    return Path(_job_ckpt_dir(source_job_id))


def _marker(binding: dict, request: dict) -> dict:
    return {
        "schema": "native-checkpoint-store-v1",
        "store_id": request["store_id"],
        "allocation_sha256": request["allocation_sha256"],
        **binding,
    }


@contextlib.contextmanager
def _native_lock(native: Path, *, exclusive: bool):
    native.parent.mkdir(parents=True, exist_ok=True)
    if native.parent.resolve() != native.parent:
        raise ValueError("native checkpoint parent is not canonical")
    lock = Path(f"{native}.lock")
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or _identity(lock.lstat()) != _identity(info)
        ):
            raise ValueError("native checkpoint lock ownership changed")
        fcntl.flock(descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        try:
            yield _identity(info)
            if _identity(lock.lstat()) != _identity(info):
                raise ValueError("native checkpoint lock inode changed")
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _root_identity(root: Path, binding: dict, request: dict) -> dict:
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise ValueError("checkpoint root must be an owned real directory")
    marker = root / _MARKER
    descriptor = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        marker_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(marker_info.st_mode)
            or marker_info.st_uid != os.geteuid()
            or marker_info.st_nlink != 1
            or marker_info.st_size > 8192
            or _identity(marker.lstat()) != _identity(marker_info)
        ):
            raise ValueError("checkpoint ownership marker is not a bounded owned file")
        with os.fdopen(descriptor, "r", closefd=False) as stream:
            if json.load(stream) != _marker(binding, request):
                raise ValueError("checkpoint ownership marker differs")
    finally:
        os.close(descriptor)
    return {"root": _identity(info), "marker": _identity(marker_info)}


def _no_payload_links(root: Path) -> None:
    def failed(error):
        raise error

    for directory, directories, files in os.walk(root, followlinks=False, onerror=failed):
        for name in directories + files:
            info = (Path(directory) / name).lstat()
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError("checkpoint payload contains a symlink or special file")


def _verified_identity(root: Path, native: Path, binding: dict, request: dict, expected: dict, lock: list) -> dict:
    actual = _root_identity(root, binding, request)
    link = None
    if os.path.lexists(native):
        info = native.lstat()
        if not stat.S_ISLNK(info.st_mode) or info.st_uid != os.geteuid() or os.readlink(native) != str(root):
            raise ValueError("native checkpoint link ownership or literal target differs")
        link = _identity(info)
    actual["link"] = link
    actual["lock"] = lock
    if actual != expected:
        raise ValueError("checkpoint root, marker or native link identity changed")
    _no_payload_links(root)
    return actual


def _node_receipt(request: dict) -> dict:
    receipt = {
        "host": socket.gethostname(),
        "token": request["token"],
        "operation": request["operation"],
        "binding": dict(request["binding"]),
        "ok": False,
        "identity": None,
        "error": None,
    }
    try:
        binding = receipt["binding"]
        source = binding["source_job_id"]
        if not isinstance(source, str) or len(source) != 44 or not source.startswith("correctness-"):
            raise ValueError("unique correctness checkpoint source required")
        uuid.UUID(hex=source.removeprefix("correctness-"))
        root = _canonical_root(binding["path"])
        native = _native_path(source)
        if not native.is_absolute() or native.parent.resolve() != native.parent:
            raise ValueError("canonical native checkpoint namespace required")
        if binding.get("native_path", str(native)) != str(native):
            raise ValueError("native checkpoint path differs from the native resolver")
        binding["native_path"] = str(native)
        operation = request["operation"]
        if operation not in {"create", "verify", "close"}:
            raise ValueError("unknown checkpoint store operation")
        with _native_lock(native, exclusive=operation != "verify") as lock:
            if operation == "create":
                if os.path.lexists(native):
                    raise FileExistsError(f"native checkpoint namespace already exists: {native}")
                if root.exists():
                    info = root.lstat()
                    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or any(root.iterdir()):
                        raise ValueError("checkpoint root must be an empty owned real directory")
                else:
                    root.mkdir(mode=0o700)
                with (root / _MARKER).open("x", encoding="utf-8") as stream:
                    json.dump(_marker(binding, request), stream, sort_keys=True)
                receipt["identity"] = {**_root_identity(root, binding, request), "link": None, "lock": lock}
                os.symlink(root, native)
                receipt["identity"]["link"] = _identity(native.lstat())
            else:
                expected = request["expected"].get(receipt["host"])
                if expected is None:
                    raise ValueError("no acknowledged ownership identity for this node")
                if expected.get("closed"):
                    if operation == "verify" or os.path.lexists(root) or os.path.lexists(native):
                        raise ValueError("closed checkpoint namespace was replaced")
                    if expected["lock"] != lock:
                        raise ValueError("native checkpoint lock inode changed")
                    receipt["identity"] = expected
                else:
                    receipt["identity"] = _verified_identity(root, native, binding, request, expected, lock)
                    if operation == "verify" and receipt["identity"]["link"] is None:
                        raise ValueError("checkpoint native link is absent")
                    if operation == "close":
                        if not shutil.rmtree.avoids_symlink_attacks:
                            raise RuntimeError("checkpoint cleanup requires symlink-safe tree removal")
                        if receipt["identity"]["link"] is not None:
                            native.unlink()
                            receipt["identity"]["link"] = None
                        shutil.rmtree(root)
                        receipt["identity"] = {"closed": True, "lock": lock}
            receipt["ok"] = True
    except BaseException:
        receipt["error"] = traceback.format_exc()
    return receipt


class NativeCheckpointStore:
    """Keep source namespaces until all native consumers have quiesced."""

    def __init__(self, hostfile: Path, *, timeout: float = 120) -> None:
        self.hostfile = Path(hostfile).absolute()
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("checkpoint fanout timeout must be finite and positive")
        self.timeout = timeout
        self.store_id = uuid.uuid4().hex
        self._host_identity = self._hostfile_identity()
        entries = parse_hostfile(self.hostfile)
        self.hosts = {entry.host for entry in entries}
        if not self.hosts or len(self.hosts) != len(entries):
            raise ValueError("checkpoint store requires exact unique allocation nodes")
        self._check_hostfile()
        self._records: dict[str, tuple[dict, dict]] = {}
        self._closed = False

    def _hostfile_identity(self) -> dict:
        info = self.hostfile.lstat()
        target = self.hostfile.resolve(strict=True)
        actual = target.stat()
        if not stat.S_ISREG(actual.st_mode) or not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            raise ValueError("allocation hostfile must name a regular file")
        content = target.read_bytes()

        def identity(value):
            return [*_identity(value), value.st_size, value.st_mtime_ns, value.st_ctime_ns]

        if identity(info) != identity(self.hostfile.lstat()) or identity(actual) != identity(target.stat()):
            raise ValueError("allocation hostfile changed while reading")
        return {
            "link": os.readlink(self.hostfile) if stat.S_ISLNK(info.st_mode) else None,
            "target": str(target),
            "path_identity": identity(info),
            "target_identity": identity(actual),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def _check_hostfile(self) -> None:
        observed = self._hostfile_identity()
        if observed != self._host_identity:
            changes = {
                field: {"expected": expected, "observed": observed[field]}
                for field, expected in self._host_identity.items()
                if observed[field] != expected
            }
            raise CheckpointStoreError(
                f"allocation hostfile changed: path={self.hostfile}; differences={json.dumps(changes, sort_keys=True)}"
            )

    def _fanout(self, operation: str, binding: dict, identities: dict) -> dict:
        self._check_hostfile()
        request = {
            "operation": operation,
            "binding": binding,
            "store_id": self.store_id,
            "allocation_sha256": self._host_identity["sha256"],
            "token": uuid.uuid4().hex,
            "expected": {receipt["host"]: receipt["identity"] for receipt in identities.values()},
        }
        results = {}
        done = None
        try:
            source = Path(__file__).resolve(strict=True)
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            # SSH login cwd/environment need not retain the gateway's private overlay.
            import_roots = list(
                dict.fromkeys(
                    [
                        str(source.parents[3]),
                        *(str(Path(root).resolve()) for root in sys.path if isinstance(root, str)),
                    ]
                )
            )
            script = (
                f"import hashlib, json, sys\nfrom pathlib import Path\nsys.path[:] = {import_roots!r}\nsource ="
                f" Path({str(source)!r})\nexpected_sha = {source_sha!r}\nif"
                " hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha:\n    raise"
                " RuntimeError('checkpoint store source SHA256 differs')\nfrom arctic_platform.correctness.harness"
                " import storage as module\nif Path(module.__file__).resolve() != source:\n    raise"
                " RuntimeError('checkpoint store imported source path differs')\nif"
                " hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha:\n    raise"
                " RuntimeError('checkpoint store source SHA256 differs after import')\nreceipt ="
                f" module._node_receipt(json.loads({json.dumps(request)!r}))\nprint({_RECEIPT!r} +"
                " json.dumps(receipt), flush=True)\n"
            )
            # ds_ssh forwards unquoted arguments to pdsh, flattening multiline source.
            encoded = base64.b64encode(script.encode()).decode()
            done = subprocess.run(
                [_ds_ssh(), "-f", str(self.hostfile), f"echo {encoded} | base64 -d | {shlex.quote(sys.executable)} -"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            for line in done.stdout.splitlines():
                prefix, separator, raw = line.partition(_RECEIPT)
                if not separator:
                    continue
                host = prefix.strip().removesuffix(":").strip()
                if host not in self.hosts or host in results:
                    raise ValueError(f"unexpected or duplicate checkpoint receipt from {host!r}")
                receipt = json.loads(raw)
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("token") != request["token"]
                    or receipt.get("operation") != operation
                    or not isinstance(receipt.get("host"), str)
                    or type(receipt.get("ok")) is not bool
                    or not isinstance(receipt.get("binding"), dict)
                    or any(receipt["binding"].get(key) != value for key, value in binding.items())
                    or not receipt.get("host")
                ):
                    raise ValueError(f"invalid checkpoint receipt from {host}")
                if receipt["ok"]:
                    identity = receipt.get("identity")
                    if not isinstance(identity, dict) or not isinstance(receipt["binding"].get("native_path"), str):
                        raise ValueError(f"missing checkpoint ownership identity from {host}")
                    fields = ("lock",) if identity.get("closed") is True else ("root", "marker", "link", "lock")
                    if operation == "close" and identity.get("closed") is not True:
                        raise ValueError(f"checkpoint cleanup incomplete on {host}")
                    if any(
                        not isinstance(identity.get(field), list)
                        or len(identity[field]) != 4
                        or any(type(value) is not int for value in identity[field])
                        for field in fields
                    ):
                        raise ValueError(f"invalid checkpoint ownership identity from {host}")
                if host in identities and receipt["host"] != identities[host]["host"]:
                    raise ValueError(f"checkpoint node identity changed for {host}")
                if receipt["host"] in {value["host"] for value in results.values()}:
                    raise ValueError("allocation receipts describe duplicate executing nodes")
                results[host] = receipt
            self._check_hostfile()
            if done.returncode or results.keys() != self.hosts or any(not value["ok"] for value in results.values()):
                errors = {host: value.get("error") for host, value in results.items() if not value["ok"]}
                missing = sorted(self.hosts - results.keys())
                raise ValueError(
                    f"checkpoint {operation} failed: rc={done.returncode}, missing hosts={missing}, receipts={errors}"
                )
            return results
        except Exception as exc:
            output = exc if isinstance(exc, subprocess.TimeoutExpired) else done
            raise CheckpointStoreError(
                f"allocation checkpoint {operation}: {exc}",
                results,
                transport_stdout=getattr(output, "stdout", None),
                transport_stderr=getattr(output, "stderr", None),
            ) from exc

    def create(self, path: Path) -> dict:
        if self._closed:
            raise CheckpointStoreError("checkpoint store is closed")
        binding = {"source_job_id": f"correctness-{uuid.uuid4().hex}", "path": str(_canonical_root(path))}
        receipts = {}
        try:
            receipts = self._fanout("create", binding, {})
            native_paths = {value["binding"].get("native_path") for value in receipts.values()}
            if len(native_paths) != 1 or not all(isinstance(value, str) for value in native_paths):
                raise CheckpointStoreError("allocation native checkpoint paths disagree", receipts)
            binding["native_path"] = native_paths.pop()
            return dict(binding)
        except CheckpointStoreError as exc:
            receipts = exc.receipts
            if receipts:
                binding = dict(next(iter(receipts.values()))["binding"])
            raise
        finally:
            self._records[binding["source_job_id"]] = (binding, receipts)

    def verify(self, binding: dict) -> None:
        record = self._records.get(binding.get("source_job_id"))
        if record is None or record[0] != binding:
            raise CheckpointStoreError("checkpoint binding is not owned by this store")
        self._fanout("verify", record[0], record[1])

    @property
    def bindings(self) -> tuple[dict, ...]:
        return tuple(
            {key: binding.get(key) for key in ("source_job_id", "path", "native_path")}
            for binding, _ in self._records.values()
        )

    def close(self) -> None:
        failures = []
        for source, (binding, receipts) in list(self._records.items()):
            try:
                self._fanout("close", binding, receipts)
            except CheckpointStoreError as exc:
                updated = dict(receipts)
                for host, receipt in exc.receipts.items():
                    if receipt.get("identity") is not None:
                        updated[host] = receipt
                self._records[source] = (binding, updated)
                failures.append(exc)
            else:
                del self._records[source]
        if failures:
            raise CheckpointStoreError("; ".join(str(error) for error in failures)) from failures[0]
        self._closed = True


# Reference Records

SCHEMA_VERSION = 1
DEFAULT_RECORDS_DIR = Path(__file__).resolve().parents[1] / "reference_data"
CRITERIA = {
    "single-step-grads": "gradient-l2-norm-v1",
    "single-step-optimizer": "optimizer-state-l2-norm-v1",
    "e2e-sft-train-validate": "sft-trajectory-ce-v1",
}
MAX_RECORD_BYTES = 8 * 1024 * 1024


class ReferencePrerequisiteError(ValueError):
    """A missing, incompatible or corrupt baseline is not a numerical mismatch."""


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _digest(value) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


@lru_cache(maxsize=4096)
def _file_hash(path: str, size: int, mtime_ns: int, ctime_ns: int, inode: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path | str, *, refresh: bool = False) -> dict:
    path = Path(path)
    stat = path.stat()
    hash_file = _file_hash.__wrapped__ if refresh else _file_hash
    sha = hash_file(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    current = path.stat()
    stable_fields = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino", "st_dev")
    if any(getattr(current, name) != getattr(stat, name) for name in stable_fields):
        raise ReferencePrerequisiteError(f"reference source changed while hashing: {path}")
    return {"sha256": sha, "size": stat.st_size}


def model_identity(path: Path | str, *, refresh: bool = False) -> dict:
    """Hash asset contents, not machine-specific roots or file sizes alone."""
    from arctic_platform.correctness.harness.config import NON_MODEL_PROCESSOR_SIDECARS

    root = Path(path)
    if not root.is_dir():
        raise ReferencePrerequisiteError(f"reference model assets unavailable: {root}")
    assets = {
        str(item.relative_to(root)): file_identity(item, refresh=refresh)
        for item in sorted(root.rglob("*"))
        if item.is_file()
        and item.suffix in {".json", ".safetensors", ".bin", ".model", ".tiktoken", ".txt", ".py"}
        and item.name not in NON_MODEL_PROCESSOR_SIDECARS
    }
    if "config.json" not in assets or not any(name.endswith((".safetensors", ".bin")) for name in assets):
        raise ReferencePrerequisiteError(f"reference requires complete model config and weights: {root}")
    return {"assets": assets, "sha256": _digest(assets)}


def tensor_identity(tensor) -> dict:
    import torch

    value = tensor.detach().to(device="cpu").contiguous()
    metadata = {"shape": list(value.shape), "dtype": str(value.dtype)}
    digest = hashlib.sha256(_json(metadata))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {**metadata, "sha256": digest.hexdigest()}


def batch_identity(batch) -> dict:
    return {
        "name": batch.name,
        "active_tokens": batch.active_tokens,
        "tensors": {name: tensor_identity(getattr(batch, name)) for name in ("input_ids", "position_ids", "labels")},
    }


def workload_identity(ctx, check_id: str, recipe: dict) -> dict:
    """Bind HF semantics and inputs, excluding unrelated target source and local paths."""
    if check_id not in CRITERIA:
        raise ValueError(f"not an HF reference check: {check_id}")
    dependencies = {}
    for package in ("torch", "transformers", "peft", "safetensors", "flash-attn", "liger-kernel"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    # Distribution metadata can omit Torch's CUDA build suffix on an identical runtime.
    import torch

    dependencies["torch"] = str(torch.__version__)
    root = Path(__file__).resolve().parents[1]
    package = root.parent
    sources = {
        str(path.relative_to(package if path.is_relative_to(package / "model") else root)): file_identity(path)
        for path in [
            *sorted((root / "reference").glob("*.py")),
            root / "harness" / "sft.py",
            root / "harness" / "optimizer.py",
            # The reference applies the target loader's rounding through these modules.
            package / "model" / "loaders" / "huggingface.py",
            package / "model" / "implementations" / "gpu" / "vllm_rounding.py",
            package / "model" / "implementations" / "gpu" / "qwen3_5_dense_parity.py",
            *(
                [root / "harness" / "lora.py"]
                if isinstance(recipe.get("initialization"), dict)
                and recipe["initialization"].get("schema") == "shared-peft-adapter-v1"
                else []
            ),
        ]
    }
    identity = {
        "schema": SCHEMA_VERSION,
        "check_id": check_id,
        "criterion": CRITERIA[check_id],
        "config_id": ctx.config_id,
        "config_checksum": ctx.spec.config_checksum,
        "parameter_name_map": getattr(ctx.spec, "param_name_map", {}),
        "gpu_family": ctx.cfg.gpu_type,
        "reference_sources": sources,
        "dependencies": dependencies,
        "python": platform.python_version(),
        "recipe": recipe,
    }
    if check_id == "single-step-grads":
        identity["gate"] = asdict(ctx.spec.test_tolerances[check_id])
    elif check_id == "single-step-optimizer":
        identity["norm_reduction_source"] = file_identity(root / "checks" / "optimizer_state.py")
        identity["norm_schema_source"] = file_identity(root / "harness" / "optimizer.py")
    elif check_id == "e2e-sft-train-validate":
        # Shared replay/scoring helpers affect the reference semantics. Older
        # records did not bind them, so do not silently relabel those measurements.
        identity["sft_source"] = file_identity(root / "harness" / "sft.py")
    return identity


def _finite(value, trail="measurements") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReferencePrerequisiteError(f"nonfinite reference measurement: {trail}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{trail}[{index}]")


def validate_measurements(identity: dict, measurements: dict) -> None:
    if not isinstance(measurements, dict):
        raise ReferencePrerequisiteError("reference measurements must be an object")
    _finite(measurements)
    check = identity["check_id"]
    if check in {"single-step-grads", "single-step-optimizer"}:
        for key in ("loss", "microbatches", "active_tokens"):
            if key not in measurements or not isinstance(measurements[key], (int, float)):
                raise ReferencePrerequisiteError(f"incomplete reference: missing {key}")
        norms = measurements.get("grad_norms")
        if (
            not isinstance(norms, dict)
            or not norms
            or any(
                not isinstance(name, str) or not isinstance(value, (int, float)) or value < 0
                for name, value in norms.items()
            )
        ):
            raise ReferencePrerequisiteError("reference gradient norms must be complete finite nonnegative scalars")
        if measurements.get("participation") is not None:
            from arctic_platform.correctness.harness.optimizer import active_gradient_norms

            active_gradient_norms(measurements["participation"], norms)
        if check == "single-step-optimizer":
            summary = measurements.get("optimizer_norms", {})
            gate = measurements.get("optimizer_norm_gate", {})
            if summary.get("criterion") != CRITERIA[check] or gate.get("criterion") != CRITERIA[check]:
                raise ReferencePrerequisiteError("optimizer reference requires the versioned norm criterion")
            provenance = gate.get("provenance")
            if (
                gate.get("status") != "reviewed"
                or not isinstance(provenance, (str, dict))
                or not provenance
                or (isinstance(provenance, str) and not provenance.strip())
            ):
                raise ReferencePrerequisiteError("optimizer norm gate has no onboarding review provenance")
            if type(gate.get("absolute")) not in {int, float} or gate["absolute"] < 0:
                raise ReferencePrerequisiteError("optimizer norm gate must be a finite nonnegative scalar")
            if not summary.get("parameters"):
                raise ReferencePrerequisiteError("optimizer reference has no parameter norms")
            from arctic_platform.correctness.harness.optimizer import compare_participation
            from arctic_platform.correctness.harness.optimizer import validate_optimizer_norm_summary
            from arctic_platform.correctness.harness.optimizer import validate_optimizer_participation

            validate_optimizer_norm_summary(summary)
            validate_optimizer_participation(summary, backend="reference")
            if measurements.get("participation") is not None or summary.get("participation") is not None:
                compare_participation(measurements.get("participation"), summary.get("participation"))
    elif check == "e2e-sft-train-validate":
        for key in ("train_losses", "train_gradient_norms"):
            values = measurements.get(key)
            if (
                not isinstance(values, list)
                or len(values) != 100
                or not all(isinstance(value, (int, float)) for value in values)
            ):
                raise ReferencePrerequisiteError(f"reference requires all 100 scalar {key}")
        if measurements.get("optimizer_steps") != 100:
            raise ReferencePrerequisiteError("reference requires 100 optimizer steps")
        if not isinstance(measurements.get("validation_cross_entropy"), (int, float)):
            raise ReferencePrerequisiteError("reference requires validation cross-entropy")
        if not isinstance(measurements.get("scored_answer_tokens"), int) or measurements["scored_answer_tokens"] <= 0:
            raise ReferencePrerequisiteError("reference requires positive scored-answer-token count")


def _config_dir(ctx, identity: dict) -> Path:
    if (
        identity.get("check_id") not in CRITERIA
        or not isinstance(identity.get("config_id"), str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+", identity["config_id"]) is None
        or identity["config_id"] in {".", ".."}
    ):
        raise ReferencePrerequisiteError("invalid reference record config or check identity")
    root = Path(getattr(ctx, "reference_records_dir", DEFAULT_RECORDS_DIR))
    return root / identity["config_id"]


def _record_path(ctx, identity: dict) -> Path:
    directory = _config_dir(ctx, identity) / identity["check_id"]
    if identity["check_id"] == "e2e-sft-train-validate":
        return directory / "trajectory.json"
    recipe = identity.get("recipe", {})
    arm = recipe.get("synthetic_inputs", {}).get("batch", {}).get("name", recipe.get("arm"))
    if not isinstance(arm, str) or re.fullmatch(r"[A-Za-z0-9_-]+", arm) is None:
        raise ReferencePrerequisiteError("invalid reference record arm identity")
    if identity["check_id"] == "single-step-grads":
        directory /= "with-optimizer" if recipe.get("optimizer") is not None else "gradients-only"
    return directory / f"{arm}.json"


def _legacy_record_path(ctx, identity: dict) -> Path:
    return _config_dir(ctx, identity) / identity["check_id"] / f"{_digest(identity)}.json"


@contextmanager
def _publication_lock(directory: Path, *, exclusive: bool):
    import fcntl

    # Keep the lock outside the directory being replaced; never replace its inode.
    with (directory.parent / f".{directory.name}.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            if directory.is_symlink():
                raise ReferencePrerequisiteError(f"reference directory must not be a symbolic link: {directory}")
            previous = directory.parent / f".previous-{directory.name}"
            abandoned = [
                path
                for path in directory.parent.glob(f".set-{directory.name}-*")
                if path.is_dir() and not path.is_symlink()
            ]
            if previous.exists() or abandoned:
                # Recover interruption between renames, then discard abandoned staging.
                fcntl.flock(stream, fcntl.LOCK_EX)
                if previous.exists() and not directory.exists():
                    os.rename(previous, directory)
                elif previous.exists():
                    shutil.rmtree(previous)
                for path in directory.parent.glob(f".set-{directory.name}-*"):
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                if not exclusive:
                    fcntl.flock(stream, fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _publish_directory(staged: Path, current: Path) -> None:
    """Rename the old set aside, install the prepared directory, then remove the old set."""
    previous = current.parent / f".previous-{current.name}"
    if current.exists():
        os.rename(current, previous)
    try:
        os.rename(staged, current)
    except BaseException:
        if previous.exists() and not current.exists():
            os.rename(previous, current)
        raise
    fd = os.open(current.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if previous.exists():
        shutil.rmtree(previous)


def require_git_records_dir(path: Path | str) -> Path:
    """CLI publication stays in the checkout; transient tensor artifacts cannot use this directory."""
    source_root = Path(__file__).resolve().parents[3]
    root = next((parent for parent in (source_root, *source_root.parents) if (parent / ".git").exists()), source_root)
    destination = Path(path).resolve()
    if not destination.is_relative_to(root) or destination.is_relative_to(root / ".git"):
        raise ReferencePrerequisiteError(f"compact references must belong to the Git checkout: {root}")
    return destination


def _strict_record_json(raw: bytes) -> dict:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ReferencePrerequisiteError(f"duplicate reference JSON key: {key}")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ReferencePrerequisiteError(f"nonfinite reference JSON value: {value}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=invalid_constant)


def load_record(ctx, current_identity: dict) -> dict:
    identity = current_identity
    path = _record_path(ctx, identity)
    selection = ["--test-id", identity["check_id"]]
    recipe = identity.get("recipe", {})
    if identity["check_id"] == "single-step-grads" and recipe.get("optimizer") is not None:
        selection = ["--reference-tests", "single-step-grads", "single-step-optimizer"]
    options = ["--reference-records-dir", str(getattr(ctx, "reference_records_dir", DEFAULT_RECORDS_DIR))]
    if getattr(ctx, "local_gateway_full_model", False):
        options.append("--local-gateway-full-model")
    if identity["check_id"] != "e2e-sft-train-validate" and recipe.get("token_budget") is not None:
        options.extend(["--reference-token-budget", str(recipe["token_budget"])])
    command = (
        "python -m arctic_platform.correctness onboard "
        f"--config {shlex.quote(str(ctx.spec.config_path))} --prepare-references "
        f"{shlex.join(selection + options)}"
    )
    try:
        directory = _config_dir(ctx, identity)
        if not directory.parent.exists():
            raise FileNotFoundError(path)
        with _publication_lock(directory, exclusive=False):
            # Legacy reuse is exact-identity only. Never fall back past a published latest set.
            latest_files = [
                item for item in path.parent.glob("*.json") if re.fullmatch(r"[0-9a-f]{64}\.json", item.name) is None
            ]
            if not latest_files:
                path = _legacy_record_path(ctx, identity)
            if path.stat().st_size > MAX_RECORD_BYTES:
                raise ReferencePrerequisiteError(f"reference record is oversized: {path}")
            record = _strict_record_json(path.read_bytes())
    except FileNotFoundError as exc:
        raise ReferencePrerequisiteError(f"prepared HF reference missing: {path}; run {command}") from exc
    except (UnicodeError, json.JSONDecodeError, ReferencePrerequisiteError) as exc:
        if "oversized" in str(exc):
            raise
        raise ReferencePrerequisiteError(f"prepared HF reference corrupt: {path}; run {command}") from exc
    try:
        valid = (
            isinstance(record, dict)
            and record.get("schema") == SCHEMA_VERSION
            and record.get("identity") == identity
            and record.get("payload_sha256") == _digest(record.get("measurements"))
        )
    except (ValueError, TypeError) as exc:
        raise ReferencePrerequisiteError(f"prepared HF reference corrupt: {path}; run {command}") from exc
    if not valid:
        if (
            isinstance(record, dict)
            and record.get("schema") == SCHEMA_VERSION
            and record.get("identity") != identity
            and record.get("payload_sha256") == _digest(record.get("measurements"))
        ):
            raise ReferencePrerequisiteError(f"prepared HF reference incompatible: {path}; run {command}")
        raise ReferencePrerequisiteError(f"prepared HF reference corrupt or incompatible: {path}; run {command}")
    try:
        validate_measurements(identity, record["measurements"])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ReferencePrerequisiteError(
            f"prepared HF reference invalid measurements: {path}: {exc}; run {command}"
        ) from exc
    paths = getattr(ctx, "reference_record_paths", None)
    if paths is not None:
        paths[_digest(identity)] = str(path)
    print(f"Loaded prepared HF reference: {path}", flush=True)
    return record["measurements"]


def _record_bytes(identity: dict, measurements: dict) -> bytes:
    validate_measurements(identity, measurements)
    data = _json(
        {
            "schema": SCHEMA_VERSION,
            "identity": identity,
            "measurements": measurements,
            "payload_sha256": _digest(measurements),
        }
    )
    if len(data) > MAX_RECORD_BYTES:
        raise ReferencePrerequisiteError("reference JSON exceeds compact record size limit")
    return data


def _remove_replaced_legacy_records(ctx, staged: Path, identities: list[dict]) -> None:
    replaced = {_record_path(ctx, identity).parent for identity in identities}
    directory = _config_dir(ctx, identities[0])
    for check_id in {identity["check_id"] for identity in identities}:
        for path in (staged / check_id).glob("*.json"):
            if re.fullmatch(r"[0-9a-f]{64}\.json", path.name) is None:
                continue
            record = _strict_record_json(path.read_bytes())
            identity = record["identity"]
            if (
                record.get("schema") != SCHEMA_VERSION
                or identity.get("config_id") != directory.name
                or identity.get("check_id") != check_id
                or path.name != _legacy_record_path(ctx, identity).name
                or record.get("payload_sha256") != _digest(record.get("measurements"))
            ):
                raise ReferencePrerequisiteError(f"cannot identify legacy reference for replacement: {path}")
            validate_measurements(identity, record["measurements"])
            if _record_path(ctx, identity).parent in replaced:
                path.unlink()


def publish_record_set(ctx, entries, *, expected_identities) -> dict[str, Path]:
    """Publish all expected arms/checks together, retaining only their latest complete workload sets."""
    entries = list(entries)
    expected = list(expected_identities)
    if not entries or not expected:
        raise ReferencePrerequisiteError("reference publication requires a nonempty complete set")
    directory = _config_dir(ctx, expected[0])
    prepared = {}
    groups = {}
    for identity in expected:
        path = _record_path(ctx, identity)
        if _config_dir(ctx, identity) != directory:
            raise ReferencePrerequisiteError("reference set must belong to one configuration")
        if path in prepared:
            raise ReferencePrerequisiteError("reference set has duplicate workload arms")
        prepared[path] = identity
        groups.setdefault(path.parent, []).append(identity)
    for identities in groups.values():
        generations = []
        for identity in identities:
            generation = deepcopy(identity)
            recipe = generation["recipe"]
            for field in ("arm", "batch"):
                recipe.pop(field, None)
            inputs = recipe.get("synthetic_inputs")
            if isinstance(inputs, dict):
                inputs.pop("batch", None)
                inputs.pop("rows", None)
            generations.append(_digest(generation))
        if len(set(generations)) != 1:
            raise ReferencePrerequisiteError("reference set mixes incompatible workload generations")
    data_by_path = {}
    for identity, measurements in entries:
        path = _record_path(ctx, identity)
        if path not in prepared or identity != prepared[path] or path in data_by_path:
            raise ReferencePrerequisiteError("reference set has unexpected or duplicate workload identities")
        data_by_path[path] = _record_bytes(identity, measurements)
    if set(data_by_path) != set(prepared):
        raise ReferencePrerequisiteError("reference publication requires every expected workload arm")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(directory, exclusive=True):
        for parent in groups:
            existing_arms = {
                path.name for path in parent.glob("*.json") if re.fullmatch(r"[0-9a-f]{64}\.json", path.name) is None
            }
            expected_arms = {path.name for path in prepared if path.parent == parent}
            if existing_arms - expected_arms:
                raise ReferencePrerequisiteError("reference replacement must include every previously published arm")
        # A repeated identity cannot silently acquire different measurements, even in a new set.
        for path, data in data_by_path.items():
            old = path if path.exists() else _legacy_record_path(ctx, prepared[path])
            if old.exists():
                previous = _strict_record_json(old.read_bytes())
                if previous.get("identity") == prepared[path] and old.read_bytes() != data:
                    raise ReferencePrerequisiteError(f"conflicting immutable reference publication: {path}")
        unchanged = all(path.exists() and path.read_bytes() == data for path, data in data_by_path.items())
        exact_files = all(
            set(parent.iterdir()) == {path for path in prepared if path.parent == parent}
            for parent in groups
            if parent.exists()
        )
        legacy = any(list((directory / identity["check_id"]).glob("[0-9a-f]" * 64 + ".json")) for identity in expected)
        if not (unchanged and exact_files and not legacy):
            temporary = Path(tempfile.mkdtemp(prefix=f".set-{directory.name}-", dir=directory.parent))
            staged = temporary / directory.name
            try:
                if directory.exists():
                    shutil.copytree(directory, staged)
                else:
                    staged.mkdir()
                for parent in groups:
                    destination = staged / parent.relative_to(directory)
                    if destination.exists():
                        shutil.rmtree(destination)
                    destination.mkdir(parents=True)
                for path, data in data_by_path.items():
                    destination = staged / path.relative_to(directory)
                    with destination.open("xb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if destination.read_bytes() != data:
                        raise ReferencePrerequisiteError(f"staged reference changed before publication: {destination}")
                _remove_replaced_legacy_records(ctx, staged, expected)
                for parent in sorted({staged, staged.parent, *(path for path in staged.rglob("*") if path.is_dir())}):
                    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                _publish_directory(staged, directory)
                fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                shutil.rmtree(temporary)
    published = {_digest(identity): path for path, identity in prepared.items()}
    paths = getattr(ctx, "reference_record_paths", None)
    if paths is not None:
        paths.update({key: str(path) for key, path in published.items()})
    for path, data in data_by_path.items():
        print(f"Prepared HF reference: {path} ({len(data)} bytes)", flush=True)
    return published


def publish_record(ctx, identity: dict, measurements: dict) -> Path:
    """Publish a single-record workload; multi-arm preparation must use publish_record_set."""
    return publish_record_set(ctx, [(identity, measurements)], expected_identities=[identity])[_digest(identity)]
