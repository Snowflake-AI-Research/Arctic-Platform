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


"Fast-storage paths, allocation hostfiles, and interrupted-build checks."

from __future__ import annotations

import base64
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

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
