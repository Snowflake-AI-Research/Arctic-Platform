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


"Supervised gateway jobs, explicit request packing, and owned cleanup."

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig

from .config import ReviewedClientConfig
from .config import ReviewedTrainingConfig
from .config import correctness_microbatch_tokens
from .cortex import CorrectnessClient as ArcticClient
from .cortex import CortexTrainingConfig
from .storage import parse_hostfile
from .storage import require_data_fast_path


class UnsafeCleanupError(RuntimeError):
    """Owned resources may remain; do not start another correctness workload."""


@dataclass
class StepResult:
    avg_loss: float
    grad_norms: Dict[str, float]
    model_calls: Optional[int] = None
    packed_rows: Optional[int] = None
    optimizer_state_manifest: Optional[str] = None


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
    r"([r]aylet|[g]cs_server|[d]ashboard.py|[m]onitor.py|[l]og_monitor.py|[d]efault_worker.py|ray::)"
)


def _allocation_hostfile() -> Path | None:
    value = os.environ.get("DSS_GATEWAY_HOSTFILE") or os.environ.get("ARCTIC_HOSTFILE")
    if value:
        path = Path(value)
        if path.is_file():
            return path
    default = Path("/data-fast/hostfile")
    return default if default.is_file() else None


_GATEWAY_PROC_ROOT = Path("/proc")


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
    print(f"Stopping stale owned Cortex-training gateway (PID {gateway['pid']}); log: {log_path}", flush=True)
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
            record_path.unlink()
            return
        environment = (_GATEWAY_PROC_ROOT / str(gateway["pid"]) / "environ").read_bytes().split(b"\0")
        marker = f"DSS_CORRECTNESS_GATEWAY_OWNER={record['token']}".encode()
        if marker not in environment or os.getpgid(gateway["pid"]) != gateway["pid"]:
            raise ValueError("live process does not match the recorded harness-owned gateway")
        _stop_stale_gateway(gateway, record["log"])
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
def local_cortex_gateway(directory: Path) -> Iterator[str]:
    """Recover an orphan, then start and close this command's owned DSS service."""
    with _owned_gateway_slot() as record_path:
        with _started_local_cortex_gateway(directory, record_path) as url:
            yield url


@contextlib.contextmanager
def _started_local_cortex_gateway(directory: Path, record_path: Path) -> Iterator[str]:
    import requests
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    hostfile = _allocation_hostfile()
    if hostfile is None and Path("/job/hostfile").is_file():
        hostfile = Path("/job/hostfile")
    if hostfile is None:
        hostfile = directory / "hostfile"
        hostfile.write_text(f"{socket.gethostname()} slots={torch.cuda.device_count()}\n")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = directory / "gateway.log"
    print(f"\n*** Target: Starting Cortex Training Gateway ***\nGateway log: {log_path}\n", flush=True)
    previous_handlers = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}; stopping owned Cortex-training gateway")

    with log_path.open("w") as log:
        process = None
        token = uuid.uuid4().hex
        try:
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, interrupt)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "dss.ray_dss.gateway_server",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--backend",
                    "gpu",
                    "--hostfile",
                    str(hostfile),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "DSS_CORRECTNESS_GATEWAY_OWNER": token},
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
                            f"Stopping owned Cortex-training gateway (PID {process.pid}); log: {log_path}", flush=True
                        )
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=120)
                            except subprocess.TimeoutExpired as exc:
                                try:
                                    os.killpg(process.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    # The group can exit between the timeout and signal; still reap the owned child.
                                    pass
                                process.wait()
                                raise UnsafeCleanupError(
                                    "owned Cortex-training gateway did not shut down cleanly; inspect"
                                    f" {log_path} before retrying"
                                ) from exc
                        else:
                            process.wait()
                        record_path.unlink(missing_ok=True)
                        print("Owned Cortex-training gateway stopped and reaped.", flush=True)
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


def _stop_and_verify_gateway_runtime() -> None:
    """Stop every Ray process in the allocation and prove the standard rendezvous port is reusable."""
    ray_bin = Path(sys.executable).resolve().parent / "ray"
    _run_on_allocation(f"{shlex.quote(str(ray_bin))} stop --force")
    port_probe = (
        "import socket; "
        "s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
        f"s.bind(('', {_STANDARD_MASTER_PORT})); s.close()"
    )
    _run_on_allocation(
        f"! pgrep -af {shlex.quote(_RAY_PROCESS_PATTERN)} >/dev/null && "
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
    try:
        ray_cluster = _load_gateway_ray_cluster()
        previous_hostfile = ray_cluster._HOSTFILE
        if not include_peers:
            ray_cluster._HOSTFILE = os.devnull
        _stop_and_verify_gateway_runtime()
        ray_cluster.init_ray_cluster(auto_attach=False)
        yield
    finally:
        try:
            try:
                if ray_cluster is not None:
                    ray_cluster._shutdown()
            finally:
                _stop_and_verify_gateway_runtime()
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
    yield GatewaySession(workdir=workdir, slots=slots, backend=backend if backend is not None else OnPremConfig())


def _ds_optimizer(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    value = deepcopy(value)
    name = value.pop("name", value.pop("type", "AdamW"))
    return {"type": name, "params": value}


def _training_client_config(session: GatewaySession, payload: dict) -> ArcticClientConfig:
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
    )


@contextlib.contextmanager
def running_job(session: GatewaySession, payload: dict) -> Iterator[ArcticJob]:
    job_type = str(payload.get("job_type", "training"))
    if job_type != "training":
        raise ValueError("P1 supports supervised training jobs only")
    config = _training_client_config(session, payload)
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
    if processing:
        raise ValueError("P1 Cortex requests support supervised SFT only")
    lowered = {
        "args": (),
        "kwargs": tensors,
        "context": dict(body.get("meta") or {}),
    }
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
    source = next((path for path in candidates if (path / "adapter_model.safetensors").is_file()), None)
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
    debug.update({"gradient_norms_per_param": gradient_norms_per_param, "gradient_sample_max_numel": 0})
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
    )
