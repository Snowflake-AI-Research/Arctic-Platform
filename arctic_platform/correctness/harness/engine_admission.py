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

"""Resolve omitted accumulation against every automatically owned executing DSS runtime."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import inspect
import json
import marshal
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from functools import wraps
from pathlib import Path

from arctic_platform.correctness.harness.storage import correctness_environment

RUNTIME_ADMISSION_ENV = "AP_CORRECTNESS_RUNTIME_ADMISSION"
RUNTIME_ADMISSION_KEY = "correctness_runtime_admission"
_RUNTIME_MODULES = (
    "dss.ray_dss.models",
    "dss.ray_dss.jobs.gpu.training",
    "arctic_platform.common.deepspeed_engine",
    "deepspeed.runtime.config",
    "deepspeed.runtime.constants",
    "deepspeed.runtime.engine",
    "deepspeed.runtime.precision_config",
    "deepspeed.runtime.zero.config",
)
_RUNTIME_CALLABLES = {
    "payload",
    "builder",
    "spec_builder",
    "engine_spec",
    "worker_constructor",
    "shared_builder",
    "accumulation_default",
    "initializer",
    "engine_dtypes",
}


def _validate_runtime_sources(sources: dict) -> None:
    if not isinstance(sources, dict) or set(sources) != {*_RUNTIME_MODULES, "loaded_callables"}:
        raise ValueError("complete runtime source inventory required")
    for name in _RUNTIME_MODULES:
        row = sources[name]
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not Path(row["path"]).is_absolute()
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("absolute runtime source path and complete content hash required")
    callables = sources["loaded_callables"]
    if (
        not isinstance(callables, dict)
        or set(callables) != _RUNTIME_CALLABLES
        or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in callables.values())
    ):
        raise ValueError("complete loaded runtime callable identities required")


def _runtime_sources() -> dict:
    from deepspeed.runtime import config as ds_config
    from deepspeed.runtime import constants as ds_constants
    from deepspeed.runtime import engine as ds_engine
    from deepspeed.runtime import precision_config as ds_precision
    from deepspeed.runtime.zero import config as ds_zero
    from dss.ray_dss import models
    from dss.ray_dss.jobs.gpu import training

    from arctic_platform.common import deepspeed_engine as shared

    sources = {}
    modules = (models, training, shared, ds_config, ds_constants, ds_engine, ds_precision, ds_zero)
    for module in modules:
        path = Path(inspect.getsourcefile(module)).resolve()
        sources[module.__name__] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ("build_deepspeed_config", "TrainingEngineSpec", "initialize_training_engine"):
        if getattr(training, name) is not getattr(shared, name):
            raise ValueError(f"native training does not bind the admitted shared {name}")
    namespace = training._DeepSpeedWorker.__init__.__globals__
    for name in ("_build_ds_config", "initialize_training_engine"):
        if namespace.get(name) is not getattr(training, name):
            raise ValueError(f"native worker constructor binds a different {name}")
    callables = {
        "payload": models.JobConfig.to_engine_dict,
        "builder": training._build_ds_config,
        "spec_builder": training._training_engine_spec,
        "engine_spec": shared.TrainingEngineSpec.__init__,
        "worker_constructor": training._DeepSpeedWorker.__init__,
        "shared_builder": shared.build_deepspeed_config,
        "accumulation_default": shared.apply_default_grad_accum_dtype,
        "initializer": shared.initialize_training_engine,
        "engine_dtypes": ds_engine.DeepSpeedEngine.get_data_types,
    }
    sources["loaded_callables"] = {
        name: hashlib.sha256(marshal.dumps(function.__code__)).hexdigest() for name, function in callables.items()
    }
    return sources


def _runtime_engine_config(payload: dict, n_gpus: int, sp_size: int) -> dict:
    from dss.ray_dss.jobs.gpu import training as runtime
    from dss.ray_dss.models import JobConfig

    builder = getattr(runtime, "_build_ds_config", None)
    if not callable(builder):
        raise ValueError("installed DSS runtime does not expose _build_ds_config")
    cfg = JobConfig.model_validate(payload).to_engine_dict("correctness-admission")
    return builder(cfg, n_gpus, sp_size)


def _runtime_admission_receipt(request: dict) -> dict:
    """Authenticate the payload and run the actual builder and typed parsers on this node."""
    import torch

    resolved = _runtime_engine_config(request["payload"], request["n_gpus"], request["sp_size"])
    _validate_engine_config(resolved)
    sources = _runtime_sources()
    return {
        "host": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "python": sys.executable,
        "python_sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        "gpu_count": torch.cuda.device_count(),
        "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
        "admission_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": sources,
        "resolved": resolved,
    }


def _existing_allocation_transport() -> dict:
    from .inference import _held_ssh_trust

    trust = Path(correctness_environment("AP_CORRECTNESS_SSH_KNOWN_HOSTS", str(Path.home() / ".ssh/known_hosts")))
    port = int(correctness_environment("AP_CORRECTNESS_SSH_PORT", "22"))
    transport = {
        "port": port,
        "known_hosts": str(trust),
        "known_hosts_sha256": hashlib.sha256(trust.read_bytes()).hexdigest(),
    }
    if not 1 <= port <= 65535:
        raise ValueError("allocation SSH port must be in 1..65535")
    with _held_ssh_trust(transport, time.monotonic() + 30):
        pass
    return transport


def _allocation_runtime_receipts(request: dict, hostfile: Path, transport: dict | None = None) -> dict[str, dict]:
    from .inference import _strict_ssh_argv
    from .storage import parse_hostfile

    entries = parse_hostfile(hostfile)
    hosts = {entry.host for entry in entries}
    if not entries or len(hosts) != len(entries):
        raise ValueError("runtime admission requires a nonempty unique allocation hostfile")
    marker = "CORRECTNESS_ENGINE_ADMISSION "
    source = str(Path(__file__).resolve())
    expected_source = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    script = (
        "import hashlib, importlib.util, json, pathlib\n"
        f"assert hashlib.sha256(pathlib.Path({source!r}).read_bytes()).hexdigest() == {expected_source!r}\n"
        f"spec = importlib.util.spec_from_file_location('correctness_admission_probe', {source!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"receipt = module._runtime_admission_receipt(json.loads({json.dumps(request)!r}))\n"
        f"print({marker!r} + json.dumps(receipt), flush=True)\n"
    )
    encoded = base64.b64encode(script.encode()).decode()
    transport = transport or _existing_allocation_transport()
    nodes = [{"host": entry.host} for entry in entries]
    admission = {"ssh": transport, "nodes": nodes}
    expected_digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    receipts = {}
    deadline = time.monotonic() + 180
    for node in nodes:
        host = node["host"]
        pythonpath = os.environ.get("PYTHONPATH")
        environment = f"env PYTHONPATH={shlex.quote(pythonpath)} " if pythonpath is not None else ""
        command = f"echo {encoded} | base64 -d | {environment}{shlex.quote(sys.executable)} -"
        with _strict_ssh_argv(admission, node, command, deadline) as argv:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=max(1, deadline - time.monotonic()))
        if done.returncode or len(done.stdout) > 1024**2:
            raise ValueError(f"allocation runtime admission failed on {host}: {(done.stdout + done.stderr)[-2000:]}")
        rows = [line.removeprefix(marker) for line in done.stdout.splitlines() if line.startswith(marker)]
        if len(rows) != 1:
            raise ValueError(f"one authenticated runtime admission receipt required: {host}")
        receipt = json.loads(rows[0])
        _validate_runtime_sources(receipt.get("sources"))
        if (
            receipt.get("request_sha256") != expected_digest
            or receipt.get("admission_source_sha256") != expected_source
            or receipt.get("python") != sys.executable
            or not receipt.get("boot_id")
            or set(receipt.get("sources", {})) != {*_RUNTIME_MODULES, "loaded_callables"}
        ):
            raise ValueError(f"runtime admission receipt does not bind the submitted payload and interpreter: {host}")
        _validate_engine_config(receipt["resolved"])
        receipts[host] = receipt
    if hosts != receipts.keys():
        raise ValueError(f"runtime admission missing node receipts: {sorted(hosts - receipts.keys())}")
    if len({receipt["host"] for receipt in receipts.values()}) != len(receipts):
        raise ValueError("runtime admission receipts describe duplicate executing nodes")
    return receipts


def _allocation_engine_configs(request: dict, hostfile: Path, layout) -> list[dict]:
    receipts = _allocation_runtime_receipts(request, hostfile)
    for node in layout:
        receipt = receipts[node.host]
        if any(gpu < 0 or gpu >= receipt["gpu_count"] for gpu in node.gpu_ids):
            raise ValueError(f"runtime admission selected unavailable GPUs on {node.host}")
    selected = [receipts[node.host] for node in layout]
    if any(receipt["resolved"] != selected[0]["resolved"] for receipt in selected[1:]):
        raise ValueError("selected DSS workers resolve different submitted engine configurations")
    for receipt in selected:
        print("CORRECTNESS_ENGINE_ADMISSION " + json.dumps(receipt, sort_keys=True), flush=True)
    return [receipt["resolved"] for receipt in selected]


def owned_allocation_admission(config) -> dict:
    """Pin the owned allocation using existing trust, never keys acquired through ambient SSH."""
    from arctic_platform.client import CortexConfig

    from .gateway import _allocation_hostfile
    from .gateway import training_client_config
    from .storage import parse_hostfile

    hostfile = _allocation_hostfile()
    if hostfile is None:
        raise ValueError("multi-node fixture requires the owned allocation hostfile")
    raw = hostfile.read_bytes()
    entries = parse_hostfile(hostfile)
    client = training_client_config(config, CortexConfig(base_url="http://127.0.0.1:1"))
    request = {"payload": client.to_cortex()[0], "n_gpus": config.n_gpus, "sp_size": config.sp_size}
    transport = _existing_allocation_transport()
    receipts = _allocation_runtime_receipts(request, hostfile, transport)
    if hostfile.read_bytes() != raw:
        raise ValueError("allocation hostfile changed during runtime verification")
    nodes = []
    for entry in entries:
        receipt = receipts[entry.host]
        if receipt["gpu_count"] != entry.slots:
            raise ValueError(f"allocation slots do not match visible GPUs on {entry.host}")
        nodes.append(
            {
                "host": entry.host,
                "slots": entry.slots,
                "hostname": receipt["host"],
                "boot_id": receipt["boot_id"],
                "python": receipt["python"],
                "python_sha256": receipt["python_sha256"],
            }
        )
    return {
        "allocation_id": "correctness-" + uuid.uuid4().hex,
        "hostfile_content": raw.decode(),
        "hostfile_sha256": hashlib.sha256(raw).hexdigest(),
        "nodes": nodes,
        "ssh": transport,
        "runtime_request": request,
        "runtime_receipts": receipts,
    }


def _owned_local_engine_config(config, *, automatically_owned_local: bool) -> dict:
    """Use the actual submitted payload and each selected runtime, without adding overrides."""
    if not automatically_owned_local:
        raise ValueError("runtime-default admission requires the automatically owned local DSS gateway")

    import torch

    from arctic_platform.client import CortexConfig

    from .gateway import _allocation_hostfile
    from .gateway import training_client_config

    try:
        capacity = torch.cuda.device_count()
        client = training_client_config(config, CortexConfig(base_url="http://127.0.0.1:1"))
        request = {"payload": client.to_cortex()[0], "n_gpus": config.n_gpus, "sp_size": config.sp_size}
        hostfile = _allocation_hostfile()
        if hostfile is not None:
            from dss.ray_dss.placement.local import LocalPlacementBackend
            from dss.ray_dss.utils import is_local

            # An owned gateway starts a fresh placement pool. Use its real
            # selection, not the on-prem gateway's unrelated peer decision.
            placement = LocalPlacementBackend(hostfile=str(hostfile))
            layout = placement._pick_hosts(config.n_gpus)
            if any(not is_local(node.host) or any(gpu >= capacity for gpu in node.gpu_ids) for node in layout):
                return _allocation_engine_configs(request, hostfile, layout)[0]
        elif config.n_gpus > capacity:
            raise ValueError("runtime-default admission requires a same-job hostfile for nonlocal workers")
        # Without an allocation hostfile, the owned gateway writes this host
        # and its visible device count into its launch-only hostfile.
        return _runtime_engine_config(**request)
    except Exception as exc:
        raise ValueError(f"cannot resolve installed DSS runtime accumulation: {exc}") from exc


def validate_reference_accumulation(config, *, automatically_owned_local: bool = False) -> None:
    """Preserve explicit FP32 support; infer defaults only from the owned executing runtime."""
    from .config import validate_first_two_model_dtype

    validate_first_two_model_dtype(config)
    if config.is_rl:
        return
    data_types = (config.training.get("ds_config") or {}).get("data_types") or {}
    accumulation = data_types.get("grad_accum_dtype") if isinstance(data_types, dict) else "unsupported"
    if accumulation == "fp32":
        return
    if accumulation is not None:
        raise ValueError(
            f"FP32 HF reference accumulation requires grad_accum_dtype='fp32'; got {accumulation!r}. "
            'Set data_types.grad_accum_dtype to "fp32" in a supported supplied configuration; '
            "admission does not change this setting."
        )
    if not automatically_owned_local:
        raise ValueError("omitted accumulation requires an automatically owned local DSS runtime; unverified defaults")

    _validate_engine_config(_owned_local_engine_config(config, automatically_owned_local=True))


def verify_runtime_node(cfg: dict, admission: dict, host: str, world_size: int, sp_size: int) -> dict:
    verify_final_runtime_request(cfg, admission, world_size, sp_size)
    node = next((node for node in admission["nodes"] if node["host"] == host), None)
    if node is None or (
        socket.gethostname() != node["hostname"]
        or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != node["boot_id"]
        or sys.executable != node["python"]
        or hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest() != node["python_sha256"]
    ):
        raise ValueError("actual runtime node or interpreter differs from admission")
    sources = _runtime_sources()
    if sources != admission["runtime_receipts"][host]["sources"]:
        raise ValueError("actual worker runtime sources or bound callables changed")
    from dss.ray_dss.jobs.gpu import training

    resolved = training._build_ds_config(cfg, world_size, sp_size)
    _validate_engine_config(resolved)
    if resolved != admission["runtime_receipts"][host]["resolved"]:
        raise ValueError("final worker engine configuration differs from the admitted request")
    return {
        "host": host,
        "job_id": cfg["job_id"],
        "payload_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "sources": sources,
        "resolved": resolved,
    }


def verify_worker_runtime(cfg: dict, admission: dict, ranks: list, world_size: int, sp_size: int) -> dict:
    """Recheck normal runtime imports and the final engine payload in the actual rank process."""
    rank = int(os.environ["RANK"])
    if not 0 <= rank < len(ranks):
        raise ValueError("actual worker rank is outside the admitted layout")
    host, gpu = ranks[rank]
    node = next((node for node in admission["nodes"] if node["host"] == host), None)
    if node is None or (
        os.environ.get("WORLD_SIZE") != str(world_size)
        or os.environ.get("DSS_RANK_HOST") != host
        or os.environ.get("DSS_RANK_GPU_ID") != str(gpu)
        or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu)
        or not 0 <= gpu < node["slots"]
    ):
        raise ValueError("actual worker rank or GPU differs from admission")
    return {"rank": rank, **verify_runtime_node(cfg, admission, host, world_size, sp_size)}


def runtime_request_from_config(cfg: dict) -> dict | None:
    return ((cfg.get("training_config") or {}).get("debug") or {}).get(RUNTIME_ADMISSION_KEY)


def _final_engine_payload(cfg: dict) -> dict:
    payload = deepcopy(cfg)
    training = payload.get("training_config") or {}
    debug = training.get("debug")
    if isinstance(debug, dict):
        debug.pop(RUNTIME_ADMISSION_KEY, None)
        if not debug:
            training.pop("debug", None)
    for name in ("master_addr", "master_port", "zone_id", "init_event_url", "worker_event_url", "init_status_url"):
        payload.pop(name, None)
    payload.pop("_sp_model_parallelism_metadata", None)
    return payload


def verify_final_runtime_request(cfg: dict, admission: dict, world_size: int, sp_size: int) -> None:
    request = admission.get("runtime_request") or {}
    if (
        request.get("n_gpus") != world_size
        or request.get("sp_size") != sp_size
        or cfg.get("job_id") != admission.get("training_child_id")
        or _final_engine_payload(cfg) != request.get("payload")
    ):
        raise ValueError("final native request differs from gateway admission")


@contextlib.contextmanager
def gateway_runtime_scope(gateway, hostfile: Path):
    """Bind final child requests before the gateway sends them to the zone."""
    raw_allocation = correctness_environment(RUNTIME_ADMISSION_ENV)
    allocation = json.loads(raw_allocation) if raw_allocation is not None else None
    if allocation is None:
        yield
        return
    original = gateway._create_job_v0

    async def create_job(zone, config, job_id):
        if config.training_config is None:
            return await original(zone, config, job_id)
        from dss.ray_dss.jobs.gpu import training
        from dss.ray_dss.stage_info import strip_stage_info_from_job_config

        ranks = [(node.host, gpu) for node in zone.nodes for gpu in node.gpu_ids]
        slots = {node["host"]: node["slots"] for node in allocation["nodes"]}
        if (
            config.job_id != job_id
            or len(ranks) != config.num_gpus
            or len(set(ranks)) != len(ranks)
            or any(host not in slots or not 0 <= gpu < slots[host] for host, gpu in ranks)
            or hashlib.sha256(hostfile.read_bytes()).hexdigest() != allocation["hostfile_sha256"]
        ):
            raise ValueError("final gateway child or placement differs from admitted allocation")
        cfg = strip_stage_info_from_job_config(config).to_engine_dict(job_id)
        if runtime_request_from_config(cfg) is not None:
            raise ValueError("runtime admission must be supplied by the owned gateway, not its client")
        request = {
            "payload": _final_engine_payload(cfg),
            "n_gpus": len(ranks),
            "sp_size": training.sp.sp_size_of(cfg.get("training_config") or {}),
        }
        receipts = await asyncio.to_thread(_allocation_runtime_receipts, request, hostfile, allocation["ssh"])
        for host, receipt in receipts.items():
            previous = allocation["runtime_receipts"][host]
            if any(receipt[key] != previous[key] for key in ("host", "boot_id", "python", "python_sha256", "sources")):
                raise ValueError("final gateway runtime differs from allocation admission")
        if hashlib.sha256(hostfile.read_bytes()).hexdigest() != allocation["hostfile_sha256"]:
            raise ValueError("allocation hostfile changed during final gateway admission")
        retained = {
            **deepcopy(allocation),
            "runtime_request": request,
            "runtime_receipts": receipts,
            "training_child_id": job_id,
            "zone_id": zone.zone_id,
            "ranks": ranks,
        }
        debug = deepcopy(config.training_config.debug or {})
        debug[RUNTIME_ADMISSION_KEY] = retained
        config.training_config.debug = debug
        print("CORRECTNESS_GATEWAY_RUNTIME " + json.dumps(retained, sort_keys=True), flush=True)
        return await original(zone, config, job_id)

    gateway._create_job_v0 = create_job
    try:
        yield
    finally:
        gateway._create_job_v0 = original


@contextlib.contextmanager
def runtime_engine_scope(receipt: dict):
    """Check the config passed to the bound initializer before DeepSpeed creation."""
    from dss.ray_dss.jobs.gpu import training

    namespace = training._DeepSpeedWorker.__init__.__globals__
    original = namespace["initialize_training_engine"]
    signature = inspect.signature(original)
    expected_builder = training._build_ds_config
    expected_spec = training.TrainingEngineSpec
    calls = 0

    @wraps(original)
    def initialize(*args, **kwargs):
        nonlocal calls
        calls += 1
        bound = signature.bind(*args, **kwargs)
        if (
            calls != 1
            or bound.arguments.get("rank") != receipt["rank"]
            or bound.arguments.get("ds_config") != receipt["resolved"]
            or training._build_ds_config is not expected_builder
            or training.TrainingEngineSpec is not expected_spec
            or namespace.get("initialize_training_engine") is not initialize
        ):
            raise ValueError("actual bound engine initialization differs from runtime admission")
        for name in _RUNTIME_MODULES:
            row = receipt["sources"][name]
            if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("runtime source changed before engine initialization")
        _validate_engine_config(bound.arguments["ds_config"])
        result = original(*args, **kwargs)
        if namespace.get("initialize_training_engine") is not initialize:
            raise ValueError("actual bound engine initializer changed")
        return result

    namespace["initialize_training_engine"] = initialize
    try:
        yield
        if calls != 1:
            raise ValueError("admitted worker did not invoke its bound engine initializer exactly once")
    finally:
        namespace["initialize_training_engine"] = original


def _validate_engine_config(resolved: dict) -> None:
    import torch
    from deepspeed.runtime.config import DtypeEnum
    from deepspeed.runtime.config import get_data_types_params
    from deepspeed.runtime.config import get_torch_autocast_enabled
    from deepspeed.runtime.constants import GRAD_ACCUM_DTYPE_DEFAULT
    from deepspeed.runtime.precision_config import get_bfloat16_config
    from deepspeed.runtime.precision_config import get_float16_config
    from deepspeed.runtime.zero.config import get_zero_config

    # These are the runtime's typed parsers, without initializing distributed
    # groups or validating batch sizes against an uninitialized process group.
    get_zero_config(resolved)
    if (
        not get_bfloat16_config(resolved).enabled
        or get_float16_config(resolved).enabled
        or get_torch_autocast_enabled(resolved)
    ):
        raise ValueError("runtime-default reference admission requires BF16 without FP16 or autocast")
    data_types = get_data_types_params(resolved)
    param_dtype = data_types.get("param_dtype")
    if param_dtype is not None and DtypeEnum(param_dtype).value != torch.bfloat16:
        raise ValueError("runtime-default reference admission requires BF16 model parameters")
    dtype = data_types.get("grad_accum_dtype", GRAD_ACCUM_DTYPE_DEFAULT)
    # DeepSpeedEngine.get_data_types falls back to the BF16 model dtype on None.
    accumulation = torch.bfloat16 if dtype is None else DtypeEnum(dtype).value
    if accumulation != torch.float32:
        raise ValueError(
            f"installed DSS shared engine resolves {accumulation} gradient accumulation, not FP32; "
            "the unchanged recipe is unsupported by this runtime"
        )
