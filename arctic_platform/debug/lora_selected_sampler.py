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

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Select the owned scoped receiver on each actual external engine worker."""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
from pathlib import Path

from arctic_platform.debug.lora_sampler_evidence import _bind_sources
from arctic_platform.debug.lora_sampler_evidence import _require
from arctic_platform.debug.lora_sampler_evidence import engine_kwargs_for_lora
from arctic_platform.debug.lora_sampler_evidence import serving_lora_request
from arctic_platform.debug.lora_sampler_worker import LoRASamplerEvidenceWorker
from arctic_platform.debug.lora_sampler_worker import _request
from arctic_platform.debug.rl_sampler_evidence import SamplerEvidenceWorker
from arctic_platform.debug.weight_sync_evidence import VERIFY_ENV
from arctic_platform.debug.weight_sync_evidence import enabled
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import verify_sources

PATH_ENV = "ARCTIC_PLATFORM_SELECTED_LORA_BOOTSTRAP"
SHA_ENV = "ARCTIC_PLATFORM_SELECTED_LORA_BOOTSTRAP_SHA256"
EXTENSION = "arctic_platform.debug.lora_selected_sampler.SelectedLoRAWeightSyncExtension"
RPC_PATH = Path(__file__).resolve().parents[2] / ".codex-run-state/job-scripts/job3_lora_selected_rpc.py"
RPC_SHA = "0ad85782bd2b4405b713635c6bfc036acfd5f2ff1977bf1d38c4b7a0bb394230"


def _bytes(path):
    path = Path(path)
    _require(
        path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path,
        "canonical regular bootstrap/source path required",
    )

    def identity(info):
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    before = path.stat()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read()
        closed = os.fstat(stream.fileno())
    _require(
        identity(before) == identity(opened) == identity(closed) == identity(path.stat()),
        "bootstrap/source changed while reading",
    )
    return raw


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate bootstrap JSON key")
        result[key] = value
    return result


def bootstrap_descriptor(pointer):
    """Read caller-staged bytes, not a live singleton or an inferred import path."""
    _require(
        type(pointer) is dict
        and set(pointer) == {"path", "sha256"}
        and type(pointer["sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", pointer["sha256"]) is not None,
        "exact bootstrap path/digest required",
    )
    raw = _bytes(pointer["path"])
    _require(hashlib.sha256(raw).hexdigest() == pointer["sha256"], "bootstrap digest changed")
    body = json.loads(raw, object_pairs_hook=_unique_object)
    _require(
        type(body) is dict
        and set(body) == {"schema", "receiver", "prepare", "admitted_sources"}
        and body["schema"] == "job3-selected-lora-bootstrap-v1",
        "exact selected bootstrap schema required",
    )
    sources = body["admitted_sources"]
    _require(
        type(sources) is dict and sources,
        "actual nonempty bootstrap source closure required",
    )
    verify_sources(sources)
    observed = source_files(bootstrap_descriptor, LoRASamplerEvidenceWorker, engine_kwargs_for_lora)
    _require(
        observed.items() <= sources.items() and sources.get(str(RPC_PATH)) == RPC_SHA,
        "bootstrap, frozen worker and scoped RPC sources must be admitted",
    )
    for key in ("receiver", "prepare"):
        item = body[key]
        _require(
            type(item) is dict
            and set(item) == {"module", "name", "source"}
            and type(item["module"]) is str
            and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", item["module"]) is not None
            and type(item["name"]) is str
            and re.fullmatch(r"[A-Za-z_]\w*", item["name"]) is not None
            and type(item["source"]) is str
            and item["source"] in sources,
            "explicit top-level selected symbol and admitted source required",
        )
    _require(_bytes(pointer["path"]) == raw, "bootstrap changed during source verification")
    return body


def _symbol(item, sources):
    verify_sources(sources)
    module = importlib.import_module(item["module"])
    _require(
        inspect.getsourcefile(module) == item["source"],
        "selected module resolved to a different source",
    )
    # Avoid module __getattr__ selecting an unrecorded replacement.
    value = vars(module).get(item["name"])
    _require(
        value is not None and inspect.getsourcefile(value) == item["source"],
        "selected symbol source changed",
    )
    verify_sources(sources)
    return value


def _rpc_factory(sources):
    raw = _bytes(RPC_PATH)
    _require(hashlib.sha256(raw).hexdigest() == RPC_SHA, "frozen selected RPC changed")
    verify_sources(sources)
    key = "_job3_selected_sampler_rpc"
    spec = importlib.util.spec_from_file_location(key, RPC_PATH)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(key)
    sys.modules[key] = module
    try:
        exec(compile(raw, str(RPC_PATH), "exec", dont_inherit=True), module.__dict__)
        _require(_bytes(RPC_PATH) == raw, "selected RPC changed during loading")
        verify_sources(sources)
    except BaseException:
        if previous is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = previous
        raise
    return module.make_selected_receiver_extension


def selected_receiver_extension(pointer):
    """Resolve the supplied native class without constructing a model/receiver."""
    _require(enabled(), "selected receiver bootstrap requires explicit opt-in")
    body = bootstrap_descriptor(pointer)
    pointer = dict(pointer)
    sources = body["admitted_sources"]
    receiver = _symbol(body["receiver"], sources)
    prepare = _symbol(body["prepare"], sources)
    _require(
        isinstance(receiver, type) and inspect.isfunction(prepare),
        "actual receiver class and function required",
    )
    scoped = _rpc_factory(sources)(receiver, prepare=prepare, admitted_sources=sources)

    class SelectedLoRAWeightSyncExtension(scoped):
        def sync_lora_weights(self, *args, **kwargs):
            try:
                _require(
                    bootstrap_descriptor(pointer) == body,
                    "selected bootstrap changed before RPC",
                )
            except BaseException as error:
                try:
                    self.close_weight_sync()
                except BaseException as cleanup_error:
                    error.add_note("selected bootstrap cleanup also failed: " + repr(cleanup_error))
                raise
            # The frozen RPC owns its failures, including non-destructive
            # rejection of a duplicate transfer while its first scope is live.
            result = super().sync_lora_weights(*args, **kwargs)
            try:
                _require(
                    bootstrap_descriptor(pointer) == body,
                    "selected bootstrap changed during RPC",
                )
                return {
                    **result,
                    "selected_bootstrap_sha256": pointer["sha256"],
                    "selected_bootstrap_runtime_admitted": False,
                }
            except BaseException as error:
                try:
                    self.close_weight_sync()
                except BaseException as cleanup_error:
                    error.add_note("selected bootstrap cleanup also failed: " + repr(cleanup_error))
                raise

    return SelectedLoRAWeightSyncExtension


def __getattr__(name):
    if name != "SelectedLoRAWeightSyncExtension":
        raise AttributeError(name)
    _require(
        PATH_ENV in os.environ and SHA_ENV in os.environ,
        "actual engine worker lacks the owned bootstrap",
    )
    return selected_receiver_extension({"path": os.environ[PATH_ENV], "sha256": os.environ[SHA_ENV]})


def selected_engine_kwargs(engine_kwargs, *, rank):
    supplied = dict(engine_kwargs)
    if supplied.get("worker_extension_cls") == EXTENSION:
        supplied["worker_extension_cls"] = None
    kwargs = engine_kwargs_for_lora(supplied, rank=rank)
    kwargs["worker_extension_cls"] = EXTENSION
    return kwargs


def make_selected_lora_replica_pool(pool_class, request, pointer):
    _require(enabled(), "selected sampler requires explicit evidence opt-in")
    request = _request(request)
    body = bootstrap_descriptor(pointer)
    pointer = dict(pointer)
    _require(
        body["admitted_sources"] == request["admitted_sources"],
        "request and bootstrap closures differ",
    )
    _bind_sources(
        request["admitted_sources"],
        pool_class,
        SelectedLoRASamplerWorker,
        make_selected_lora_replica_pool,
    )
    import ray

    class OwnedSelectedLoRAWorker(SelectedLoRASamplerWorker):
        def __init__(self):
            super().__init__(request, pointer)

    settings = {VERIFY_ENV: "1", PATH_ENV: pointer["path"], SHA_ENV: pointer["sha256"]}
    actor = ray.remote(runtime_env={"env_vars": settings})(OwnedSelectedLoRAWorker)
    return functools.partial(pool_class, worker_cls=actor)


class SelectedLoRASamplerWorker(LoRASamplerEvidenceWorker):
    """Preserve serving/cleanup; require actual TP replies from the selected bridge."""

    def __init__(self, request, pointer):
        super().__init__(request)
        body = bootstrap_descriptor(pointer)
        _require(
            body["admitted_sources"] == self._request["admitted_sources"],
            "bootstrap/request closure mismatch",
        )
        self._bootstrap_pointer = dict(pointer)
        self._bootstrap_environment = None

    def _restore_bootstrap_environment(self):
        previous = self._bootstrap_environment
        self._bootstrap_environment = None
        if previous is not None:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    async def initialize(self, engine_kwargs, extra_env=None):
        _require(
            self.llm is None and self._bootstrap_environment is None,
            "selected sampler already initialized",
        )
        settings = {
            PATH_ENV: self._bootstrap_pointer["path"],
            SHA_ENV: self._bootstrap_pointer["sha256"],
        }
        _require(
            all(os.environ.get(key) in (None, value) for key, value in settings.items())
            and (not extra_env or all(extra_env.get(key, value) == value for key, value in settings.items())),
            "another owner's bootstrap environment cannot be replaced",
        )
        self._bootstrap_environment = {key: os.environ.get(key) for key in settings}
        try:
            if extra_env:
                os.environ.update(extra_env)
            os.environ.update(settings)
            _require(enabled(), "selected sampler opt-in must reach the worker")
            bootstrap_descriptor(self._bootstrap_pointer)
            import vllm.plugins
            from arctic_inference.envs import arctic_inference_effective_enabled
            from arctic_inference.server.metrics import RingStatLogger

            vllm.plugins.load_general_plugins()
            from vllm.v1.engine.async_llm import AsyncLLM

            if arctic_inference_effective_enabled():
                from arctic_inference.vllm.args import ArcticAsyncEngineArgs as EngineArgs
            else:
                from vllm.engine.arg_utils import AsyncEngineArgs as EngineArgs
            kwargs = selected_engine_kwargs(engine_kwargs, rank=self._request["lora_config"]["r"])
            if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
                kwargs.setdefault("attention_config", {"backend": "FLASH_ATTN"})
            args = EngineArgs(**kwargs)
            _require(
                type(args.tensor_parallel_size) is int and args.tensor_parallel_size == 8,
                "actual SAMPLE8 required",
            )
            self._tp = args.tensor_parallel_size
            self._origins = _bind_sources(
                self._request["admitted_sources"],
                type(self),
                SamplerEvidenceWorker,
                EngineArgs,
                args.create_engine_config,
                AsyncLLM.from_vllm_config,
                AsyncLLM.collective_rpc,
                AsyncLLM.generate,
                AsyncLLM.abort,
                AsyncLLM.shutdown,
                serving_lora_request,
            )
            from arctic_platform.debug.lora_sync_hooks import maybe_admit_sampler_construction
            from arctic_platform.debug.lora_sync_hooks import maybe_verify_sampler_construction

            maybe_admit_sampler_construction(self, args, AsyncLLM.from_vllm_config, engine_kwargs)
            self.llm = AsyncLLM.from_vllm_config(args.create_engine_config(), stat_loggers=[RingStatLogger])
            maybe_verify_sampler_construction(self)
            verify_sources(self._origins)
            bootstrap_descriptor(self._bootstrap_pointer)
            self.state = "ready"
        except BaseException as error:
            try:
                if self.llm is not None:
                    self.shutdown()
            except BaseException as cleanup_error:
                error.add_note("selected sampler shutdown also failed: " + repr(cleanup_error))
            finally:
                self._restore_bootstrap_environment()
            raise

    async def sync_weights(self, *args, **kwargs):
        try:
            bootstrap_descriptor(self._bootstrap_pointer)
            result = await super().sync_weights(*args, **kwargs)
            rows = result["rank_results"]
            _require(
                len(rows) == 8
                and all(
                    row.get("selected_bootstrap_sha256") == self._bootstrap_pointer["sha256"]
                    and row.get("selected_bootstrap_runtime_admitted") is False
                    and (kwargs.get("engine_only") is True or row.get("selected_scope_retained_for_serving") is True)
                    for row in rows
                ),
                "every actual TP rank must report the same selected bootstrap",
            )
            bootstrap_descriptor(self._bootstrap_pointer)
            return result
        except BaseException as error:
            try:
                await self.close_weight_sync()
            except BaseException as cleanup_error:
                error.add_note("selected sampler transport cleanup also failed: " + repr(cleanup_error))
            raise

    def shutdown(self):
        try:
            super().shutdown()
        finally:
            self._restore_bootstrap_environment()
