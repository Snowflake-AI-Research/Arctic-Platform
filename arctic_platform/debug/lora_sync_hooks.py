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

"""Explicit, pinned LoRA protocol selection at existing native call boundaries."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from contextlib import nullcontext
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from types import CodeType
from types import MethodType

from arctic_platform.debug.lora_sampler_evidence import _bind_sources
from arctic_platform.debug.lora_sampler_evidence import _require
from arctic_platform.debug.lora_training_protocol import prepare_lora_training_workers
from arctic_platform.debug.lora_training_protocol import prepare_lora_worker
from arctic_platform.debug.lora_training_protocol import release_lora_worker
from arctic_platform.debug.lora_training_protocol import send_lora_worker
from arctic_platform.debug.weight_sync_evidence import VERIFY_ENV
from arctic_platform.debug.weight_sync_evidence import enabled

REQUEST_ENV = "ARCTIC_PLATFORM_LORA_SYNC_REQUEST"
DIGEST_ENV = "ARCTIC_PLATFORM_LORA_SYNC_REQUEST_SHA256"
INITIALIZATION_ENV = "ARCTIC_PLATFORM_INITIALIZATION_REQUEST"
INITIALIZATION_SHA_ENV = "ARCTIC_PLATFORM_INITIALIZATION_REQUEST_SHA256"
MODEL_CONSTRUCTION_ENV = "ARCTIC_PLATFORM_MODEL_CONSTRUCTION_REQUEST"
MODEL_CONSTRUCTION_SHA_ENV = "ARCTIC_PLATFORM_MODEL_CONSTRUCTION_REQUEST_SHA256"
_MODEL_CONSTRUCTOR_FRAME = ContextVar("arctic_platform_owned_model_constructor_frame", default=None)


def _initialization_request(
    *,
    environment=None,
    path_env=INITIALIZATION_ENV,
    digest_env=INITIALIZATION_SHA_ENV,
    schema="explicit-native-initialization-request-v1",
):
    settings = os.environ if environment is None else environment
    path, digest = settings.get(path_env), settings.get(digest_env)
    if path is None and digest is None:
        return None
    _require(
        settings.get(VERIFY_ENV) == "1",
        "initialization request requires explicit evidence opt-in",
    )
    _require(
        type(path) is str
        and Path(path).is_relative_to("/data-fast")
        and type(digest) is str
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        "selected initialization requires an explicit FAST request and SHA256",
    )
    from arctic_platform.debug.lora_selected_sampler import _bytes
    from arctic_platform.debug.lora_selected_sampler import _unique_object

    raw = _bytes(path)
    _require(hashlib.sha256(raw).hexdigest() == digest, "initialization request changed")
    request = json.loads(raw, object_pairs_hook=_unique_object)
    fields = {"schema", "admitted_sources", "runtime", "runtime_options", "verify_capacity", "capacity_request"}
    if schema == "explicit-native-model-construction-request-v1":
        fields.add("factory_route")
    _require(
        type(request) is dict and set(request) == fields and request["schema"] == schema,
        "exact source-bound initialization request required",
    )
    sources = request["admitted_sources"]
    _require(type(sources) is dict and sources, "initialization requires actual source origins")
    for source, expected in sources.items():
        _require(
            type(source) is str
            and type(expected) is str
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected)
            and hashlib.sha256(_bytes(source)).hexdigest() == expected,
            "initialization source identity differs",
        )
    for key in ("runtime", "verify_capacity"):
        item = request[key]
        _require(
            type(item) is dict
            and set(item) == {"module", "name", "source"}
            and type(item["module"]) is str
            and all(part.isidentifier() for part in item["module"].split("."))
            and type(item["name"]) is str
            and item["name"].isidentifier()
            and item["source"] in sources,
            "explicit admitted initialization provider required",
        )
    options = request["runtime_options"]
    _require(
        type(options) is dict
        and set(options) == {"release_bound", "wrapper_sources"}
        and type(options["release_bound"]) is dict
        and set(options["release_bound"]) == {"path", "sha256"}
        and type(options["release_bound"]["path"]) is str
        and type(options["release_bound"]["sha256"]) is str
        and sources.get(options["release_bound"]["path"]) == options["release_bound"]["sha256"]
        and type(options["wrapper_sources"]) is dict
        and options["wrapper_sources"].items() <= sources.items()
        and type(request["capacity_request"]) is dict
        and request["capacity_request"],
        "explicit existing runtime release and complete capacity supplier request required",
    )
    _bind_sources(sources, _initialization_request, _bytes, _unique_object)
    _require(_bytes(path) == raw, "initialization request changed during admission")
    return {"path": path, "sha256": digest}, request


def _initialization_code(actual, sources):
    from arctic_platform.debug.lora_selected_sampler import _bytes

    path = actual.co_filename
    _require(path in sources, "initialization callable origin is not admitted")
    code = compile(_bytes(path), path, "exec", dont_inherit=True, optimize=sys.flags.optimize)
    pending, matches = [code], []
    while pending:
        current = pending.pop()
        if current.co_qualname == actual.co_qualname:
            matches.append(current)
        pending.extend(value for value in current.co_consts if isinstance(value, CodeType))
    _require(
        len(matches) == 1 and actual == matches[0] and actual.co_linetable == matches[0].co_linetable,
        "loaded initialization callable differs from admitted source",
    )


def _initialization_function(function, sources):
    _require(inspect.isfunction(function), "source-derived initialization Python callable required")
    _require(
        function.__code__.co_filename == inspect.getsourcefile(function)
        and function.__code__.co_qualname == function.__qualname__,
        "loaded initialization callable differs from admitted source",
    )
    _initialization_code(function.__code__, sources)


def _initialization_slot(selected, subject, *surfaces, request_reader=_initialization_request):
    from arctic_platform.debug.lora_selected_sampler import _symbol

    pointer, request = selected
    sources = request["admitted_sources"]
    runtime_class = _symbol(request["runtime"], sources)
    capacity = _symbol(request["verify_capacity"], sources)
    _require(
        inspect.isclass(runtime_class) and inspect.isfunction(capacity) and capacity.__closure__ is None,
        "explicit native runtime class and top-level capacity verifier required",
    )
    _bind_sources(
        sources,
        _initialization_slot,
        _symbol,
        runtime_class,
        runtime_class.guard,
        runtime_class.verify_engine,
        capacity,
        request_reader,
        *surfaces,
    )
    for function in (capacity, runtime_class.__init__, runtime_class.guard, runtime_class.verify_engine):
        _initialization_function(function, sources)
    options = request["runtime_options"]
    runtime = runtime_class(deepcopy(options["release_bound"]), wrapper_sources=deepcopy(options["wrapper_sources"]))
    slot = {
        "pointer": pointer,
        "request": request,
        "request_reader": request_reader,
        "runtime_class": runtime_class,
        "runtime": runtime,
        "capacity": capacity,
        "capacity_code": capacity.__code__,
        "capacity_defaults": deepcopy((capacity.__defaults__, capacity.__kwdefaults__)),
        "subject": subject,
        "status": "armed",
        "observations": [],
        "runtime_methods": {
            name: (
                getattr(runtime_class, name),
                getattr(runtime_class, name).__code__,
                deepcopy((getattr(runtime_class, name).__defaults__, getattr(runtime_class, name).__kwdefaults__)),
            )
            for name in ("guard", "verify_engine")
        },
    }
    _guard_initialization(slot)
    return slot


def _guard_initialization(slot):
    from arctic_platform.debug.lora_selected_sampler import _symbol

    _require(
        slot["request_reader"]() == (slot["pointer"], slot["request"]),
        "armed initialization request or source closure changed",
    )
    request = slot["request"]
    _require(
        _symbol(request["runtime"], request["admitted_sources"]) is slot["runtime_class"]
        and _symbol(request["verify_capacity"], request["admitted_sources"]) is slot["capacity"]
        and slot["capacity"].__code__ is slot["capacity_code"],
        "captured initialization provider changed",
    )
    _require(
        (slot["capacity"].__defaults__, slot["capacity"].__kwdefaults__) == slot["capacity_defaults"],
        "captured capacity verifier defaults changed",
    )
    _require_initialization_runtime_methods(slot)
    slot["runtime"].guard()
    _require_initialization_runtime_methods(slot)


def _require_initialization_runtime_methods(slot):
    for name, (function, code, defaults) in slot["runtime_methods"].items():
        method = getattr(slot["runtime"], name)
        _require(
            inspect.ismethod(method)
            and method.__self__ is slot["runtime"]
            and method.__func__ is function
            and function.__code__ is code,
            "captured runtime verification method changed",
        )
        _require(
            (function.__defaults__, function.__kwdefaults__) == defaults,
            "captured runtime verification method defaults changed",
        )


def _verify_initialization_capacity(slot, phase, subject):
    _guard_initialization(slot)
    request = deepcopy(slot["request"]["capacity_request"])
    result = slot["capacity"](request, phase=phase, subject=dict(subject), runtime=slot["runtime"])
    _require(
        type(result) is dict
        and set(result) == {"phase", "request", "evidence"}
        and result["phase"] == phase
        and request == result["request"] == slot["request"]["capacity_request"]
        and type(result["evidence"]) is dict
        and result["evidence"],
        "actual phase-bound capacity supplier evidence required",
    )
    _guard_initialization(slot)
    observation = {
        "schema": "native-initialization-hook-observation-v1",
        "phase": phase,
        "provider_evidence": deepcopy(result["evidence"]),
        "capacity_admitted": False,
        "numerical": None,
    }
    slot["observations"].append(observation)
    return observation


def _model_construction_request():
    return _initialization_request(
        path_env=MODEL_CONSTRUCTION_ENV,
        digest_env=MODEL_CONSTRUCTION_SHA_ENV,
        schema="explicit-native-model-construction-request-v1",
    )


def _model_construction_route(frames, route, sources):
    _require(
        type(route) is dict and set(route) == {"factory", "caller", "constructor"},
        "explicit factory, loader and constructor route required",
    )
    for name, frame in zip(("factory", "caller", "constructor"), frames, strict=True):
        descriptor = route[name]
        _require(
            type(descriptor) is dict
            and set(descriptor) == {"module", "qualname", "source"}
            and frame is not None
            and descriptor
            == {
                "module": frame.f_globals.get("__name__"),
                "qualname": frame.f_code.co_qualname,
                "source": frame.f_code.co_filename,
            },
            "actual model construction route differs from selected caller",
        )
        _initialization_code(frame.f_code, sources)


def _guard_model_constructor_frame(binding):
    owner_class, original, wrapper = binding["class"], binding["original"], binding["wrapper"]
    _require(
        vars(owner_class).get("__init__") is wrapper
        and original.__code__ is binding["original_code"]
        and wrapper.__code__ is binding["wrapper_code"]
        and getattr(wrapper, "__wrapped__", None) is original,
        "owned native constructor binding changed",
    )
    closure = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__ or (), strict=True))
    _require(
        "original" in closure
        and "worker_class" in closure
        and closure["original"].cell_contents is original
        and closure["worker_class"].cell_contents is owner_class,
        "owned native constructor closure changed",
    )
    caller = binding["frame"]
    _require(
        caller.f_code is binding["wrapper_code"]
        and caller.f_locals.get("worker") is binding["worker"]
        and caller.f_locals.get("original") is original
        and caller.f_locals.get("worker_class") is owner_class,
        "actual owned native wrapper frame required",
    )
    current = sys.modules.get(owner_class.__module__)
    for component in owner_class.__qualname__.split("."):
        _require(current is not None and component in vars(current), "actual native class origin required")
        current = vars(current)[component]
    _require(current is owner_class, "native module no longer owns the constructor class")
    _require(
        (original.__defaults__, original.__kwdefaults__, wrapper.__defaults__, wrapper.__kwdefaults__)
        == binding["defaults"],
        "owned native constructor defaults changed",
    )
    _require(
        _model_construction_request() == binding["selected"],
        "owned native constructor request or sources changed",
    )
    sources = binding["selected"][1]["admitted_sources"]
    _bind_sources(
        sources,
        owned_model_constructor_frame,
        _OwnedModelConstructorFrame,
        _guard_model_constructor_frame,
        _model_constructor_binding,
        owner_class,
        original,
        wrapper,
    )
    _initialization_function(original, sources)
    _initialization_code(wrapper.__code__, sources)


class _OwnedModelConstructorFrame:
    def __init__(self, binding):
        self.binding = binding
        self.token = None

    def __enter__(self):
        _require(
            self.token is None and _MODEL_CONSTRUCTOR_FRAME.get() is None,
            "nested or reused native constructor ownership rejected",
        )
        _guard_model_constructor_frame(self.binding)
        self.token = _MODEL_CONSTRUCTOR_FRAME.set(self.binding)
        self.binding["active"] = True

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None:
                _guard_model_constructor_frame(self.binding)
        finally:
            self.binding["active"] = False
            _MODEL_CONSTRUCTOR_FRAME.reset(self.token)
            self.binding["frame"] = None


def owned_model_constructor_frame(worker, *, constructor, wrapper, admitted_sources):
    """Bind the exact owning wrapper frame while it calls its native constructor."""
    selected = _model_construction_request()
    if selected is None:
        return nullcontext()
    owner_class = type(worker)
    _require(
        type(owner_class) is type
        and inspect.isfunction(constructor)
        and inspect.isfunction(wrapper)
        and constructor.__qualname__ == owner_class.__qualname__ + ".__init__"
        and constructor.__module__ == owner_class.__module__
        and type(admitted_sources) is dict
        and admitted_sources == selected[1]["admitted_sources"],
        "exact native constructor and selected source closure required",
    )
    frame = inspect.currentframe()
    try:
        binding = {
            "class": owner_class,
            "worker": worker,
            "original": constructor,
            "original_code": constructor.__code__,
            "wrapper": wrapper,
            "wrapper_code": wrapper.__code__,
            "frame": frame.f_back,
            "selected": selected,
            "defaults": deepcopy(
                (constructor.__defaults__, constructor.__kwdefaults__, wrapper.__defaults__, wrapper.__kwdefaults__)
            ),
            "active": False,
        }
        _guard_model_constructor_frame(binding)
        return _OwnedModelConstructorFrame(binding)
    finally:
        del frame


def _model_constructor_binding(owner, constructor, sources):
    installed = type(owner).__init__
    _require(inspect.isfunction(installed), "actual native constructor owner differs from selected code")
    if installed.__code__ is constructor.f_code:
        _initialization_function(installed, sources)
        return installed, installed, None
    binding = _MODEL_CONSTRUCTOR_FRAME.get()
    _require(
        binding is not None
        and binding["active"]
        and binding["worker"] is owner
        and binding["class"] is type(owner)
        and binding["original_code"] is constructor.f_code
        and constructor.f_back is binding["frame"]
        and sources == binding["selected"][1]["admitted_sources"],
        "actual native constructor owner differs from selected code",
    )
    _guard_model_constructor_frame(binding)
    return binding["original"], binding["wrapper"], binding


def maybe_admit_model_construction(spec, parallel_groups):
    selected = _model_construction_request()
    if selected is None:
        return None
    from arctic_platform.model.config import ModelSpec

    frame = inspect.currentframe()
    factory = frame.f_back
    caller = factory.f_back if factory is not None else None
    constructor = caller.f_back if caller is not None else None
    try:
        sources = selected[1]["admitted_sources"]
        frames = (factory, caller, constructor)
        route = selected[1]["factory_route"]
        _model_construction_route(frames, route, sources)
        _require(type(spec) is ModelSpec, "actual native ModelSpec required before model construction")
        _require(parallel_groups is None or type(parallel_groups) is dict, "actual selected parallel groups required")
        owner, config = constructor.f_locals.get("self"), constructor.f_locals.get("cfg")
        _require(owner is not None and type(config) is dict, "actual native constructor owner and cfg required")
        constructor_binding = _model_constructor_binding(owner, constructor, sources)
        build_model = factory.f_globals.get("build_model")
        _require(
            inspect.isfunction(build_model) and build_model.__code__ is factory.f_code,
            "actual native build_model callable required",
        )
        spec_before, config_before = deepcopy(spec), deepcopy(config)
        groups_before = None if parallel_groups is None else dict(parallel_groups)
        subject = {
            "worker": owner,
            "config": config,
            "model_spec": spec,
            "parallel_groups": parallel_groups,
            "build_model": build_model,
            "factory_route": deepcopy(route),
        }
        slot = _initialization_slot(
            selected,
            subject,
            maybe_admit_model_construction,
            _model_construction_route,
            _model_constructor_binding,
            ModelSpec,
            type(owner),
            *constructor_binding[:2],
            build_model,
            request_reader=_model_construction_request,
        )
        result = _verify_initialization_capacity(slot, "before-build-model", subject)
        _model_construction_route(frames, route, sources)
        _require(spec == spec_before and config == config_before, "capacity verifier changed native spec or cfg")
        _require(
            (parallel_groups is None and groups_before is None)
            or (
                type(parallel_groups) is dict
                and parallel_groups.keys() == groups_before.keys()
                and all(parallel_groups[key] is value for key, value in groups_before.items())
            ),
            "capacity verifier changed native parallel groups",
        )
        _require(
            factory.f_globals.get("build_model") is build_model
            and build_model.__code__ is factory.f_code
            and all(
                actual is captured
                for actual, captured in zip(
                    _model_constructor_binding(owner, constructor, sources),
                    constructor_binding,
                    strict=True,
                )
            ),
            "capacity verifier replaced native factory or constructor code",
        )
        return result
    finally:
        del frame, factory, caller, constructor


def maybe_observe_model_materialization(model, adapter, config, parallel_dims, ep_mesh):
    """Supply the actual post-sharding model before destination allocation."""
    selected = _model_construction_request()
    if selected is None:
        return None
    from arctic_platform.model.config import ModelSpec
    from arctic_platform.model.implementations.debug.gpu_memory import observe_native_model_inventory

    frame = inspect.currentframe()
    load_caller = frame.f_back
    factory = caller = constructor = None
    try:
        sources = selected[1]["admitted_sources"]
        descriptor = selected[1]["factory_route"]["factory"]
        active = load_caller
        matches = []
        while active is not None:
            if descriptor == {
                "module": active.f_globals.get("__name__"),
                "qualname": active.f_code.co_qualname,
                "source": active.f_code.co_filename,
            }:
                matches.append(active)
            active = active.f_back
        _require(len(matches) == 1, "one active owning model factory required before materialization")
        factory = matches[0]
        caller = factory.f_back
        constructor = caller.f_back if caller is not None else None
        frames = (factory, caller, constructor)
        _model_construction_route(frames, selected[1]["factory_route"], sources)
        _initialization_code(load_caller.f_code, sources)
        owner, native_config = constructor.f_locals.get("self"), constructor.f_locals.get("cfg")
        spec, groups = factory.f_locals.get("spec"), factory.f_locals.get("parallel_groups")
        _require(
            owner is not None and type(native_config) is dict and type(spec) is ModelSpec,
            "actual native constructor owner, cfg and ModelSpec required before materialization",
        )
        _require(groups is None or type(groups) is dict, "actual selected parallel groups required")
        binding = _model_constructor_binding(owner, constructor, sources)
        _require(
            load_caller.f_locals.get("model") is model
            and load_caller.f_locals.get("adapter") is adapter
            and load_caller.f_locals.get("config") is config
            and load_caller.f_locals.get("parallel_dims") is parallel_dims
            and load_caller.f_locals.get("ep_mesh") is ep_mesh,
            "actual loader model, adapter, config and placement required",
        )
        load = adapter.load_dcp_from_hf
        _initialization_function(load, sources)
        load_code, load_defaults = load.__code__, deepcopy((load.__defaults__, load.__kwdefaults__))
        build_model = factory.f_globals.get("build_model")
        _require(
            inspect.isfunction(build_model) and build_model.__code__ is factory.f_code,
            "actual native build_model callable required",
        )
        _bind_sources(
            sources, maybe_observe_model_materialization, observe_native_model_inventory, type(model), ModelSpec
        )
        _initialization_function(observe_native_model_inventory, sources)
        retained = []
        inventory = observe_native_model_inventory(model, retained_objects=retained)
        before = deepcopy((adapter, config, parallel_dims, spec, native_config))
        groups_before = None if groups is None else dict(groups)
        subject = {
            "worker": owner,
            "model": model,
            "model_spec": spec,
            "native_config": native_config,
            "parallel_groups": groups,
            "adapter": adapter,
            "config": config,
            "parallel_dims": parallel_dims,
            "ep_mesh": ep_mesh,
            "native_inventory": deepcopy(inventory),
            "build_model": build_model,
            "factory_route": deepcopy(selected[1]["factory_route"]),
        }
        slot = _initialization_slot(
            selected,
            subject,
            maybe_observe_model_materialization,
            observe_native_model_inventory,
            type(model),
            ModelSpec,
            build_model,
            load,
            *binding[:2],
            request_reader=_model_construction_request,
        )
        result = _verify_initialization_capacity(slot, "before-model-materialization", subject)
        _model_construction_route(frames, selected[1]["factory_route"], sources)
        _require(
            (adapter, config, parallel_dims, spec, native_config) == before,
            "capacity verifier changed the selected loading recipe",
        )
        _require(
            factory.f_globals.get("build_model") is build_model
            and build_model.__code__ is factory.f_code
            and factory.f_locals.get("spec") is spec
            and factory.f_locals.get("parallel_groups") is groups
            and constructor.f_locals.get("cfg") is native_config
            and constructor.f_locals.get("self") is owner
            and all(
                load_caller.f_locals.get(name) is value
                for name, value in (
                    ("model", model),
                    ("adapter", adapter),
                    ("config", config),
                    ("parallel_dims", parallel_dims),
                    ("ep_mesh", ep_mesh),
                )
            ),
            "capacity verifier changed the active model factory or loader inputs",
        )
        _require(
            (groups is None and groups_before is None)
            or (
                type(groups) is dict
                and groups.keys() == groups_before.keys()
                and all(groups[key] is value for key, value in groups_before.items())
            ),
            "capacity verifier changed native parallel groups",
        )
        _require(
            adapter.load_dcp_from_hf is load
            and load.__code__ is load_code
            and (load.__defaults__, load.__kwdefaults__) == load_defaults,
            "capacity verifier changed the selected checkpoint loader",
        )
        _require(
            observe_native_model_inventory(model, retained_objects=retained) == inventory,
            "capacity verifier changed native parameters, buffers or storage",
        )
        _require(
            all(
                actual is captured
                for actual, captured in zip(
                    _model_constructor_binding(owner, constructor, sources),
                    binding,
                    strict=True,
                )
            ),
            "capacity verifier changed the owning native constructor",
        )
        _bind_sources(sources, maybe_observe_model_materialization, observe_native_model_inventory, type(model), load)
        _initialization_function(load, sources)
        _initialization_function(observe_native_model_inventory, sources)
        return result
    finally:
        del frame, load_caller, factory, caller, constructor


def maybe_observe_checkpoint_reader(reader, model, state_dict, config, parallel_dims):
    """Observe one selected instance's finalized plan before its original reads."""
    selected = _model_construction_request()
    if selected is None:
        return reader
    from arctic_platform.model.implementations.debug.gpu_memory import observe_native_model_inventory

    frame = inspect.currentframe()
    loader = frame.f_back
    sources = selected[1]["admitted_sources"]
    try:
        _initialization_code(loader.f_code, sources)
        _require(
            all(
                loader.f_locals.get(name) is value
                for name, value in (
                    ("model", model),
                    ("state_dict", state_dict),
                    ("config", config),
                    ("parallel_dims", parallel_dims),
                )
            ),
            "actual checkpoint builder inputs required",
        )
        _require(type(state_dict) is dict and state_dict, "actual filtered checkpoint destinations required")
        original = reader.read_data
        _require(
            inspect.ismethod(original)
            and original.__self__ is reader
            and "read_data" not in vars(reader)
            and original.__func__ is type(reader).read_data,
            "unwrapped selected reader instance required",
        )
        _initialization_function(original.__func__, sources)
        function, code = original.__func__, original.__func__.__code__
        defaults = deepcopy((function.__defaults__, function.__kwdefaults__))
        _bind_sources(sources, maybe_observe_checkpoint_reader, observe_native_model_inventory, function)
        descriptor = selected[1]["factory_route"]["factory"]
        active, matches = loader, []
        while active is not None:
            if descriptor == {
                "module": active.f_globals.get("__name__"),
                "qualname": active.f_code.co_qualname,
                "source": active.f_code.co_filename,
            }:
                matches.append(active)
            active = active.f_back
        _require(len(matches) == 1, "one active owning model factory required for checkpoint reader")
        factory = matches[0]
        caller = factory.f_back
        constructor = caller.f_back if caller is not None else None
        frames = (factory, caller, constructor)
        _model_construction_route(frames, selected[1]["factory_route"], sources)
        owner = constructor.f_locals.get("self")
        binding = _model_constructor_binding(owner, constructor, sources)
        spec, groups = factory.f_locals.get("spec"), factory.f_locals.get("parallel_groups")
        native_config = constructor.f_locals.get("cfg")
        build_model = factory.f_globals.get("build_model")
        _require(
            inspect.isfunction(build_model) and build_model.__code__ is factory.f_code,
            "actual native build_model callable required for checkpoint reader",
        )
        recipe = deepcopy((config, parallel_dims, spec, native_config))
        group_items = None if groups is None else tuple(groups.items())
        state_items = tuple(state_dict.items())
        cache_path, threads = reader.path, reader.thread_count
        status = "armed"

        def guard():
            _require(_model_construction_request() == selected, "selected checkpoint request changed")
            _model_construction_route(frames, selected[1]["factory_route"], sources)
            current = inspect.currentframe()
            active = current.f_back
            try:
                while active is not None and active is not loader:
                    active = active.f_back
                _require(active is loader, "selected checkpoint builder is no longer active")
            finally:
                del current, active
            _require(
                factory.f_globals.get("build_model") is build_model
                and build_model.__code__ is factory.f_code
                and factory.f_locals.get("spec") is spec
                and factory.f_locals.get("parallel_groups") is groups
                and constructor.f_locals.get("self") is owner
                and constructor.f_locals.get("cfg") is native_config
                and all(
                    loader.f_locals.get(name) is value
                    for name, value in (
                        ("model", model),
                        ("state_dict", state_dict),
                        ("config", config),
                        ("parallel_dims", parallel_dims),
                    )
                )
                and (config, parallel_dims, spec, native_config) == recipe,
                "selected checkpoint construction recipe changed",
            )
            _require(
                (groups is None and group_items is None)
                or (
                    type(groups) is dict
                    and tuple(groups) == tuple(name for name, _ in group_items)
                    and all(groups[name] is value for name, value in group_items)
                ),
                "selected checkpoint parallel groups changed",
            )
            _require(
                type(state_dict) is dict
                and tuple(state_dict) == tuple(name for name, _ in state_items)
                and all(state_dict[name] is value for name, value in state_items),
                "selected checkpoint destination binding changed",
            )
            _require(
                reader.read_data is installed
                and type(reader).read_data is function
                and inspect.ismethod(original)
                and original.__self__ is reader
                and original.__func__ is function
                and function.__code__ is code
                and (function.__defaults__, function.__kwdefaults__) == defaults
                and reader.path == cache_path
                and reader.thread_count == threads,
                "selected checkpoint reader binding, cache or concurrency changed",
            )
            _require(
                all(
                    actual is captured
                    for actual, captured in zip(
                        _model_constructor_binding(owner, constructor, sources), binding, strict=True
                    )
                ),
                "selected checkpoint native constructor changed",
            )
            _initialization_function(function, sources)
            _initialization_function(observe_native_model_inventory, sources)

        def observed_read(actual_reader, plan, planner):
            nonlocal status
            _require(actual_reader is reader and status == "armed", "selected checkpoint reader is single-use")
            status = "observing"
            guard()
            _require(
                type(planner.state_dict) is dict
                and planner.state_dict.keys() == state_dict.keys()
                and all(planner.state_dict[name] is value for name, value in state_items),
                "selected planner must retain the actual filtered destination tensors",
            )
            planner_dict = planner.state_dict
            methods = {}
            for name in ("resolve_tensor", "commit_tensor", "finish_plan"):
                method = getattr(planner, name)
                _require(inspect.ismethod(method) and method.__self__ is planner, "actual selected planner required")
                _initialization_function(method.__func__, sources)
                methods[name] = (
                    method.__func__,
                    method.__func__.__code__,
                    deepcopy((method.__func__.__defaults__, method.__func__.__kwdefaults__)),
                )
            retained = []
            inventory = observe_native_model_inventory(
                model, retained_objects=retained, checkpoint_destinations=state_dict
            )
            before_plan, before_storage = deepcopy(plan), deepcopy(reader.storage_data)
            subject = {
                "worker": owner,
                "model": model,
                "model_spec": spec,
                "native_config": native_config,
                "parallel_groups": groups,
                "config": config,
                "parallel_dims": parallel_dims,
                "reader": reader,
                "planner": planner,
                "load_plan": plan,
                "checkpoint_destinations": state_dict,
                "native_inventory": deepcopy(inventory),
                "reader_storage": deepcopy(before_storage),
                "cache_path": cache_path,
                "reader_threads": threads,
                "factory_route": deepcopy(selected[1]["factory_route"]),
            }
            slot = _initialization_slot(
                selected,
                subject,
                maybe_observe_checkpoint_reader,
                observe_native_model_inventory,
                guard,
                observed_read,
                function,
                *[item[0] for item in methods.values()],
                build_model,
                *binding[:2],
                request_reader=_model_construction_request,
            )
            _verify_initialization_capacity(slot, "before-checkpoint-read", subject)
            guard()
            _require(
                plan == before_plan and reader.storage_data == before_storage,
                "capacity verifier changed the finalized checkpoint plan or file bindings",
            )
            _require(
                planner.state_dict is planner_dict
                and planner_dict.keys() == state_dict.keys()
                and all(planner_dict[name] is value for name, value in state_items)
                and all(
                    getattr(planner, name).__self__ is planner
                    and getattr(planner, name).__func__ is method
                    and method.__code__ is captured_code
                    and (method.__defaults__, method.__kwdefaults__) == captured_defaults
                    for name, (method, captured_code, captured_defaults) in methods.items()
                ),
                "capacity verifier changed the selected checkpoint planner",
            )
            _require(
                observe_native_model_inventory(model, retained_objects=retained, checkpoint_destinations=state_dict)
                == inventory,
                "capacity verifier changed checkpoint backing storage",
            )
            _initialization_function(guard, sources)
            _initialization_function(observed_read, sources)
            status = "reading"
            try:
                return original(plan, planner)
            finally:
                status = "finished"

        installed = MethodType(observed_read, reader)
        reader.read_data = installed
        return reader
    finally:
        del frame


def maybe_admit_worker_construction(worker, spec, ds_config, parallel_groups, build_model, initialize):
    selected = _initialization_request()
    if selected is None:
        return None
    _require(not hasattr(worker, "_ap_initialization_admission"), "worker initialization is single-use")
    subject = {
        "worker": worker,
        "model_spec": spec,
        "ds_config": ds_config,
        "parallel_groups": parallel_groups,
        "build_model": build_model,
        "initialize": initialize,
    }
    slot = _initialization_slot(
        selected,
        subject,
        maybe_admit_worker_construction,
        maybe_verify_worker_engine,
        type(worker),
        worker.initialize,
        worker.destroy,
        build_model,
        initialize,
    )
    slot["destroy"] = worker.destroy
    slot["destroy_code"] = slot["destroy"].__func__.__code__
    slot["destroy_defaults"] = deepcopy(
        (slot["destroy"].__func__.__defaults__, slot["destroy"].__func__.__kwdefaults__)
    )
    slot["worker_initialize"] = worker.initialize.__func__
    slot["ds_config"] = deepcopy(ds_config)
    result = _verify_initialization_capacity(slot, "before-build-model", subject)
    _require(ds_config == slot["ds_config"], "capacity verifier changed the native training recipe")
    _require(_worker_cleanup_matches(slot, worker), "capacity verifier changed worker cleanup")
    worker._ap_initialization_admission = slot
    return result


def _worker_cleanup_matches(slot, worker):
    destroy = getattr(worker, "destroy", None)
    captured = slot["destroy"]
    return (
        inspect.ismethod(destroy)
        and destroy.__self__ is worker
        and destroy.__func__ is captured.__func__
        and captured.__func__.__code__ is slot["destroy_code"]
        and (captured.__func__.__defaults__, captured.__func__.__kwdefaults__) == slot["destroy_defaults"]
    )


def maybe_verify_worker_engine(worker):
    slot = getattr(worker, "_ap_initialization_admission", None)
    if slot is None:
        _require(_initialization_request() is None, "selected engine has no armed pre-build admission")
        return None
    _require(slot["subject"]["worker"] is worker and slot["status"] == "armed", "actual armed worker required")
    engine = worker.engine
    try:
        _guard_initialization(slot)
        _require(
            worker.initialize.__func__ is slot["worker_initialize"] and _worker_cleanup_matches(slot, worker),
            "captured worker lifecycle changed",
        )
        _require(slot["subject"]["ds_config"] == slot["ds_config"], "native training recipe changed")
        runtime_result = slot["runtime"].verify_engine(
            engine,
            sp_info=worker.get_sequence_parallel_info(),
            sp_group=worker.sp_group,
        )
        _require(type(runtime_result) is dict and runtime_result, "actual selected engine verification required")
        result = _verify_initialization_capacity(
            slot,
            "after-engine",
            {**slot["subject"], "engine": engine},
        )
        _require(worker.engine is engine, "capacity verifier replaced the actual native engine")
        _require(slot["subject"]["ds_config"] == slot["ds_config"], "native training recipe changed")
        _require(_worker_cleanup_matches(slot, worker), "capacity verifier changed worker cleanup")
        slot["runtime_result"] = runtime_result
        slot["status"] = "verified"
        return result
    except BaseException as error:
        slot["status"] = "failed"
        # A source-file change does not invalidate still-owned cleanup code.
        if worker.engine is engine:
            try:
                _require(_worker_cleanup_matches(slot, worker), "worker cleanup changed; owned cleanup not attempted")
                slot["destroy"]()
            except BaseException as cleanup_error:
                error.add_note("owned worker initialization cleanup also failed: " + repr(cleanup_error))
        raise


def maybe_admit_sampler_construction(worker, args, constructor, engine_kwargs):
    selected = _initialization_request()
    if selected is None:
        return None
    _require(not hasattr(worker, "_ap_initialization_admission"), "sampler initialization is single-use")
    subject = {"worker": worker, "engine_args": args, "constructor": constructor, "engine_kwargs": engine_kwargs}
    slot = _initialization_slot(
        selected,
        subject,
        maybe_admit_sampler_construction,
        maybe_verify_sampler_construction,
        type(worker),
        worker.initialize,
        type(args),
        args.create_engine_config,
        constructor,
    )
    recipe = deepcopy(engine_kwargs)
    args_state = deepcopy(vars(args))
    result = _verify_initialization_capacity(slot, "before-sampler-constructor", subject)
    _require(
        engine_kwargs == recipe and vars(args) == args_state,
        "capacity verifier changed the native sampler recipe",
    )
    worker._ap_initialization_admission = slot
    return result


def maybe_verify_sampler_construction(worker):
    slot = getattr(worker, "_ap_initialization_admission", None)
    if slot is None:
        _require(_initialization_request() is None, "selected sampler has no armed constructor admission")
        return None
    _require(
        slot["subject"]["worker"] is worker and slot["status"] == "armed" and worker.llm is not None,
        "actual newly constructed sampler required",
    )
    _guard_initialization(slot)
    slot["status"] = "verified"
    return {"source_after_verified": True, "capacity_admitted": False, "numerical": None}


def selected_lora_request():
    """Read the caller-staged request, not a discovered or fabricated recipe."""
    path, expected = os.environ.get(REQUEST_ENV), os.environ.get(DIGEST_ENV)
    if path is None and expected is None:
        return None
    _require(enabled(), "pinned LoRA sync request requires explicit evidence opt-in")
    _require(
        type(path) is str
        and type(expected) is str
        and len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected),
        "exact caller-staged request path and SHA256 required",
    )
    selected = Path(path)
    _require(
        selected.is_absolute()
        and selected.is_relative_to("/data-fast")
        and selected.is_file()
        and not selected.is_symlink()
        and selected.resolve(strict=True) == selected,
        "canonical regular FAST LoRA request required",
    )
    before = selected.stat()
    raw = selected.read_bytes()
    after = selected.stat()

    def identity(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    _require(
        identity(before) == identity(after) and hashlib.sha256(raw).hexdigest() == expected,
        "caller-staged LoRA request changed",
    )
    request = json.loads(raw)
    _require(
        type(request) is dict
        and set(request) == {"schema", "training", "sampling"}
        and request["schema"] == "explicit-native-lora-sync-request-v1"
        and type(request["training"]) is dict
        and type(request["sampling"]) is dict
        and request["training"].get("admitted_sources") == request["sampling"].get("admitted_sources"),
        "exact shared source-bound training and sampling request required",
    )
    return request


def maybe_lora_replica_pool(pool_class):
    request = selected_lora_request()
    if request is None:
        from arctic_platform.debug.rl_sampler_evidence import maybe_replica_pool

        return maybe_replica_pool(pool_class)
    from arctic_platform.debug.lora_sampler_worker import make_lora_replica_pool

    return make_lora_replica_pool(pool_class, request["sampling"])


def maybe_prepare_lora_worker(worker, request):
    selected = selected_lora_request()
    _require(
        selected is not None and selected["training"] == request, "actual staged actor preparation request required"
    )
    return prepare_lora_worker(worker, request)


def maybe_send_lora_worker(worker):
    selected = selected_lora_request()
    if selected is None:
        from arctic_platform.debug.weight_sync_evidence import maybe_send_weights

        weights = worker.get_weights()
        sent_l2_sq = worker._weights_l2_sq(weights) if worker._weight_sender is not None else None
        result = maybe_send_weights(worker, weights)
        if worker._weight_sender is not None:
            result["sent_l2_sq"] = sent_l2_sq
        return result
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    _require(
        type(slot) is dict and slot["request"] == selected["training"],
        "native preparation differs from current staged request",
    )
    sender = worker._weight_sender
    binding = slot.get("sender_binding")
    _require(
        type(binding) is dict
        and binding["sender"] is sender
        and sender.group_id == 0
        and sender.targets == binding["targets"]
        and sender.ports == binding["ports"],
        "actual native sender targets or ports differ from captured initialization",
    )
    result = send_lora_worker(worker)
    _require(
        worker._weight_sender is sender and sender.targets == binding["targets"] and sender.ports == binding["ports"],
        "actual native sender routing changed during transfer",
    )
    return result


def maybe_record_lora_sender(worker, group, schedule, base_port):
    request = selected_lora_request()
    if request is None:
        return None
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    _require(
        type(slot) is dict and slot["request"] == request["training"] and worker.rank == 0,
        "actual prepared global-rank-zero actor required before sender initialization",
    )
    _require(
        group.group_id == 0
        and group.sender_train_rank == 0
        and group.targets == [(0, rank) for rank in range(8)]
        and schedule.training_gpus == 8
        and schedule.inference_tp == 8
        and schedule.inference_replicas == 1
        and type(base_port) is int
        and 0 < base_port <= 65528,
        "actual full8 sender schedule and finite native ports required",
    )
    sender = worker._weight_sender
    _require(sender is not None and "sender_binding" not in slot, "actual single initialized native sender required")
    _bind_sources(
        request["training"]["admitted_sources"],
        maybe_record_lora_sender,
        type(sender),
        type(group),
        type(schedule),
        schedule.port_for_target,
    )
    targets = list(group.targets)
    ports = [schedule.port_for_target(replica, tp, base_port) for replica, tp in targets]
    _require(
        sender.group_id == group.group_id
        and sender.targets == targets
        and sender.ports == ports
        and ports == list(range(base_port, base_port + 8)),
        "actual initialized sender differs from the retained native target and port recipe",
    )
    slot["sender_binding"] = {"sender": sender, "targets": targets, "ports": ports}
    return {"sender_train_rank": 0, "targets": targets, "ports": ports, "training_gpus": 8, "inference_tp": 8}


def maybe_release_lora_worker(worker):
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    if slot is not None:
        return release_lora_worker(worker, slot)
    return None


def maybe_lora_transfer_schedule(schedule_class, **kwargs):
    native = schedule_class.build(**kwargs)
    request = selected_lora_request()
    if request is None:
        return native
    _require(
        kwargs
        == {
            "training_sharding": "dp",
            "training_gpus": 8,
            "inference_replicas": 1,
            "inference_tp": 8,
        },
        "actual original TRAIN8 and SAMPLE8 schedule geometry required",
    )
    _require(
        len(native.groups) == 8
        and [group.sender_train_rank for group in native.groups] == list(range(8))
        and all(group.world_size == 2 for group in native.groups),
        "actual native eight-sender default schedule required before LoRA projection",
    )
    targets = [target for group in native.groups for target in group.targets]
    _require(
        targets == [(0, rank) for rank in range(8)],
        "every native TP target must occur exactly once in retained native order",
    )
    origins = request["training"]["admitted_sources"]
    group_class = type(native.groups[0])
    _bind_sources(origins, maybe_lora_transfer_schedule, schedule_class.build, schedule_class, group_class)
    # A prepared actor-local list is replayable; the native sender consumes it for each target.
    native.groups = [group_class(group_id=0, sender_train_rank=0, targets=targets)]
    _require(
        native.training_gpus == 8
        and native.inference_tp == 8
        and native.inference_replicas == 1
        and native.groups[0].targets == targets
        and native.groups[0].world_size == 2,
        "projected sender policy must retain actual native allocation geometry and targets",
    )
    _bind_sources(origins, maybe_lora_transfer_schedule, schedule_class, group_class)
    return native


async def maybe_prepare_lora_training(server, workers, pool, schedule):
    request = selected_lora_request()
    if request is None:
        return None
    _require(
        server.weight_sync_ready is False
        and not getattr(server, "_ap_lora_prepare_attempted", False)
        and not hasattr(server, "_ap_lora_training_preparation"),
        "native LoRA transfer is single-use; no unchanged collective or target retry",
    )
    _require(
        pool.num_replicas == 1
        and pool.tp_size == 8
        and len(schedule.groups) == 1
        and schedule.groups[0].sender_train_rank == 0
        and schedule.groups[0].world_size == 2
        and schedule.groups[0].replica_ids == [0],
        "actual original one-replica SAMPLE8 rank-zero pairwise schedule required",
    )
    origins = request["training"]["admitted_sources"]
    _bind_sources(origins, maybe_prepare_lora_training, type(server), type(pool), pool.sync_weights, type(schedule))
    server._ap_lora_prepare_attempted = True
    prepared = await prepare_lora_training_workers(workers, request["training"])
    _require(selected_lora_request() == request, "staged request changed during native preparation")
    _bind_sources(origins, maybe_prepare_lora_training, type(server), type(pool), pool.sync_weights, type(schedule))
    server._ap_lora_training_preparation = prepared
    return prepared


async def maybe_lora_max_param_bytes(server, worker):
    if selected_lora_request() is None:
        return (await server._await_training_refs(lambda: [worker.max_param_bytes.remote()]))[0]
    prepared = getattr(server, "_ap_lora_training_preparation", None)
    _require(type(prepared) is dict and prepared["all_training_ranks_prepared"] is True, "all8 preparation required")
    manifest = prepared["rank_preparations"][0].get("emitted_manifest")
    _require(type(manifest) is dict and manifest, "actual retained rank-zero adapter manifest required")
    widths = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}
    sizes = []
    for value in manifest.values():
        _require(
            type(value) is dict
            and value.get("dtype") in widths
            and type(value.get("numel")) is int
            and value["numel"] > 0,
            "actual positive floating adapter tensor geometry required",
        )
        sizes.append(value["numel"] * widths[value["dtype"]])
    return max(sizes)


async def maybe_collect_lora_sync(server, receive_task, send_tasks, *, schedule=None):
    request = selected_lora_request()
    received, sent = await server._await_weight_sync_refs(receive_task, send_tasks)
    if request is None:
        from arctic_platform.debug.rl_sampler_evidence import maybe_sync_weights_result

        result = maybe_sync_weights_result(received, sent)
        if schedule is not None:
            from arctic_platform.common.utils.weight_sync import received_checksums
            from arctic_platform.common.weight_checksum import compare_weight_checksums

            workers_results = received.get("workers", [])
            result.update(
                compare_weight_checksums(
                    {
                        f"training rank {group.sender_train_rank}": receipt.get("sent_l2_sq")
                        for group, receipt in zip(schedule.groups, sent)
                    },
                    received_checksums(
                        [f"sampling replica {index}" for index in range(len(workers_results))], workers_results
                    ),
                )
            )
        return result
    _require(
        len(sent) == 1
        and type(sent[0]) is dict
        and sent[0].get("weight_format") == "lora"
        and type(received) is dict
        and type(received.get("workers")) is list
        and len(received["workers"]) == 1,
        "actual single native LoRA sender and sampler receipt required",
    )
    worker = received["workers"][0]
    _require(
        type(worker) is dict
        and worker.get("weight_format") == "lora"
        and type(worker.get("all_rank_adapter_installation")) is list
        and len(worker["all_rank_adapter_installation"]) == 8
        and worker.get("adapter_storage_verified") is False
        and worker.get("adapter_application_verified") is False
        and worker.get("full_parameter_coverage") is False
        and worker.get("capacity_admitted") is False,
        "allTP adapter installation is not storage/application/full-model/capacity admission",
    )
    _require(selected_lora_request() == request, "staged request changed during native transfer")
    _bind_sources(request["training"]["admitted_sources"], maybe_collect_lora_sync)
    return {
        "status": "ok",
        "weight_format": "lora",
        "targets": [{"send": sent, "recv": received}],
        "training_preparation": server._ap_lora_training_preparation,
        "adapter_storage_verified": False,
        "adapter_application_verified": False,
        "full_parameter_coverage": False,
        "capacity_admitted": False,
        "parameter_only_judge_admitted": False,
        "numerical": None,
    }
