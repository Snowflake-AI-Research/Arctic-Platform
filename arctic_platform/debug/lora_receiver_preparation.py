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

"""Resolve selected-worker build inputs without inventing a receiver or model."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import stat
from pathlib import Path

from arctic_platform.debug.lora_selected_sampler import PATH_ENV
from arctic_platform.debug.lora_selected_sampler import SHA_ENV
from arctic_platform.debug.lora_selected_sampler import _bytes
from arctic_platform.debug.lora_selected_sampler import _unique_object
from arctic_platform.debug.lora_selected_sampler import bootstrap_descriptor
from arctic_platform.debug.weight_sync_evidence import enabled
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import verify_sources

PREPARATION_FILENAME = "receiver-preparation.json"
BOUND_KEYS = ("max_source_bytes", "max_layers", "max_total_source_bytes")


def _require(value, message):
    if not value:
        raise ValueError(message)


def preparation_settings():
    """Consume settings sealed into the same bootstrap's source/data closure."""
    _require(enabled(), "selected receiver preparation requires explicit opt-in")
    _require(
        PATH_ENV in os.environ and SHA_ENV in os.environ,
        "actual selected worker bootstrap required",
    )
    pointer = {"path": os.environ[PATH_ENV], "sha256": os.environ[SHA_ENV]}
    body = bootstrap_descriptor(pointer)
    path = Path(pointer["path"]).with_name(PREPARATION_FILENAME)
    sources = body["admitted_sources"]
    _require(str(path) in sources, "preparation settings must be bootstrap-bound")
    raw = _bytes(path)
    _require(
        hashlib.sha256(raw).hexdigest() == sources[str(path)],
        "preparation settings changed",
    )
    settings = json.loads(raw, object_pairs_hook=_unique_object)
    _require(
        type(settings) is dict
        and set(settings) == {"schema", "publication_root", "distribution_names", *BOUND_KEYS}
        and settings["schema"] == "job3-selected-lora-preparation-v1",
        "exact selected preparation settings required",
    )
    _require(
        all(type(settings[key]) is int and settings[key] > 0 for key in BOUND_KEYS),
        "explicit positive publication bounds required",
    )
    names = settings["distribution_names"]
    _require(
        type(names) is list
        and names
        and all(type(name) is str and name for name in names)
        and len(set(names)) == len(names),
        "explicit distinct selected distributions required",
    )
    _require(
        type(settings["publication_root"]) is str,
        "explicit private publication root required",
    )
    verify_sources(sources)
    _require(_bytes(path) == raw, "preparation settings changed during resolution")
    return settings, dict(sources)


def _actual_components():
    # These imports run on the already-selected native worker, never at module load.
    from vllm.distributed import get_world_group
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.utils import parse_fine_tuned_lora_name

    from arctic_platform.common.peft import normalize_lora_peft_config
    from arctic_platform.debug.lora_adapter_tensor_binding import observe_lora_build
    from arctic_platform.inference.server.weight_sync.receiver import _build_lora_model
    from arctic_platform.inference.server.weight_sync.receiver import _install_lora_adapter
    from arctic_platform.inference.server.weight_sync.receiver import _remap_lora_key
    from arctic_platform.inference.server.weight_sync.receiver import _served_lm_module_prefix

    return {
        "observer": observe_lora_build,
        "builder": _build_lora_model,
        "installer": _install_lora_adapter,
        "normalize_config": normalize_lora_peft_config,
        "native_model_class": LoRAModel,
        "parse_name": parse_fine_tuned_lora_name,
        "remap_key": _remap_lora_key,
        "served_prefix_for_model": _served_lm_module_prefix,
    }, get_world_group


def _rank_output(root, rank):
    root = Path(root)
    _require(
        root.is_absolute()
        and root.is_relative_to("/data-fast")
        and root.resolve(strict=True) == root
        and root.is_dir()
        and not root.is_symlink()
        and stat.S_IMODE(root.stat().st_mode) == 0o700,
        "canonical private FAST publication root required",
    )
    output = root / f"tp-rank-{rank}"
    output.mkdir(mode=0o700, exist_ok=False)
    # A newly created NFS directory can inherit setgid; never chmod an old path.
    output.chmod(0o700)
    _require(
        output.resolve(strict=True) == output and stat.S_IMODE(output.stat().st_mode) == 0o700,
        "new rank publication directory is not private",
    )
    return output


def prepare_selected_lora_receiver(receiver):
    """Supply the frozen RPC's exact options from actual selected-worker imports."""
    settings, sources = preparation_settings()
    components, get_world_group = _actual_components()
    observed = source_files(
        prepare_selected_lora_receiver,
        _actual_components,
        get_world_group,
        *components.values(),
    )
    _require(
        observed.items() <= sources.items(),
        "actual selected builder/parser/rank sources must be admitted",
    )
    verify_sources(sources)
    group = get_world_group()
    _require(
        type(group.rank) is int and type(group.world_size) is int and group.world_size == 8 and 0 <= group.rank < 8,
        "actual original SAMPLE8 rank required",
    )
    _require(
        receiver.model_runner.model is not None and receiver.model_runner.lora_manager._adapter_manager is not None,
        "already-loaded selected receiver and manager required",
    )
    distributions = tuple(importlib.metadata.distribution(name) for name in settings["distribution_names"])
    verify_sources(sources)
    output = _rank_output(settings["publication_root"], group.rank)
    verify_sources(sources)
    return {
        **components,
        "distributions": distributions,
        "output": output,
        **{key: settings[key] for key in BOUND_KEYS},
    }
