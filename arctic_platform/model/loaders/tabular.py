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
"""Loader for tabular foundation models: NVIDIA Kumo Tabular from ``structured-data-models``.

The built module is the raw in-context-learning network, ``forward(x, y, categorical_mask) -> outputs``:

- ``x``: ``[..., R, C]`` float features, the ``R_context`` labeled rows first, then the query rows.
- ``y``: ``[..., R_context]`` context targets, class indices for classification, floats for regression.
- ``categorical_mask``: ``[C]`` bool, which columns were categorical before preprocessing.
- outputs: ``[..., R_query, 10]`` class logits, or ``[..., R_query, 999]`` quantiles (levels 0.001 to 0.999).

Table preprocessing (sdm's recipe) runs on the client, so the server sees plain tensors.

``model_path_or_name`` is a directory whose ``config.json`` has ``model_type: kumo_tabular`` and a
``checkpoint_files`` map from size and task to a weights file relative to the directory, as published in
``nvidia/Kumo-Tabular``. Weights-only checkpoints saved by training use the same layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel
from pydantic import ConfigDict

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.config import Patches
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader

KUMO_TABULAR_MODEL_TYPE = "kumo_tabular"
TABULAR_LOADER = "tabular"


class TabularOptions(BaseModel):
    """Validated ``loader_options`` for the tabular loader."""

    model_config = ConfigDict(extra="forbid")

    task: Literal["classification", "regression"]
    size: Literal["small", "medium", "large"] = "small"


def _validate_spec(spec: ModelSpec) -> None:
    if spec.dtype not in ("float32", "bfloat16"):
        raise ValueError(f"the tabular loader supports float32 or bfloat16 parameters, got {spec.dtype!r}")
    if spec.patches != Patches():
        raise ValueError("the tabular loader does not apply model patches (liger, PEFT, checkpointing, compile, ...)")
    if spec.parallelism.sequence_parallel != 1 or spec.parallelism.expert_parallel != 1:
        raise ValueError("the tabular loader supports data parallelism only (sequence/expert parallel size must be 1)")


def weights_file(model_dir: str | Path, options: TabularOptions) -> Path:
    """The weights file ``config.json`` names for ``options.size`` and ``options.task``."""
    root = Path(model_dir)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != KUMO_TABULAR_MODEL_TYPE:
        raise ValueError(
            f"{root}/config.json has model_type {config.get('model_type')!r}; the tabular loader reads "
            f"{KUMO_TABULAR_MODEL_TYPE!r}"
        )
    try:
        relative = config["checkpoint_files"][options.size][options.task]
    except KeyError as exc:
        raise ValueError(
            f"{root}/config.json has no checkpoint_files entry for {options.size}/{options.task}"
        ) from exc
    return root / relative


def read_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(f"unsupported tabular weights file {path}; expected .safetensors or .pt")


def save_safetensors(state: dict[str, torch.Tensor], path: str | Path) -> None:
    """Write ``state`` as safetensors.

    Kumo Tabular's query and key rotary transforms share one ``inv_freq`` buffer, and safetensors refuses tensors
    that share storage, so every tensor is written as its own copy. ``load_state_dict`` restores the sharing, which
    lives in the module.
    """
    from safetensors.torch import save_file

    save_file({key: value.detach().to("cpu").contiguous().clone() for key, value in state.items()}, str(path))


def build_network(options: TabularOptions) -> torch.nn.Module:
    """The untrained Kumo Tabular network for ``options``, as ``sdm`` builds it."""
    from sdm.models import KumoTabular

    return KumoTabular(task=options.task, size=options.size, pretrained=False, device="cpu").models[options.task]


def convert_to_safetensors(
    src_dir: str | Path,
    dst_dir: str | Path,
    sizes: tuple[str, ...] | None = None,
) -> None:
    """Write a copy of an ``nvidia/Kumo-Tabular`` snapshot whose weights are all safetensors.

    Run once before uploading the base model, so serving never unpickles. ``config.json`` is rewritten to point at
    the converted files and lists only ``sizes`` (default: every size the snapshot names).
    """
    src, dst = Path(src_dir), Path(dst_dir)
    config = json.loads((src / "config.json").read_text())
    if config.get("model_type") != KUMO_TABULAR_MODEL_TYPE:
        raise ValueError(f"{src}/config.json is not a {KUMO_TABULAR_MODEL_TYPE} config")
    if sizes is not None:
        unknown = set(sizes) - set(config["checkpoint_files"])
        if unknown:
            raise ValueError(f"{src}/config.json names no checkpoint_files for sizes {sorted(unknown)}")
        config["checkpoint_files"] = {size: config["checkpoint_files"][size] for size in sizes}
        if config.get("default_size") not in config["checkpoint_files"]:
            config["default_size"] = sizes[0]
    for size, tasks in config["checkpoint_files"].items():
        for task, relative in tasks.items():
            target = Path(relative).with_suffix(".safetensors")
            (dst / target).parent.mkdir(parents=True, exist_ok=True)
            save_safetensors(read_state_dict(src / relative), dst / target)
            config["checkpoint_files"][size][task] = str(target)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")


@register_loader(TABULAR_LOADER, options=TabularOptions, validate_spec=_validate_spec)
def load_tabular(ctx: LoaderContext) -> LoadedModel:
    options = TabularOptions.model_validate(ctx.spec.loader_options)
    model = build_network(options)
    model.load_state_dict(read_state_dict(weights_file(ctx.spec.model_path_or_name, options)), strict=True)
    return LoadedModel(model=model.to(getattr(torch, ctx.spec.dtype)))
