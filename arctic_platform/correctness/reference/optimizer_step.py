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

"""Single-device FP32-master AdamW golden step and named optimizer-state artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from arctic_platform.common.optimizer import OptimizerSettings
from arctic_platform.correctness.harness.storage import require_data_fast_path

NO_DECAY_NAMES = ("bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight")
ADAMW_NAMES = {"adam", "adamw", "fusedadam", "fused_adam", "torch_adamw", "deepspeed_cpu_adam"}


@dataclass
class OptimizerStepArtifact:
    manifest_path: Path
    gradient_norm_before_clip: float


def _uses_weight_decay(name: str) -> bool:
    lowered = name.lower()
    return not any(part in lowered for part in NO_DECAY_NAMES)


def master_dtype(optimizer_dtype: str):
    """The precision the FP32-master copy and the Adam moments are held in."""
    import torch

    dtype = getattr(torch, optimizer_dtype, None)
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"unsupported optimizer dtype {optimizer_dtype!r}")
    return dtype


def assert_adamw_semantics(optimizer_config: dict) -> None:
    """Refuse a config whose optimizer is not one whose step this module reproduces."""
    settings = OptimizerSettings.model_validate(optimizer_config)
    optimizer_name = settings.name.strip().lower().replace("-", "_")
    if optimizer_name not in ADAMW_NAMES:
        raise ValueError(f"optimizer-state correctness supports AdamW semantics, got {optimizer_name!r}")


def build_adamw(named_masters, optimizer_config: dict, *, learning_rate: float):
    """One AdamW over the config's hyperparameters, with weight decay withheld from biases and norms.

    Single-step and trajectory references both use independent PyTorch AdamW with fused and foreach
    implementations disabled. The single-step reference runs it in the precision the product's implementation
    computes in and rounds to the optimizer dtype on store; other product rounding remains a measured difference.
    """
    import torch

    decay = [master for name, master in named_masters if _uses_weight_decay(name)]
    no_decay = [master for name, master in named_masters if not _uses_weight_decay(name)]
    settings = OptimizerSettings.model_validate(optimizer_config)
    weight_decay = settings.weight_decay
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})

    kwargs = {
        "lr": float(learning_rate),
        "betas": tuple(settings.betas),
        "eps": settings.eps,
    }
    return torch.optim.AdamW(groups, foreach=False, fused=False, **kwargs)


def _write_adamw_parameter(
    name,
    source,
    gradients,
    optimizer_config,
    *,
    dtype,
    compute_dtype,
    learning_rate,
    coefficient,
    foreach_clip,
    path,
    consume_gradients,
):
    master = initial_master = optimizer = state = tensors = None
    try:
        master = torch.nn.Parameter(source.detach().to(dtype).to(compute_dtype, copy=True), requires_grad=True)
        # Initialization precision conversion is not part of the optimizer update.
        initial_master = master.detach().float().clone()
        gradient = gradients.pop(name) if consume_gradients else gradients[name]
        clipped = gradient.detach().to(device=master.device, dtype=dtype).clone()
        del gradient
        if coefficient is not None:
            if foreach_clip:
                torch._foreach_mul_([clipped], coefficient)
            else:
                clipped.mul_(coefficient)
        master.grad = clipped.to(compute_dtype)
        del clipped
        optimizer = build_adamw([(name, master)], optimizer_config, learning_rate=learning_rate)
        optimizer.step()
        master.grad = None
        state = optimizer.state[master]
        tensors = {
            "parameter_update": master.detach().to(dtype).float().sub(initial_master).cpu(),
            "exp_avg": state["exp_avg"].detach().to(dtype).cpu(),
            "exp_avg_sq": state["exp_avg_sq"].detach().to(dtype).cpu(),
        }
        torch.save(tensors, path)
        return {
            "name": name,
            "file": path.name,
            "shape": list(master.shape),
            "parameter_update_norm": float(torch.linalg.vector_norm(tensors["parameter_update"])),
            "exp_avg_norm": float(torch.linalg.vector_norm(tensors["exp_avg"])),
            "exp_avg_sq_norm": float(torch.linalg.vector_norm(tensors["exp_avg_sq"])),
        }
    finally:
        if master is not None:
            master.grad = None
        if state is not None:
            state.clear()
        if optimizer is not None:
            optimizer.state.clear()
            for group in optimizer.param_groups:
                group["params"].clear()
            optimizer.param_groups.clear()
        if tensors is not None:
            tensors.clear()
        master = initial_master = optimizer = state = tensors = None


def run_adamw_step(
    model,
    gradients: Dict[str, "torch.Tensor"],
    optimizer_config: dict,
    *,
    learning_rate: float,
    gradient_clipping: float | None,
    optimizer_dtype: str,
    output_dir: Path,
    optimizer_compute_dtype: str | None = None,
    consume_gradients: bool = False,
    participation: dict | None = None,
) -> OptimizerStepArtifact:
    """Apply one AdamW step, optionally consuming selected gradients after full clipping preflight."""
    output_dir = require_data_fast_path(output_dir, label="reference optimizer output")
    import torch
    from torch.utils._foreach_utils import _has_foreach_support

    dtype = master_dtype(optimizer_dtype)
    compute_dtype = master_dtype(optimizer_compute_dtype or optimizer_dtype)
    assert_adamw_semantics(optimizer_config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("reference optimizer output directory must be empty")

    inventory = None
    if participation is not None:
        from arctic_platform.correctness.harness.optimizer import validate_participation

        inventory = validate_participation(participation)
        actual = {
            name: list(parameter.shape) for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if actual != {name: entry["shape"] for name, entry in inventory.items()}:
            raise ValueError("optimizer participation does not cover the actual trainable inventory")
        active = {name for name, entry in inventory.items() if entry["state"] == "active"}
        if set(gradients) != active:
            raise ValueError("optimizer gradients do not match actual backward participation")

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in gradients
    ]
    if not named_parameters:
        raise ValueError("optimizer step found no trainable parameters with gradients")

    device = named_parameters[0][1].device
    for name, parameter in named_parameters:
        gradient = gradients[name]
        if parameter.device != device or device.type == "meta":
            raise ValueError("optimizer step requires parameters on one materialized device")
        if type(gradient) is not torch.Tensor or gradient.layout != torch.strided:
            raise ValueError(f"optimizer gradient {name!r} must be a dense Tensor")
        if gradient.shape != parameter.shape:
            raise ValueError(f"optimizer gradient {name!r} shape does not match its parameter")

    clipping = gradient_clipping is not None and float(gradient_clipping) > 0
    reported_norms, clipping_norms = [], []
    foreach_clip = False
    # Preserve both reductions: clip_grad_norm_'s default foreach norm need not equal the reported norm.
    for name, parameter in named_parameters:
        gradient = gradients[name].detach().to(device=device, dtype=dtype).clone()
        try:
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError(f"optimizer gradient {name!r} is not finite in {dtype}")
            reported_norms.append(torch.linalg.vector_norm(gradient))
            if clipping:
                foreach_clip = _has_foreach_support([gradient], device)
                clipping_norms.append(torch._foreach_norm([gradient], 2.0)[0] if foreach_clip else reported_norms[-1])
        finally:
            del gradient
    gradient_norm = torch.linalg.vector_norm(torch.stack(reported_norms))
    coefficient = None
    if clipping:
        clipping_norm = torch.linalg.vector_norm(torch.stack(clipping_norms))
        coefficient = (float(gradient_clipping) / (clipping_norm + 1e-6)).clamp(max=1.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    inactive_weights = {}
    if inventory is not None:
        from arctic_platform.correctness.harness.optimizer import validate_inactive_weights
        from arctic_platform.correctness.harness.optimizer import weight_identity

        for name, source in model.named_parameters():
            if name not in inventory or inventory[name]["state"] != "inactive":
                continue
            master = torch.nn.Parameter(source.detach().to(dtype).clone(), requires_grad=True)
            parameter_before, master_before = weight_identity(source), weight_identity(master)
            optimizer = build_adamw([(name, master)], optimizer_config, learning_rate=learning_rate)
            optimizer.step()
            inactive_weights[name] = {
                "parameter": {"before": parameter_before, "after": weight_identity(source)},
                "master": {"before": master_before, "after": weight_identity(master)},
            }
            if optimizer.state:
                raise ValueError(f"inactive reference parameter {name} acquired Adam state")
            del optimizer, master
        validate_inactive_weights(participation, inactive_weights)
    entries = []
    for index, (name, source) in enumerate(named_parameters):
        entries.append(
            _write_adamw_parameter(
                name,
                source,
                gradients,
                optimizer_config,
                dtype=dtype,
                compute_dtype=compute_dtype,
                learning_rate=learning_rate,
                coefficient=coefficient,
                foreach_clip=foreach_clip,
                path=output_dir / f"{index:06d}.pt",
                consume_gradients=consume_gradients,
            )
        )

    manifest_path = output_dir / "manifest.json"
    temporary_manifest = output_dir / ".manifest.json.tmp"
    manifest = {"step": 1, "parameters": entries}
    if participation is not None:
        manifest.update(participation=participation, inactive_weights=inactive_weights)
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    return OptimizerStepArtifact(
        manifest_path=manifest_path,
        gradient_norm_before_clip=float(gradient_norm.detach().cpu()),
    )
