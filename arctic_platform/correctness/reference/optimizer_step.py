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
    optimizer_name = str(optimizer_config.get("name", "AdamW")).strip().lower().replace("-", "_")
    if optimizer_name not in ADAMW_NAMES:
        raise ValueError(f"optimizer-state correctness supports AdamW semantics, got {optimizer_name!r}")


def build_adamw(named_masters, optimizer_config: dict, *, learning_rate: float, backend: str = "torch"):
    """One AdamW over the config's hyperparameters, with weight decay withheld from biases and norms.

    The one-step reference uses PyTorch so its moments are independent evidence. A multi-step trajectory uses the
    product backend because backend scalar rounding compounds across its hundred steps. Both retain the same
    parameter grouping, master precision, and configured hyperparameters.
    """
    import torch

    decay = [master for name, master in named_masters if _uses_weight_decay(name)]
    no_decay = [master for name, master in named_masters if not _uses_weight_decay(name)]
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})

    betas = optimizer_config.get("betas", [0.9, 0.999])
    kwargs = {
        "lr": float(learning_rate),
        "betas": (float(betas[0]), float(betas[1])),
        "eps": float(optimizer_config.get("eps", 1e-8)),
    }
    if backend == "fused_adam":
        from deepspeed.ops.adam import FusedAdam

        return FusedAdam(groups, adam_w_mode=True, **kwargs)
    if backend != "torch":
        raise ValueError(f"unknown AdamW backend: {backend!r}")
    return torch.optim.AdamW(groups, foreach=False, fused=False, **kwargs)


def run_adamw_step(
    model,
    gradients: Dict[str, "torch.Tensor"],
    optimizer_config: dict,
    *,
    learning_rate: float,
    gradient_clipping: float | None,
    optimizer_dtype: str,
    output_dir: Path,
) -> OptimizerStepArtifact:
    """Apply one independent AdamW step and write each named state without an HTTP payload."""
    import torch

    dtype = master_dtype(optimizer_dtype)
    assert_adamw_semantics(optimizer_config)

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in gradients
    ]
    if not named_parameters:
        raise ValueError("optimizer step found no trainable parameters with gradients")

    masters = {
        name: torch.nn.Parameter(parameter.detach().to(dtype).clone(), requires_grad=True)
        for name, parameter in named_parameters
    }
    optimizer = build_adamw(
        [(name, masters[name]) for name, _ in named_parameters], optimizer_config, learning_rate=learning_rate
    )
    for name, _ in named_parameters:
        masters[name].grad = gradients[name].detach().to(device=masters[name].device, dtype=dtype).clone()

    gradient_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(masters[name].grad) for name, _ in named_parameters])
    )
    if gradient_clipping is not None and float(gradient_clipping) > 0:
        torch.nn.utils.clip_grad_norm_(list(masters.values()), float(gradient_clipping))
    optimizer.step()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    source_parameters = dict(named_parameters)
    for index, (name, _) in enumerate(named_parameters):
        master = masters[name]
        state = optimizer.state[master]
        source = source_parameters[name]
        tensors = {
            "parameter_update": master.detach().sub(source.detach().float()).cpu(),
            "exp_avg": state["exp_avg"].detach().cpu(),
            "exp_avg_sq": state["exp_avg_sq"].detach().cpu(),
        }
        filename = f"{index:06d}.pt"
        torch.save(tensors, output_dir / filename)
        entries.append(
            {
                "name": name,
                "file": filename,
                "shape": list(master.shape),
                "parameter_update_norm": float(torch.linalg.vector_norm(tensors["parameter_update"])),
                "exp_avg_norm": float(torch.linalg.vector_norm(tensors["exp_avg"])),
                "exp_avg_sq_norm": float(torch.linalg.vector_norm(tensors["exp_avg_sq"])),
            }
        )
        state.clear()
        master.grad = None

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"step": 1, "parameters": entries}, indent=2, sort_keys=True) + "\n")
    return OptimizerStepArtifact(
        manifest_path=manifest_path,
        gradient_norm_before_clip=float(gradient_norm.detach().cpu()),
    )
