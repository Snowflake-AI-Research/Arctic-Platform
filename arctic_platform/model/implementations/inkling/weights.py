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

"""Official Inkling checkpoint loading and serving export."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.distributed.tensor import DTensor

from arctic_platform.model.loader import canonical_parameter_name


def build_iter_vllm_weights(model):
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights

    native_weights = build_iter_full_hf_weights(model)

    def serving_name(name):
        if name == "lm_head.weight":
            return "model.llm.unembed.weight"
        name = name.replace("model.embed_tokens.", "model.embed.")
        for source, target in (
            (".self_attn.q_proj.", ".attn.wq_du."),
            (".self_attn.k_proj.", ".attn.wk_dv."),
            (".self_attn.v_proj.", ".attn.wv_dv."),
            (".self_attn.r_proj.", ".attn.wr_du."),
            (".self_attn.o_proj.", ".attn.wo_ud."),
            (".self_attn.", ".attn."),
            (".input_layernorm.", ".attn_norm."),
            (".post_attention_layernorm.", ".mlp_norm."),
            (".gate.e_score_correction_bias", ".gate.bias"),
            (".conv1d.weight", ".weight"),
            (".mlp.experts.down_proj", ".mlp.experts.w2_weight"),
            (".mlp.shared_experts.down_proj", ".mlp.shared_experts.shared_w2_weight"),
            (".mlp.down_proj.weight", ".mlp.w2_md.weight"),
        ):
            name = name.replace(source, target)
        return name.replace("model.", "model.llm.", 1)

    def iterator():
        pending = {}
        for name, tensor in native_weights(include_skipped=True):
            if name.endswith(".mlp.experts.gate_up_proj"):
                gate, up = tensor.chunk(2, dim=1)
                tensor = torch.stack((gate, up), dim=2).flatten(1, 2)
                name = name.removesuffix("gate_up_proj") + "w13_weight"
            elif name.endswith(
                (
                    ".mlp.gate_proj.weight",
                    ".mlp.up_proj.weight",
                    ".mlp.shared_experts.gate_proj",
                    ".mlp.shared_experts.up_proj",
                )
            ):
                prefix, projection = name.removesuffix(".weight").rsplit(".", 1)
                pair = pending.setdefault(prefix, {})
                pair[projection] = tensor
                if len(pair) != 2:
                    continue
                gate, up = pair["gate_proj"], pair["up_proj"]
                if prefix.endswith("shared_experts"):
                    shape = (model.config.n_shared_experts, -1, gate.shape[-1])
                    tensor = torch.stack((gate.reshape(shape), up.reshape(shape)), dim=2).flatten(1, 2)
                    name = prefix + ".shared_w13_weight"
                else:
                    tensor = torch.stack((gate, up), dim=1).flatten(0, 1)
                    name = prefix + ".w13_dn.weight"
                del pending[prefix]
            yield serving_name(name), tensor
        if pending:
            raise ValueError(f"Incomplete Inkling gate/up projections: {sorted(pending)}")

    return iterator


def weight_targets(name: str) -> list[tuple[str, int | Literal["gate", "up"] | None]]:
    """Target name and gate/up parity for interleaved checkpoint tensors."""
    if name.startswith(("model.audio.", "model.visual.", "model.mtp.")):
        return []
    if name == "model.llm.unembed.weight":
        return [("lm_head.weight", None)]
    name = name.replace("model.llm.", "model.")
    name = name.replace("model.embed.weight", "model.embed_tokens.weight")
    if name.endswith("mlp.experts.gate_up_proj"):
        prefix = name.removesuffix("gate_up_proj")
        return [(prefix + "w1", "gate"), (prefix + "w3", "up")]
    if name.endswith("mlp.experts.down_proj"):
        return [(name.removesuffix("down_proj") + "w2", None)]
    for source, target in (
        ("attn.wq_du", "self_attn.q_proj"),
        ("attn.wk_dv", "self_attn.k_proj"),
        ("attn.wv_dv", "self_attn.v_proj"),
        ("attn.wr_du", "self_attn.r_proj"),
        ("attn.wo_ud", "self_attn.o_proj"),
        (".attn.", ".self_attn."),
        (".attn_norm.", ".input_layernorm."),
        (".mlp_norm.", ".post_attention_layernorm."),
        (".gate.bias", ".gate.e_score_correction_bias"),
    ):
        name = name.replace(source, target)
    name = re.sub(r"(attn_sconv|mlp_sconv|k_sconv|v_sconv)\.weight$", r"\1.conv1d.weight", name)
    for source, targets in (
        ("experts.w13_weight", ("experts.w1", "experts.w3")),
        ("shared_experts.shared_w13_weight", ("shared_experts.gate_proj", "shared_experts.up_proj")),
        ("mlp.w13_dn.weight", ("mlp.gate_proj.weight", "mlp.up_proj.weight")),
    ):
        if name.endswith(source):
            return [(name.removesuffix(source) + target, parity) for parity, target in enumerate(targets)]
    for source, target in (
        ("experts.w2_weight", "experts.w2"),
        ("shared_experts.shared_w2_weight", "shared_experts.down_proj"),
        ("mlp.w2_md.weight", "mlp.down_proj.weight"),
    ):
        if name.endswith(source):
            name = name.removesuffix(source) + target
            break
    return [(name, None)]


@torch.no_grad()
def load_weights(model, config, parallel_dims):
    path = Path(config.name)
    if not path.is_dir():
        path = Path(snapshot_download(config.name, allow_patterns=["*.json", "model*.safetensors"]))
    index = path / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    )
    model.to_empty(device=torch.cuda.current_device())
    targets = {canonical_parameter_name(name): tensor for name, tensor in model.state_dict().items()}
    loaded = set()
    for filename in files:
        with safe_open(path / filename, framework="pt", device="cpu") as source:
            for name in source.keys():
                for target_name, parity in weight_targets(name):
                    if target_name not in targets:
                        raise ValueError(f"Unsupported Inkling checkpoint weight: {name} -> {target_name}")
                    if target_name in loaded:
                        raise ValueError(f"Duplicate Inkling checkpoint weight: {target_name}")
                    target = targets[target_name]
                    sliced = source.get_slice(name)
                    shape = sliced.get_shape()
                    slices = [slice(None)] * len(shape)
                    if isinstance(target, DTensor):
                        rank = target.device_mesh.get_local_rank()
                        local = target.to_local()
                        slices[0] = slice(rank * local.shape[0], (rank + 1) * local.shape[0])
                    else:
                        local = target
                    if isinstance(parity, str):
                        halfway = shape[-2] // 2
                        slices[-2] = slice(0, halfway) if parity == "gate" else slice(halfway, None)
                    elif parity is not None:
                        slices[-2] = slice(parity, None, 2)
                    value = sliced[tuple(slices)]
                    if value.shape != local.shape:
                        raise ValueError(
                            f"Inkling weight {name}: checkpoint shape {value.shape}, expected {local.shape}"
                        )
                    local.copy_(value)
                    loaded.add(target_name)
                    del value
    missing = targets.keys() - loaded
    if missing:
        raise ValueError(f"Missing Inkling checkpoint weights: {sorted(missing)}")
    model.init_buffers_post_meta()
