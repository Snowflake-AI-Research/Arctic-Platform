"""PrimeRL → vLLM ``/sync-weights`` packing shared by generic MoE families."""

from __future__ import annotations

import torch
import torch.nn as nn


def named_weight_sync_tensors(model: nn.Module):
    yield from model.named_parameters()
    yield from (
        (name, buffer) for name, buffer in model.named_buffers() if name.endswith(".expert_bias")
    )


def raise_unmapped_mlp_keys(packer: str, leftover: list[str]) -> None:
    if leftover:
        raise RuntimeError(
            f"{packer} vLLM packer left unmapped mlp keys: "
            + ", ".join(sorted(leftover))
        )


def pack_routed_experts_to_vllm(layer_sd: dict, prefix: str) -> None:
    """Prime-RL fused experts (router + w1/w2/w3) → vLLM Triton fused-MoE keys."""
    router_key = f"{prefix}.mlp.router.gate.weight"
    if router_key in layer_sd:
        layer_sd[f"{prefix}.mlp.gate.weight"] = layer_sd.pop(router_key)

    w1_key = f"{prefix}.mlp.experts.w1"
    w3_key = f"{prefix}.mlp.experts.w3"
    w2_key = f"{prefix}.mlp.experts.w2"
    if w1_key in layer_sd and w3_key in layer_sd:
        w1 = layer_sd.pop(w1_key)
        w3 = layer_sd.pop(w3_key)
        layer_sd[f"{prefix}.mlp.experts.w13_weight"] = torch.cat([w1, w3], dim=1)
    if w2_key in layer_sd:
        layer_sd[f"{prefix}.mlp.experts.w2_weight"] = layer_sd.pop(w2_key)

    for buf_key in (f"{prefix}.mlp.expert_bias", f"{prefix}.mlp.tokens_per_expert"):
        layer_sd.pop(buf_key, None)

    packed_experts = {
        f"{prefix}.mlp.experts.w13_weight",
        f"{prefix}.mlp.experts.w2_weight",
    }
    leftover = [
        key
        for key in layer_sd
        if key.startswith(f"{prefix}.mlp.router.")
        or (
            key.startswith(f"{prefix}.mlp.experts.") and key not in packed_experts
        )
    ]
    raise_unmapped_mlp_keys("generic routed-expert", leftover)


def pack_shared_expert_to_vllm(layer_sd: dict, prefix: str) -> None:
    """PrimeRL ``mlp.shared_expert.w{1,2,3}`` → vLLM ``shared_experts.{gate,down,up}_proj``."""
    src = f"{prefix}.mlp.shared_expert"
    dst = f"{prefix}.mlp.shared_experts"
    pairs = (
        ("w1", "gate_proj.weight"),
        ("w2", "down_proj.weight"),
        ("w3", "up_proj.weight"),
    )
    for tt_name, hf_leaf in pairs:
        for old in (f"{src}.{tt_name}", f"{src}.{tt_name}.weight"):
            if old in layer_sd:
                layer_sd[f"{dst}.{hf_leaf}"] = layer_sd.pop(old)
                break
    leftover = [
        key for key in layer_sd if key.startswith(f"{prefix}.mlp.shared_expert.")
    ]
    raise_unmapped_mlp_keys("generic shared-expert", leftover)


def convert_generic_moe_layer_to_vllm(
    layer_sd: dict,
    layer_idx: int,
    *,
    layer_prefix: str | None = None,
) -> dict:
    """Pack routed experts for PrimeRL families that share DSS's MoE API."""
    if layer_idx < 0:
        return layer_sd
    prefix = layer_prefix if layer_prefix is not None else f"model.layers.{layer_idx}"
    pack_routed_experts_to_vllm(layer_sd, prefix)
    pack_shared_expert_to_vllm(layer_sd, prefix)
    return layer_sd


def to_vllm_vlm_name(name: str) -> str:
    """Rename a Prime-RL VLM parameter key to vLLM's VLM internal layout."""
    if name.startswith("model.visual."):
        return "visual." + name[len("model.visual.") :]
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model.") :]
    if name.startswith("lm_head."):
        return "language_model.lm_head." + name[len("lm_head.") :]
    raise RuntimeError(
        f"to_vllm_vlm_name: no Prime-RL -> vLLM VLM rule for parameter "
        f"name {name!r}. Either Prime-RL grew a new top-level submodule "
        f"or this model is not the Qwen3.5-MoE VLM shape this helper "
        f"was written for. Extend to_vllm_vlm_name with a new rule."
    )


def vllm_layer_converter(model: nn.Module):
    cls_name = type(model).__name__
    if cls_name.startswith("Glm5Next"):
        from arctic_platform.model.implementations.glm53.vllm_weights import (
            convert_glm5_next_layer_to_vllm,
        )

        return convert_glm5_next_layer_to_vllm
    if cls_name.startswith("Qwen3_5Moe"):
        raise RuntimeError(
            "Qwen3.5-MoE vLLM packing stays on the qwen3_5_moe path"
        )
    if cls_name.startswith("NemotronH"):
        from arctic_platform.model.implementations.nemotron_h.vllm_weights import (
            convert_nemotron_h_layer_to_vllm,
        )

        return convert_nemotron_h_layer_to_vllm
    if cls_name.startswith("MiniMaxM2"):
        from arctic_platform.model.implementations.minimax_m2.vllm_weights import (
            convert_minimax_m2_layer_to_vllm,
        )

        return convert_minimax_m2_layer_to_vllm
    return convert_generic_moe_layer_to_vllm


def build_iter_full_vllm_weights(model: nn.Module):
    """Iterator yielding ``(vllm_name, full_tensor)`` on rank 0 for weight sync."""
    convert_layer = vllm_layer_converter(model)

    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist

    is_vlm = bool(getattr(model, "_is_vlm", False))
    layer_prefix_pattern = (
        "model.language_model.layers.{i}" if is_vlm else "model.layers.{i}"
    )

    def _strip_ac_wrapper(name: str) -> str:
        return name.replace("._checkpoint_wrapped_module", "")

    def _layer_idx(name: str) -> int:
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
            try:
                return int(parts[2])
            except ValueError:
                pass
        if (
            len(parts) >= 4
            and parts[0] == "model"
            and parts[1] == "language_model"
            and parts[2] == "layers"
        ):
            try:
                return int(parts[3])
            except ValueError:
                pass
        return -1

    def _iter():
        is_master = dist.get_rank() == 0
        by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
        for name, tensor in named_weight_sync_tensors(model):
            by_layer.setdefault(_layer_idx(name), []).append((name, tensor))

        ordered_layers: list[int] = []
        if -1 in by_layer:
            ordered_layers.append(-1)
        ordered_layers.extend(sorted(k for k in by_layer if k >= 0))

        for layer_idx in ordered_layers:
            layer_sd: dict[str, torch.Tensor] = {}
            for name, tensor in by_layer[layer_idx]:
                if (
                    hasattr(tensor, "group_name")
                    and getattr(tensor, "allreduce", True) is False
                ):
                    ep_pg = ds_groups._get_expert_parallel_group(tensor.group_name)
                    local = tensor.data.contiguous()
                    shards = [
                        torch.empty_like(local)
                        for _ in range(dist.get_world_size(group=ep_pg))
                    ]
                    dist.all_gather(shards, local, group=ep_pg)
                    if is_master:
                        layer_sd[name] = torch.cat(shards, dim=0)
                elif is_master:
                    layer_sd[name] = tensor.data

            if not is_master:
                continue

            layer_sd = {_strip_ac_wrapper(k): v for k, v in layer_sd.items()}

            convert_layer(
                layer_sd,
                layer_idx,
                layer_prefix=(
                    layer_prefix_pattern.format(i=layer_idx) if layer_idx >= 0 else None
                ),
            )

            for name, tensor in layer_sd.items():
                out_name = to_vllm_vlm_name(name) if is_vlm else name
                yield out_name, tensor

    return _iter
