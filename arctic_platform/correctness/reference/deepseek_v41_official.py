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

"""DeepSeek-V4.1 reference: DeepSeek's own ``inference/model.py``, shipped in the fixture, run with autograd.

transformers has no DeepSeek-V4.1, so the independent implementation is the one DeepSeek released beside the
weights. That code is inference-only: its forward runs under ``torch.inference_mode`` and every quantizer, GEMM,
sparse attention and Hyper-Connection mix is a tilelang kernel. Here the ``kernel`` import is replaced by the
PyTorch functions below, which follow those kernels' numerics (fake quantization passes gradients straight
through), and each row is a fresh prefill with empty caches.

Weights are dequantized to bfloat16 on load, as the training loader does for full-parameter runs, and every
linear runs in bfloat16 without simulated FP8 activations. The n-gram table stays FP8 on the CPU and only the
rows a batch looks up reach the GPU. Gradients are reported under Arctic Platform's module names.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import time
import types
from pathlib import Path
from typing import Dict
from typing import List

FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
WEIGHT_BLOCK = 32
ATTENTION_QUERY_CHUNK = 512
CACHE_BUFFERS = ("window_kv_cache", "compress_kv_cache", "k_cache", "kv_state", "score_state", "cache")
LORA_MODULES = {"q_a_proj": "wq_a", "q_b_proj": "wq_b", "kv_proj": "wkv", "o_b_proj": "wo_b"}
OFFICIAL_MODULES = ("engram", "image_processor", "vision", "kernel", "model")

# DeepSeek checkpoint names to Arctic Platform's, as its V4.1 conversion chain renames them.
_LAYER_RENAMES = (
    ("attn_norm.weight", "input_layernorm.weight"),
    ("ffn_norm.weight", "post_attention_layernorm.weight"),
    ("hc_attn_fn", "attn_hc.fn"),
    ("hc_attn_base", "attn_hc.base"),
    ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_fn", "ffn_hc.fn"),
    ("hc_ffn_base", "ffn_hc.base"),
    ("hc_ffn_scale", "ffn_hc.scale"),
    ("attn.wq_a.weight", "self_attn.q_a_proj.weight"),
    ("attn.q_norm.weight", "self_attn.q_a_norm.weight"),
    ("attn.wq_b.weight", "self_attn.q_b_proj.weight"),
    ("attn.wkv.weight", "self_attn.kv_proj.weight"),
    ("attn.kv_norm.weight", "self_attn.kv_norm.weight"),
    ("attn.wo_a.weight", "self_attn.o_a_proj.weight"),
    ("attn.wo_b.weight", "self_attn.o_b_proj.weight"),
    ("attn.attn_sink", "self_attn.sinks"),
    ("ffn.gate.weight", "mlp.router.gate.weight"),
    ("ffn.gate.bias", "mlp.expert_bias"),
    ("ffn.shared_experts.w1.weight", "mlp.shared_expert.w1"),
    ("ffn.shared_experts.w2.weight", "mlp.shared_expert.w2"),
    ("ffn.shared_experts.w3.weight", "mlp.shared_expert.w3"),
)
_LAYER = re.compile(r"^layers\.(\d+)\.(.*)$")
_ROUTED = re.compile(r"^ffn\.experts\.\d+\.(w[123])\.weight$")


def is_deepseek_v41(fixture: str | Path) -> bool:
    config = json.loads((Path(fixture) / "config.json").read_text())
    return config.get("model_type") in {"deepseek_v41", "deepseek_v41_text"}


# ---- PyTorch versions of inference/kernel.py -------------------------------------------------------------


def _pow2_ceil(x):
    """``2 ** ceil(log2(x))`` exactly, as the kernels' exponent-bit arithmetic computes it."""
    import torch

    mantissa, exponent = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(mantissa == 0.5, exponent - 1, exponent))


def _straight_through(x, quantize):
    import torch

    with torch.no_grad():
        quantized = quantize(x)
    return quantized + (x - x.detach())


def fake_fp8(x, block_size: int, round_scale: bool):
    """``act_quant(..., inplace=True)``'s values: per-block E4M3 with an amax scale, optionally a power of two."""
    import torch

    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = _pow2_ceil(amax * (1 / FP8_MAX)) if round_scale else amax * (1 / FP8_MAX)
    quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale
    return quantized.flatten(-2).to(x.dtype)


def round_to_fp4(values):
    """Round to the nearest E2M1 value, ties to the even code, as the hardware conversion does."""
    import torch

    grid = torch.tensor(FP4_VALUES, device=values.device)
    midpoints = (grid[:-1] + grid[1:]) / 2
    magnitude = values.abs()
    code = torch.bucketize(magnitude, midpoints)
    tie = (code < len(midpoints)) & (magnitude == midpoints[code.clamp(max=len(midpoints) - 1)])
    code = torch.where(tie & (code % 2 == 1), code + 1, code)
    return torch.copysign(grid[code], values)


def fake_fp4(x, block_size: int, e4m3_scale: bool):
    """``fp4_act_quant(..., inplace=True)``'s values: E8M0 scales for the indexer, E4M3 ones for compressed KV."""
    import torch

    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        scale = (amax.clamp_min(FP4_MAX * 2.0**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        scale = _pow2_ceil(amax.clamp_min(FP4_MAX * 2.0**-126) * (1 / FP4_MAX))
    quantized = round_to_fp4((blocks / scale).clamp(-FP4_MAX, FP4_MAX)) * scale
    return quantized.flatten(-2).to(x.dtype)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
    if not inplace:
        raise NotImplementedError("the reference runs bfloat16 weights, so no linear asks for FP8 activations")
    x.copy_(_straight_through(x, lambda v: fake_fp8(v, block_size, scale_fmt is not None)))
    return x


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
    import torch

    if not inplace:
        raise NotImplementedError("only the in-place quantize-dequantize form is used")
    x.copy_(_straight_through(x, lambda v: fake_fp4(v, block_size, scale_dtype == torch.float8_e4m3fn)))
    return x


def _attend(q, kv, attn_sink, topk_idxs, softmax_scale):
    import torch

    batch = torch.arange(kv.size(0), device=kv.device)[:, None, None]
    valid = topk_idxs >= 0
    gathered = kv[batch, topk_idxs.long().clamp_min(0)].float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
    # As the kernel: the running max skips the sink (and starts at -1e30, so a row with no keys
    # gives zeros), and P is rounded to bfloat16 for P @ V.
    peak = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30).detach()
    weights = torch.exp(scores - peak)
    total = weights.sum(dim=-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, -1, 1) - peak)
    out = torch.einsum("bmhk,bmkd->bmhd", weights.to(torch.bfloat16).float(), gathered) / total
    return out.to(q.dtype)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Each query attends over its listed KV rows (-1 for none) plus a per-head sink logit.

    Query chunks are recomputed in backward, so only the gathered rows of one chunk are ever held.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    chunks = []
    for start in range(0, q.size(1), ATTENTION_QUERY_CHUNK):
        end = start + ATTENTION_QUERY_CHUNK
        chunks.append(
            checkpoint(
                _attend, q[:, start:end], kv, attn_sink, topk_idxs[:, start:end], softmax_scale, use_reentrant=False
            )
        )
    return torch.cat(chunks, dim=1)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    import torch

    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _quantized_gemm(*args, **kwargs):
    raise NotImplementedError("the reference dequantizes every weight, so no quantized GEMM runs")


def kernel_module() -> types.ModuleType:
    module = types.ModuleType("kernel")
    module.act_quant = act_quant
    module.fp4_act_quant = fp4_act_quant
    module.fp8_gemm = _quantized_gemm
    module.fp4_gemm = _quantized_gemm
    module.sparse_attn = sparse_attn
    module.hc_split_sinkhorn = hc_split_sinkhorn
    return module


def import_official(inference_dir: str | Path) -> types.ModuleType:
    """Import ``inference/model.py`` against the PyTorch kernels, leaving ``sys.modules`` as it was."""
    inference_dir = Path(inference_dir)
    saved = {name: sys.modules.get(name) for name in OFFICIAL_MODULES}
    sys.modules["kernel"] = kernel_module()
    try:
        for name in OFFICIAL_MODULES[:3]:
            spec = importlib.util.spec_from_file_location(name, inference_dir / f"{name}.py")
            sys.modules[name] = module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        spec = importlib.util.spec_from_file_location("deepseek_v41_official_model", inference_dir / "model.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return model


# ---- model construction and loading -----------------------------------------------------------------------


def model_args(official, config: dict, *, rows: int, seq_len: int):
    """``ModelArgs`` from the fixture's HF config, text only: no MTP/DSpark layers and no vision tower."""
    text = config.get("text_config", config)
    rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
    layers = int(text["num_hidden_layers"])
    return official.ModelArgs(
        max_batch_size=rows,
        max_seq_len=seq_len,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=text["vocab_size"],
        dim=text["hidden_size"],
        moe_inter_dim=text["moe_intermediate_size"],
        n_layers=layers,
        n_mtp_layers=0,
        n_heads=text["num_attention_heads"],
        n_routed_experts=text["n_routed_experts"],
        n_shared_experts=text["n_shared_experts"],
        n_activated_experts=text["num_experts_per_tok"],
        score_func=text["scoring_func"],
        norm_topk_prob=text["norm_topk_prob"],
        route_scale=text["routed_scaling_factor"],
        swiglu_limit=text["swiglu_limit"],
        q_lora_rank=text["q_lora_rank"],
        head_dim=text["head_dim"],
        rope_head_dim=text["qk_rope_head_dim"],
        norm_eps=text["rms_norm_eps"],
        o_groups=text["o_groups"],
        o_lora_rank=text["o_lora_rank"],
        window_size=text["sliding_window"],
        compress_ratios=tuple(text["compress_ratios"][:layers]),
        kv_source_layers=tuple(text["kv_source_layer_ids"]),
        index_source_layers=tuple(text["index_source_layer_ids"]),
        compress_rope_theta=text["compress_rope_theta"],
        original_seq_len=rope.get("original_max_position_embeddings", 0),
        rope_theta=text["rope_theta"],
        rope_factor=rope.get("factor", 1),
        beta_fast=rope.get("beta_fast", 32),
        beta_slow=rope.get("beta_slow", 1),
        index_n_heads=text["index_n_heads"],
        index_head_dim=text["index_head_dim"],
        index_topk=text["index_topk"],
        candidate_source_layer=text["candidate_source_layer_id"],
        candidate_topk_blocks=text["candidate_topk_blocks"],
        candidate_block_size=text["candidate_block_size"],
        hc_mult=text["hc_mult"],
        hc_sinkhorn_iters=text["hc_sinkhorn_iters"],
        hc_eps=text["hc_eps"],
        engram_layer_ids=tuple(text["engram_layer_ids"]),
        engram_num_embeddings=tuple(text["engram_num_embeddings"]),
        engram_max_ngram_size=text["engram_max_ngram_size"],
        engram_vocab_size=text["engram_vocab_size"],
        engram_n_heads=text["engram_n_heads"],
        engram_head_dim=text["engram_head_dim"],
        engram_pad_id=text["engram_pad_token_id"],
        engram_compressed_vocab_size=text["engram_compressed_vocab_size"],
        vision_n_layers=0,
        dspark_block_size=0,
    )


def dequantize(weight, scale, shape):
    """FP8 weights carry one E8M0 scale per 32x32 block; packed FP4 (int8, two per byte, low nibble first) one per 32."""
    import torch

    if weight.dtype == torch.int8:
        nibbles = weight.view(torch.uint8)
        table = torch.tensor(FP4_VALUES + tuple(-v for v in FP4_VALUES), device=weight.device)
        values = torch.stack([table[(nibbles & 0x0F).long()], table[(nibbles >> 4).long()]], dim=-1).flatten(-2)
        return values * scale.float().repeat_interleave(WEIGHT_BLOCK, dim=-1)[..., : values.size(-1)]
    rows, cols = shape
    expanded = scale.float().repeat_interleave(WEIGHT_BLOCK, 0)[:rows].repeat_interleave(WEIGHT_BLOCK, 1)[:, :cols]
    return weight.float() * expanded


def mapped_tensor(path: Path, name: str):
    """A read-only uint8 view of one safetensors entry, backed by the page cache rather than a private copy."""
    import numpy as np
    import torch

    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        entry = json.loads(handle.read(header_len))[name]
    start, end = entry["data_offsets"]
    shape = entry["shape"]
    row_bytes = (end - start) // shape[0]
    array = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len + start, shape=(shape[0], row_bytes))
    return torch.from_numpy(array)


def _engram_lookup(self, indices):
    """Gather only the looked-up rows of the CPU-resident FP8 table, then dequantize them on the device."""
    import torch

    flat = indices.reshape(-1).cpu()
    rows = self.table.index_select(0, flat).to(indices.device).view(torch.float8_e4m3fn)
    scales = self.table_scale.index_select(0, flat).to(indices.device).view(torch.float8_e8m0fnu)
    values = rows.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16).view(*indices.shape, -1)


def build_model(fixture: Path, *, rows: int, seq_len: int, device: str):
    """The official model on ``device`` with dequantized weights; the n-gram tables stay FP8 on the CPU."""
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    official = import_official(fixture / "inference")
    config = json.loads((fixture / "config.json").read_text())
    args = model_args(official, config, rows=rows, seq_len=seq_len)
    tokenizer = AutoTokenizer.from_pretrained(str(fixture))

    table_init = official.ParallelEngramEmbedding.__init__

    def init_on_meta(self, *a, **kw):
        with torch.device("meta"):
            table_init(self, *a, **kw)

    official.ParallelEngramEmbedding.__init__ = init_on_meta
    official.ParallelEngramEmbedding.forward = _engram_lookup
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = official.Transformer(args, tokenizer)
    finally:
        torch.set_default_dtype(previous)
        official.ParallelEngramEmbedding.__init__ = table_init

    weight_map = json.loads((fixture / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def tensor(name):
        file = weight_map[name]
        if file not in handles:
            handles[file] = safe_open(str(fixture / file), framework="pt", device="cpu")
        return handles[file].get_tensor(name)

    params = dict(model.named_parameters())
    loaded, unused = set(), []
    for name in sorted(weight_map):
        if name.endswith(".scale") and name.removesuffix("scale") + "weight" in weight_map:
            continue
        if name not in params:
            unused.append(name)
            continue
        param = params[name]
        if ".engram.embed." in name:
            scale_name = name.removesuffix("weight") + "scale"
            module = model.get_submodule(name.rsplit(".", 1)[0])
            module.table = mapped_tensor(fixture / weight_map[name], name)
            module.table_scale = mapped_tensor(fixture / weight_map[scale_name], scale_name)
            loaded.update({name, scale_name})
            continue
        value = tensor(name)
        scale_name = name.removesuffix("weight") + "scale"
        if name.endswith(".weight") and scale_name in weight_map:
            value = dequantize(value.to(device), tensor(scale_name).to(device), tuple(param.shape))
        with torch.no_grad():
            param.copy_(value.to(device=device, dtype=param.dtype).view_as(param))
        loaded.add(name)
    missing = sorted(set(params) - loaded)
    if missing:
        raise ValueError(f"fixture lacks {len(missing)} official parameters, e.g. {missing[:5]}")
    unexpected = [name for name in unused if not name.endswith(".gate.bias_vl")]
    if unexpected:
        raise ValueError(f"fixture holds {len(unexpected)} tensors the text model has no place for: {unexpected[:5]}")
    return official, model


# ---- autograd forward ------------------------------------------------------------------------------------


def _fresh_caches(official, model) -> None:
    """Every row is a prefill from position 0: caches start empty and carry no autograd history."""
    import torch

    for module in model.modules():
        for name in CACHE_BUFFERS:
            buffer = module._buffers.get(name)
            if buffer is not None:
                fill = -torch.inf if name == "score_state" else 0
                module._buffers[name] = torch.full_like(buffer, fill)
    official.shared_attn.__init__()


def _undecorated(function):
    return getattr(function, "__wrapped__", function)


def final_hidden(official, model, input_ids):
    """``Transformer.forward`` up to the final norm, with autograd and without sampling."""
    import torch

    _fresh_caches(official, model)
    hashes = None
    if model.engram_hash is not None:
        with torch.no_grad():
            hashes = _undecorated(type(model.engram_hash).forward)(model.engram_hash, input_ids, 0, None)
    h = model.embed(input_ids).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = official.make_identity_pre_mix(h, model.hc_mult)
    for layer in model.layers:
        if layer.engram is not None:
            h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
        h, pre_mix = layer(h, 0, pre_mix, None)
    return model.norm(layer.hc_pre(h, pre_mix))


def _indexer_without_grad(forward):
    import torch

    def run(self, *args, **kwargs):
        # The indexer only picks positions; its top-k carries no gradient back to the loss.
        with torch.no_grad():
            return forward(self, *args, **kwargs)

    return run


# ---- LoRA --------------------------------------------------------------------------------------------------


def attach_lora(model, adapter: Path, device: str) -> Dict[str, object]:
    """PEFT's LoRA forward on the official projections; returns the factors under PEFT's names."""
    import torch
    from safetensors.torch import load_file

    config = json.loads((adapter / "adapter_config.json").read_text())
    scaling = config["lora_alpha"] / config["r"]
    if config.get("use_rslora"):
        scaling = config["lora_alpha"] / math.sqrt(config["r"])
    pattern = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.self_attn\.(\w+)\.lora_([AB])\.weight$")
    factors: Dict[tuple, torch.Tensor] = {}
    for name, value in load_file(str(adapter / "adapter_model.safetensors")).items():
        match = pattern.match(name)
        if match is None or match.group(2) not in LORA_MODULES:
            raise ValueError(f"unexpected adapter tensor {name}")
        factors[(int(match.group(1)), match.group(2), match.group(3))] = value
    params = {}
    for (layer, proj, which), value in sorted(factors.items()):
        if which != "A":
            continue
        a = torch.nn.Parameter(value.to(device=device, dtype=torch.float32))
        b = torch.nn.Parameter(factors[(layer, proj, "B")].to(device=device, dtype=torch.float32))
        module = getattr(model.layers[layer].attn, LORA_MODULES[proj])
        base = module.forward

        def forward(x, base=base, a=a, b=b):
            result = base(x)
            return (result + (x.to(a.dtype) @ a.t()) @ b.t() * scaling).to(result.dtype)

        module.forward = forward
        prefix = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
        params[f"{prefix}.lora_A.default.weight"] = a
        params[f"{prefix}.lora_B.default.weight"] = b
    return params


# ---- names -------------------------------------------------------------------------------------------------


def platform_name(name: str) -> str:
    """Arctic Platform's name for an official parameter; routed experts fold into one ``mlp.experts.wN``."""
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name == "head.weight":
        return "lm_head.weight"
    match = _LAYER.match(name)
    if match is None:
        raise ValueError(f"no Arctic Platform name for {name}")
    layer, rest = match.groups()
    routed = _ROUTED.match(rest)
    if routed:
        rest = f"mlp.experts.{routed.group(1)}"
    else:
        for old, new in _LAYER_RENAMES:
            if rest == old:
                rest = new
                break
        else:
            rest = rest.replace("attn.compressor.", "self_attn.compressor.").replace(
                "attn.indexer.", "self_attn.indexer."
            )
    return f"model.layers.{layer}.{rest}"


def platform_norms(gradients: Dict[str, "object"], device: str) -> Dict[str, float]:
    """L2 norms under the names ``harness.names.align`` gives the target's tensors."""
    import torch

    from ..harness.names import _canonical_dss

    squares: Dict[str, float] = {}
    for name, grad in gradients.items():
        key = platform_name(name)
        squares[key] = squares.get(key, 0.0) + float(torch.linalg.vector_norm(grad.to(device)).double() ** 2)
    canonical = _canonical_dss({key: math.sqrt(value) for key, value in squares.items()})
    return {key: entry.value for key, entry in canonical.items()}


# ---- run ---------------------------------------------------------------------------------------------------


def run(
    fixture: str | Path,
    batch,
    *,
    token_budget: int = 65536,
    ce_chunk: int = 2048,
    peft_adapter_path: str | None = None,
    device: str = "cuda",
):
    """Forward-backward the whole request and return one L2 gradient norm per parameter."""
    import torch
    import torch.nn.functional as F

    from ..harness.batches import IGNORE_INDEX
    from .hf_single_gpu import ReferenceResult
    from .hf_single_gpu import row_groups

    fixture = Path(fixture)
    seq_len = int(batch.input_ids.shape[1])
    groups = row_groups([seq_len] * int(batch.input_ids.shape[0]), seq_len, token_budget)
    official, model = build_model(fixture, rows=max(len(g) for g in groups), seq_len=seq_len, device=device)
    official.Indexer.forward = _indexer_without_grad(official.Indexer.forward)

    lora = attach_lora(model, Path(peft_adapter_path), device) if peft_adapter_path else {}
    trainable = lora or {name: param for name, param in model.named_parameters() if ".engram.embed." not in name}
    for param in model.parameters():
        param.requires_grad_(False)
    for param in trainable.values():
        param.requires_grad_(True)

    input_ids = batch.input_ids.to(device)
    labels = batch.shifted_labels().to(device)
    active_tokens = int((labels != IGNORE_INDEX).sum())
    head = model.head.weight
    accumulator: Dict[str, torch.Tensor] = {}
    total_loss = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    with torch.device(device):
        for rows in groups:
            index = torch.tensor(rows, device=device)
            hidden = final_hidden(official, model, input_ids.index_select(0, index))
            detached = hidden.detach().requires_grad_(True)
            flat = detached.reshape(-1, detached.size(-1))
            flat_labels = labels.index_select(0, index).reshape(-1)
            for start in range(0, flat_labels.numel(), ce_chunk):
                chunk_labels = flat_labels[start : start + ce_chunk]
                if int((chunk_labels != IGNORE_INDEX).sum()) == 0:
                    continue
                # ParallelHead keeps the head in float32 and projects a float32 input.
                logits = F.linear(flat[start : start + ce_chunk].float(), head)
                loss = (
                    F.cross_entropy(logits, chunk_labels, ignore_index=IGNORE_INDEX, reduction="sum") / active_tokens
                )
                loss.backward()
                total_loss += float(loss.item())
            hidden.backward(detached.grad)
            for name, param in trainable.items():
                if param.grad is None:
                    continue
                grad = param.grad.detach().to(device="cpu", dtype=torch.float32)
                accumulator[name] = accumulator[name].add_(grad) if name in accumulator else grad
                param.grad = None
            print(f"[reference] rows {rows} loss so far {total_loss:.6f} ({time.time() - t0:.0f}s)", flush=True)

    skipped = sorted(name for name in trainable if name not in accumulator)
    if lora:
        norms = {name: float(torch.linalg.vector_norm(g.to(device))) for name, g in accumulator.items()}
    else:
        norms = platform_norms(accumulator, device)
    return ReferenceResult(
        loss=total_loss,
        grad_norms=norms,
        active_tokens=active_tokens,
        microbatches=len(groups),
        peak_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
        skipped_no_grad=skipped,
    )


def check_model_args(inference_dir: str | Path, config: dict) -> List[str]:
    """Fields where ``model_args`` disagrees with DeepSeek's own ``inference/config.json`` for the same model."""
    official = import_official(inference_dir)
    ours = model_args(official, config, rows=1, seq_len=1)
    theirs = json.loads((Path(inference_dir) / "config.json").read_text())
    skip = {"max_batch_size", "max_seq_len", "dtype", "expert_dtype", "n_mtp_layers", "compress_ratios"}
    diffs = []
    for key, value in theirs.items():
        if key in skip or key.startswith(("vision_", "dspark_")) or not hasattr(ours, key):
            continue
        mine = getattr(ours, key)
        if (tuple(value) if isinstance(value, list) else value) != mine:
            diffs.append(f"{key}: ours={mine!r} official={value!r}")
    if tuple(theirs["compress_ratios"][: ours.n_layers]) != ours.compress_ratios:
        diffs.append("compress_ratios")
    return diffs
