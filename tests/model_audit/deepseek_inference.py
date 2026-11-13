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

"""Reference built from the inference code a DeepSeek checkpoint ships (``inference/model.py``).

transformers has no implementation of these models, so the upstream code is the independent reference. Its tilelang
kernels are replaced by plain torch: quantized weights are dequantized into ordinary bf16/fp32 linears, activation
quantization is the identity (the trainer computes in bf16 too), and ``sparse_attn`` / ``hc_split_sinkhorn`` follow
the kernels' math. Autograd runs through all of it, so the reference also supplies gradients.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import audit_lib
import torch
from torch.utils.checkpoint import checkpoint

_FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_ATTN_QUERY_CHUNK = 128

# ModelArgs field -> HF text_config key, where the names differ; every other field has the same name in both.
_TEXT_KEYS = {
    "dim": "hidden_size",
    "moe_inter_dim": "moe_intermediate_size",
    "n_layers": "num_hidden_layers",
    "n_mtp_layers": "num_nextn_predict_layers",
    "n_heads": "num_attention_heads",
    "n_activated_experts": "num_experts_per_tok",
    "score_func": "scoring_func",
    "route_scale": "routed_scaling_factor",
    "rope_head_dim": "qk_rope_head_dim",
    "norm_eps": "rms_norm_eps",
    "window_size": "sliding_window",
    "kv_source_layers": "kv_source_layer_ids",
    "index_source_layers": "index_source_layer_ids",
    "candidate_source_layer": "candidate_source_layer_id",
    "engram_pad_id": "engram_pad_token_id",
    "dspark_n_activated_experts": "dspark_num_experts_per_tok",
}
_ROPE_KEYS = {"original_seq_len": "original_max_position_embeddings", "rope_factor": "factor"}
_VISION_KEYS = {
    "vision_n_layers": "num_hidden_layers",
    "vision_dim": "hidden_size",
    "vision_n_heads": "num_attention_heads",
    "vision_inter_dim": "intermediate_size",
    "vision_patch_size": "patch_size",
    "vision_rope_theta": "rope_theta",
    "vision_downsample_ratio": "downsample_ratio",
    "vision_max_n_token": "max_image_tokens",
    "vision_min_pixels": "min_pixels",
    "vision_max_wh_ratio": "max_wh_ratio",
}


def inference_dir(checkpoint: str) -> Path:
    path = audit_lib._resolve_file(checkpoint, "inference/model.py")
    assert path is not None, f"{checkpoint} ships no inference/model.py"
    return Path(path).parent


# ---------------------------------------------------------------------------------------------------------------------
# torch stand-ins for the tilelang kernels
# ---------------------------------------------------------------------------------------------------------------------


def _act_quant(x, *args, **kwargs):
    if len(args) >= 4 and args[3] is True or kwargs.get("inplace"):
        return x
    return x, None


def _fp4_act_quant(x, *args, **kwargs):
    return x


def _quantized_gemm(*args, **kwargs):
    raise AssertionError("quantized GEMM reached: the reference dequantizes every weight")


def _sparse_attn_chunk(q, kv, attn_sink, idxs, softmax_scale):
    valid = idxs >= 0
    gathered = kv[0][idxs[0].clamp_min(0).long()].float()  # [m, topk, d]
    scores = torch.einsum("mhd,mtd->mht", q[0].float(), gathered) * softmax_scale
    scores = scores.masked_fill(~valid[0].unsqueeze(1), -math.inf)
    peak = torch.maximum(scores.amax(-1), attn_sink.float().view(1, -1)).clamp_min(-1e30)
    weights = torch.exp(scores - peak.unsqueeze(-1))
    denom = weights.sum(-1) + torch.exp(attn_sink.float().view(1, -1) - peak)
    out = torch.einsum("mht,mtd->mhd", weights, gathered) / denom.unsqueeze(-1)
    return out.unsqueeze(0).to(q.dtype)


def _sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.sparse_attn: each query attends to the kv rows its index list names (-1 = none), plus a sink logit."""
    assert q.size(0) == 1, "the reference runs one sequence per call"
    outs = []
    for start in range(0, q.size(1), _ATTN_QUERY_CHUNK):
        sl = slice(start, start + _ATTN_QUERY_CHUNK)
        args = (q[:, sl], kv, attn_sink, topk_idxs[:, sl], softmax_scale)
        outs.append(
            checkpoint(_sparse_attn_chunk, *args, use_reentrant=False)
            if q.requires_grad
            else _sparse_attn_chunk(*args)
        )
    return torch.cat(outs, dim=1)


def _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """kernel.hc_split_sinkhorn: pre/post gates and a Sinkhorn-normalized comb matrix from one projection."""
    hc = hc_mult
    mixes = mixes.float()
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _kernel_module() -> types.ModuleType:
    kernel = types.ModuleType("kernel")
    kernel.act_quant = _act_quant
    kernel.fp4_act_quant = _fp4_act_quant
    kernel.fp8_gemm = _quantized_gemm
    kernel.fp4_gemm = _quantized_gemm
    kernel.sparse_attn = _sparse_attn
    kernel.hc_split_sinkhorn = _hc_split_sinkhorn
    return kernel


@functools.lru_cache(maxsize=None)
def load_module(checkpoint: str) -> types.ModuleType:
    """The checkpoint's ``inference/model.py``, importing its sibling modules and the torch kernels above."""
    directory = inference_dir(checkpoint)
    saved = {name: sys.modules.get(name) for name in ("kernel", "engram", "image_processor", "vision")}
    sys.modules["kernel"] = _kernel_module()
    sys.path.insert(0, str(directory))
    try:
        for name in ("engram", "image_processor", "vision"):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(
            f"deepseek_inference_{abs(hash(str(directory)))}", directory / "model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


# ---------------------------------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------------------------------


def model_args(module, config: dict, *, quantized: bool, max_seq_len: int = 8192):
    """``ModelArgs`` for an HF ``config.json``. ``quantized`` keeps the checkpoint's FP8/FP4 parameter dtypes."""
    import dataclasses

    text = audit_lib.text_config_of(config)
    fields = {f.name for f in dataclasses.fields(module.ModelArgs)}
    kwargs = {}
    for field in fields:
        key = _TEXT_KEYS.get(field, field)
        if key in text:
            kwargs[field] = text[key]
    rope = text.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type")) == "yarn":
        for field, key in _ROPE_KEYS.items():
            kwargs[field] = rope[key]
        kwargs["beta_fast"], kwargs["beta_slow"] = rope["beta_fast"], rope["beta_slow"]
    vision = config.get("vision_config") or {}
    for field, key in _VISION_KEYS.items():
        if key in vision:
            kwargs[field] = vision[key]
    if "image_token_id" in config:
        kwargs["image_token_id"] = config["image_token_id"]
    for field, value in kwargs.items():
        if isinstance(value, list):
            kwargs[field] = tuple(value)
    quant = config.get("quantization_config") or {}
    if quantized and quant:
        assert quant.get("quant_method") == "fp8" and quant.get("weight_block_size") == [32, 32], quant
        kwargs["dtype"], kwargs["expert_dtype"] = "fp8", quant.get("expert_dtype")
    else:
        kwargs["dtype"], kwargs["expert_dtype"] = "bf16", None
    kwargs.update(max_batch_size=1, max_seq_len=max_seq_len)
    return module.ModelArgs(**kwargs)


def shipped_args_mismatches(checkpoint: str) -> dict:
    """Fields where ``model_args(config.json)`` differs from the ``inference/config.json`` DeepSeek ships."""
    module = load_module(checkpoint)
    config = json.loads(Path(audit_lib._resolve_file(checkpoint, "config.json")).read_text())
    args = model_args(module, config, quantized=True)
    shipped = json.loads((inference_dir(checkpoint) / "config.json").read_text())
    out = {}
    for key, value in shipped.items():
        ours = getattr(args, key, "<missing>")
        if (tuple(value) if isinstance(value, list) else value) != ours:
            out[key] = {"shipped": value, "derived": ours}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Quantization in the published formats
# ---------------------------------------------------------------------------------------------------------------------


def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of the two quantizers, from the stored tensors alone."""
    if weight.dtype == torch.int8:
        u8 = weight.view(torch.uint8)
        table = _FP4_TABLE.to(weight.device)
        values = torch.stack([table[(u8 & 0xF).long()], table[(u8 >> 4).long()]], dim=-1).flatten(1)
    else:
        values = weight.float()
    rows, cols = values.shape
    br, bc = -(-rows // scale.shape[0]), -(-cols // scale.shape[1])
    expanded = scale.float().repeat_interleave(br, 0)[:rows].repeat_interleave(bc, 1)[:, :cols]
    return values * expanded


def _tokenizer(directory: str):
    import transformers

    return transformers.AutoTokenizer.from_pretrained(directory)


# ---------------------------------------------------------------------------------------------------------------------
# Reference forward
# ---------------------------------------------------------------------------------------------------------------------


def _dequantized_state(ckpt: str) -> dict[str, torch.Tensor]:
    raw = audit_lib.read_checkpoint_tensors(Path(ckpt))
    out = {}
    for name, value in raw.items():
        if name.endswith(".scale") and name[: -len("scale")] + "weight" in raw and "engram.embed" not in name:
            continue
        scale = raw.get(name[: -len("weight")] + "scale") if name.endswith("weight") else None
        if scale is not None and "engram.embed" not in name:
            value = dequantize(value, scale)
        out[name] = value
    return out


class Reference:
    """The upstream model on one GPU, without inference_mode, returning full-sequence logits."""

    def __init__(self, checkpoint: str, ckpt: str, dtype: torch.dtype, max_seq_len: int):
        module = load_module(checkpoint)
        self.module = module
        config = json.loads(Path(ckpt, "config.json").read_text())
        args = model_args(module, config, quantized=False, max_seq_len=max_seq_len)
        torch.manual_seed(0)
        with torch.device("cuda"), module.set_dtype(dtype):
            model = module.Transformer(args, tokenizer=_tokenizer(ckpt))
        state = _dequantized_state(ckpt)
        own = dict(model.named_parameters())
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        assert not missing and not unexpected, f"missing {missing[:8]}, unexpected {unexpected[:8]}"
        stored = (torch.float8_e4m3fn, torch.float8_e8m0fnu)
        for name, param in own.items():
            target = param.dtype if param.dtype in stored or dtype == torch.bfloat16 else torch.float32
            param.data = state[name].to(device="cuda", dtype=target)
            param.requires_grad_(param.dtype not in stored)
        if dtype == torch.float32:
            for mod in model.modules():
                if isinstance(mod, module.ParallelEngramEmbedding):
                    # its lookup always returns bf16; FP8 rows times a power-of-two scale are exact there
                    mod.forward = functools.partial(lambda m, idx: type(m).forward(m, idx).float(), mod)
        self.model = model
        self.dtype = dtype

    def _reset_state(self) -> None:
        for mod in self.model.modules():
            for name, buf in list(mod._buffers.items()):
                if buf is not None and ("cache" in name or "state" in name):
                    mod._buffers[name] = (
                        torch.zeros_like(buf) if "score" not in name else torch.full_like(buf, -math.inf)
                    )
        shared = self.module.shared_attn
        shared.compress_kv = shared.index_k = shared.topk_idxs = shared.candidates = None

    def named_parameters(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [1, s] -> fp32 logits [1, s, vocab]; mirrors Transformer.forward at start_pos 0, text only."""
        self._reset_state()
        model, module = self.model, self.module
        ids = ids.cuda()
        # the upstream code allocates index tensors on the default device
        with torch.device("cuda"):
            with torch.inference_mode(False):
                hashes = (
                    model.engram_hash.forward.__wrapped__(model.engram_hash, ids, 0, None)
                    if model.engram_hash is not None
                    else None
                )
            h = model.embed(ids)
            if self.dtype == torch.float32:
                h = h.float()
            h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
            pre_mix = module.make_identity_pre_mix(h, model.hc_mult)
            for layer in model.layers:
                if layer.engram is not None:
                    h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
                h, pre_mix = layer(h, 0, pre_mix, None)
            h = model.layers[-1].hc_pre(h, pre_mix)
            return model.head(model.norm(h), full_logits=True)
