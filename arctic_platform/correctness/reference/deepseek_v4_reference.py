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

"""Upstream DeepSeek-V4 brought to the semantics of DeepSeek's ``inference/model.py``.

- RMSNorm returns the input dtype. transformers keeps the norm weights in float32 and multiplies them after
  casting, so a bfloat16 input leaves the norm as float32 and the next bfloat16 projection rejects it.
- Attention keys/values are FP8-simulated on their non-rope channels (``act_quant(kv[..., :-rd], 64, ...)``),
  which transformers omits. Gradients pass straight through.
"""

from __future__ import annotations

FP8_MAX = 448.0
KV_TILE = 64
KV_FLOOR = 1e-4


def rms_norm_forward(self, hidden_states):
    import torch

    dtype = hidden_states.dtype
    x = hidden_states.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
    return (self.weight * x).to(dtype)


def patch_rms_norm() -> None:
    from transformers.models.deepseek_v4 import modeling_deepseek_v4

    modeling_deepseek_v4.DeepseekV4RMSNorm.forward = rms_norm_forward


def fake_quant_kv(x, rope_dim: int, *, floor_scale: bool):
    """FP8 e4m3 round trip of each 64-channel tile before the trailing ``rope_dim``, with a power-of-two scale.

    The sliding-window cache floors a tile's amax at 1e-4 and the compressor cache floors its scale there.
    """
    import torch

    nope = x.shape[-1] - rope_dim
    tiles = x[..., :nope].float().unflatten(-1, (nope // KV_TILE, KV_TILE))
    amax = tiles.abs().amax(-1, keepdim=True)
    raw = (amax / FP8_MAX).clamp_min(KV_FLOOR) if floor_scale else amax.clamp_min(KV_FLOOR) / FP8_MAX
    mantissa, exponent = torch.frexp(raw)
    scale = torch.ldexp(torch.ones_like(raw), exponent - (mantissa == 0.5).to(exponent.dtype))
    quantized = (tiles / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale
    simulated = torch.cat([quantized.flatten(-2).to(x.dtype), x[..., nope:]], dim=-1)
    return x + (simulated - x).detach()


def install_kv_fake_quant(model) -> int:
    """Hook every attention's window KV (``kv_norm``, before rope touches only the trailing channels) and its
    own compressor's output; the indexer's compressor only ranks slots and is left alone."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention

    hooked = 0
    for module in model.modules():
        if not isinstance(module, DeepseekV4Attention):
            continue
        rope_dim = module.config.qk_rope_head_dim
        module.kv_norm.register_forward_hook(
            lambda _m, _i, out, rd=rope_dim: fake_quant_kv(out, rd, floor_scale=False)
        )
        if module.compressor is not None:
            module.compressor.register_forward_hook(
                lambda _m, _i, out, rd=rope_dim: (fake_quant_kv(out[0], rd, floor_scale=True), *out[1:])
            )
        hooked += 1
    return hooked
