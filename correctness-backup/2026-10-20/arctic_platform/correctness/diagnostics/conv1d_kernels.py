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

"""Do the short-convolution choices in the gated delta net agree?

The raw varlen kernel can call ``causal_conv1d_fn`` with a segment index, while a segmented reference calls
``nn.Conv1d`` once per packed row and slices the causal tail. They are supposed to compute the same depthwise
convolution followed by SiLU, so any difference between them lands on ``conv1d.weight`` and everything the
convolution feeds.

The Qwen3.6 product path deliberately avoids the raw segmented-index kernel for multi-row packed batches.
This diagnostic therefore prints both the raw kernel comparison and the product-selected branch comparison.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained  # noqa: E402

ROW_TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
ROWS = int(os.environ.get("PROBE_ROWS", 8))
LAYERS = int(os.environ.get("PROBE_LAYERS", 4))
SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"


def report(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a32, b32 = a.detach().float(), b.detach().float()
    diff = (a32 - b32).abs()
    norm_a, norm_b = float(a32.norm()), float(b32.norm())
    print(
        f"{name:26} norm {norm_a:>12.6f} against {norm_b:>12.6f}  "
        f"norm diff {abs(norm_a - norm_b):>10.3e}  max element {float(diff.max()):>10.3e}  "
        f"exactly equal {int((diff == 0).sum()):>9}/{diff.numel()}"
    )


def run_segmented_conv(conv, x: torch.Tensor, tokens_per_row: int) -> torch.Tensor:
    conv_outs = []
    for start in range(0, x.shape[-1], tokens_per_row):
        stop = start + tokens_per_row
        conv_outs.append(conv(x[:, :, start:stop])[:, :, :tokens_per_row])
    return F.silu(torch.cat(conv_outs, dim=-1))


def run_product_selected_conv(conv, activation: str, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        _has_multiple_packed_sequences,
    )
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        _packed_sequence_indices,
    )
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        _segmented_causal_conv1d,
    )

    if _has_multiple_packed_sequences(cu_seqlens):
        return _segmented_causal_conv1d(conv, x, cu_seqlens)

    from causal_conv1d import causal_conv1d_fn

    return causal_conv1d_fn(
        x=x,
        weight=conv.weight.squeeze(1),
        bias=conv.bias,
        activation=activation,
        seq_idx=_packed_sequence_indices(cu_seqlens, batch_size=x.shape[0], seq_len=x.shape[-1], device=x.device),
    )


def run_paths(
    conv, activation: str, x: torch.Tensor, upstream: torch.Tensor, seq_idx: torch.Tensor, cu_seqlens: torch.Tensor
) -> tuple:
    from causal_conv1d import causal_conv1d_fn

    outputs, weight_grads, input_grads = {}, {}, {}
    paths = {
        "segmented": lambda xi: run_segmented_conv(conv, xi, ROW_TOKENS),
        "raw_seq_idx": lambda xi: causal_conv1d_fn(
            x=xi,
            weight=conv.weight.squeeze(1),
            bias=conv.bias,
            activation=activation,
            seq_idx=seq_idx,
        ),
        "product_selected": lambda xi: run_product_selected_conv(conv, activation, xi, cu_seqlens),
    }
    for name, run_path in paths.items():
        conv.zero_grad(set_to_none=True)
        xi = x.detach().clone().requires_grad_(True)
        out = run_path(xi)
        out.backward(upstream)
        outputs[name] = out.detach()
        weight_grads[name] = conv.weight.grad.detach().clone()
        input_grads[name] = xi.grad.detach().clone()
    return outputs, weight_grads, input_grads


def print_comparison(
    title: str, baseline: str, candidate: str, outputs: dict, weight_grads: dict, input_grads: dict
) -> None:
    print("")
    print(title)
    print(f"{'quantity':26} {baseline:>17}     {candidate:>16}")
    report("forward output", outputs[baseline], outputs[candidate])
    report("conv1d.weight gradient", weight_grads[baseline], weight_grads[candidate])
    report("input gradient", input_grads[baseline], input_grads[candidate])


def main() -> int:
    from transformers import AutoModelForCausalLM

    path = materialize_pretrained(SOURCE, f"{CACHE_ROOT}/Qwen3.8-27B-{LAYERS}L", LAYERS).cache_path
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True)
    inner = getattr(model, "model", model)
    gdn = next(layer.linear_attn for layer in inner.layers if hasattr(layer, "linear_attn"))
    conv = gdn.conv1d.cuda()
    channels = conv.weight.shape[0]
    causal_conv = getattr(gdn, "causal_conv1d_fn", getattr(gdn, "_causal_conv1d_fn", None))
    print(
        f"conv1d: {channels} channels, kernel {conv.weight.shape[-1]}, groups {conv.groups}, "
        f"bias {conv.bias is not None}, activation {gdn.activation}, {ROWS} rows x {ROW_TOKENS:,} tokens"
    )
    print(f"causal_conv1d_fn available on the module: {causal_conv is not None}")

    torch.manual_seed(SEED)
    tokens = ROWS * ROW_TOKENS
    # AP's packed path flattens rows to one batch element and passes row ids as ``seq_idx``.
    x = torch.randn(1, tokens, channels, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    upstream = torch.randn(1, tokens, channels, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    packed_seq_idx = (
        torch.arange(ROWS, dtype=torch.int32, device="cuda").repeat_interleave(ROW_TOKENS).reshape(1, tokens)
    )
    cu_seqlens = torch.arange(0, tokens + 1, ROW_TOKENS, dtype=torch.int32, device="cuda")

    outputs, weight_grads, input_grads = run_paths(conv, gdn.activation, x, upstream, packed_seq_idx, cu_seqlens)
    print_comparison(
        "raw causal_conv1d_fn(seq_idx=...) on packed 8-row flattened shape",
        "segmented",
        "raw_seq_idx",
        outputs,
        weight_grads,
        input_grads,
    )
    print_comparison(
        "product-selected branch on packed 8-row flattened shape",
        "segmented",
        "product_selected",
        outputs,
        weight_grads,
        input_grads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
