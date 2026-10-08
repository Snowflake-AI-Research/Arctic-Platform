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

"""Do the two short-convolution kernels in the gated delta net agree?

The engine's varlen path calls ``causal_conv1d_fn`` with a segment index, while a stock HuggingFace
forward calls ``nn.Conv1d`` and slices the causal tail. They compute the same depthwise convolution
followed by SiLU, so any difference between them is the kernel, not the model -- and it lands on
``conv1d.weight`` and on everything the convolution feeds, which is where the residual disagreement sits.

Both paths are given the same input and the same upstream gradient.
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

TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
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


def main() -> int:
    from causal_conv1d import causal_conv1d_fn
    from transformers import AutoModelForCausalLM

    path = materialize_pretrained(SOURCE, f"{CACHE_ROOT}/Qwen3.8-27B-{LAYERS}L", LAYERS).cache_path
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True)
    inner = getattr(model, "model", model)
    gdn = next(layer.linear_attn for layer in inner.layers if hasattr(layer, "linear_attn"))
    conv = gdn.conv1d.cuda()
    channels = conv.weight.shape[0]
    print(
        f"conv1d: {channels} channels, kernel {conv.weight.shape[-1]}, groups {conv.groups}, "
        f"bias {conv.bias is not None}, activation {gdn.activation}, {TOKENS:,} tokens"
    )
    print(f"causal_conv1d_fn available on the module: {gdn.causal_conv1d_fn is not None}")

    torch.manual_seed(SEED)
    # causal_conv1d_fn accepts a segment index only in channel-last layout, which is what the engine hands
    # it: the projection produces [batch, tokens, channels] and transposes the view without copying.
    x = torch.randn(1, TOKENS, channels, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    upstream = torch.randn(1, TOKENS, channels, dtype=torch.bfloat16, device="cuda").transpose(1, 2)

    outputs, weight_grads, input_grads = [], [], []
    for use_kernel in (False, True):
        conv.zero_grad(set_to_none=True)
        xi = x.detach().clone().requires_grad_(True)
        if use_kernel:
            seq_idx = torch.zeros(1, TOKENS, dtype=torch.int32, device="cuda")
            out = causal_conv1d_fn(
                x=xi, weight=conv.weight.squeeze(1), bias=conv.bias, activation=gdn.activation, seq_idx=seq_idx
            )
        else:
            out = F.silu(conv(xi)[..., :TOKENS])
        out.backward(upstream)
        outputs.append(out)
        weight_grads.append(conv.weight.grad.detach().clone())
        input_grads.append(xi.grad.detach().clone())

    print("")
    print(f"{'quantity':26} {'nn.Conv1d':>17}     {'causal_conv1d_fn':>12}")
    report("forward output", outputs[0], outputs[1])
    report("conv1d.weight gradient", weight_grads[0], weight_grads[1])
    report("input gradient", input_grads[0], input_grads[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
