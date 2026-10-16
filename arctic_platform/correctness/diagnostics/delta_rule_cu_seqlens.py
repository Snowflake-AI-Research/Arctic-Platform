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

"""Does the chunked delta rule return the same gradients when it is told the sequence boundaries?

The engine's packed mixer passes ``cu_seqlens`` to ``chunk_gated_delta_rule``; the stock mixer does not.
With one sequence filling the window the boundaries are the ends of the window, so the two calls describe
the same problem. Everything the rule returns a gradient for is compared: query, key, value, the gate and
the decay.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained  # noqa: E402

TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
LAYERS = int(os.environ.get("PROBE_LAYERS", 4))
SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"


def compare(name, a, b):
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    na, nb = float(a.norm()), float(b.norm())
    print(
        f"{name:20} {na:>14.6f} {nb:>14.6f} {abs(na - nb):>11.3e} {float(diff.max()):>11.3e} "
        f"{int((diff == 0).sum()):>10}/{diff.numel()}"
    )


def main() -> int:
    from transformers import AutoModelForCausalLM

    path = materialize_pretrained(SOURCE, f"{CACHE_ROOT}/Qwen3.8-27B-{LAYERS}L", LAYERS).cache_path
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True)
    inner = getattr(model, "model", model)
    gdn = next(layer.linear_attn for layer in inner.layers if hasattr(layer, "linear_attn")).cuda()
    rule = gdn.chunk_gated_delta_rule
    heads_v, heads_k = gdn.num_v_heads, gdn.num_k_heads
    print(
        f"{heads_v} value heads, {heads_k} key heads, head_k {gdn.head_k_dim}, head_v {gdn.head_v_dim}, "
        f"{TOKENS:,} tokens in one sequence"
    )

    torch.manual_seed(SEED)

    def make(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda") * 0.5

    query = make(1, TOKENS, heads_v, gdn.head_k_dim)
    key = make(1, TOKENS, heads_v, gdn.head_k_dim)
    value = make(1, TOKENS, heads_v, gdn.head_v_dim)
    beta = torch.rand(1, TOKENS, heads_v, dtype=torch.bfloat16, device="cuda")
    gate = -torch.rand(1, TOKENS, heads_v, dtype=torch.float32, device="cuda")
    upstream = make(1, TOKENS, heads_v, gdn.head_v_dim)
    cu_seqlens = torch.tensor([0, TOKENS], dtype=torch.int32, device="cuda")

    results = []
    for boundaries in (None, cu_seqlens):
        tensors = [t.detach().clone().requires_grad_(True) for t in (query, key, value, beta, gate)]
        out, _ = rule(
            tensors[0],
            tensors[1],
            tensors[2],
            g=tensors[4],
            beta=tensors[3],
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=boundaries,
        )
        out.backward(upstream)
        results.append((out.detach(), [t.grad.detach() for t in tensors]))

    print("")
    print(f"{'quantity':20} {'without':>14} {'with cu_seqlens':>14} {'norm diff':>11} {'max elem':>11}  equal")
    compare("forward output", results[0][0], results[1][0])
    for name, left, right in zip(("query", "key", "value", "beta", "g"), results[0][1], results[1][1]):
        compare(f"{name} gradient", left, right)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
