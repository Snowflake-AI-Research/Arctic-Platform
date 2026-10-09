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

"""Which argument of the packed gated-delta-net invocation moves a whole model's gradients?

At module level the engine's packed forward and the stock HuggingFace one are the same function apart from
two arguments: a segment index for the short convolution, and sequence boundaries for the delta rule. Given
both, the stock forward reproduces the packed one bit for bit; given neither, the convolution weight's
gradient differs by 5.786e-02 on a 2048-token sequence.

Stock HuggingFace reads both from its keyword arguments (``seq_idx`` and ``cu_seq_lens_q``) and builds
neither, so a caller that does not pass them gets the batched delta-rule kernel. This runs a whole model
four ways on one GPU to see which tensors that choice reaches and by how much.

Full-attention layers are removed so that flash attention's nondeterministic backward -- 6.106e-03 on the
input embedding between two identical runs -- cannot be mistaken for the effect being measured, while the
attention implementation stays as configured. The control arm is the same call twice and bounds what the
remaining kernels contribute on their own.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from attention_ablation import build_ablated  # noqa: E402
from attention_ablation import layer_type_field  # noqa: E402

from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained  # noqa: E402

IGNORE_INDEX = -100
TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
LAYERS = int(os.environ.get("PROBE_LAYERS", 4))
ATTN = os.environ.get("PROBE_ATTN", "flash_attention_3")
# ``full`` removes the full-attention layers; ``none`` keeps the stack as configured, which puts flash
# attention's nondeterministic backward back into every arm.
DROP = os.environ.get("PROBE_DROP", "full")
CE_CHUNK = int(os.environ.get("PROBE_CE_CHUNK", 2048))
SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"


def grads_for(model, ids, labels, active, hidden_size, call_kwargs):
    """One forward-backward of the whole model, returning float32 gradients keyed by parameter name."""
    model.zero_grad(set_to_none=True)
    decoder = model.get_decoder()
    head = model.get_output_embeddings()
    head_name = next(name for name, p in model.named_parameters() if p is head.weight)

    hidden = decoder(input_ids=ids, use_cache=False, **call_kwargs).last_hidden_state
    detached = hidden.detach().requires_grad_(True)
    flat = labels.reshape(-1)

    accumulator = {}
    total_loss = 0.0
    for start in range(0, flat.numel(), CE_CHUNK):
        chunk_labels = flat[start : start + CE_CHUNK]
        if int((chunk_labels != IGNORE_INDEX).sum()) == 0:
            continue
        chunk_hidden = detached.view(-1, hidden_size)[start : start + CE_CHUNK]
        logits = F.linear(chunk_hidden, head.weight).float()
        loss = F.cross_entropy(logits, chunk_labels, ignore_index=IGNORE_INDEX, reduction="sum") / active
        loss.backward()
        total_loss += float(loss.item())
        if head.weight.grad is not None:
            if head_name in accumulator:
                accumulator[head_name].add_(head.weight.grad.detach().float())
            else:
                accumulator[head_name] = head.weight.grad.detach().float().clone()
            head.weight.grad = None

    hidden.backward(detached.grad)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name in accumulator:
            accumulator[name].add_(param.grad.detach().float())
        else:
            accumulator[name] = param.grad.detach().float().clone()
    model.zero_grad(set_to_none=True)
    return total_loss, accumulator


def watch_first_mixer(model):
    """Print the arguments the first gated delta net receives, once per arm."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    if getattr(Qwen3_5GatedDeltaNet.forward, "_probe_watched", False):
        return
    original = Qwen3_5GatedDeltaNet.forward
    state = {"armed": False}

    def watched(self, hidden_states, *args, **kwargs):
        if state["armed"]:
            state["armed"] = False
            received = {
                k: tuple(v.shape) if hasattr(v, "shape") else v
                for k, v in kwargs.items()
                if k in ("seq_idx", "cu_seq_lens_q")
            }
            print(f"    mixer received {received or 'neither argument'}", flush=True)
        return original(self, hidden_states, *args, **kwargs)

    watched._probe_watched = True
    Qwen3_5GatedDeltaNet.forward = watched
    return state


def compare(name, a, b):
    diff = (a - b).abs()
    na, nb = float(a.norm()), float(b.norm())
    over = "  over 1e-3" if abs(na - nb) > 1e-3 else ""
    print(
        f"{name:52} {na:>13.6f} {nb:>13.6f} {abs(na - nb):>11.3e} {float(diff.max()):>11.3e} "
        f"{int((diff == 0).sum()):>10}/{diff.numel()}{over}"
    )


def main() -> int:
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM

    intact = materialize_pretrained(SOURCE, f"{CACHE_ROOT}/Qwen3.8-27B-{LAYERS}L", LAYERS).cache_path
    source_cfg = AutoConfig.from_pretrained(intact, trust_remote_code=True)
    text = getattr(source_cfg, "text_config", source_cfg)
    types = layer_type_field(text)
    full_types = {t for t in types if "full" in t or "self" in t or t == "attention"}
    if DROP == "none":
        path, keep = intact, types
    else:
        path, keep, _ = build_ablated(intact, f"{CACHE_ROOT}/Qwen3.8-27B-{LAYERS}L-no-full-attn", full_types)

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation=ATTN, trust_remote_code=True
    ).cuda()
    model.train()
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    inner_cfg = getattr(cfg, "text_config", cfg)
    hidden_size = inner_cfg.hidden_size

    batch = build_batch("gas1", 1, TOKENS, inner_cfg.vocab_size, seed=SEED)
    ids = batch.input_ids.cuda()
    labels = batch.shifted_labels().cuda()
    active = int((labels != IGNORE_INDEX).sum())
    print(
        f"{len(keep)} layers of {len(types)} kept (dropping {DROP}), attention implementation {ATTN}, "
        f"{TOKENS:,} tokens in one sequence, {active:,} scored",
        flush=True,
    )

    seq_idx = torch.zeros(1, TOKENS, dtype=torch.int32, device="cuda")
    boundaries = torch.tensor([0, TOKENS], dtype=torch.int32, device="cuda")
    position_ids = torch.arange(TOKENS, device="cuda").unsqueeze(0)

    arms = {
        "no arguments": {},
        "no arguments, again": {},
        "segment index only": {"seq_idx": seq_idx},
        "boundaries only": {
            "cu_seq_lens_q": boundaries,
            "cu_seq_lens_k": boundaries,
            "max_length_q": TOKENS,
            "max_length_k": TOKENS,
            "position_ids": position_ids,
        },
        "both": {
            "seq_idx": seq_idx,
            "cu_seq_lens_q": boundaries,
            "cu_seq_lens_k": boundaries,
            "max_length_q": TOKENS,
            "max_length_k": TOKENS,
            "position_ids": position_ids,
        },
    }
    watch = watch_first_mixer(model)
    results = {}
    for label, call_kwargs in arms.items():
        if watch is not None:
            watch["armed"] = True
        loss, grads = grads_for(model, ids, labels, active, hidden_size, call_kwargs)
        results[label] = grads
        print(f"[{label}] loss {loss:.6f}", flush=True)

    pairs = (
        ("no arguments against itself, the same call twice", "no arguments", "no arguments, again"),
        ("no arguments against both", "no arguments", "both"),
        ("segment index only against both", "segment index only", "both"),
        ("boundaries only against both", "boundaries only", "both"),
    )
    for title, left, right in pairs:
        print("")
        print(f"--- {title}")
        print(f"{'tensor':52} {'left':>13} {'right':>13} {'norm diff':>11} {'max elem':>11}  equal")
        for name in sorted(set(results[left]) & set(results[right])):
            compare(name, results[left][name], results[right][name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
