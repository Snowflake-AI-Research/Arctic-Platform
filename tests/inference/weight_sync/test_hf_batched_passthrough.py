#!/usr/bin/env python3
"""Regression: the ``weight_format="hf"`` batched path is UNCHANGED by the
``_ShardAwareFusedWriter`` interception added for the vLLM-fused families.

``sync_weights_broadcast(weight_format="hf")`` routes through the same
``_load_batched`` that now builds a ``_ShardAwareFusedWriter``.  HF checkpoints
carry the *un-fused* names (per-expert MoE ``experts.{e}.gate_proj.weight`` and
separate GDN ``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_b`` / ``in_proj_a``),
which must NOT match the writer's fused wire names.  Therefore ``writer.feed``
returns False and each ``(name, tensor)`` is forwarded to ``model.load_weights``
*verbatim* -- byte-for-byte the behaviour before this change.

This test proves the invariant directly, without depending on any particular
model's ability to load HF MoE weights: it spies on ``model.load_weights`` and
asserts that EVERY fed HF item is forwarded unchanged (same name, same tensor
identity) and that the writer intercepted NONE of them.

Run (dss env == vllm 0.26):
    python tests/weight_sync/test_hf_batched_passthrough.py
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

NUM_EXPERTS = 8
HIDDEN = 128
INTERMEDIATE = 512
KEY_DIM = 256
VALUE_DIM = 512
NUM_V_HEADS = 32
DTYPE = torch.bfloat16


def _init_single_rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29711")
    torch.cuda.set_device(0)
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    init_distributed_environment(
        world_size=1, rank=0,
        distributed_init_method="tcp://127.0.0.1:29711",
        local_rank=0, backend="nccl",
    )


class _FakeEngine:
    def __init__(self, items):
        self._items = items

    def receive_weights(self):
        for name, tensor in self._items:
            yield name, tensor


def main() -> int:
    device = torch.device("cuda", 0)
    _init_single_rank()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

    from vllm.distributed import initialize_model_parallel

    cfg_ctx = set_current_vllm_config(VllmConfig())
    cfg_ctx.__enter__()
    initialize_model_parallel(tensor_model_parallel_size=1)
    experts = FusedMoEFactory(
        num_experts=NUM_EXPERTS, top_k=2, hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE, params_dtype=DTYPE,
        renormalize=True, quant_config=None, prefix="experts",
    ).routed_experts.to(device)
    qkvz = MergedColumnParallelLinear(
        input_size=HIDDEN,
        output_sizes=[KEY_DIM, KEY_DIM, VALUE_DIM, VALUE_DIM],
        bias=False, prefix="linear_attn.in_proj_qkvz").to(device)
    ba = MergedColumnParallelLinear(
        input_size=HIDDEN, output_sizes=[NUM_V_HEADS, NUM_V_HEADS],
        bias=False, prefix="linear_attn.in_proj_ba").to(device)

    class FakeGDN(GatedDeltaNetAttention):
        def __init__(self, qkvz, ba):
            nn.Module.__init__(self)
            self.in_proj_qkvz = qkvz
            self.in_proj_ba = ba

        def get_state_shape(self):
            return ()

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = FakeGDN(qkvz, ba)
            self.mlp = nn.Module()
            self.mlp.experts = experts

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    forwarded = []

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def load_weights(self, weights):
            # Spy: record exactly what falls through to model.load_weights.
            for name, tensor in weights:
                forwarded.append((name, id(tensor)))
            return [n for n, _ in weights]

    model = Model()

    # HF checkpoint names (un-fused) — what weight_format="hf" broadcasts.
    hf_items = []
    for e in range(NUM_EXPERTS):
        base = f"model.layers.0.mlp.experts.{e}"
        hf_items.append((f"{base}.gate_proj.weight",
                         torch.randn(INTERMEDIATE, HIDDEN, dtype=DTYPE, device=device)))
        hf_items.append((f"{base}.up_proj.weight",
                         torch.randn(INTERMEDIATE, HIDDEN, dtype=DTYPE, device=device)))
        hf_items.append((f"{base}.down_proj.weight",
                         torch.randn(HIDDEN, INTERMEDIATE, dtype=DTYPE, device=device)))
    for leaf, rows in (("in_proj_qkv", 2 * KEY_DIM + VALUE_DIM),
                       ("in_proj_z", VALUE_DIM),
                       ("in_proj_b", NUM_V_HEADS), ("in_proj_a", NUM_V_HEADS)):
        hf_items.append((f"model.layers.0.linear_attn.{leaf}.weight",
                         torch.randn(rows, HIDDEN, dtype=DTYPE, device=device)))

    # (1) writer must not claim any HF name.
    from arctic_platform.inference.server.weight_sync.utils import _ShardAwareFusedWriter
    probe = _ShardAwareFusedWriter(model, device)
    intercepted = [n for n, _ in hf_items if n in probe._handlers]

    # (2) drive the real seam; every HF item must be forwarded verbatim.
    from arctic_platform.inference.server.weight_sync.receiver import WeightSyncExtension
    ext = SimpleNamespace(device=device)
    WeightSyncExtension._load_batched(ext, model, _FakeEngine(hf_items))

    expected = [(n, id(t)) for n, t in hf_items]
    forwarded_ok = forwarded == expected

    print("=" * 72)
    print("HF batched passthrough regression")
    print("=" * 72)
    print(f"fed HF items          : {len(hf_items)}")
    print(f"writer intercepted    : {len(intercepted)}  {intercepted[:3]}")
    print(f"forwarded to load_wts : {len(forwarded)}")
    print(f"forwarded verbatim    : {forwarded_ok}")
    ok = (len(intercepted) == 0) and forwarded_ok and (len(forwarded) == len(hf_items))
    print("\nRESULT:", "PASSED — hf path unchanged (writer transparent to HF names)"
          if ok else "FAILED — hf path altered by writer")
    print("=" * 72)
    cfg_ctx.__exit__(None, None, None)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
