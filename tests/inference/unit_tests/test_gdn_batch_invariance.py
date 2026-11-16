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

import os
import sys
import types

import pytest
import torch

from arctic_platform.inference.vllm import gdn_batch_invariance

DECODE_ENV = {"VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": "0", "VLLM_GDN_DECODE_KERNEL": "triton"}


class _Autotuner:
    def __init__(self, configs, fn):
        self.configs = list(configs)
        self.cache = {"stale": configs[-1]}
        self.fn = fn


class _Heuristics:
    def __init__(self, fn):
        self.fn = fn


@pytest.fixture
def fake_modules(monkeypatch):
    autotuner_module = types.ModuleType("triton.runtime.autotuner")
    autotuner_module.Autotuner = _Autotuner
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    monkeypatch.setitem(sys.modules, "triton.runtime", types.ModuleType("triton.runtime"))
    monkeypatch.setitem(sys.modules, "triton.runtime.autotuner", autotuner_module)

    kernels = {}
    prefix = "vllm.third_party.flash_linear_attention.ops"
    for name in ("vllm", "vllm.third_party", "vllm.third_party.flash_linear_attention", prefix):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name in gdn_batch_invariance._FLA_AUTOTUNED_MODULES:
        module = types.ModuleType(f"{prefix}.{name}")
        kernel = _Heuristics(_Autotuner(["a", "b", "c"], fn=object()))
        module.kernel = kernel
        kernels[name] = kernel
        monkeypatch.setitem(sys.modules, f"{prefix}.{name}", module)
    layernorm_guard = types.ModuleType(f"{prefix}.layernorm_guard")
    layernorm_guard.calc_rows_per_block = lambda M, device: 8
    monkeypatch.setitem(sys.modules, f"{prefix}.layernorm_guard", layernorm_guard)
    sys.modules[prefix].layernorm_guard = layernorm_guard

    backend = type("GDNAttentionBackend", (), {"supports_batch_invariance": classmethod(lambda cls: False)})
    gdn_attn = types.ModuleType("vllm.v1.attention.backends.gdn_attn")
    gdn_attn.GDNAttentionBackend = backend
    for name in ("vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.gdn_attn", gdn_attn)

    monkeypatch.setattr(gdn_batch_invariance, "_PINNED", False)
    monkeypatch.setattr(gdn_batch_invariance, "_APPLIED", False)
    return kernels, layernorm_guard, backend


def test_pin_keeps_one_config_through_wrappers_and_one_row_per_block(fake_modules):
    kernels, layernorm_guard, _ = fake_modules
    assert gdn_batch_invariance.pin_gdn_kernels() == len(kernels)
    for kernel in kernels.values():
        assert kernel.fn.configs == ["a"] and kernel.fn.cache == {}
    assert layernorm_guard.calc_rows_per_block(4096, None) == 1


def test_patch_is_opt_in(fake_modules, monkeypatch):
    kernels, _, backend = fake_modules
    for name in (gdn_batch_invariance.ENV_VAR, *DECODE_ENV):
        monkeypatch.delenv(name, raising=False)
    gdn_batch_invariance.ensure_gdn_batch_invariance()
    assert backend.supports_batch_invariance() is False
    assert all(len(kernel.fn.configs) == 3 for kernel in kernels.values())
    assert not set(DECODE_ENV) & set(os.environ)

    monkeypatch.setenv(gdn_batch_invariance.ENV_VAR, "1")
    gdn_batch_invariance.ensure_gdn_batch_invariance()
    assert backend.supports_batch_invariance() is True
    assert all(kernel.fn.configs == ["a"] for kernel in kernels.values())
    assert {name: os.environ[name] for name in DECODE_ENV} == DECODE_ENV


def test_a_decoded_token_reaches_the_same_recurrent_kernel_with_or_without_a_prefill_in_its_step(monkeypatch):
    """vLLM's GDN core, its kernels recorded: decodes alone and beside a prefill reach one recurrent kernel."""
    import vllm.envs as envs
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    monkeypatch.setattr(GDNAttentionBackend, "supports_batch_invariance", None)
    monkeypatch.setattr(gdn_batch_invariance, "pin_gdn_kernels", lambda: 0)
    monkeypatch.setattr(gdn_batch_invariance, "_APPLIED", False)
    for name in DECODE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(gdn_batch_invariance.ENV_VAR, "1")
    gdn_batch_invariance.ensure_gdn_batch_invariance()

    heads, dim, slots = 2, 4, 8
    conv_dim = 3 * heads * dim
    calls = []

    def recurrent(name):
        def kernel(**kw):
            tokens = kw["a"].shape[0]  # one per decoded sequence
            calls.append((name, kw["a"].clone(), kw["ssm_state_indices"][:tokens].clone()))
            return torch.zeros(1, tokens, heads, dim), None

        return kernel

    def chunk(q, **kw):
        calls.append(("chunk", None, None))
        return torch.zeros(1, q.shape[1], heads, dim), torch.zeros(kw["initial_state"].shape)

    def post_conv_prep(conv_output, a, b, **kw):
        q, k, v = conv_output.view(-1, 3, heads, dim).unbind(1)
        return q, k, v, a, b

    monkeypatch.setattr(gdn, "fused_sigmoid_gating_delta_rule_update", recurrent("recurrent"))
    monkeypatch.setattr(gdn, "fused_recurrent_gated_delta_rule_packed_decode", recurrent("packed_recurrent"))
    monkeypatch.setattr(gdn, "causal_conv1d_update", lambda x, *args, **kw: x)
    monkeypatch.setattr(gdn, "causal_conv1d_fn", lambda x, *args, **kw: x)
    monkeypatch.setattr(gdn, "fused_post_conv_prep", post_conv_prep)
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: True)
    context = types.SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(gdn, "get_forward_context", lambda: context)

    cls = gdn.QwenGatedDeltaNetAttention
    layer = types.SimpleNamespace(
        prefix="gdn", tp_size=1, num_k_heads=heads, head_k_dim=dim, head_v_dim=dim, key_dim=heads * dim,
        value_dim=heads * dim, activation="silu", A_log=torch.zeros(heads), dt_bias=torch.zeros(heads),
        conv1d=types.SimpleNamespace(weight=torch.ones(conv_dim, 1, 4), bias=None), chunk_gated_delta_rule=chunk,
        kv_cache=(torch.zeros(slots, conv_dim, 3), torch.zeros(slots, heads, dim, dim)),
        enable_packed_recurrent_decode=envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE,  # as the layer's __init__ reads it
    )  # fmt: skip
    for name in ("rearrange_mixed_qkv", "_forward_core_decode_non_spec"):
        setattr(layer, name, types.MethodType(getattr(cls, name), layer))

    decodes, prefill = 2, 3
    qkv, a, b = (torch.randn(decodes + prefill, width) for width in (conv_dim, heads, heads))
    alone = GDNAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=decodes, num_decode_tokens=decodes, num_spec_decodes=0,
        num_spec_decode_tokens=0, num_actual_tokens=decodes, non_spec_query_start_loc=torch.tensor([0, 1, 2]),
        non_spec_state_indices_tensor=torch.tensor([3, 5]),
    )  # fmt: skip
    shared = GDNAttentionMetadata(
        num_prefills=1, num_prefill_tokens=prefill, num_decodes=decodes, num_decode_tokens=decodes,
        num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=decodes + prefill,
        has_initial_state=torch.tensor([True, True, False]), non_spec_query_start_loc=torch.tensor([0, 1, 2, 5]),
        non_spec_state_indices_tensor=torch.tensor([3, 5, 6]), prefill_query_start_loc=torch.tensor([0, prefill]),
        prefill_state_indices=torch.tensor([6]), prefill_has_initial_state=torch.tensor([False]),
    )  # fmt: skip

    def decode_kernels(md, tokens):
        calls.clear()
        context.attn_metadata = {"gdn": md}
        cls._forward_core(layer, qkv[:tokens], b[:tokens], a[:tokens], torch.zeros(tokens, heads, dim))
        return [call for call in calls if call[0] != "chunk"]

    [(alone_kernel, alone_a, alone_slots)] = decode_kernels(alone, decodes)
    [(shared_kernel, shared_a, shared_slots)] = decode_kernels(shared, decodes + prefill)
    assert alone_kernel == shared_kernel == "recurrent"
    assert torch.equal(alone_a, a[:decodes]) and torch.equal(shared_a, a[:decodes])
    assert torch.equal(alone_slots, shared_slots)
    assert envs.VLLM_GDN_DECODE_KERNEL == "triton"  # so no layer enables the fused CUDA speculative decode
