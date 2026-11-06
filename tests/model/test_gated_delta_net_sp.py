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

"""Shared GatedDeltaNet sequence-parallel integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from arctic_platform.model.implementations.gpu.sp import gated_delta_net
from arctic_platform.model.implementations.gpu.sp import transformers as sp_transformers


def _fake_sp_group(monkeypatch, world_size=1):
    """Stand in for a one-rank SP group: all-gather returns the caller's own window."""

    def all_gather(destinations, tensor, group=None):
        for destination in destinations:
            destination.copy_(tensor)

    monkeypatch.setattr(gated_delta_net.dist, "get_world_size", lambda process_group: world_size)
    monkeypatch.setattr(gated_delta_net.dist, "get_rank", lambda process_group: 0)
    monkeypatch.setattr(gated_delta_net.dist, "all_gather", all_gather)
    return "sp-group"


def _hf_model_with_gated_delta_net():
    """A model shaped like a hybrid HF backbone: a top-level forward, one linear-attention module."""
    import torch
    from torch import nn

    class FakeGatedDeltaNet(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_size = 3
            self.conv_kernel_size = 4
            self.num_k_heads = 2
            self.num_v_heads = 4
            self.head_k_dim = 1
            self.head_v_dim = 1
            self.activation = "silu"
            self.in_proj_qkv = nn.Linear(hidden_size, 8, bias=False)
            self.in_proj_z = nn.Linear(hidden_size, 4, bias=False)
            self.in_proj_b = nn.Linear(hidden_size, 4, bias=False)
            self.in_proj_a = nn.Linear(hidden_size, 4, bias=False)
            self.conv1d = nn.Conv1d(
                8,
                8,
                kernel_size=4,
                groups=8,
                padding=3,
                bias=False,
            )
            self.A_log = nn.Parameter(torch.zeros(4))
            self.dt_bias = nn.Parameter(torch.ones(4))
            self.norm = FakeGatedNorm()
            self.out_proj = nn.Linear(4, hidden_size, bias=False)
            self.causal_conv1d_fn = None
            self.chunk_gated_delta_rule = lambda q, k, v, g, beta, **_kwargs: (
                v,
                None,
            )

        def forward(self, hidden_states):
            return hidden_states

    class FakeGatedNorm(nn.Module):
        def forward(self, hidden_states, _gate):
            return hidden_states

    class FakeConfig:
        layer_types = ["linear_attention"]

        def get_text_config(self):
            return self

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = FakeConfig()
            self.gated_delta_net = FakeGatedDeltaNet()

        def forward(self, input_ids=None, position_ids=None, **_kwargs):
            hidden_states = self.gated_delta_net(torch.zeros(1, input_ids.shape[1], 3))
            return SimpleNamespace(last_hidden_state=hidden_states)

    class FakeCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeBackbone()
            self.config = self.model.config
            self.lm_head = nn.Linear(3, 7, bias=False)

        @property
        def base_model(self):
            return self.model

        def forward(self, input_ids=None, position_ids=None, **_kwargs):
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
            )
            return SimpleNamespace(logits=self.lm_head(outputs.last_hidden_state))

    return FakeCausalLM()


def _record_head_parallel_calls(monkeypatch):
    calls = []

    def fake_head_parallel(
        _causal_convolution,
        _gated_delta_rule,
        q,
        k,
        v,
        b,
        a,
        **kwargs,
    ):
        calls.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "b": b,
                "a": a,
                **kwargs,
            }
        )
        return torch.zeros_like(v), None

    import torch

    monkeypatch.setattr(
        gated_delta_net,
        "head_parallel_gated_delta_net",
        fake_head_parallel,
    )
    return calls


def test_hf_gated_delta_net_resets_state_at_every_packed_row_boundary(monkeypatch):
    """A packed call holds several rows, and FLA has to be told where each one ends.

    Recurrent state and the convolution's left pad carry forward across every token FLA believes belongs to one
    sequence. Describing a packed group as a single sequence therefore starts each row from the previous row's
    state, which is a cross-document leak whose only symptom is a loss that is quietly wrong.

    The boundaries have to be global rather than shard-local: a row can span the cut between two ranks, so no
    rank can see from its own window where that row began.
    """
    import torch

    model = _hf_model_with_gated_delta_net()
    process_group = _fake_sp_group(monkeypatch)
    calls = _record_head_parallel_calls(monkeypatch)

    assert sp_transformers.configure_transformers_sequence_parallel_model(model, process_group, fp32_beta=False) == 1

    # Three tokens, then five: the positions restart at 0 wherever a packed row begins.
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    model(input_ids=torch.ones(1, 8, dtype=torch.long), position_ids=position_ids)

    assert len(calls) == 1
    assert calls[0]["global_cu_seqlens"].tolist() == [0, 3, 8]
    assert calls[0]["q"].shape == (1, 8, 2, 1)
    assert calls[0]["v"].shape == (1, 8, 4, 1)


def test_hf_gated_delta_net_refuses_a_call_it_cannot_place_in_the_global_sequence(monkeypatch):
    """Without positions the row boundaries are unknowable, so the call has to fail rather than guess."""
    import torch

    model = _hf_model_with_gated_delta_net()
    process_group = _fake_sp_group(monkeypatch)
    calls = _record_head_parallel_calls(monkeypatch)
    sp_transformers.configure_transformers_sequence_parallel_model(model, process_group, fp32_beta=False)

    with pytest.raises(ValueError, match="position_ids"):
        model(input_ids=torch.ones(1, 8, dtype=torch.long))

    assert calls == []


def test_chunked_lm_head_prepares_gated_delta_net_global_boundaries(monkeypatch):
    """Packed-row boundaries stay global when the logprob pass goes through the chunked LM head.

    The labels are the caller's to supply: the chunked head refuses to derive a next-token target from the
    window's own ``input_ids``, because a window may end mid-row and the target for its last position is then a
    token it does not hold. The whole row exists before the split, which is where ``ensure_next_token_labels``
    in the caller's data plane builds them.
    """
    import torch

    from arctic_platform.model.implementations.gpu.lm_head import enable_chunked_lm_head_logprobs

    model = _hf_model_with_gated_delta_net()
    process_group = _fake_sp_group(monkeypatch)
    calls = _record_head_parallel_calls(monkeypatch)
    assert (
        sp_transformers.configure_transformers_sequence_parallel_model(
            model,
            process_group,
            fp32_beta=False,
        )
        == 1
    )
    enable_chunked_lm_head_logprobs(
        model,
        token_chunk_size=2,
        vocab_chunk_size=3,
        fp32_lm_head=False,
    )

    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    input_ids = torch.ones(1, 8, dtype=torch.long)
    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=torch.roll(input_ids, shifts=-1, dims=-1),
        dss_compute_logprobs=True,
    )

    assert output["logprobs"].shape == (1, 8)
    assert len(calls) == 1
    assert calls[0]["global_cu_seqlens"].tolist() == [0, 3, 8]


def test_hybrid_model_rejects_missing_gated_delta_net_adapter():
    from torch import nn

    class FakeConfig:
        layer_types = ["linear_attention", "full_attention"]

        def get_text_config(self):
            return self

    model = nn.Module()
    model.config = FakeConfig()

    with pytest.raises(RuntimeError, match="no compatible GatedDeltaNet"):
        gated_delta_net.apply_gated_delta_net_sequence_parallelism(
            model,
            "sp-group",
            fp32_beta=False,
        )


def test_transformers_sp_model_disables_cache_before_adapting_linear_attention(monkeypatch):
    from torch import nn

    class FakeTextConfig:
        use_cache = True

    class FakeConfig:
        use_cache = True

        def __init__(self):
            self.text_config = FakeTextConfig()

        def get_text_config(self):
            return self.text_config

    calls = []
    monkeypatch.setattr(
        sp_transformers,
        "apply_gated_delta_net_sequence_parallelism",
        lambda model, process_group, **_kwargs: calls.append((model, process_group)) or 2,
    )
    model = nn.Module()
    model.config = FakeConfig()

    adapted_modules = sp_transformers.configure_transformers_sequence_parallel_model(
        model,
        "sp-group",
        fp32_beta=False,
    )

    assert adapted_modules == 2
    assert model.config.use_cache is False
    assert model.config.text_config.use_cache is False
    assert calls == [(model, "sp-group")]


def test_prime_rl_gated_delta_net_uses_full_temporal_core_head_parallelism(monkeypatch):
    import torch

    monkeypatch.setenv("TILELANG_CACHE_DIR", "/tmp/tilelang-cache")
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import configuration_qwen3_5_moe
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import modeling_qwen3_5_moe

    calls = []

    def fake_head_parallel(
        _causal_convolution,
        _gated_delta_rule,
        q,
        k,
        v,
        b,
        a,
        **kwargs,
    ):
        calls.append((q, k, v, b, a, kwargs))
        return torch.zeros_like(v), None

    monkeypatch.setattr(
        modeling_qwen3_5_moe,
        "head_parallel_gated_delta_net",
        fake_head_parallel,
    )
    config = configuration_qwen3_5_moe.Qwen3_5MoeConfig(
        hidden_size=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
    )
    module = modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet(config)
    module.norm = type(
        "FakeGatedNorm",
        (torch.nn.Module,),
        {"forward": lambda self, hidden_states, gate: hidden_states},
    )()
    global_cu_seqlens = torch.tensor([0, 6, 16], dtype=torch.int32)
    module.cp_group = "sp-group"
    module._dss_sp_global_cu_seqlens = global_cu_seqlens

    result = module(torch.randn(1, 8, config.hidden_size))

    assert result.shape == (1, 8, config.hidden_size)
    q, k, v, b, a, kwargs = calls[0]
    assert q.shape == (1, 8, 2, 2)
    assert k.shape == q.shape
    assert v.shape == (1, 8, 4, 2)
    assert b.shape == a.shape == (1, 8, 4)
    assert kwargs["global_cu_seqlens"] is global_cu_seqlens
    assert kwargs["process_group"] == "sp-group"


def test_shared_expert_gate_is_sequence_partition_invariant(monkeypatch):
    import torch
    from torch import nn

    monkeypatch.setenv("TILELANG_CACHE_DIR", "/tmp/tilelang-cache")
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import modeling_qwen3_5_moe

    torch.manual_seed(2718)
    hidden_states = torch.randn(32, 64, dtype=torch.bfloat16)
    gate = nn.Linear(64, 1, bias=False, dtype=torch.bfloat16)

    full = modeling_qwen3_5_moe._shared_expert_gate(hidden_states, gate)
    partitioned = torch.cat(
        [
            modeling_qwen3_5_moe._shared_expert_gate(hidden_states[:11], gate),
            modeling_qwen3_5_moe._shared_expert_gate(hidden_states[11:], gate),
        ]
    )

    assert full.dtype == hidden_states.dtype
    assert torch.equal(full, partitioned)


def test_shared_expert_gate_backward_is_finite(monkeypatch):
    import torch
    from torch import nn

    monkeypatch.setenv("TILELANG_CACHE_DIR", "/tmp/tilelang-cache")
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import modeling_qwen3_5_moe

    torch.manual_seed(3141)
    hidden_states = torch.randn(
        17,
        32,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gate = nn.Linear(32, 1, bias=False, dtype=torch.bfloat16)
    output = modeling_qwen3_5_moe._shared_expert_gate(hidden_states, gate)
    output.float().sum().backward()

    assert hidden_states.grad is not None
    assert gate.weight.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert torch.isfinite(gate.weight.grad).all()


def test_head_parallel_rule_preserves_repeated_qk_heads_and_global_boundaries(monkeypatch):
    import torch

    process_group = _fake_sp_group(monkeypatch)
    native_q = torch.randn(1, 8, 2, 3)
    native_k = torch.randn(1, 8, 2, 3)
    q = native_q.repeat_interleave(2, dim=2)
    k = native_k.repeat_interleave(2, dim=2)
    v = torch.randn(1, 8, 4, 5)
    g = torch.randn(1, 8, 4, dtype=torch.float32)
    beta = torch.randn(1, 8, 4)
    global_cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    calls = []

    def rule(q_arg, k_arg, v_arg, g_arg, beta_arg, **kwargs):
        calls.append((q_arg, k_arg, v_arg, g_arg, beta_arg, kwargs))
        return v_arg + 1, None

    output, final_state = gated_delta_net.head_parallel_gated_delta_rule(
        rule,
        q,
        k,
        v,
        g,
        beta,
        process_group=process_group,
        global_cu_seqlens=global_cu_seqlens,
        num_key_heads=2,
        num_value_heads=4,
    )

    q_arg, k_arg, v_arg, g_arg, beta_arg, kwargs = calls[0]
    assert torch.equal(q_arg, q)
    assert torch.equal(k_arg, k)
    assert torch.equal(v_arg, v)
    assert torch.equal(g_arg, g)
    assert torch.equal(beta_arg, beta)
    assert kwargs["cu_seqlens"] is global_cu_seqlens
    assert kwargs["cp_context"] is None
    assert kwargs["initial_state"] is None
    assert kwargs["output_final_state"] is False
    assert torch.equal(output, v + 1)
    assert final_state is None


def test_head_parallel_rule_rejects_native_qk_heads(monkeypatch):
    import torch

    process_group = _fake_sp_group(monkeypatch)
    q = torch.randn(1, 8, 2, 3)
    v = torch.randn(1, 8, 4, 5)
    gates = torch.randn(1, 8, 4)

    with pytest.raises(RuntimeError, match="repeat Q/K"):
        gated_delta_net.head_parallel_gated_delta_rule(
            lambda *_args, **_kwargs: (v, None),
            q,
            q,
            v,
            gates,
            gates,
            process_group=process_group,
            global_cu_seqlens=torch.tensor([0, 8], dtype=torch.int32),
            num_key_heads=2,
            num_value_heads=4,
        )


def test_mixed_dtype_head_exchange_preserves_gradients(monkeypatch):
    import torch

    process_group = _fake_sp_group(monkeypatch)
    q = torch.randn(1, 6, 2, 3, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(1, 6, 4, dtype=torch.float32, requires_grad=True)

    exchanged = gated_delta_net._pack_and_exchange_head_tensors(
        {"q": q, "g": g},
        process_group=process_group,
    )
    assert exchanged["q"].dtype == torch.bfloat16
    assert exchanged["g"].dtype == torch.float32
    assert torch.equal(exchanged["q"], q)
    assert torch.equal(exchanged["g"], g)

    exchanged["q"].float().sum().backward()
    exchanged["g"].sum().backward()
    assert torch.equal(q.grad, torch.ones_like(q))
    assert torch.equal(g.grad, torch.ones_like(g))


def test_head_parallel_rule_rejects_cached_recurrent_state(monkeypatch):
    import torch

    process_group = _fake_sp_group(monkeypatch)
    q = torch.randn(1, 8, 2, 3).repeat_interleave(2, dim=2)
    v = torch.randn(1, 8, 4, 5)
    g = torch.randn(1, 8, 4)
    beta = torch.randn(1, 8, 4)

    with pytest.raises(RuntimeError, match="initial recurrent state"):
        gated_delta_net.head_parallel_gated_delta_rule(
            lambda *_args, **_kwargs: (v, None),
            q,
            q,
            v,
            g,
            beta,
            process_group=process_group,
            global_cu_seqlens=torch.tensor([0, 8], dtype=torch.int32),
            num_key_heads=2,
            num_value_heads=4,
            initial_state=torch.zeros(1, 4, 3, 5),
        )
