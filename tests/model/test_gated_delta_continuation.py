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

"""GatedDeltaNet prefix continuation: a continuation resumes its root's conv inputs and recurrent state."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from arctic_platform.common.packing.layout import PackedLayout
from arctic_platform.model.implementations.gpu.sp import gated_delta_net
from arctic_platform.model.implementations.gpu.sp.gated_delta_net import gated_delta_continuation
from arctic_platform.model.implementations.gpu.sp.gated_delta_net import packed_causal_convolution
from arctic_platform.model.implementations.gpu.sp.gated_delta_net import packed_gated_delta_rule

KERNEL = 4
HIDDEN = 5
KEY_HEADS = 2
VALUE_HEADS = 4
KEY_DIM = 3
VALUE_DIM = 2
CHANNELS = 2 * KEY_HEADS * KEY_DIM + VALUE_HEADS * VALUE_DIM
_SAME = object()


def reference_rule(
    q,
    k,
    v,
    g,
    beta,
    *,
    initial_state=None,
    output_final_state=False,
    cu_seqlens,
    use_qk_l2norm_in_kernel=False,
    scale=None,
):
    """Token-by-token gated delta rule with FLA's varlen contract: one state per ``cu_seqlens`` segment."""
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
    groups = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(groups, dim=2)
    k = k.repeat_interleave(groups, dim=2)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    state_dtype = torch.promote_types(v.dtype, torch.float32)
    outputs, finals = [], []
    for segment, (start, stop) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())):
        if initial_state is None:
            state = v.new_zeros(v.shape[2], q.shape[-1], v.shape[-1], dtype=state_dtype)
        else:
            state = initial_state[segment]
        for t in range(start, stop):
            state = state * g[0, t].exp()[:, None, None]
            memory = (state * k[0, t][:, :, None]).sum(-2)
            delta = (v[0, t] - memory) * beta[0, t][:, None]
            state = state + k[0, t][:, :, None] * delta[:, None, :]
            outputs.append((state * (q[0, t] * scale)[:, :, None]).sum(-2))
        finals.append(state)
    output = torch.stack(outputs).unsqueeze(0).to(v.dtype)
    return output, (torch.stack(finals) if output_final_state else None)


class Mixer(torch.nn.Module):
    """The token-mixing part of a GatedDeltaNet layer, on packed ``[1, T, HIDDEN]`` hidden states."""

    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(HIDDEN, CHANNELS, bias=False)
        self.a = torch.nn.Linear(HIDDEN, VALUE_HEADS, bias=False)
        self.b = torch.nn.Linear(HIDDEN, VALUE_HEADS, bias=False)
        self.conv = torch.nn.Parameter(torch.randn(CHANNELS, KERNEL) * 0.5)

    def forward(self, hidden, cu_seqlens, continuation, *, rule=reference_rule, convolution_continuation=_SAME):
        conv_plan = continuation if convolution_continuation is _SAME else convolution_continuation
        mixed = packed_causal_convolution(
            None,
            self.qkv(hidden).transpose(1, 2),
            weight=self.conv,
            bias=None,
            activation="silu",
            cu_seqlens=cu_seqlens,
            continuation=conv_plan,
        ).transpose(1, 2)
        tokens = hidden.shape[1]
        q, k, v = torch.split(mixed, [KEY_HEADS * KEY_DIM, KEY_HEADS * KEY_DIM, VALUE_HEADS * VALUE_DIM], dim=-1)
        return packed_gated_delta_rule(
            rule,
            q.reshape(1, tokens, KEY_HEADS, KEY_DIM),
            k.reshape(1, tokens, KEY_HEADS, KEY_DIM),
            v.reshape(1, tokens, VALUE_HEADS, VALUE_DIM),
            -F.softplus(self.a(hidden)),
            self.b(hidden).sigmoid(),
            cu_seqlens=cu_seqlens,
            continuation=continuation,
            use_qk_l2norm_in_kernel=True,
        )


def _rows(lengths, shared, groups):
    """Padded hidden rows whose groups agree on their first ``shared`` tokens."""
    torch.manual_seed(0)
    width = max(lengths)
    hidden = torch.randn(len(lengths), width, HIDDEN, dtype=torch.float64)
    for rows, n in zip(groups, shared):
        for row in rows[1:]:
            hidden[row, :n] = hidden[rows[0], :n]
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    input_ids = (hidden[..., 0] * 1000).long()
    return hidden, mask, input_ids


def _per_row(mixer, hidden, lengths):
    """Each row on its own, the share-off reference."""
    outputs = []
    for row, length in enumerate(lengths):
        cu = torch.tensor([0, length], dtype=torch.int32)
        outputs.append(mixer(hidden[row : row + 1, :length], cu, None)[0])
    return outputs


def _packed(mixer, layout, hidden, **kwargs):
    packed = layout.pack(hidden).unsqueeze(0)
    plan = gated_delta_continuation(layout, KERNEL, "cpu") if layout.has_sharing else None
    output = mixer(packed, layout.cu_seqlens, plan, **kwargs)[0]
    return [
        output[layout.row_index[row, :length]] for row, length in enumerate(layout.row_index.ge(0).sum(1).tolist())
    ]


def _loss(outputs, weights):
    return sum((output * weight[: output.shape[0]]).sum() for output, weight in zip(outputs, weights))


def _grads(mixer, outputs, weights):
    mixer.zero_grad()
    _loss(outputs, weights).backward()
    return {name: parameter.grad.clone() for name, parameter in mixer.named_parameters()}


@pytest.mark.parametrize("shared_length", [5, 8])
def test_sharing_matches_each_row_run_alone(shared_length):
    """Outputs and every parameter gradient match the unshared rows, including two continuations of one root."""
    lengths = [11, 9, 13, 7]
    groups = [[0, 1, 2], [3]]
    hidden, mask, input_ids = _rows(lengths, [shared_length, 0], groups)
    layout = PackedLayout.build(mask, groups, [shared_length, 0], input_ids=input_ids)
    assert layout.has_sharing and layout.num_roots == 2

    mixer = Mixer().double()
    weights = [torch.randn(length, VALUE_HEADS, VALUE_DIM, dtype=torch.float64) for length in lengths]
    rows = _per_row(mixer, hidden, lengths)
    expected = _grads(mixer, rows, weights)
    shared = _packed(mixer, layout, hidden)
    actual = _grads(mixer, shared, weights)

    for row, (want, got) in enumerate(zip(rows, shared)):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10, msg=f"row {row}")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-9, atol=1e-10, msg=name)


def test_unshared_layout_is_one_packed_call():
    lengths = [6, 9]
    hidden, mask, _ = _rows(lengths, [0], [[0], [1]])
    layout = PackedLayout.build(mask, [[0], [1]], [0, 0])
    mixer = Mixer().double()
    for want, got in zip(_per_row(mixer, hidden, lengths), _packed(mixer, layout, hidden)):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dropped", ["state", "conv"])
def test_dropping_either_handoff_changes_the_continuation(dropped):
    """Negative controls: restarting the state or the conv at the continuation must not match the rows."""
    lengths = [10, 12]
    hidden, mask, input_ids = _rows(lengths, [6], [[0, 1]])
    layout = PackedLayout.build(mask, [[0, 1]], [6], input_ids=input_ids)
    plan = gated_delta_continuation(layout, KERNEL, "cpu")
    mixer = Mixer().double()
    rows = _per_row(mixer, hidden, lengths)

    if dropped == "conv":
        no_seed = gated_delta_net.GatedDeltaContinuation(
            carry_index=torch.arange(layout.num_tokens),
            carry_cu_seqlens=layout.cu_seqlens,
            keep_index=torch.arange(layout.num_tokens),
            root_cu_seqlens=plan.root_cu_seqlens,
            continuation_cu_seqlens=plan.continuation_cu_seqlens,
            parent=plan.parent,
            root_tokens=plan.root_tokens,
        )
        shared = _packed(mixer, layout, hidden, convolution_continuation=no_seed)
    else:

        def stateless(*args, initial_state=None, **kwargs):
            return reference_rule(*args, initial_state=None, **kwargs)

        shared = _packed(mixer, layout, hidden, rule=stateless)

    # The shared prefix is the root itself either way; only the continuation tokens can differ.
    assert torch.allclose(shared[0][:6], rows[0][:6])
    assert not torch.allclose(shared[1][6:], rows[1][6:])


def test_conv_carry_refuses_a_root_shorter_than_the_seed():
    lengths = [5, 6]
    hidden, mask, input_ids = _rows(lengths, [2], [[0, 1]])
    layout = PackedLayout.build(mask, [[0, 1]], [2], input_ids=input_ids)
    with pytest.raises(ValueError, match="kernel_size - 1 = 3"):
        gated_delta_continuation(layout, KERNEL, "cpu")


def test_plan_indices_seed_each_continuation_with_its_parents_tail():
    lengths = [7, 8, 5]
    hidden, mask, input_ids = _rows(lengths, [4], [[0, 1], [2]])
    layout = PackedLayout.build(mask, [[0, 1], [2]], [4, 0], input_ids=input_ids)
    plan = gated_delta_continuation(layout, KERNEL, "cpu")
    # Roots: [0, 4) shared, [4, 9) row 2. Continuations: row 0 [9, 12), row 1 [12, 16), both after root 0.
    assert layout.cu_seqlens.tolist() == [0, 4, 9, 12, 16]
    assert plan.carry_index.tolist() == [*range(9), 1, 2, 3, 9, 10, 11, 1, 2, 3, 12, 13, 14, 15]
    assert plan.carry_cu_seqlens.tolist() == [0, 4, 9, 15, 22]
    assert plan.keep_index.tolist() == [*range(9), 12, 13, 14, 18, 19, 20, 21]
    assert plan.root_cu_seqlens.tolist() == [0, 4, 9]
    assert plan.continuation_cu_seqlens.tolist() == [0, 3, 7]
    assert plan.parent.tolist() == [0, 0]
    assert gated_delta_continuation(layout, KERNEL, "cpu") is plan


def test_conv_carry_hands_causal_conv1d_a_channel_last_input():
    """causal-conv1d takes ``seq_idx`` only with a channel-last ``[1, C, T]`` x, the layout the models pass in."""
    lengths = [7, 8, 5]
    hidden, mask, input_ids = _rows(lengths, [4], [[0, 1], [2]])
    layout = PackedLayout.build(mask, [[0, 1], [2]], [4, 0], input_ids=input_ids)
    plan = gated_delta_continuation(layout, KERNEL, "cpu")
    weight = torch.randn(CHANNELS, KERNEL, dtype=torch.float64)
    x = torch.randn(1, layout.num_tokens, CHANNELS, dtype=torch.float64).transpose(1, 2)
    strides = []

    def channel_last_conv(*, x, weight, bias, activation, seq_idx):
        strides.append(x.stride(1))
        counts = torch.unique_consecutive(seq_idx[0], return_counts=True)[1]
        output = gated_delta_net._fallback_depthwise_causal_convolution(
            x, weight=weight, bias=bias, activation=activation, global_cu_seqlens=F.pad(counts.cumsum(0), (1, 0))
        )
        return output.transpose(1, 2).contiguous().transpose(1, 2)

    def run(kernel):
        return packed_causal_convolution(
            kernel, x, weight=weight, bias=None, activation="silu", cu_seqlens=layout.cu_seqlens, continuation=plan
        )

    got = run(channel_last_conv)
    assert strides == [1]
    assert got.stride(1) == 1
    torch.testing.assert_close(got, run(None), rtol=0, atol=0)


def test_rule_refuses_caller_state():
    lengths = [6, 7]
    hidden, mask, input_ids = _rows(lengths, [4], [[0, 1]])
    layout = PackedLayout.build(mask, [[0, 1]], [4], input_ids=input_ids)
    plan = gated_delta_continuation(layout, KERNEL, "cpu")
    q = torch.zeros(1, layout.num_tokens, 1, 2)
    with pytest.raises(RuntimeError, match="initial_state"):
        packed_gated_delta_rule(
            reference_rule,
            q,
            q,
            q,
            q[..., 0],
            q[..., 0],
            cu_seqlens=layout.cu_seqlens,
            continuation=plan,
            initial_state=torch.zeros(2, 1, 2, 2),
        )


class FakeGatedNorm(torch.nn.Module):
    def forward(self, hidden_states, gate):
        return hidden_states * F.silu(gate)


class FakeGatedDeltaNet(torch.nn.Module):
    """Shaped like transformers' Qwen3.5 / Qwen3-Next GatedDeltaNet; the stock forward reads ``cu_seq_lens_q``."""

    def __init__(self, fused=False):
        super().__init__()
        self.conv_kernel_size = KERNEL
        self.num_k_heads, self.num_v_heads = KEY_HEADS, VALUE_HEADS
        self.head_k_dim, self.head_v_dim = KEY_DIM, VALUE_DIM
        self.activation = "silu"
        if fused:
            self.in_proj_qkvz = torch.nn.Linear(HIDDEN, CHANNELS + VALUE_HEADS * VALUE_DIM, bias=False)
            self.in_proj_ba = torch.nn.Linear(HIDDEN, 2 * VALUE_HEADS, bias=False)
        else:
            self.in_proj_qkv = torch.nn.Linear(HIDDEN, CHANNELS, bias=False)
            self.in_proj_z = torch.nn.Linear(HIDDEN, VALUE_HEADS * VALUE_DIM, bias=False)
            self.in_proj_b = torch.nn.Linear(HIDDEN, VALUE_HEADS, bias=False)
            self.in_proj_a = torch.nn.Linear(HIDDEN, VALUE_HEADS, bias=False)
        self.conv1d = torch.nn.Conv1d(CHANNELS, CHANNELS, KERNEL, groups=CHANNELS, padding=KERNEL - 1, bias=False)
        self.A_log = torch.nn.Parameter(torch.rand(VALUE_HEADS))
        self.dt_bias = torch.nn.Parameter(torch.rand(VALUE_HEADS))
        self.norm = FakeGatedNorm()
        self.out_proj = torch.nn.Linear(VALUE_HEADS * VALUE_DIM, HIDDEN, bias=False)
        self.causal_conv1d_fn = None
        self.chunk_gated_delta_rule = reference_rule
        self.stock_calls = 0

    def fix_query_key_value_ordering(self, qkvz, ba):
        # Qwen3-Next's grouping: per key head, [q, k, v-group, z-group] and [b-group, a-group].
        batch, tokens, _ = qkvz.shape
        per = VALUE_HEADS // KEY_HEADS
        qkvz = qkvz.view(batch, tokens, KEY_HEADS, 2 * KEY_DIM + 2 * per * VALUE_DIM)
        ba = ba.view(batch, tokens, KEY_HEADS, 2 * per)
        q, k, v, z = torch.split(qkvz, [KEY_DIM, KEY_DIM, per * VALUE_DIM, per * VALUE_DIM], dim=3)
        b, a = torch.split(ba, [per, per], dim=3)
        v = v.reshape(batch, tokens, -1, VALUE_DIM)
        z = z.reshape(batch, tokens, -1, VALUE_DIM)
        return q, k, v, z, b.reshape(batch, tokens, VALUE_HEADS), a.reshape(batch, tokens, VALUE_HEADS)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        self.stock_calls += 1
        batch, tokens, _ = hidden_states.shape
        if hasattr(self, "in_proj_qkvz"):
            q, k, v, z, b, a = self.fix_query_key_value_ordering(
                self.in_proj_qkvz(hidden_states), self.in_proj_ba(hidden_states)
            )
            mixed = torch.cat([x.reshape(batch, tokens, -1) for x in (q, k, v)], dim=-1)
        else:
            mixed = self.in_proj_qkv(hidden_states)
            z = self.in_proj_z(hidden_states).reshape(batch, tokens, -1, VALUE_DIM)
            b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
        cu = kwargs["cu_seq_lens_q"]
        mixed = gated_delta_net._fallback_depthwise_causal_convolution(
            mixed.transpose(1, 2),
            weight=self.conv1d.weight.squeeze(1),
            bias=None,
            activation="silu",
            global_cu_seqlens=cu,
        ).transpose(1, 2)
        q, k, v = torch.split(mixed, [KEY_HEADS * KEY_DIM, KEY_HEADS * KEY_DIM, VALUE_HEADS * VALUE_DIM], dim=-1)
        q = q.reshape(batch, tokens, KEY_HEADS, KEY_DIM).repeat_interleave(VALUE_HEADS // KEY_HEADS, dim=2)
        k = k.reshape(batch, tokens, KEY_HEADS, KEY_DIM).repeat_interleave(VALUE_HEADS // KEY_HEADS, dim=2)
        v = v.reshape(batch, tokens, VALUE_HEADS, VALUE_DIM)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        out, _ = reference_rule(q, k, v, g, b.sigmoid(), cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
        out = self.norm(out.reshape(-1, VALUE_DIM), z.reshape(-1, VALUE_DIM)).reshape(batch, tokens, -1)
        return self.out_proj(out)


class FakeHybrid(torch.nn.Module):
    def __init__(self, fused=False):
        super().__init__()
        self.linear_attn = FakeGatedDeltaNet(fused)

    def forward(self, hidden_states, **kwargs):
        return self.linear_attn(hidden_states=hidden_states, cache_params=None, attention_mask=None, **kwargs)


@pytest.fixture
def verified_kernels(monkeypatch):
    """The reference kernels stand in for FLA / causal-conv1d, which the ``[gdn]`` extra pins."""
    monkeypatch.setattr(gated_delta_net, "prefix_continuation_kernel_blocker", lambda conv, rule: None)


def _hybrid(fused=False):
    torch.manual_seed(1)
    model = FakeHybrid(fused).double()
    assert gated_delta_net.apply_gated_delta_net_prefix_continuation(model) == 1
    return model


def _stock_rows(model, hidden, lengths):
    outputs = []
    for row, length in enumerate(lengths):
        cu = torch.tensor([0, length], dtype=torch.int32)
        outputs.append(model(hidden[row : row + 1, :length], cu_seq_lens_q=cu)[0])
    return outputs


@pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
def test_sp1_adapter_matches_the_stock_rows_under_sharing(verified_kernels, fused):
    from arctic_platform.common.packing.layout import using_layout

    lengths = [70, 66, 72]
    groups = [[0, 1, 2]]
    hidden, mask, input_ids = _rows(lengths, [64], groups)
    layout = PackedLayout.build(mask, groups, [64], input_ids=input_ids)
    model = _hybrid(fused)
    weights = [torch.randn(length, HIDDEN, dtype=torch.float64) for length in lengths]

    rows = _stock_rows(model, hidden, lengths)
    expected = _grads(model, rows, weights)
    stock_calls = model.linear_attn.stock_calls
    packed = layout.pack(hidden).unsqueeze(0)
    with using_layout(model, layout):
        # Stale unshared boundaries from the client must not be read under a layout.
        output = model(packed, cu_seq_lens_q=torch.tensor([0, 70, 136, 208], dtype=torch.int32))[0]
        shared = [output[layout.row_index[row, :length]] for row, length in enumerate(lengths)]
        actual = _grads(model, shared, weights)

    assert model.linear_attn.stock_calls == stock_calls
    for row, (want, got) in enumerate(zip(rows, shared)):
        torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-10, msg=f"row {row}")
    for name in expected:
        # g is computed from A_log.float() / a.float(), as in transformers, so those gradients sum in fp32.
        fp32 = name.endswith(("A_log", "in_proj_a.weight", "in_proj_ba.weight"))
        tolerance = dict(rtol=1e-6, atol=1e-8) if fp32 else dict(rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(actual[name], expected[name], **tolerance, msg=name)


def test_sp1_adapter_keeps_the_stock_forward_without_a_layout(verified_kernels):
    model = _hybrid()
    hidden = torch.randn(1, 9, HIDDEN, dtype=torch.float64)
    model(hidden, cu_seq_lens_q=torch.tensor([0, 4, 9], dtype=torch.int32))
    assert model.linear_attn.stock_calls == 1


@pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
def test_sp1_adapter_gives_an_unshared_layout_its_segments(verified_kernels, fused):
    from arctic_platform.common.packing.layout import using_layout

    lengths = [6, 9]
    hidden, mask, _ = _rows(lengths, [0], [[0], [1]])
    layout = PackedLayout.build(mask, [[0], [1]], [0, 0])
    model = _hybrid(fused)
    packed = layout.pack(hidden).unsqueeze(0)
    padded = torch.cat([packed, packed.new_zeros(1, 3, HIDDEN)], 1)
    padded_cu = torch.tensor([0, 6, 15, 18], dtype=torch.int32)
    want = model(packed, cu_seq_lens_q=layout.cu_seqlens)
    want_padded = model(padded, cu_seq_lens_q=padded_cu)
    with using_layout(model, layout):
        # The layout's own call needs no boundaries from the caller.
        got = model(packed)
        # Any other call keeps the stock forward and the caller's boundaries.
        got_padded = model(padded, cu_seq_lens_q=padded_cu)
    assert model.linear_attn.stock_calls == 3
    torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(got_padded, want_padded, rtol=0, atol=0)


def test_sharing_refuses_a_root_off_the_gdn_chunk(verified_kernels):
    from arctic_platform.common.packing.layout import using_layout

    hidden, mask, input_ids = _rows([70, 66], [40], [[0, 1]])
    layout = PackedLayout.build(mask, [[0, 1]], [40], input_ids=input_ids)
    with pytest.raises(ValueError, match="multiple of its 64-token chunk"):
        with using_layout(_hybrid(), layout):
            pass


def test_sharing_refuses_an_unadapted_gated_delta_net():
    from arctic_platform.common.packing.layout import using_layout

    hidden, mask, input_ids = _rows([70, 66], [64], [[0, 1]])
    layout = PackedLayout.build(mask, [[0, 1]], [64], input_ids=input_ids)
    with pytest.raises(ValueError, match="FakeGatedDeltaNet"):
        with using_layout(FakeHybrid().double(), layout):
            pass


def test_sharing_refuses_fallback_kernels():
    from arctic_platform.common.packing.layout import using_layout

    hidden, mask, input_ids = _rows([70, 66], [64], [[0, 1]])
    layout = PackedLayout.build(mask, [[0, 1]], [64], input_ids=input_ids)
    with pytest.raises(ValueError, match="flash-linear-attention's chunk_gated_delta_rule"):
        with using_layout(_hybrid(), layout):
            pass


def _sharing_rows():
    hidden, mask, input_ids = _rows([70, 66], [64], [[0, 1]])
    return hidden, PackedLayout.build(mask, [[0, 1]], [64], input_ids=input_ids)


def test_sp1_adapter_refuses_cached_state_under_a_sharing_layout(verified_kernels):
    from arctic_platform.common.packing.layout import using_layout

    hidden, layout = _sharing_rows()
    model = _hybrid()
    with using_layout(model, layout):
        with pytest.raises(RuntimeError, match="cached recurrent"):
            model.linear_attn(layout.pack(hidden).unsqueeze(0), cache_params=object())


def test_sp1_adapter_without_a_stash_refuses_inside_another_models_sharing_context(verified_kernels):
    from arctic_platform.common.packing.layout import using_layout

    hidden, layout = _sharing_rows()
    unshared = PackedLayout.build(layout.row_index.ge(0).long(), [[0], [1]], [0, 0])
    packed = unshared.pack(hidden).unsqueeze(0)
    other, model = _hybrid(), _hybrid()
    with using_layout(other, layout):
        with pytest.raises(ValueError, match="no PackedLayout stashed"):
            model(packed, cu_seq_lens_q=unshared.cu_seqlens)
        # Its own unshared layout, nested in the other model's sharing context, is accepted.
        with using_layout(model, unshared):
            model(packed)
    with using_layout(other, unshared):
        model(packed, cu_seq_lens_q=unshared.cu_seqlens)
    assert model.linear_attn.stock_calls == 1


def test_sp_boundary_wrapper_reads_its_own_stash_inside_another_sharing_context(verified_kernels, monkeypatch):
    from arctic_platform.common.packing.layout import using_layout

    def all_gather(gathered, tensor, group=None):
        gathered[0].copy_(tensor)

    monkeypatch.setattr(gated_delta_net.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(gated_delta_net.dist, "all_gather", all_gather)
    _, sharing = _sharing_rows()
    # Positions that do not start at 0, so only the stash gives the right segments.
    own = PackedLayout.build(
        torch.ones(2, 3, dtype=torch.long), [[0], [1]], [0, 0], torch.tensor([[5, 6, 7], [2, 3, 4]])
    )
    backbone, layer = torch.nn.Module(), torch.nn.Module()
    backbone.forward = lambda **kwargs: None
    gated_delta_net._install_global_boundary_wrapper(backbone, [layer], "sp-group")
    positions = own.position_ids.unsqueeze(0)
    with using_layout(_hybrid(), sharing):
        with using_layout(backbone, own):
            backbone.forward(position_ids=positions)
        torch.testing.assert_close(layer._dss_sp_global_cu_seqlens, own.cu_seqlens.int(), rtol=0, atol=0)
        with pytest.raises(ValueError, match="no PackedLayout stashed"):
            backbone.forward(position_ids=positions)


def _sp2_worker(rank, rendezvous, lengths, shared):
    """SP=2 adapter vs the SP=1 adapter on the whole packed call: outputs and parameter gradients summed over ranks."""
    from datetime import timedelta

    import torch.distributed as dist

    from arctic_platform.common.packing.layout import using_layout

    gated_delta_net.prefix_continuation_kernel_blocker = lambda conv, rule: None
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        groups = [list(range(len(lengths)))] if shared else [[row] for row in range(len(lengths))]
        shared_lengths = [shared] * len(groups)
        hidden, mask, input_ids = _rows(lengths, shared_lengths, groups)
        layout = PackedLayout.build(mask, groups, shared_lengths, input_ids=input_ids)
        assert layout.has_sharing == bool(shared) and layout.num_tokens % 2 == 0
        packed = layout.pack(hidden).unsqueeze(0)
        torch.manual_seed(2)
        weight = torch.randn(1, layout.num_tokens, HIDDEN, dtype=torch.float64)

        reference = _hybrid()
        with using_layout(reference, layout):
            want = reference(packed, cu_seq_lens_q=layout.cu_seqlens)
            want_grads = _grads(reference, [want[0]], [weight[0]])

        torch.manual_seed(1)
        model = FakeHybrid().double()
        group = dist.group.WORLD
        gated_delta_net._adapt_gated_delta_net_module(model.linear_attn, group, fp32_beta=False, stock_numerics=True)
        gated_delta_net._install_global_boundary_wrapper(model, [model.linear_attn], group)
        window = slice(rank * layout.num_tokens // 2, (rank + 1) * layout.num_tokens // 2)
        with using_layout(model, layout):
            out = model(packed[:, window], position_ids=layout.position_ids.unsqueeze(0)[:, window])
            model.zero_grad()
            (out * weight[:, window]).sum().backward()
        assert model.linear_attn.stock_calls == 0

        gathered = [torch.empty_like(out) for _ in range(2)]
        dist.all_gather(gathered, out.detach().contiguous())
        torch.testing.assert_close(torch.cat(gathered, dim=1), want.detach(), rtol=1e-9, atol=1e-10)
        for name, parameter in model.named_parameters():
            grad = parameter.grad.clone()
            dist.all_reduce(grad)
            fp32 = name.endswith(("A_log", "in_proj_a.weight"))
            tolerance = dict(rtol=1e-6, atol=1e-8) if fp32 else dict(rtol=1e-9, atol=1e-12)
            torch.testing.assert_close(grad, want_grads[name], **tolerance, msg=lambda m: f"{name}: {m}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "lengths, shared",
    [([70, 66, 72], 64), ([104, 94], 64), ([70, 66], 0)],
    ids=["root-crosses-ranks", "continuation-crosses-ranks", "unshared-row-crosses-ranks"],
)
def test_sp2_adapter_matches_sp1_with_segments_crossing_ranks(tmp_path, lengths, shared):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    torch.multiprocessing.spawn(_sp2_worker, args=(rendezvous, lengths, shared), nprocs=2, join=True)
