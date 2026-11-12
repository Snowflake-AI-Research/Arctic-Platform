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

import torch

from arctic_platform.model.implementations.vllm_aligned.glm5_next import _AlignedLayer


def test_aligned_layer_returns_the_aligned_value_with_the_stock_gradient():
    torch.manual_seed(0)
    stock = torch.nn.Linear(6, 6)
    streams = torch.randn(5, 6, requires_grad=True)
    nudge = torch.full((5, 6), 1e-3)

    out = _AlignedLayer.apply(streams, lambda s: torch.tanh(stock(s)) + nudge, lambda s: torch.tanh(stock(s)))
    assert torch.equal(out, torch.tanh(stock(streams)).detach() + nudge)
    weights = torch.randn(5, 6)
    (out * weights).sum().backward()

    want_streams = streams.detach().clone().requires_grad_(True)
    reference = torch.nn.Linear(6, 6)
    reference.load_state_dict(stock.state_dict())
    (torch.tanh(reference(want_streams)) * weights).sum().backward()
    assert torch.equal(streams.grad, want_streams.grad)
    assert torch.equal(stock.weight.grad, reference.weight.grad)
    assert torch.equal(stock.bias.grad, reference.bias.grad)


def test_aligned_layers_chain_like_checkpointed_layers():
    torch.manual_seed(1)
    layers = [torch.nn.Linear(4, 4) for _ in range(3)]
    x = torch.randn(3, 4, requires_grad=True)
    h = x
    for layer in layers:
        h = _AlignedLayer.apply(h, lambda s, layer=layer: torch.relu(layer(s)), lambda s, layer=layer: torch.relu(layer(s)))
    h.sum().backward()
    y = x.detach().clone().requires_grad_(True)
    clones = [torch.nn.Linear(4, 4) for _ in layers]
    for clone, layer in zip(clones, layers):
        clone.load_state_dict(layer.state_dict())
    g = y
    for clone in clones:
        g = torch.relu(clone(g))
    g.sum().backward()
    assert torch.equal(x.grad, y.grad)
    for clone, layer in zip(clones, layers):
        assert torch.equal(clone.weight.grad, layer.weight.grad)


def test_merged_fp8_stacks_weights_and_block_scales():
    import types

    import pytest

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import _merged_fp8

    def linear(rows, cols, fill):
        return types.SimpleNamespace(weight=torch.full((rows, cols), fill), weight_scale_inv=torch.full((rows // 128, 1), fill))

    weight, scale = _merged_fp8(linear(256, 128, 1.0), linear(128, 128, 2.0))
    assert weight.shape == (384, 128) and scale.shape == (3, 1)
    assert scale[:, 0].tolist() == [1.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        _merged_fp8(linear(100, 128, 1.0), linear(128, 128, 2.0))


def test_causal_topk_and_buffer_width_follow_vllm():
    import types

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import causal_topk
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import topk_buffer_width

    assert topk_buffer_width(types.SimpleNamespace(index_topk=2048, index_kpool=4)) == 2176
    rows = causal_topk(torch.tensor([0, 2]), 4)
    assert rows.tolist() == [[0, -1, -1, -1], [0, 1, 2, -1]]


def test_mla_kv_up_weights_split_kv_b_proj_per_head():
    import types

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import mla_kv_up_weights

    config = types.SimpleNamespace(num_attention_heads=2, kv_lora_rank=3, qk_nope_head_dim=4, v_head_dim=5)
    weight = torch.arange(2 * 9 * 3, dtype=torch.float32).view(2 * 9, 3)
    w_uk_t, w_uv = mla_kv_up_weights(types.SimpleNamespace(weight=weight), config)
    assert w_uk_t.shape == (2, 4, 3) and w_uv.shape == (2, 3, 5)
    per_head = weight.view(2, 9, 3)
    assert torch.equal(w_uk_t[1], per_head[1, :4])
    assert torch.equal(w_uv[1], per_head[1, 4:].t())


def test_dequant_fp8_block_matches_vllm_formula_with_partial_blocks():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import dequant_fp8_block

    torch.manual_seed(2)
    weight = torch.randn(200, 130).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 2) + 0.5
    got = dequant_fp8_block(weight, scale)
    full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:200, :130]
    assert got.dtype == torch.bfloat16 and torch.equal(got, (weight.float() * full).to(torch.bfloat16))


def test_prompt_lengths_run_through_the_first_trained_position():
    import torch

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import prompt_lengths

    cu = torch.tensor([0, 5, 8, 12], dtype=torch.int32)
    trained = torch.zeros(12, dtype=torch.bool)
    trained[[2, 3]] = True  # request 0: prompt of 3 tokens, then decoded
    trained[5] = True  # request 1: trained from its first token
    assert prompt_lengths(trained, cu).tolist() == [3, 1, 4]


def test_stock_recompute_replays_the_forward_choices_and_vllm_mla_weights():
    import types

    import torch

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
            self.weight_scale_inv = torch.nn.Parameter(torch.ones(1, 1), requires_grad=False)
            self.block_size = 128

        def forward(self, x):
            return "fp8"

    seen = {}

    class Layer(torch.nn.Module):
        block_type = "deepseek_sparse_attention"
        _replay_layer_idx = 2

        def __init__(self):
            super().__init__()
            self.self_attn = types.SimpleNamespace(q_a_proj=Projection(), kv_a_proj_with_mqa=Projection(),
                                                   q_b_proj=Projection(), o_proj=Projection())  # fmt: skip

        def forward(self, streams, **kwargs):
            seen["routes"] = kwargs.get("routed_experts")
            seen["topk"] = getattr(self.self_attn, glm5_next.REPLAY_TOPK_ATTR, None)
            seen["projection"] = self.self_attn.q_a_proj.forward
            return streams, None

    layer = Layer()
    experts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    topk = torch.tensor([[0, -1], [0, 1], [0, 1]], dtype=torch.int32)
    replay = {id(layer): {"experts": experts, "topk": topk}}
    run = glm5_next._stock_layer_runner(None, torch.zeros(1, 3, dtype=torch.long), torch.arange(3)[None], {}, replay)
    run(layer, torch.zeros(3, 4, 8))

    assert seen["routes"].shape == (1, 3, 3, 2) and torch.equal(seen["routes"][0, :, 2], experts)
    assert torch.equal(seen["topk"][0], topk)
    assert seen["projection"].__func__ is glm5_next._bf16_linear_forward
    assert not hasattr(layer.self_attn, glm5_next.REPLAY_TOPK_ATTR)
    assert layer.self_attn.q_a_proj(torch.zeros(1)) == "fp8"


def _peft_projection(rows: int = 6, cols: int = 8, r: int = 2, alpha: int = 4):
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear as LoraLinear

    class Projection(torch.nn.Linear):
        def __init__(self):
            super().__init__(cols, rows, bias=False)
            self.weight_scale_inv = torch.nn.Parameter(torch.full((1, 1), 0.5), requires_grad=False)
            self.block_size = 128

        def forward(self, x):
            raise AssertionError("the W8A8 forward runs")

    wrapped = LoraLinear(Projection(), "default", LoraConfig(r=r, lora_alpha=alpha), r=r, lora_alpha=alpha)
    torch.nn.init.normal_(wrapped.lora_B["default"].weight)
    fp8 = torch.randn(rows, cols).to(torch.float8_e4m3fn)
    wrapped.base_layer.weight = torch.nn.Parameter(fp8, requires_grad=False)
    return wrapped


def test_lora_factors_cast_to_bf16_then_fold_the_scaling_like_vllm():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import bf16_weight
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import lora_factors

    torch.manual_seed(3)
    projection = _peft_projection(r=2, alpha=4)
    a, b = lora_factors(projection)
    assert a.dtype == b.dtype == torch.bfloat16
    assert torch.equal(a, projection.lora_A["default"].weight.to(torch.bfloat16))
    assert torch.equal(b, projection.lora_B["default"].weight.to(torch.bfloat16) * 2.0)
    assert torch.equal(bf16_weight(projection), (projection.base_layer.weight.float() * 0.5).to(torch.bfloat16))
    projection.enable_adapters(False)
    assert lora_factors(projection) is None
    assert lora_factors(torch.nn.Linear(2, 2)) is None


def test_add_lora_stacks_merged_slices_like_vllm(monkeypatch):
    import sys
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    def lora_shrink(x, a_stack, buffer, *meta_and_scale):
        assert len(meta_and_scale) == 8
        for i, a in enumerate(a_stack):
            buffer[i] = (x.float() @ a[0].float().t()) * meta_and_scale[-1]

    def lora_expand(buffer, b_stack, y, *meta, offset_start, add_inputs):
        assert add_inputs and offset_start == 0
        offset = 0
        for i, b in enumerate(b_stack):
            width = b.shape[1]
            y[:, offset : offset + width] += (buffer[i].to(b.dtype) @ b[0].t()).to(y.dtype)
            offset += width

    class Meta:
        @staticmethod
        def make(max_loras, tokens, device):
            return Meta()

        def prepare_tensors(self, mapping):
            assert mapping.tolist() == [0] * mapping.numel()

        def meta_args(self, tokens, specialize):
            return (None,) * 7

    ops = types.ModuleType("vllm.lora.ops.triton_ops")
    utils = types.ModuleType("vllm.lora.ops.triton_ops.utils")
    utils._LORA_A_PTR_DICT, utils._LORA_B_PTR_DICT = {(1,): "stale"}, {(1,): "stale"}
    ops.LoRAKernelMeta, ops.lora_shrink, ops.lora_expand, ops.utils = Meta, lora_shrink, lora_expand, utils
    for name, module in (("vllm", types.ModuleType("vllm")), ("vllm.lora", types.ModuleType("vllm.lora")),
                         ("vllm.lora.ops", types.ModuleType("vllm.lora.ops")), (ops.__name__, ops),
                         (utils.__name__, utils)):  # fmt: skip
        monkeypatch.setitem(sys.modules, name, module)

    torch.manual_seed(4)
    x = torch.randn(5, 8, dtype=torch.bfloat16)
    y = torch.randn(5, 7, dtype=torch.bfloat16)
    q = (torch.randn(2, 8, dtype=torch.bfloat16), torch.randn(3, 2, dtype=torch.bfloat16))
    kv = (torch.randn(2, 8, dtype=torch.bfloat16), torch.randn(2, 2, dtype=torch.bfloat16))  # 2 of 4 rows: no rope rows
    out = glm5_next.add_lora(y.clone(), x, [q, kv], (3, 4))

    def delta(a, b):
        return ((x.float() @ a.float().t()).to(torch.bfloat16) @ b.t()).to(torch.bfloat16)

    assert torch.equal(out[:, :3], y[:, :3] + delta(*q))
    assert torch.equal(out[:, 3:5], y[:, 3:5] + delta(*kv))
    assert torch.equal(out[:, 5:], y[:, 5:])
    assert not utils._LORA_A_PTR_DICT and not utils._LORA_B_PTR_DICT
    only_kv = glm5_next.add_lora(y.clone(), x, [None, kv], (3, 4))
    assert torch.equal(only_kv[:, :3], y[:, :3]) and torch.equal(only_kv[:, 3:5], out[:, 3:5])
    assert glm5_next.add_lora(y, x, [None, None], (3, 4)) is y


def test_stock_recompute_keeps_the_peft_adapter_on_the_bf16_projection():
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    torch.manual_seed(5)
    projection = _peft_projection()
    x = torch.randn(3, 8)
    seen = {}

    class Layer(torch.nn.Module):
        block_type = "deepseek_sparse_attention"

        def __init__(self):
            super().__init__()
            self.self_attn = types.SimpleNamespace(q_a_proj=projection)

        def forward(self, streams, **kwargs):
            seen["out"] = self.self_attn.q_a_proj(x)
            return streams, None

    run = glm5_next._stock_layer_runner(None, torch.zeros(1, 3, dtype=torch.long), torch.arange(3)[None], {})
    run(Layer(), torch.zeros(3, 4, 8))

    lora = projection.lora_B["default"](projection.lora_A["default"](x)) * projection.scaling["default"]
    base = torch.nn.functional.linear(x, glm5_next.bf16_weight(projection).float())
    assert torch.allclose(seen["out"], base + lora)
    seen["out"].sum().backward()
    assert projection.lora_B["default"].weight.grad.abs().sum() > 0
    assert "forward" not in vars(projection) and "forward" not in vars(projection.base_layer)


def test_substitute_takes_the_exact_value_and_the_reference_gradient():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import substitute

    w = torch.tensor([2.0, 3.0], requires_grad=True)
    reference = (w * 1.5).unsqueeze(0)
    exact = torch.tensor([3.25, 4.25], dtype=torch.float64)
    out = substitute(reference, exact)
    assert out.shape == (1, 2) and out.dtype == torch.float64 and out.tolist() == [[3.25, 4.25]]
    out.sum().backward()
    assert w.grad.tolist() == [1.5, 1.5]


def test_fake_fp8_is_vllms_group_quantization_with_a_straight_through_gradient():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import fake_fp8

    torch.manual_seed(9)
    x = (torch.randn(3, 256) * 5).to(torch.bfloat16).requires_grad_(True)
    groups = x.detach().float().reshape(3, 2, 128)
    scale = groups.abs().amax(-1, keepdim=True) / 448.0
    expected = ((groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).reshape(3, 256)
    assert torch.equal(fake_fp8(x, False), expected.to(torch.bfloat16))
    pow2 = torch.exp2(torch.ceil(torch.log2(scale)))
    expected_ue8m0 = ((groups / pow2).to(torch.float8_e4m3fn).float() * pow2).reshape(3, 256)
    out = fake_fp8(x, True)
    assert torch.equal(out, expected_ue8m0.to(torch.bfloat16)) and not torch.equal(out, fake_fp8(x, False))
    out.float().sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    odd = torch.randn(2, 100)
    assert torch.equal(fake_fp8(odd, True), odd)


def test_vllm_backward_chains_the_exact_sublayer_values_with_stock_gradients(monkeypatch):
    import contextlib
    import sys
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    n, hidden, tokens = 2, 4, 3

    def remix(fused, fp32, post, comb, out, res):
        return post[..., None] * out[..., None, :] + comb.transpose(-1, -2) @ res

    modeling = types.ModuleType("arctic_platform.model.implementations.glm53.modeling_glm5_next")
    modeling.fused_hyper_connection_applies = lambda hc, res: False
    modeling._remix = remix
    package = types.ModuleType("arctic_platform.model.implementations.glm53")
    package.modeling_glm5_next = modeling
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)

    class HC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(1.2))

        def forward(self, res):
            post = torch.sigmoid(self.a) * torch.ones(1, tokens, n)
            return post, torch.eye(n).expand(1, tokens, n, n) * self.a, res.mean(2) * self.a

    class Scale(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(value))

        def forward(self, hidden_states, **kwargs):
            return hidden_states * self.w

    class Layer(torch.nn.Module):
        block_type, _fp32_parity, _replay_layer_idx = "linear_attention", True, 0

        def __init__(self):
            super().__init__()
            self.attn_hc, self.ffn_hc = HC(), HC()
            self.input_layernorm, self.post_attention_layernorm = Scale(0.9), Scale(1.1)
            self.self_attn, self.mlp = Scale(1.7), Scale(2.3)

    def exact_pre(streams, hc, norm, config):
        return torch.full((tokens, n), 0.5), torch.eye(n).expand(tokens, n, n) * 1.25, streams.mean(1) * 1.5

    def exact_post(x, streams, post, comb):
        return remix(False, True, post, comb, x, streams)

    monkeypatch.setattr(glm5_next, "hc_pre", exact_pre)
    monkeypatch.setattr(glm5_next, "hc_post", exact_post)
    monkeypatch.setattr(glm5_next, "_mixer", lambda layer, x, ctx: x * 2.0)
    monkeypatch.setattr(glm5_next, "_mlp", lambda layer, x, ctx: x * 3.0)
    monkeypatch.setattr(glm5_next, "_vllm_mlp_quantization", lambda layer: contextlib.nullcontext())

    layer = Layer()
    ctx = types.SimpleNamespace(config=types.SimpleNamespace(), replay={})
    run = glm5_next._vllm_backward_runner(torch.zeros(tokens, dtype=torch.long), torch.arange(tokens), {}, ctx)
    torch.manual_seed(10)
    streams = torch.randn(tokens, n, hidden, requires_grad=True)
    out = run(layer, streams)

    with torch.no_grad():
        post, comb, x = exact_pre(streams, None, None, None)
        s1 = exact_post(x * 2.0, streams, post, comb)
        post, comb, x = exact_pre(s1, None, None, None)
        expected = exact_post(x * 3.0, s1, post, comb)
    assert torch.equal(out, expected)
    out.sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert streams.grad is not None and streams.grad.abs().sum() > 0


def test_tp_sum_adds_rank_partials_in_fp32_in_rank_order_and_rounds_once():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import tp_sum

    torch.manual_seed(6)
    partials = [torch.randn(5, 7).to(torch.bfloat16) for _ in range(4)]
    expected = partials[0].float()
    for p in partials[1:]:
        expected = expected + p.float()
    out = tp_sum(partials)
    assert out.dtype == torch.bfloat16 and torch.equal(out, expected.to(torch.bfloat16))
    assert tp_sum(partials[:1]) is partials[0]


def test_fp8_and_lora_shards_follow_vllm_tensor_parallel_layouts():
    import pytest

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    weight = torch.arange(512 * 3).reshape(512, 3)
    scale = torch.arange(4 * 1).reshape(4, 1).float()
    w, s = glm5_next._fp8_shard(weight, scale, 2, 1, 0)
    assert torch.equal(w, weight[256:]) and torch.equal(s, scale[2:])
    with pytest.raises(ValueError, match="128-blocks"):
        glm5_next._fp8_shard(weight, scale, 8, 0, 0)
    experts = torch.arange(2 * 3 * 512).reshape(2, 3, 512)
    e, es = glm5_next._fp8_shard(experts, torch.zeros(2, 1, 4), 4, 3, 2)
    assert torch.equal(e, experts[:, :, 384:]) and es.shape == (2, 1, 1)

    a, b = torch.randn(2, 8), torch.randn(6, 2)
    col_a, col_b = glm5_next._column_lora((a, b), 3, 2)
    assert torch.equal(col_a, a) and torch.equal(col_b, b[4:])
    row_a, row_b = glm5_next._row_lora((a, b), 4, 1)
    assert torch.equal(row_a, a[:, 2:4]) and torch.equal(row_b, b)
    assert glm5_next._row_lora(None, 4, 1) is None and glm5_next._column_lora((a, b), 1, 0) == (a, b)


def test_tensor_parallel_check_needs_whole_heads_and_fp8_blocks():
    import types

    import pytest

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import check_tensor_parallel

    config = types.SimpleNamespace(num_attention_heads=64, linear_num_heads=32, intermediate_size=12288,
                                   moe_intermediate_size=1536, n_shared_experts=1)  # fmt: skip
    check_tensor_parallel(config, 4)
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        check_tensor_parallel(config, 8)
    with pytest.raises(ValueError, match="linear_num_heads"):
        check_tensor_parallel(types.SimpleNamespace(**{**vars(config), "linear_num_heads": 6}), 4)


def test_kda_rank_weights_are_vllm_tensor_parallel_shards():
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    heads, head_dim, hidden, kernel = 4, 2, 5, 3
    projection = heads * head_dim

    def lin(rows, cols):
        return types.SimpleNamespace(weight=torch.randn(rows, cols))

    torch.manual_seed(7)
    gate = types.SimpleNamespace(f_a_proj=lin(head_dim, hidden), f_b_proj=lin(projection, head_dim),
                                 A_log=torch.randn(heads), dt_bias=torch.randn(projection))  # fmt: skip
    attn = types.SimpleNamespace(num_heads=heads, head_dim=head_dim, q_proj=lin(projection, hidden),
                                 k_proj=lin(projection, hidden), v_proj=lin(projection, hidden), b_proj=lin(heads, hidden),
                                 g_a_proj=lin(head_dim, hidden), g_b_proj=lin(projection, head_dim), forget_gate=gate,
                                 conv1d=types.SimpleNamespace(weight=torch.randn(3 * projection, 1, kernel)),
                                 o_norm=types.SimpleNamespace(weight=torch.randn(head_dim)), o_proj=lin(hidden, projection))  # fmt: skip
    full = glm5_next._kda_weights(attn)
    assert full.heads == heads and full.merged.shape == (3 * projection + heads + 2 * head_dim, hidden)
    w = glm5_next._kda_weights(attn, tp=2, rank=1)
    half = slice(projection // 2, projection)
    expected = torch.cat([attn.q_proj.weight[half], attn.k_proj.weight[half], attn.v_proj.weight[half],
                          attn.b_proj.weight[2:], gate.f_a_proj.weight, attn.g_a_proj.weight])  # fmt: skip
    assert w.heads == 2 and torch.equal(w.merged, expected)
    conv = attn.conv1d.weight.reshape(3 * projection, kernel)
    assert torch.equal(w.conv, torch.cat([conv[half], conv[projection:][half], conv[2 * projection:][half]]))
    assert torch.equal(w.f_b, gate.f_b_proj.weight[half]) and torch.equal(w.g_b, attn.g_b_proj.weight[half])
    assert torch.equal(w.a_log, gate.A_log[2:]) and torch.equal(w.dt_bias, gate.dt_bias[half])
    assert torch.equal(w.o_weight, attn.o_proj.weight[:, half]) and w.o_lora is None


def test_lm_head_logits_are_the_ranks_vocabulary_slices(monkeypatch):
    import types

    from arctic_platform.model.implementations.vllm_aligned import ops

    shapes = []

    def head(x, weight):
        shapes.append(tuple(weight.shape))
        return x @ weight.t()

    monkeypatch.setattr(ops, "_batch_invariant", lambda: types.SimpleNamespace(linear_batch_invariant=head))
    monkeypatch.setattr(ops, "_token_logprobs_kernel",
                        lambda logits, ids: torch.log_softmax(logits.float(), -1).gather(1, ids[:, None])[:, 0])  # fmt: skip
    torch.manual_seed(8)
    hidden, weight = torch.randn(3, 4), torch.randn(100, 4)
    labels = torch.tensor([0, 50, 99])
    one = ops.lm_head_logprobs_kernel(hidden, weight, labels, 1.0, 8, 64, tensor_parallel=1)
    assert shapes == [(128, 4)]
    shapes.clear()
    four = ops.lm_head_logprobs_kernel(hidden, weight, labels, 1.0, 8, 64, tensor_parallel=4)
    assert shapes == [(32, 4)] * 4
    assert torch.allclose(one, four)


def test_aligned_layer_backward_runs_with_deterministic_algorithms_and_restores_the_setting():
    seen = []

    def stock(s):
        import torch.utils.deterministic

        seen.append((torch.are_deterministic_algorithms_enabled(), torch.utils.deterministic.fill_uninitialized_memory))
        return s * 2

    streams = torch.randn(3, 4, requires_grad=True)
    assert not torch.are_deterministic_algorithms_enabled()
    _AlignedLayer.apply(streams, lambda s: s * 2, stock).sum().backward()
    assert seen == [(True, False)]
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.utils.deterministic.fill_uninitialized_memory
    assert torch.equal(streams.grad, torch.full((3, 4), 2.0))

def test_prefill_rounds_cut_each_prompt_every_chunk_tokens_from_its_start():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import _prefill_rounds

    rounds = _prefill_rounds([0, 5, 8], [4, 3, 10], 4, torch.device("cpu"))
    assert [r.rows.tolist() for r in rounds] == [
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17],
    ]
    assert [r.cu.tolist() for r in rounds] == [[0, 4, 7, 11], [0, 4], [0, 2]]
    assert [r.slots.tolist() for r in rounds] == [[1, 2, 3], [3], [3]]
    assert [r.has_initial_state.tolist() for r in rounds] == [[False] * 3, [True], [True]]
    assert all(r.cu.dtype == torch.int32 and r.slots.dtype == torch.int32 for r in rounds)
