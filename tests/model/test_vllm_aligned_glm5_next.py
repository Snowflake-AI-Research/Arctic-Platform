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
