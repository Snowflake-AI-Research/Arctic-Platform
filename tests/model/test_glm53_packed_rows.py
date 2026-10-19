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

import types

import pytest
import torch
from torch import nn

from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import require_torch_multi_gpu
from arctic_platform.testing_utils import reserve_free_port

pytest.importorskip("transformers.models.glm5_next")


def _tiny_glm_sparse_model(*, with_linear_attention=False):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    from arctic_platform.model.implementations.glm53.context_parallel import apply_context_parallelism

    hidden_size = 256 if with_linear_attention else 16
    layer_types = (
        ["deepseek_sparse_attention", "linear_attention"] if with_linear_attention else ["deepseek_sparse_attention"]
    )
    config = Glm5NextTextConfig(
        vocab_size=64,
        pad_token_id=0,
        hidden_size=hidden_size,
        intermediate_size=128 if with_linear_attention else 32,
        moe_intermediate_size=16,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        v_head_dim=4,
        qk_nope_head_dim=4,
        layer_types=layer_types,
        indexer_types=["full"] * len(layer_types),
        mlp_layer_types=["dense"] * len(layer_types),
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        index_kpool=2,
        linear_head_dim=128,
        linear_num_heads=2,
        linear_conv_kernel_dim=4,
        hc_mult=1,
        max_position_embeddings=32,
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Glm5NextTextModel(config)
            self.config = type("TextConfig", (), {"model_type": "test"})()

        def forward(self, **kwargs):
            return self.model(**kwargs)

    model = Model().to(torch.bfloat16).eval()
    apply_context_parallelism(model, 1, None)
    return model


@pytest.mark.parametrize("layout", ["zero", "continuous_with_cu_seqlens"])
def test_glm53_sparse_packed_rows_match_isolated_rows(layout):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = _tiny_glm_sparse_model()

    def run(tokens, positions, **kwargs):
        hidden_states = model.model.embed_tokens(tokens).detach().requires_grad_(True)
        output = model(
            inputs_embeds=hidden_states,
            position_ids=positions,
            use_cache=False,
            **kwargs,
        ).last_hidden_state
        return hidden_states, output

    for row_a_length, row_b_length in ((2, 4), (3, 5), (1, 3)):
        tokens = torch.arange(row_a_length + row_b_length)[None] + 1
        if layout == "zero":
            positions = torch.cat((torch.arange(row_a_length), torch.arange(row_b_length)))[None]
            varlen = {}
        else:
            positions = torch.arange(row_a_length + row_b_length)[None] + 7
            varlen = {
                "cu_seq_lens_q": torch.tensor(
                    [0, row_a_length, row_a_length + row_b_length],
                    dtype=torch.int32,
                )
            }

        packed_input, packed = run(tokens, positions, **varlen)
        isolated_varlen = {"cu_seq_lens_q": torch.tensor([0, row_b_length], dtype=torch.int32)} if varlen else {}
        isolated_input, isolated = run(
            tokens[:, row_a_length:],
            positions[:, row_a_length:],
            **isolated_varlen,
        )
        torch.testing.assert_close(packed[:, row_a_length:], isolated, rtol=1e-5, atol=1e-6)

        packed[:, row_a_length:].square().sum().backward()
        isolated.square().sum().backward()
        torch.testing.assert_close(
            packed_input.grad[:, row_a_length:],
            isolated_input.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            packed_input.grad[:, :row_a_length],
            torch.zeros_like(packed_input.grad[:, :row_a_length]),
        )

        changed = tokens.clone()
        changed[:, :row_a_length] += 20
        _, perturbed = run(changed, positions, **varlen)
        torch.testing.assert_close(perturbed[:, row_a_length:], isolated, rtol=1e-5, atol=1e-6)


def test_glm53_sparse_forward_without_explicit_positions():
    model = _tiny_glm_sparse_model()
    output = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
    assert output.last_hidden_state.shape == (1, 4, 16)


def test_glm53_packed_boundaries_isolate_padded_tail():
    from arctic_platform.model.implementations.glm53.context_parallel import _normalize_cu_seqlens

    normalized = _normalize_cu_seqlens(
        torch.tensor([0, 3, 5], dtype=torch.int32),
        sequence_length=8,
        device=torch.device("cpu"),
    )
    assert normalized.tolist() == [0, 3, 5, 6, 7, 8]


@pytest.mark.parametrize(
    "boundaries",
    [
        [1, 3],
        [0, 3, 2],
        [0, 2, 2],
        [0, 9],
    ],
)
def test_glm53_rejects_malformed_packed_boundaries(boundaries):
    from arctic_platform.model.implementations.glm53.context_parallel import _normalize_cu_seqlens

    with pytest.raises(ValueError, match="packed boundaries"):
        _normalize_cu_seqlens(
            torch.tensor(boundaries, dtype=torch.int32),
            sequence_length=8,
            device=torch.device("cpu"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed KDA kernels require CUDA")
def test_glm53_sp1_packed_kda_outputs_and_gradients_match_isolated_row():
    torch.manual_seed(23)
    model = _tiny_glm_sparse_model(with_linear_attention=True).cuda()

    def run(tokens, lengths, loss_start):
        model.zero_grad(set_to_none=True)
        cumulative_lengths = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        sequence_indices = torch.repeat_interleave(
            torch.arange(len(lengths), dtype=torch.int32, device="cuda"),
            torch.tensor(lengths, device="cuda"),
        )[None]
        positions = torch.cat([torch.arange(length, device="cuda") for length in lengths])[None]
        hidden_states = model.model.embed_tokens(tokens).detach().requires_grad_(True)
        output = model(
            inputs_embeds=hidden_states,
            position_ids=positions,
            use_cache=False,
            cu_seq_lens_q=cumulative_lengths,
            cu_seq_lens_k=cumulative_lengths,
            seq_idx=sequence_indices,
            max_length_q=max(lengths),
            max_length_k=max(lengths),
        ).last_hidden_state
        selected_output = output[:, loss_start:]
        selected_output.float().square().sum().backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return selected_output.detach(), hidden_states.grad.detach(), gradients

    row_a_length, row_b_length = 3, 5
    tokens = (torch.arange(row_a_length + row_b_length)[None] + 1).cuda()
    packed_output, packed_input_gradient, packed_gradients = run(
        tokens,
        (row_a_length, row_b_length),
        row_a_length,
    )
    isolated_output, isolated_input_gradient, isolated_gradients = run(
        tokens[:, row_a_length:],
        (row_b_length,),
        0,
    )
    changed = tokens.clone()
    changed[:, :row_a_length] += 20
    changed_output, _, changed_gradients = run(
        changed,
        (row_a_length, row_b_length),
        row_a_length,
    )

    torch.testing.assert_close(packed_output, isolated_output, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        packed_input_gradient[:, row_a_length:],
        isolated_input_gradient,
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(
        packed_input_gradient[:, :row_a_length],
        torch.zeros_like(packed_input_gradient[:, :row_a_length]),
    )
    torch.testing.assert_close(changed_output, packed_output, rtol=1e-2, atol=2e-2)
    for name in packed_gradients:
        torch.testing.assert_close(
            packed_gradients[name],
            isolated_gradients[name],
            rtol=3e-2,
            atol=3e-2,
        )
        torch.testing.assert_close(
            changed_gradients[name],
            packed_gradients[name],
            rtol=3e-2,
            atol=3e-2,
        )


def test_glm53_kda_convolution_shards_select_matching_qkv_channels(monkeypatch):
    import torch.distributed as dist

    from arctic_platform.model.implementations.gpu.sp.gated_delta_net import _shard_convolution_parameter

    num_heads, head_dim, world_size = 4, 3, 2
    channels = num_heads * head_dim
    weight = torch.arange(3 * channels, dtype=torch.float32)[:, None].repeat(1, 4)
    bias = torch.arange(3 * channels, dtype=torch.float32)
    weight_shards, bias_shards = [], []
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: world_size)
    for rank in range(world_size):
        monkeypatch.setattr(dist, "get_rank", lambda group=None, rank=rank: rank)
        for value, shards in ((weight, weight_shards), (bias, bias_shards)):
            shards.append(
                _shard_convolution_parameter(
                    value,
                    name="conv1d",
                    num_key_heads=num_heads,
                    num_value_heads=num_heads,
                    key_head_dim=head_dim,
                    value_head_dim=head_dim,
                    process_group=None,
                )
            )

    for weight_shard, bias_shard in zip(weight_shards, bias_shards, strict=True):
        assert torch.equal(weight_shard[:, 0], bias_shard)
    local_channels = channels // world_size
    reassembled = torch.cat(
        [
            torch.cat([shard[part * local_channels : (part + 1) * local_channels] for shard in bias_shards])
            for part in range(3)
        ]
    )
    assert torch.equal(reassembled, bias)


def test_glm53_head_parallel_kda_requires_global_cu_seqlens():
    from arctic_platform.model.implementations.glm53 import context_parallel as cp

    model = _tiny_glm_sparse_model(with_linear_attention=True)
    attention = model.model.layers[1].self_attn
    hidden_states = torch.zeros(1, 8, model.model.config.hidden_size, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="global cu_seqlens"):
        cp._linear_attention_forward(attention, hidden_states)


def _unfused_causal_conv1d_module(global_cu_seqlens):
    from arctic_platform.model.implementations.gpu.sp.gated_delta_net import _fallback_depthwise_causal_convolution

    def causal_conv1d_fn(x, weight, bias, activation, seq_idx):
        return _fallback_depthwise_causal_convolution(
            x,
            weight=weight,
            bias=bias,
            activation=activation,
            global_cu_seqlens=global_cu_seqlens,
        )

    module = types.ModuleType("causal_conv1d")
    module.causal_conv1d_fn = causal_conv1d_fn
    return module


def _head_parallel_kda_worker(rank, world_size, rendezvous, fallback_convolution):
    import sys
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    from arctic_platform.model.implementations.glm53 import context_parallel as cp

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(29)
        model = _tiny_glm_sparse_model(with_linear_attention=True).cuda()
        attention = model.model.layers[1].self_attn
        attention.conv1d.bias = nn.Parameter(0.1 * torch.randn_like(attention.conv1d.weight[:, 0, 0]))
        global_length = 8
        local_length = global_length // world_size
        # The second packed sequence starts inside rank 0's shard and continues into rank 1's.
        global_cu_seqlens = torch.tensor([0, 3, global_length], dtype=torch.int32).cuda()
        setattr(attention, cp._GLOBAL_CU_SEQLENS, global_cu_seqlens)
        hidden_states = torch.randn(1, global_length, model.model.config.hidden_size, dtype=torch.bfloat16).cuda()

        def run(forward, inputs):
            attention.zero_grad(set_to_none=True)
            inputs = inputs.detach().requires_grad_(True)
            output = forward(attention, inputs)
            output.float().square().sum().backward()
            gradients = {
                name: parameter.grad.detach().float()
                for name, parameter in attention.named_parameters()
                if parameter.grad is not None
            }
            return output.detach(), inputs.grad.detach(), gradients

        # The fallback case must not need a loadable causal_conv1d, including for the local reference.
        reference_modules = (
            {"causal_conv1d": _unfused_causal_conv1d_module(global_cu_seqlens)} if fallback_convolution else {}
        )
        with patch.dict(sys.modules, reference_modules):
            local_output, local_input_gradient, local_gradients = run(
                cp._local_linear_attention_forward, hidden_states
            )
        attention._cp_group = dist.group.WORLD
        attention._cp_world_size = world_size
        shard = slice(rank * local_length, (rank + 1) * local_length)
        parallel_modules = {"causal_conv1d": None} if fallback_convolution else {}
        with patch.dict(sys.modules, parallel_modules), patch.object(cp, "get_logger") as get_logger:
            output, input_gradient, gradients = run(cp._linear_attention_forward, hidden_states[:, shard])
            with torch.no_grad():
                cp._linear_attention_forward(attention, hidden_states[:, shard])
        fallback_logs = [
            call for call in get_logger.return_value.info.call_args_list if "CONVOLUTION_FALLBACK" in call.args[0]
        ]

        outputs = [torch.empty_like(output) for _ in range(world_size)]
        dist.all_gather(outputs, output.contiguous())
        for gradient in gradients.values():
            dist.all_reduce(gradient)
    finally:
        dist.destroy_process_group()

    assert len(fallback_logs) == (1 if fallback_convolution else 0)
    torch.testing.assert_close(torch.cat(outputs, dim=1), local_output, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(input_gradient, local_input_gradient[:, shard], rtol=3e-2, atol=3e-2)
    assert gradients.keys() == local_gradients.keys()
    assert "conv1d.bias" in gradients
    for name, gradient in gradients.items():
        torch.testing.assert_close(gradient, local_gradients[name], rtol=3e-2, atol=3e-2, msg=name)


def _spawn_head_parallel_kda(world_size, fallback_convolution):
    import torch.multiprocessing as mp

    if not fallback_convolution:
        pytest.importorskip("causal_conv1d", exc_type=ImportError)
    port = reserve_free_port(get_unique_port_number() + 1)
    mp.spawn(
        _head_parallel_kda_worker,
        args=(world_size, f"tcp://127.0.0.1:{port}", fallback_convolution),
        nprocs=world_size,
        join=True,
    )


_CONVOLUTIONS = pytest.mark.parametrize(
    "fallback_convolution",
    [False, True],
    ids=["causal_conv1d", "fallback_convolution"],
)


@require_torch_gpu
@_CONVOLUTIONS
def test_glm53_head_parallel_kda_matches_local_path(fallback_convolution):
    _spawn_head_parallel_kda(1, fallback_convolution)


@require_torch_multi_gpu
@_CONVOLUTIONS
def test_glm53_head_parallel_kda_multi_gpu_matches_local_path(fallback_convolution):
    _spawn_head_parallel_kda(2, fallback_convolution)
