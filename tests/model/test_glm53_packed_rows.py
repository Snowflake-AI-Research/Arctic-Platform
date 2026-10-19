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

import pytest
import torch
from torch import nn

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed KDA kernels require CUDA")
def test_glm53_head_parallel_kda_matches_local_path(tmp_path):
    import torch.distributed as dist

    from arctic_platform.model.implementations.glm53 import context_parallel as cp

    created = not dist.is_initialized()
    if created:
        dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'group'}", world_size=1, rank=0)
    try:
        torch.manual_seed(29)
        model = _tiny_glm_sparse_model(with_linear_attention=True).cuda()
        attention = model.model.layers[1].self_attn
        setattr(attention, cp._GLOBAL_CU_SEQLENS, torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda"))
        attention._cp_group = dist.group.WORLD
        attention._cp_world_size = 1
        hidden_states = torch.randn(1, 8, model.model.config.hidden_size, device="cuda", dtype=torch.bfloat16)

        def run(forward):
            attention.zero_grad(set_to_none=True)
            inputs = hidden_states.detach().requires_grad_(True)
            output = forward(attention, inputs)
            output.float().square().sum().backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in attention.named_parameters()
                if parameter.grad is not None
            }
            return output.detach(), inputs.grad.detach(), gradients

        local_output, local_input_gradient, local_gradients = run(cp._local_linear_attention_forward)
        parallel_output, parallel_input_gradient, parallel_gradients = run(cp._linear_attention_forward)
    finally:
        if created:
            dist.destroy_process_group()

    torch.testing.assert_close(parallel_output, local_output, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(parallel_input_gradient, local_input_gradient, rtol=3e-2, atol=3e-2)
    assert parallel_gradients.keys() == local_gradients.keys()
    for name, gradient in parallel_gradients.items():
        torch.testing.assert_close(gradient, local_gradients[name], rtol=3e-2, atol=3e-2, msg=name)
