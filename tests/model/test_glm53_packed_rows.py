import pytest
import torch
from torch import nn

pytest.importorskip("transformers.models.glm5_next")


def _tiny_glm_sparse_model():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    from arctic_platform.model.implementations.glm53.context_parallel import apply_context_parallelism

    config = Glm5NextTextConfig(
        vocab_size=64,
        pad_token_id=0,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        v_head_dim=4,
        qk_nope_head_dim=4,
        layer_types=["deepseek_sparse_attention"],
        indexer_types=["full"],
        mlp_layer_types=["dense"],
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        index_kpool=2,
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
        isolated_varlen = (
            {"cu_seq_lens_q": torch.tensor([0, row_b_length], dtype=torch.int32)}
            if varlen
            else {}
        )
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
