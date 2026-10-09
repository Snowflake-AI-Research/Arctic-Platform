import pytest
import torch

pytest.importorskip("transformers.models.qwen4_exp")


def test_qsa_packed_rows_match_isolated_rows(monkeypatch):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    from arctic_platform.model.implementations.qwen38 import qsa_flex as qsa

    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = Qwen4ExpTextConfig(
        vocab_size=32, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, num_experts=2, num_experts_per_tok=1,
        hc_count=2, hc_lowrank=4, layer_types=['qwen_sparse_attention'],
        indexer_n_heads=1, indexer_kv_heads=1, indexer_head_dim=4,
        indexer_budget=2, indexer_compress_ratio=2,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000, 'mrope_section': [1, 1, 0]},
    )
    qsa.register_qsa_flex_backend()
    config._attn_implementation = 'qsa_flex'
    model = Qwen4ExpForCausalLM(config).to(torch.bfloat16).eval()
    qsa.apply_qsa_flex(model)
    # Exercise the real attention implementation on CPU; replace only its CUDA/BF16 entry guard.
    monkeypatch.setattr(
        qsa, "flex_sparse_gqa_attention",
        lambda q, k, v, ids, softmax_scale: qsa._SparseGQAAttention.apply(q, k, v, ids, softmax_scale, 128),
    )

    for lengths in ((2, 4), (3, 5), (1, 3)):
        a, b = lengths
        ids = torch.arange(a + b)[None] + 1
        positions = torch.cat((torch.arange(a), torch.arange(b)))[None]
        for mask in (None, torch.ones_like(ids)):
            def run(tokens, pos, amask):
                x = model.model.embed_tokens(tokens).detach().requires_grad_(True)
                y = model(inputs_embeds=x, position_ids=pos, attention_mask=amask, use_cache=False).logits
                return x, y
            x, packed = run(ids, positions, mask)
            bx, isolated = run(ids[:, a:], positions[:, a:], None if mask is None else mask[:, a:])
            torch.testing.assert_close(packed[:, a:], isolated, rtol=1e-5, atol=1e-6)
            packed[:, a:].square().sum().backward()
            isolated.square().sum().backward()
            torch.testing.assert_close(x.grad[:, a:], bx.grad, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(x.grad[:, :a], torch.zeros_like(x.grad[:, :a]))
            changed = ids.clone()
            changed[:, :a] += 10
            _, perturbed = run(changed, positions, mask)
            torch.testing.assert_close(perturbed[:, a:], isolated, rtol=1e-5, atol=1e-6)
            print(f'packed lengths={lengths} mask={mask is not None}: output/input-gradient/isolation PASS')
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        implicit = model(input_ids=ids, use_cache=False).logits
        broadcast = model(input_ids=ids, position_ids=torch.arange(3)[None], use_cache=False).logits
        explicit = model(input_ids=ids, position_ids=torch.arange(3)[None].expand(2, -1), use_cache=False).logits
        torch.testing.assert_close(implicit, explicit)
        torch.testing.assert_close(broadcast, explicit)
