# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the per-op backward of the vLLM-aligned GLM-5.3-Flash forward (``glm5_next_backward``)."""

import torch

from arctic_platform.model.implementations.vllm_aligned import glm5_next_backward as bw


def test_vllm_op_takes_the_kernels_value_and_the_references_gradient():
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    w = torch.tensor(0.5, requires_grad=True)
    out = bw.vllm_op(lambda x, w: (x * w).round(), lambda x, w: x * w * 3.0, x, w)
    assert out.tolist() == [0.0, -1.0, 2.0]
    out.sum().backward()
    assert x.grad.tolist() == [1.5, 1.5, 1.5] and w.grad.item() == 6.0


def test_op_routes_through_vllm_op_only_in_differentiable_mode():
    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    x = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(2, 4)
    kernel = lambda x, weight, *, scale: x @ weight.t() * scale  # noqa: E731
    assert torch.equal(glm5_next._op(kernel, "ref_linear", x, weight, scale=1.0), x @ weight.t())
    with glm5_next.differentiable():
        out = glm5_next._op(kernel, "ref_linear", x, weight, scale=2.0)
    assert torch.equal(out, x @ weight.t() * 2.0)
    out.sum().backward()
    assert torch.allclose(x.grad, weight.sum(0).expand(3, 4))  # the reference's gradient: no factor 2


def _naive_kda(q, k, v, g, beta, a_log, dt_bias, lower_bound):
    heads, dim = q.shape[1:]
    h = torch.zeros(heads, v.shape[-1], dim, dtype=torch.float64)
    out = []
    for t in range(q.shape[0]):
        qt = q[t].double() / torch.sqrt((q[t].double() ** 2).sum(-1, keepdim=True) + 1e-6) * dim**-0.5
        kt = k[t].double() / torch.sqrt((k[t].double() ** 2).sum(-1, keepdim=True) + 1e-6)
        gate = lower_bound / (1 + torch.exp(-torch.exp(a_log.double()).unsqueeze(-1) * (g[t].double() + dt_bias.double().view(heads, dim))))
        h = h * torch.exp(gate).unsqueeze(1)
        vt = (v[t].double() - torch.einsum("hvk,hk->hv", h, kt)) * torch.sigmoid(beta[t].double()).unsqueeze(-1)
        h = h + torch.einsum("hv,hk->hvk", vt, kt)
        out.append(torch.einsum("hvk,hk->hv", h, qt))
    return torch.stack(out)


def test_kda_reference_is_the_gated_delta_rule_per_request_and_chunk_invariant():
    torch.manual_seed(0)
    tokens, heads, dim = 7, 2, 4
    q, k, v, g = (torch.randn(1, tokens, heads, dim) for _ in range(4))
    beta, a_log, dt_bias = torch.randn(tokens, heads), torch.randn(heads), torch.randn(heads * dim)
    cu = torch.tensor([0, 3, 7])
    out = bw.ref_kda_core(q, k, v, g, beta, a_log, dt_bias, cu, -5.0, chunk=2)
    assert torch.allclose(out, bw.ref_kda_core(q, k, v, g, beta, a_log, dt_bias, cu, -5.0, chunk=64))
    for s0, s1 in ((0, 3), (3, 7)):
        expected = _naive_kda(q[0, s0:s1], k[0, s0:s1], v[0, s0:s1], g[0, s0:s1], beta[s0:s1], a_log, dt_bias, -5.0)
        assert torch.allclose(out[s0:s1].double(), expected, atol=1e-5)
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta, a_log, dt_bias)]
    bw.ref_kda_core(*leaves, cu, -5.0, chunk=2).sum().backward()
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in leaves)


def test_sparse_mla_reference_attends_to_each_tokens_rows():
    torch.manual_seed(1)
    tokens, heads, rope, latent, width = 5, 2, 3, 4, 3
    q_rope, q_v = torch.randn(tokens, heads, rope), torch.randn(tokens, heads, latent)
    k_cache, v_cache = torch.randn(tokens, 1, 1, rope), torch.randn(tokens, 1, 1, latent)
    rows = torch.tensor([[0, -1, -1], [0, 1, -1], [1, 2, -1], [0, 2, 3], [4, 3, -1]], dtype=torch.int32)
    valid = (rows >= 0).sum(-1).to(torch.int32)
    out = bw.ref_sparse_mla(q_rope, k_cache, v_cache, q_v, rows, valid, 0.7, chunk=2)
    k, v = k_cache.reshape(tokens, rope), v_cache.reshape(tokens, latent)
    for t in range(tokens):
        sel = rows[t, : valid[t]].long()
        scores = (q_rope[t] @ k[sel].t() + q_v[t] @ v[sel].t()) * 0.7
        assert torch.allclose(out[t], torch.softmax(scores, -1) @ v[sel], atol=1e-6)


def test_causal_conv_reference_restarts_at_each_request():
    torch.manual_seed(2)
    x, weight = torch.randn(6, 3), torch.randn(3, 4)
    out = bw.ref_causal_conv_silu(x, weight, torch.tensor([0, 2, 6]))
    for s0, s1 in ((0, 2), (2, 6)):
        for t in range(s0, s1):
            acc = sum(weight[:, 3 - j] * x[t - j] for j in range(4) if t - j >= s0)
            assert torch.allclose(out[t], torch.nn.functional.silu(acc), atol=1e-6)


def test_router_weights_are_renormalized_sigmoid_scores_at_the_ids():
    logits = torch.tensor([[0.0, 1.0, -1.0, 2.0]])
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    s = torch.sigmoid(logits[0, [3, 1]])
    assert torch.allclose(bw.ref_router_weights(logits, ids, True, 2.5), (s / s.sum() * 2.5).unsqueeze(0))
    assert torch.allclose(bw.ref_router_weights(logits, ids, False, 1.0), s.unsqueeze(0))


def test_add_lora_reference_adds_each_slices_delta_and_pads_narrow_b():
    torch.manual_seed(3)
    y, x = torch.randn(4, 7), torch.randn(4, 5)
    a0, b0, a1, b1 = torch.randn(2, 5), torch.randn(3, 2), torch.randn(2, 5), torch.randn(2, 2)
    out = bw.ref_add_lora(y, x, a0, b0, a1, b1, slices=(3, 4))
    assert torch.allclose(out[:, :3], y[:, :3] + x @ a0.t() @ b0.t(), atol=1e-6)
    assert torch.allclose(out[:, 3:5], y[:, 3:5] + x @ a1.t() @ b1.t(), atol=1e-6)
    assert torch.equal(out[:, 5:], y[:, 5:])


def test_mhc_pre_reference_mixes_streams_with_a_doubly_normalized_comb():
    torch.manual_seed(4)
    n, hidden, tokens = 4, 8, 3
    streams = torch.randn(tokens, n, hidden).to(torch.bfloat16)
    fn, scale, base = torch.randn(2 * n + n * n, n * hidden) * 0.1, torch.ones(3), torch.zeros(2 * n + n * n)
    post, comb, x = bw.ref_mhc_pre(streams, fn, scale, base, torch.ones(hidden), 1e-6, 1e-6, 1e-6, 2.0, 20, 1e-6)
    assert post.shape == (tokens, n) and comb.shape == (tokens, n, n) and x.shape == (tokens, hidden)
    assert torch.allclose(comb.sum(-2), torch.ones(tokens, n), atol=1e-4)
    assert torch.allclose(comb.sum(-1), torch.ones(tokens, n), atol=1e-2)
    assert bool((post > 0).all() and (post < 2.0).all())
