import pytest
import torch

from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9, reason="SonicMoE needs SM90+"
)


def _reference(x, idx, scores, w1, w2, w3):
    """FP32 sum over each token's local experts of score * down(silu(gate(x)) * up(x))."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    xf = x.float()
    for k in range(idx.shape[1]):
        for e in idx[:, k].unique().tolist():
            if e < 0:
                continue
            rows = idx[:, k] == e
            h = torch.nn.functional.silu(xf[rows] @ w1[e].float().T) * (xf[rows] @ w3[e].float().T)
            out[rows] += scores[rows, k, None].float() * (h @ w2[e].float().T)
    return out


def test_sonic_routed_experts_match_fp32_reference_and_keep_weights():
    from arctic_platform.model.implementations.kernels.sonic_experts import sonic_routed_experts
    from arctic_platform.model.implementations.kernels.sonic_experts import sonic_routed_experts_applicable

    torch.manual_seed(0)
    tokens, topk, experts, hidden, inter = 3000, 4, 8, 512, 256
    module = GroupedExperts(dim=hidden, hidden_dim=inter, num_experts=experts, use_grouped_mm=True).cuda().bfloat16()
    with torch.no_grad():
        for w in (module.w1, module.w2, module.w3):
            w.normal_(std=0.05)
            w.requires_grad_(False)
    assert sonic_routed_experts_applicable(module)
    w1, w2, w3 = (w.detach().clone() for w in (module.w1, module.w2, module.w3))
    idx = torch.randint(0, experts, (tokens, topk), device="cuda")
    idx = torch.where(torch.rand(tokens, topk, device="cuda") < 0.3, -1, idx)  # rows with non-local slots
    idx[0] = -1  # a token with no local expert
    scores = torch.rand(tokens, topk, device="cuda").requires_grad_(True)
    x = torch.randn(tokens, hidden, device="cuda").bfloat16().requires_grad_(True)
    dy = torch.randn(tokens, hidden, device="cuda").bfloat16()

    out = sonic_routed_experts(module, x, idx, scores)
    gx, gs = torch.autograd.grad(out, (x, scores), dy)

    xr = x.detach().float().requires_grad_(True)
    sr = scores.detach().clone().requires_grad_(True)
    ref = _reference(xr, idx, sr, w1, w2, w3)
    rx, rs = torch.autograd.grad(ref, (xr, sr), dy.float())

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    assert rel(out, ref) < 1e-2
    assert rel(gx, rx) < 1e-2
    assert rel(gs[idx >= 0], rs[idx >= 0]) < 1e-2
    assert torch.all(out[0] == 0)
    # gate|up were restacked in place: the parameters are views of one buffer with unchanged values
    assert module.w1.data_ptr() == module._arl_gate_up.data_ptr()
    assert torch.equal(module.w1, w1) and torch.equal(module.w3, w3)


def test_sonic_routed_experts_train_the_expert_weights():
    from arctic_platform.model.implementations.kernels.sonic_experts import sonic_routed_experts

    torch.manual_seed(0)
    tokens, topk, experts, hidden, inter = 2000, 4, 8, 512, 256
    module = GroupedExperts(dim=hidden, hidden_dim=inter, num_experts=experts, use_grouped_mm=True).cuda().bfloat16()
    with torch.no_grad():
        for w in (module.w1, module.w2, module.w3):
            w.normal_(std=0.05)
    idx = torch.randint(0, experts, (tokens, topk), device="cuda")
    idx = torch.where(torch.rand(tokens, topk, device="cuda") < 0.3, -1, idx)
    scores = torch.rand(tokens, topk, device="cuda")
    x = torch.randn(tokens, hidden, device="cuda").bfloat16().requires_grad_(True)
    dy = torch.randn(tokens, hidden, device="cuda").bfloat16()

    out = sonic_routed_experts(module, x, idx, scores)
    grads = torch.autograd.grad(out, (x, module.w1, module.w2, module.w3), dy)

    ref_w = [w.detach().float().requires_grad_(True) for w in (module.w1, module.w2, module.w3)]
    xr = x.detach().float().requires_grad_(True)
    ref = _reference(xr, idx, scores, *ref_w)
    ref_grads = torch.autograd.grad(ref, (xr, *ref_w), dy.float())

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    assert rel(out, ref) < 1e-2
    for got, want in zip(grads, ref_grads):
        assert rel(got, want) < 2e-2


def test_sonic_routed_experts_carry_peft_expert_lora():
    """Called through PEFT's wrappers, SonicMoE sees the LoRA-merged weights and trains A and B like the grouped path."""
    from peft import LoraConfig
    from peft import get_peft_model

    torch.manual_seed(0)
    tokens, topk, experts, hidden, inter = 2000, 4, 8, 512, 256
    module = GroupedExperts(dim=hidden, hidden_dim=inter, num_experts=experts, use_grouped_mm=True)
    module.set_ep_comm_backend("deepep")
    module = module.cuda().bfloat16()
    with torch.no_grad():
        for w in (module.w1, module.w2, module.w3):
            w.normal_(std=0.05)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = module

    model = get_peft_model(
        Tiny(),
        LoraConfig(r=8, lora_alpha=16, target_modules=[], target_parameters=["experts.w1", "experts.w2", "experts.w3"]),
    ).cuda()
    lora = {name: p for name, p in model.named_parameters() if "lora_" in name}
    with torch.no_grad():
        for p in lora.values():
            p.normal_(std=0.05)
    wrapped = model.base_model.model.experts

    idx = torch.randint(0, experts, (tokens, topk), device="cuda")
    idx = torch.where(torch.rand(tokens, topk, device="cuda") < 0.3, -1, idx)
    scores = torch.rand(tokens, topk, device="cuda")
    x = torch.randn(tokens, hidden, device="cuda").bfloat16().requires_grad_(True)
    dy = torch.randn(tokens, hidden, device="cuda").bfloat16()

    def grads_of(out):
        got = torch.autograd.grad(out, (x, *lora.values()), dy)
        return out.detach(), got

    sonic_out, sonic_grads = grads_of(wrapped(x, None, scores, expert_indices=idx))

    # The grouped path on the same rows: sort the (token, expert) pairs by expert, run, scale and sum per token.
    mask = idx >= 0
    token_ids = torch.arange(tokens, device="cuda").unsqueeze(1).expand(tokens, topk)[mask]
    expert_ids, pair_scores = idx[mask], scores[mask]
    order = torch.argsort(expert_ids, stable=True)
    counts = torch.bincount(expert_ids, minlength=experts)
    rows = wrapped(x[token_ids[order]], counts)
    grouped = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float32)
    grouped = grouped.index_add(0, token_ids[order], rows.float() * pair_scores[order, None])
    grouped_out, grouped_grads = grads_of(grouped.bfloat16())

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    assert rel(sonic_out, grouped_out) < 1e-2
    for got, want in zip(sonic_grads, grouped_grads):
        assert rel(got, want) < 2e-2
