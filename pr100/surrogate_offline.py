"""Offline proof: the grpo-as-surrogate trick reproduces exact parameter gradients.

Path A (direct)    : real GRPO loss, backward through the model.
Path B (surrogate) : forward-no-grad -> client-side loss on a leaf -> w = dL/dlogprobs
                     -> server runs `grpo` with old_log_probs=lp0 and advantages=-w*N
                     -> backward through the model.

If the trick is sound, both paths produce bit-comparable gradients on every
parameter, because at ratio == 1 GRPO's gradient wrt logprobs is -advantages/N.

Uses the real `_grpo_loss` shipped in the deployed image, not a reimplementation.
"""

import torch

from arctic_training.arctic_rl.processors.grpo import _grpo_loss

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

B, S, V, H = 2, 6, 32, 8
N = float(B * S)


class TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(V, H)
        self.proj = torch.nn.Linear(H, V)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.emb(input_ids))


def logprobs_of(model: TinyLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token logprobs under the roll(-1) convention the server uses."""
    logits = model(input_ids)
    labels = torch.roll(input_ids, shifts=-1, dims=-1)
    return torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def grads(model: TinyLM) -> dict:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def run(loss_mask: torch.Tensor, label: str) -> None:
    model = TinyLM()
    input_ids = torch.randint(0, V, (B, S))

    # The "real" objective: a GRPO step against a behavioural policy that is not
    # the current one, so the ratio is genuinely off 1 and the loss is non-trivial.
    old_log_probs = logprobs_of(model, input_ids).detach() - 0.05
    advantages = torch.randn(B, S)
    real_ctx = {
        "old_log_probs_shifted": old_log_probs,
        "advantages": advantages,
        "loss_mask": loss_mask,
    }
    cfg = {"batch_num_tokens": N, "dp_size": 1}

    # ---- Path A: direct, gradients flow from the real loss through the model.
    model.zero_grad()
    lp = logprobs_of(model, input_ids)
    loss_a, _ = _grpo_loss({"logprobs": lp}, real_ctx, cfg, "cpu")
    loss_a.backward()
    grad_a = grads(model)

    # ---- Path B: two passes, loss evaluated on the client.
    model.zero_grad()
    lp0 = logprobs_of(model, input_ids).detach()          # what /forward returns
    leaf = lp0.clone().requires_grad_(True)
    loss_client, _ = _grpo_loss({"logprobs": leaf}, real_ctx, cfg, "cpu")
    (w,) = torch.autograd.grad(loss_client, leaf)          # w = dL/dlogprobs

    # The surrogate call: ratio pinned to 1, advantages carry -w*N so that
    # d(surrogate)/d(logprobs) == w exactly.
    lp2 = logprobs_of(model, input_ids)
    surrogate_ctx = {
        "old_log_probs_shifted": lp0,
        "advantages": -w * N,
        "loss_mask": torch.ones_like(loss_mask),
    }
    loss_b, _ = _grpo_loss({"logprobs": lp2}, surrogate_ctx, cfg, "cpu")
    loss_b.backward()
    grad_b = grads(model)

    worst = max(
        (grad_a[n] - grad_b[n]).abs().max().item() / max(grad_a[n].abs().max().item(), 1e-30)
        for n in grad_a
    )
    print(f"  {label}:")
    print(f"    client loss value   = {float(loss_client):.12f}  (direct = {float(loss_a):.12f})")
    print(f"    worst relative grad difference across all params = {worst:.3e}")
    print(f"    {'PASS' if worst < 1e-9 else 'FAIL'}")


print("grpo-as-surrogate: parameter-gradient equivalence")
run(torch.ones(B, S), "loss_mask = all ones")

mask = torch.ones(B, S)
mask[0, :2] = 0.0
mask[1, -1] = 0.0
run(mask, "loss_mask = partially masked")
