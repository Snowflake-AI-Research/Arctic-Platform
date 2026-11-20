"""Expert SwiGLU on BF16 inputs: the forward and the up gradient keep their rounding, the gate gradient rounds once."""

import pytest
import torch


@pytest.mark.parametrize("limit", [None, 10.0])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))]
)
def test_expert_swiglu_preserves_forward_and_widens_gate_gradient(device, limit):
    from arctic_platform.model.implementations.moe.layers.moe import _apply_swiglu

    # Element zero distinguishes both placements of the rounding; the rest exercise the clamp boundaries.
    gate = torch.tensor([0.21875, 10, 11, -1], dtype=torch.bfloat16, device=device, requires_grad=True)
    up = torch.tensor([1.15625, -10, 10, 11], dtype=torch.bfloat16, device=device, requires_grad=True)
    grad = torch.full_like(up, 1.4375)
    output = _apply_swiglu(gate, up, limit)
    output.backward(grad)

    clamped_gate, clamped_up = gate.detach(), up.detach()
    if limit is not None:
        clamped_gate, clamped_up = clamped_gate.clamp(max=limit), clamped_up.clamp(-limit, limit)
    silu = torch.nn.functional.silu(clamped_gate.double()).bfloat16()
    expected_up = grad * silu
    sig = clamped_gate.double().sigmoid()
    expected_gate = grad.double() * clamped_up * sig * (1 + clamped_gate * (1 - sig))
    if limit is not None:
        expected_gate *= gate.detach() <= limit
        expected_up *= up.detach().abs() <= limit
    assert torch.equal(output, silu * clamped_up)
    assert torch.equal(up.grad, expected_up)
    assert torch.equal(gate.grad, expected_gate.bfloat16())


def test_expert_swiglu_up_gradient_retains_its_dependence_on_gate():
    from arctic_platform.model.implementations.moe.layers.moe import _apply_swiglu

    gate = torch.tensor([0.21875, 2], dtype=torch.bfloat16, requires_grad=True)
    up = torch.tensor(1.15625, dtype=torch.bfloat16, requires_grad=True)
    (up_grad,) = torch.autograd.grad(_apply_swiglu(gate, up, None).sum(), up, create_graph=True)
    (mixed,) = torch.autograd.grad(up_grad, gate)
    (expected,) = torch.autograd.grad(torch.nn.functional.silu(gate).sum(), gate)
    assert torch.equal(mixed, expected)
