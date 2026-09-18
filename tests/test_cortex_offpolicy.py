"""Sampler log-probs must reach Cortex, and reach it on the right alignment.

Cortex's server-side grpo loss falls back to ``old_log_probs = logprobs.detach()``
when ``context.old_log_probs_shifted`` is absent, which pins π_old ≡ π_new. That
is correct on-policy and wrong as soon as a batch is replayed across policy
versions — and it fails silently, as an importance ratio stuck at 1 rather than
an error. These tests pin both the presence and the shift.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from arctic_platform.rl._cortex_dispatch import _CortexClientShim  # noqa: E402


class _RecordingClient:
    """Stands in for the Cortex SnowAPI client; keeps the payload."""

    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def fwd_bwd(self, payload: dict) -> dict:
        self.payload = payload
        return {"loss": 0.0}


def _dispatch() -> tuple[_CortexClientShim, _RecordingClient]:
    d = _CortexClientShim.__new__(_CortexClientShim)
    client = _RecordingClient()
    d._client = client

    class _Cfg:
        training_gpus = 1

    d._unified_config = _Cfg()
    return d, client


def _batch(**extra: Any) -> dict:
    base = {
        "input_ids": torch.tensor([[10, 11, 12, 13]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "loss_mask": torch.tensor([[0, 0, 1, 1]]),
        "advantages": torch.tensor([[0.0, 0.0, 0.5, 0.5]]),
    }
    base.update(extra)
    return base


@pytest.mark.asyncio
async def test_absent_logprobs_stay_absent():
    """No sampler log-probs means no key — the server's on-policy default."""
    d, client = _dispatch()
    await d.fwd_bwd(_batch())
    assert "old_log_probs_shifted" not in client.payload["context"]


@pytest.mark.asyncio
async def test_old_log_probs_are_rolled_to_the_shifted_contract():
    """``old_log_probs`` is aligned to input_ids; the server wants entry i to
    be the log-prob of token i+1."""
    d, client = _dispatch()
    lp = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    await d.fwd_bwd(_batch(old_log_probs=lp))

    got = client.payload["context"]["old_log_probs_shifted"]
    # roll(-1) with the wrapped last position zeroed: there is no token 4 to
    # score, and leaving the wrapped value would score token 0's log-prob
    # against the end of the sequence.
    assert torch.allclose(got, torch.tensor([[-2.0, -3.0, -4.0, 0.0]]))


@pytest.mark.asyncio
async def test_already_shifted_input_is_passed_through_untouched():
    d, client = _dispatch()
    pre = torch.tensor([[-2.0, -3.0, -4.0, 0.0]])
    await d.fwd_bwd(_batch(old_log_probs_shifted=pre))
    assert torch.allclose(client.payload["context"]["old_log_probs_shifted"], pre)


@pytest.mark.asyncio
async def test_shifted_wins_and_unshifted_does_not_leak_into_kwargs():
    d, client = _dispatch()
    await d.fwd_bwd(_batch(
        old_log_probs_shifted=torch.tensor([[-9.0, -9.0, -9.0, 0.0]]),
        old_log_probs=torch.tensor([[-1.0, -2.0, -3.0, -4.0]]),
    ))
    ctx = client.payload["context"]
    assert torch.allclose(ctx["old_log_probs_shifted"], torch.tensor([[-9.0, -9.0, -9.0, 0.0]]))
    # Neither form may ride along as a model kwarg; the model signature has no
    # such parameter and would raise on the server.
    assert "old_log_probs" not in client.payload["kwargs"]
    assert "old_log_probs_shifted" not in client.payload["kwargs"]
