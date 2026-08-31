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

"""CPU tests for the adapter's server-side-loss plumbing: GRPO closure
introspection (``_extract_grpo_ingredients``) and TRL-frame re-alignment
(``_to_server_bs``)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("trl.experimental.api")

from arctic_platform.integrations.trl.client import ArcticTrainingClient  # noqa: E402
from arctic_platform.integrations.trl.client import _extract_grpo_ingredients  # noqa: E402
from arctic_platform.integrations.trl.client import _to_server_bs  # noqa: E402


class _FakeAccelerator:
    def __init__(self, num_processes: int = 1) -> None:
        self.num_processes = num_processes


class _FakeTrainer:
    """Minimal stand-in for TRL's AsyncGRPOTrainer (the closure's ``self``)."""

    def __init__(self, num_processes: int = 1) -> None:
        self.accelerator = _FakeAccelerator(num_processes)
        self.epsilon_low = 0.2
        self.epsilon_high = 0.3
        self.current_gradient_accumulation_steps = 2


def _make_trl_like_loss_fn(trainer, old, adv, mask, tokens):
    """Build a closure with the same freevar names TRL's compute_loss creates."""
    shifted_old_log_probs = old
    shifted_advantages = adv
    shifted_completion_mask = mask
    tokens_per_rank = tokens
    self = trainer

    def loss_fn(log_probs):
        coef_1 = torch.exp(log_probs - shifted_old_log_probs)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token = -torch.min(coef_1 * shifted_advantages, coef_2 * shifted_advantages)
        loss = (per_token * shifted_completion_mask).sum() / tokens_per_rank
        return loss / self.current_gradient_accumulation_steps

    return loss_fn


def test_extract_recovers_closure_values():
    trainer = _FakeTrainer()
    old = torch.tensor([[0.1, 0.2, 0.3]])
    adv = torch.tensor([[1.0, -1.0, 0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    tokens = torch.tensor(2.0)

    loss_fn = _make_trl_like_loss_fn(trainer, old, adv, mask, tokens)
    ing = _extract_grpo_ingredients(loss_fn)

    assert torch.equal(ing["old_log_probs"], old)
    assert torch.equal(ing["advantages"], adv)
    assert torch.equal(ing["completion_mask"], mask)
    assert ing["tokens_per_rank"] is tokens
    assert ing["epsilon_low"] == pytest.approx(0.2)
    assert ing["epsilon_high"] == pytest.approx(0.3)
    assert ing["grad_accum_steps"] == 2


def test_extract_raises_on_plain_callable():
    def not_a_closure(log_probs):
        return log_probs.sum()

    with pytest.raises(RuntimeError, match="loss_placement=client"):
        _extract_grpo_ingredients(not_a_closure)


def test_extract_raises_on_missing_freevar():
    # A closure that omits shifted_advantages/etc must fail loudly (TRL drift guard).
    other = torch.tensor([[1.0]])

    def partial_loss_fn(log_probs):
        return (log_probs * other).sum()

    with pytest.raises(RuntimeError, match="freevars"):
        _extract_grpo_ingredients(partial_loss_fn)


def test_extract_raises_on_multiprocess_trainer():
    trainer = _FakeTrainer(num_processes=2)
    loss_fn = _make_trl_like_loss_fn(
        trainer, torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2), torch.tensor(2.0)
    )
    with pytest.raises(RuntimeError, match="single-process"):
        _extract_grpo_ingredients(loss_fn)


def test_to_server_bs_places_shifted_into_padded_rows():
    # TRL-frame [1, T-1] -> server [B, S]: unshift appends a trailing 0, unpack by seq_lens.
    shifted = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # T-1 = 4
    seq_lens = torch.tensor([3, 2])  # T = 5

    out = _to_server_bs(shifted, seq_lens)

    expected = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]])
    assert torch.equal(out, expected)


class _RecordingClient:
    """Records the last ``fwd_bwd`` payload and returns preset [B, S] logprobs/entropy."""

    def __init__(self, logprobs, entropy, avg_loss):
        self._logprobs = logprobs
        self._entropy = entropy
        self._avg_loss = avg_loss
        self.last_payload = None

    def fwd_bwd(self, payload):
        self.last_payload = payload
        return {"batch": {"logprobs": self._logprobs, "entropy": self._entropy}, "avg_loss": self._avg_loss}


def test_forward_backward_server_branch_wiring():
    # Two packed sequences of length 3 and 2 -> T=5, T-1=4.
    input_ids = torch.tensor([[10, 11, 12, 20, 21]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    completion_mask = torch.tensor([[0, 1, 1, 0, 1]])

    trainer = _FakeTrainer()  # epsilon 0.2/0.3, grad_accum 2, num_processes 1
    old = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    adv = torch.tensor([[1.0, -1.0, 0.5, 2.0]])
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    tokens_per_rank = torch.tensor(4.0)
    loss_fn = _make_trl_like_loss_fn(trainer, old, adv, mask, tokens_per_rank)

    server_logprobs = torch.tensor([[0.0, -0.1, -0.2], [0.0, -0.3, 0.0]])  # [B=2, S=3]
    server_entropy = torch.tensor([[1.0, 1.1, 1.2], [1.3, 1.4, 0.0]])
    client = _RecordingClient(server_logprobs, server_entropy, avg_loss=0.5)

    adapter = ArcticTrainingClient(client=client, temperature=0.7, server_side_loss=True)
    out = adapter.forward_backward(
        model=None,
        input_ids=input_ids,
        position_ids=position_ids,
        completion_mask=completion_mask,
        loss_fn=loss_fn,
    )

    payload = client.last_payload
    assert payload is not None
    # Ingredients shipped as server [B, S] frame.
    assert payload["batch"]["old_log_probs"].shape == (2, 3)
    assert payload["batch"]["advantages"].shape == (2, 3)
    assert payload["batch"]["loss_mask"].shape == (2, 3)
    # Meta carries the GRPO params + the opt-in batch return.
    meta = payload["meta"]
    assert meta["epsilon_low"] == pytest.approx(0.2)
    assert meta["epsilon_high"] == pytest.approx(0.3)
    assert meta["batch_num_tokens"] == 4
    assert meta["grad_accum_steps"] == 2
    assert meta["return_fwd_batch"] is True
    assert meta["temperature"] == 0.7
    # Fused single-pass server loss.
    assert payload["processing"]["loss_fn"] == "arctic_platform.integrations.trl.loss.trl_grpo"
    assert payload["processing"]["post"] == ["apply_temperature", "compute_entropy_and_logprobs"]

    # log_probs/entropy come back in TRL frame [1, T-1].
    assert out.log_probs.shape == (1, 4)
    assert out.entropy.shape == (1, 4)
    assert out.aux_loss is None
    # No-op backward leaf carrying the server's reported loss.
    assert out.loss.requires_grad
    assert out.loss.item() == pytest.approx(0.5, abs=1e-6)
    out.loss.backward()  # must not raise (leaf tensor)


def test_server_side_loss_forwards_logits_optimization_memory():
    input_ids = torch.tensor([[10, 11, 12, 20, 21]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    completion_mask = torch.tensor([[0, 1, 1, 0, 1]])
    trainer = _FakeTrainer()
    loss_fn = _make_trl_like_loss_fn(
        trainer,
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        torch.tensor([[1.0, -1.0, 0.5, 2.0]]),
        torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
        torch.tensor(4.0),
    )
    client = _RecordingClient(
        torch.tensor([[0.0, -0.1, -0.2], [0.0, -0.3, 0.0]]),
        torch.tensor([[1.0, 1.1, 1.2], [1.3, 1.4, 0.0]]),
        avg_loss=0.5,
    )
    adapter = ArcticTrainingClient(
        client=client,
        temperature=0.7,
        server_side_loss=True,
        logits_optimization="memory",
        logits_optimization_peak_mem_size_in_gib=8,
    )
    adapter.forward_backward(
        model=None,
        input_ids=input_ids,
        position_ids=position_ids,
        completion_mask=completion_mask,
        loss_fn=loss_fn,
    )
    meta = client.last_payload["meta"]
    assert meta["logits_optimization"] == "memory"
    assert meta["logits_optimization_peak_mem_size_in_gib"] == 8
