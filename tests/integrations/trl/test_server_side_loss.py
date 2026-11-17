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

"""CPU tests for the adapter's server-side-loss plumbing: explicit shifted-frame
tensors, multiprocess rejection, and TRL-frame re-alignment (``_to_server_bs``)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("trl.experimental.async_grpo.async_grpo_trainer")

from arctic_platform.integrations.trl.client import ArcticTrainingClient  # noqa: E402
from arctic_platform.integrations.trl.client import _reject_multiprocess_trl  # noqa: E402
from arctic_platform.integrations.trl.client import _to_server_bs  # noqa: E402


def _loss_kwargs(old, adv, mask, tokens, *, world_size=1, epsilon_low=0.2, epsilon_high=0.3):
    return dict(
        old_log_probs=old,
        advantages=adv,
        loss_mask=mask,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        tokens_per_rank=tokens,
        world_size=world_size,
    )


def test_server_loss_rejects_multiprocess_trainer():
    with pytest.raises(RuntimeError, match="single-process"):
        _reject_multiprocess_trl(2)


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

    old = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    adv = torch.tensor([[1.0, -1.0, 0.5, 2.0]])
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    tokens_per_rank = torch.tensor(4.0)

    server_logprobs = torch.tensor([[0.0, -0.1, -0.2], [0.0, -0.3, 0.0]])  # [B=2, S=3]
    server_entropy = torch.tensor([[1.0, 1.1, 1.2], [1.3, 1.4, 0.0]])
    client = _RecordingClient(server_logprobs, server_entropy, avg_loss=0.5)

    adapter = ArcticTrainingClient(client=client, temperature=0.7, server_side_loss=True)
    out = adapter.forward_backward(
        input_ids,
        position_ids,
        completion_mask,
        **_loss_kwargs(old, adv, mask, tokens_per_rank),
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
    assert "grad_accum_steps" not in meta
    assert meta["return_fwd_batch"] is True
    assert meta["temperature"] == 0.7
    # Fused single-pass server loss.
    assert payload["processing"]["loss_fn"] == "arctic_platform.integrations.trl.loss.trl_grpo"
    assert payload["processing"]["post"] == ["apply_temperature", "compute_entropy_and_logprobs"]

    # log_probs/entropy come back in TRL frame [1, T-1].
    assert out.log_probs.shape == (1, 4)
    assert out.entropy.shape == (1, 4)
    # No-op backward leaf carrying the server's reported loss.
    assert out.loss.requires_grad
    assert out.loss.item() == pytest.approx(0.5, abs=1e-6)
    out.loss.backward()  # must not raise (leaf tensor)


def test_server_side_loss_forwards_logits_optimization_memory():
    input_ids = torch.tensor([[10, 11, 12, 20, 21]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    completion_mask = torch.tensor([[0, 1, 1, 0, 1]])
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
        input_ids,
        position_ids,
        completion_mask,
        **_loss_kwargs(
            torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            torch.tensor([[1.0, -1.0, 0.5, 2.0]]),
            torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
            torch.tensor(4.0),
        ),
    )
    meta = client.last_payload["meta"]
    assert meta["logits_optimization"] == "memory"
    assert meta["logits_optimization_peak_mem_size_in_gib"] == 8
