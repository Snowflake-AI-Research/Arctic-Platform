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

"""CPU tests for ``engine_old_log_probs`` roll(-1) -> current-token conversion."""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytest.importorskip("trl.experimental.api")

from arctic_platform.integrations.trl.client import engine_old_log_probs  # noqa: E402


class _FakeClient:
    """Records the last ``fwd_no_grad`` payload and returns a preset ``logprobs`` tensor as [B, S]."""

    def __init__(self, logprobs: Any) -> None:
        self._logprobs = logprobs
        self.last_payload: dict | None = None

    def fwd_no_grad(self, payload: dict) -> dict:
        self.last_payload = payload
        return {"batch": {"logprobs": self._logprobs}}


def test_empty_rows_short_circuits():
    client = _FakeClient(torch.zeros((0, 0)))
    assert (
        engine_old_log_probs(client, [], temperature=1.0, pad_token_id=0, rollout_n=1, max_token_len_per_gpu=64) == []
    )
    assert client.last_payload is None  # no forward issued


def test_roll_minus_one_to_current_token_frame():
    # Server roll(-1) logprobs for a [B=2, S=3] batch; row1's last slot is padding (ignored, len 2).
    server_lp = torch.tensor([[0.5, -1.0, 2.0], [3.0, -4.0, 9.9]])
    client = _FakeClient(server_lp)

    out = engine_old_log_probs(
        client, [[10, 11, 12], [30, 31]], temperature=1.0, pad_token_id=0, rollout_n=1, max_token_len_per_gpu=64
    )

    # current[p] = server[p-1] for p>=1, current[0] = 0 (unconditioned first token), per row length.
    assert out == [[0.0, 0.5, -1.0], [0.0, 3.0]]


def test_lengths_match_input_rows_and_first_is_zero():
    server_lp = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 1.0, 2.0, 3.0]])
    rows = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]  # lengths 4, 2, 3
    client = _FakeClient(server_lp)

    out = engine_old_log_probs(client, rows, temperature=0.7, pad_token_id=0, rollout_n=1, max_token_len_per_gpu=128)

    assert [len(o) for o in out] == [4, 2, 3]
    for o in out:
        assert o[0] == 0.0  # first token is always unconditioned -> 0


def test_accepts_nontensor_logprobs():
    # Wire codecs may hand back nested lists; the helper must torch.as_tensor them.
    client = _FakeClient([[0.1, 0.2, 0.3]])
    out = engine_old_log_probs(
        client, [[1, 2, 3]], temperature=1.0, pad_token_id=0, rollout_n=1, max_token_len_per_gpu=64
    )
    # torch.as_tensor makes fp32, so 0.1/0.2 aren't bit-exact after .tolist(); allow fp32 round-trip noise.
    assert out[0] == pytest.approx([0.0, 0.1, 0.2], abs=1e-6)


def test_request_envelope_and_padded_batch():
    server_lp = torch.tensor([[0.5, -1.0, 2.0], [3.0, -4.0, 0.0]])
    client = _FakeClient(server_lp)

    engine_old_log_probs(
        client, [[10, 11, 12], [30, 31]], temperature=0.5, pad_token_id=7, rollout_n=4, max_token_len_per_gpu=256
    )

    payload = client.last_payload
    assert payload is not None
    # Same forward contract as the trainer's `new` log-prob pass: forward-only post-processors, no loss.
    assert payload["processing"] == {
        "post": ["apply_temperature", "compute_entropy_and_logprobs"],
        "loss_fn": None,
    }
    meta = payload["meta"]
    assert meta["temperature"] == 0.5
    assert meta["pad_token_id"] == 7
    assert meta["rollout_n"] == 4
    assert meta["max_token_len_per_gpu"] == 256
    assert meta["calculate_entropy"] is False
    assert meta["drop_position_ids"] is True

    batch = payload["batch"]
    # Dense right-padded [B, S]; non-zero pad token lands only in the pad slot of the short row.
    assert torch.equal(batch["input_ids"], torch.tensor([[10, 11, 12], [30, 31, 7]], dtype=torch.long))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long))
    assert batch["prompts"].shape == (2, 0)
