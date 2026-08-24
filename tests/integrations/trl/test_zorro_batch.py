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

"""CPU tests for the ZoRRo adapter path: packed->structured batch conversion,
prompt/response boundary recovery, and the response-window <-> TRL-shifted
round-trip verified against the server pipeline's own conversion functions."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("trl.experimental.api")

from arctic_platform.integrations.trl.client import (  # noqa: E402
    ArcticTrainingClient,
    _place_response_window,
    _response_window_to_shifted,
    _zorro_layout,
    _zorro_structured_batch,
)

# Oracle: the server pipeline's REAL response-window <-> 1D conversions. The whole point of the
# round-trip test is that the adapter helpers are the exact inverse of these. Fall back to verbatim
# copies if pipeline.py isn't CPU-importable (it pulls the training stack transitively).
try:
    from arctic_platform.rl.processors.pipeline import (  # noqa: E402
        compute_packing_info_for_batch as _packinfo,
        padded_tensor_2d_full_to_unpadded_tensor_1d_response as _full_to_1d,
        unpadded_tensor_1d_response_to_padded_tensor_2d_full as _1d_to_full,
    )
except Exception:  # pragma: no cover - exercised only when the training stack is unavailable

    def _full_to_1d(tensor_2d, attention_mask_2d_bool, max_prompt_len):
        tensor_2d_response = tensor_2d[:, max_prompt_len:]
        attn_response = attention_mask_2d_bool[:, max_prompt_len:]
        return tensor_2d_response[attn_response].unsqueeze(0)

    def _1d_to_full(tensor_1d, attention_mask_2d_bool, max_prompt_len):
        tensor_2d = torch.zeros(
            attention_mask_2d_bool.shape, dtype=tensor_1d.dtype, device=tensor_1d.device
        )
        tensor_2d_response = tensor_2d[:, max_prompt_len:]
        attn_response = attention_mask_2d_bool[:, max_prompt_len:]
        tensor_2d_response[attn_response] = tensor_1d.view(-1)
        return tensor_2d

    def _packinfo(tensor_dict):
        attention_mask = tensor_dict["attention_mask"]
        tensor_dict["sequence_offsets"] = attention_mask.long().sum(dim=1).cumsum(dim=0)
        prompt_ids = tensor_dict["prompts"]
        tensor_dict["response_lens"] = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        return tensor_dict


# Two rollout groups (shared prompt within a group, different prompt lengths across groups):
#   seq0: prompt [5,6,7]  resp [8,9]      len 5
#   seq1: prompt [5,6,7]  resp [8,10]     len 5   (same prompt as seq0)
#   seq2: prompt [1,2]    resp [3,4,5]    len 5
#   seq3: prompt [1,2]    resp [3,6]      len 4   (same prompt as seq2)
# T = 19, T-1 = 18, max_prompt_len = 3.
PACKED_INPUT_IDS = torch.tensor([[5, 6, 7, 8, 9, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 1, 2, 3, 6]])
PACKED_POSITION_IDS = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3]])
PACKED_COMPLETION_MASK = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1]])
SEQ_LENS = torch.tensor([5, 5, 5, 4])
RESPONSE_LEN = 4  # configured max_completion_length (>= longest completion = 3)
PAD = 0


def test_boundary_recovery():
    layout = _zorro_layout(SEQ_LENS, PACKED_COMPLETION_MASK)
    assert layout["starts"] == [0, 5, 10, 15]
    assert layout["prompt_lens"] == [3, 3, 2, 2]
    assert layout["resp_lens"] == [2, 2, 3, 2]
    assert layout["max_prompt_len"] == 3
    assert layout["T"] == 19
    assert layout["B"] == 4


def test_structured_batch_layout_and_shared_prefix():
    batch, layout = _zorro_structured_batch(
        PACKED_INPUT_IDS, PACKED_COMPLETION_MASK, SEQ_LENS, PAD, RESPONSE_LEN
    )
    mp = layout["max_prompt_len"]  # 3
    s = mp + RESPONSE_LEN  # 7

    assert batch["input_ids"].shape == (4, s)
    assert batch["prompts"].shape == (4, mp)
    assert "position_ids" not in batch  # drop_position_ids: server reconstructs from attention_mask

    # Prompts left-padded (right-aligned at max_prompt_len); responses right-padded from column mp.
    assert torch.equal(batch["input_ids"][0], torch.tensor([5, 6, 7, 8, 9, PAD, PAD]))
    assert torch.equal(batch["input_ids"][2], torch.tensor([PAD, 1, 2, 3, 4, 5, PAD]))
    assert torch.equal(batch["input_ids"][3], torch.tensor([PAD, 1, 2, 3, 6, PAD, PAD]))

    # attention_mask 0 on both pad regions.
    assert torch.equal(batch["attention_mask"][0], torch.tensor([1, 1, 1, 1, 1, 0, 0]))
    assert torch.equal(batch["attention_mask"][2], torch.tensor([0, 1, 1, 1, 1, 1, 0]))
    assert torch.equal(batch["attention_mask"][3], torch.tensor([0, 1, 1, 1, 1, 0, 0]))

    # response_mask marks only the response window.
    assert torch.equal(batch["response_mask"][0], torch.tensor([0, 0, 0, 1, 1, 0, 0]))
    assert torch.equal(batch["response_mask"][2], torch.tensor([0, 0, 0, 1, 1, 1, 0]))

    # A rollout group's shared prefix must be byte-identical (find_prompt_groups uses torch.equal).
    assert torch.equal(batch["prompts"][0], batch["prompts"][1])
    assert torch.equal(batch["prompts"][2], batch["prompts"][3])
    assert not torch.equal(batch["prompts"][0], batch["prompts"][2])

    # compute_packing_info_for_batch must recover the true response lengths from the mask.
    info = _packinfo({k: v.clone() for k, v in batch.items()})
    assert torch.equal(info["response_lens"], torch.tensor([2, 2, 3, 2]))


def test_response_len_overflow_raises():
    with pytest.raises(ValueError, match="response_len"):
        _zorro_structured_batch(PACKED_INPUT_IDS, PACKED_COMPLETION_MASK, SEQ_LENS, PAD, response_len=2)


def _completion_positions(layout):
    """Set of TRL-shifted [1, T-1] indices that carry a completion-token value."""
    filled = torch.zeros(layout["T"] - 1, dtype=torch.bool)
    for start, pl, rl in zip(layout["starts"], layout["prompt_lens"], layout["resp_lens"]):
        q0 = start + pl - 1
        filled[q0 : q0 + rl] = True
    return filled


def test_round_trip_identity_through_pipeline():
    """place -> pipeline(full->1d) -> pipeline(1d->full) -> to_shifted recovers the completion values.

    This is the correctness gate: the adapter's response-window helpers must be the exact inverse of
    the server pipeline's conversions, so a value placed for completion token (i, j) lands back at the
    same TRL-shifted index after a full server round-trip.
    """
    layout = _zorro_layout(SEQ_LENS, PACKED_COMPLETION_MASK)
    batch, _ = _zorro_structured_batch(PACKED_INPUT_IDS, PACKED_COMPLETION_MASK, SEQ_LENS, PAD, RESPONSE_LEN)
    attn_bool = batch["attention_mask"].bool()
    mp = layout["max_prompt_len"]

    # Distinct per-index values so any misalignment shows up.
    x = torch.arange(1, layout["T"], dtype=torch.float32).unsqueeze(0)  # [1, T-1]

    full_in = _place_response_window(x, layout, RESPONSE_LEN, torch.float32)
    one_d = _full_to_1d(full_in, attn_bool, mp)  # what the model/loss actually sees on the server
    full_out = _1d_to_full(one_d.view(-1).clone(), attn_bool, mp)  # server post-fwd scatter
    x_rec = _response_window_to_shifted(full_out, layout, RESPONSE_LEN)

    filled = _completion_positions(layout)
    expected = torch.where(filled.unsqueeze(0), x, torch.zeros_like(x))
    assert torch.equal(x_rec, expected)
    # Nothing leaked into prompt/boundary positions.
    assert torch.equal(x_rec[0, ~filled], torch.zeros(int((~filled).sum())))


def test_pipeline_1d_is_response_row_major():
    """The 1D vector the server sees is the per-row response values concatenated in sample order."""
    layout = _zorro_layout(SEQ_LENS, PACKED_COMPLETION_MASK)
    batch, _ = _zorro_structured_batch(PACKED_INPUT_IDS, PACKED_COMPLETION_MASK, SEQ_LENS, PAD, RESPONSE_LEN)
    attn_bool = batch["attention_mask"].bool()
    mp = layout["max_prompt_len"]

    x = torch.arange(1, layout["T"], dtype=torch.float32).unsqueeze(0)
    one_d = _full_to_1d(_place_response_window(x, layout, RESPONSE_LEN, torch.float32), attn_bool, mp)

    # Expected: for each seq, x[q0 : q0+rl]; concatenated over seqs.
    parts = []
    for start, pl, rl in zip(layout["starts"], layout["prompt_lens"], layout["resp_lens"]):
        q0 = start + pl - 1
        parts.append(x[0, q0 : q0 + rl])
    expected = torch.cat(parts).unsqueeze(0)
    assert torch.equal(one_d, expected)
    assert one_d.shape[1] == sum(layout["resp_lens"])  # 9


def test_one_hot_advantage_index_alignment():
    """A one-hot advantage on completion token (seq2, j=1) lands at the right 1D slot and round-trips."""
    layout = _zorro_layout(SEQ_LENS, PACKED_COMPLETION_MASK)
    batch, _ = _zorro_structured_batch(PACKED_INPUT_IDS, PACKED_COMPLETION_MASK, SEQ_LENS, PAD, RESPONSE_LEN)
    attn_bool = batch["attention_mask"].bool()
    mp = layout["max_prompt_len"]

    # seq2 (i=2), response token j=1 -> TRL-shifted index q = start + prompt_len + j - 1 = 10 + 2 + 1 - 1 = 12.
    q = 10 + 2 + 1 - 1
    # row-major 1D slot = sum(resp_lens[:2]) + j = (2 + 2) + 1 = 5.
    slot = (2 + 2) + 1

    adv = torch.zeros(1, layout["T"] - 1, dtype=torch.float32)
    adv[0, q] = 1.0

    one_d = _full_to_1d(_place_response_window(adv, layout, RESPONSE_LEN, torch.float32), attn_bool, mp)
    assert one_d.sum().item() == pytest.approx(1.0)
    assert one_d[0, slot].item() == pytest.approx(1.0)

    full_out = _1d_to_full(one_d.view(-1).clone(), attn_bool, mp)
    adv_rec = _response_window_to_shifted(full_out, layout, RESPONSE_LEN)
    assert adv_rec[0, q].item() == pytest.approx(1.0)
    assert adv_rec.sum().item() == pytest.approx(1.0)


class _RecordingClient:
    """Records the last ``fwd_bwd`` payload; returns preset response-window logprobs/entropy."""

    def __init__(self, logprobs, entropy, avg_loss):
        self._logprobs = logprobs
        self._entropy = entropy
        self._avg_loss = avg_loss
        self.last_payload = None

    def fwd_bwd(self, payload):
        self.last_payload = payload
        return {"batch": {"logprobs": self._logprobs, "entropy": self._entropy}, "avg_loss": self._avg_loss}


class _FakeAccelerator:
    def __init__(self, num_processes=1):
        self.num_processes = num_processes


class _FakeTrainer:
    def __init__(self):
        self.accelerator = _FakeAccelerator(1)
        self.epsilon_low = 0.2
        self.epsilon_high = 0.3
        self.current_gradient_accumulation_steps = 2


def _make_loss_fn(trainer, old, adv, mask, tokens):
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


def test_zorro_forward_backward_wiring():
    """End-to-end wiring: zorro path ships a structured batch + response-window ingredients and maps back."""
    layout = _zorro_layout(SEQ_LENS, PACKED_COMPLETION_MASK)
    mp = layout["max_prompt_len"]
    s = mp + RESPONSE_LEN

    trainer = _FakeTrainer()
    old = torch.zeros(1, layout["T"] - 1)
    adv = torch.zeros(1, layout["T"] - 1)
    mask = torch.zeros(1, layout["T"] - 1)
    mask[0, _completion_positions(layout)] = 1.0
    tokens = torch.tensor(float(sum(layout["resp_lens"])))
    loss_fn = _make_loss_fn(trainer, old, adv, mask, tokens)

    # Server returns response-window logprobs/entropy (values only in the response columns).
    server_logprobs = torch.zeros(4, s)
    server_entropy = torch.zeros(4, s)
    client = _RecordingClient(server_logprobs, server_entropy, avg_loss=0.25)

    adapter = ArcticTrainingClient(
        client=client, temperature=0.7, server_side_loss=True, zorro_train_enable=True, response_len=RESPONSE_LEN
    )
    out = adapter.forward_backward(
        model=None,
        input_ids=PACKED_INPUT_IDS,
        position_ids=PACKED_POSITION_IDS,
        completion_mask=PACKED_COMPLETION_MASK,
        loss_fn=loss_fn,
    )

    payload = client.last_payload
    assert payload is not None
    b = payload["batch"]
    # Structured zorro batch on the wire.
    assert b["input_ids"].shape == (4, s)
    assert b["prompts"].shape == (4, mp)
    assert "position_ids" not in b
    # GRPO ingredients live in the full [B, mp+response_len] frame (response window filled).
    assert b["old_log_probs"].shape == (4, s)
    assert b["advantages"].shape == (4, s)
    assert b["loss_mask"].shape == (4, s)
    meta = payload["meta"]
    assert meta["max_prompt_len"] == mp
    assert meta["max_response_len"] == RESPONSE_LEN
    assert meta["return_fwd_batch"] is True
    assert meta["zorro_train_enable"] is True
    assert meta["load_balancer"] is False  # experimental; off by default (verl-only bin-packing assumption)
    assert payload["processing"]["loss_fn"] == "arctic_platform.integrations.trl.loss.trl_grpo"

    # Outputs mapped back to TRL's shifted [1, T-1] frame.
    assert out.log_probs.shape == (1, layout["T"] - 1)
    assert out.entropy.shape == (1, layout["T"] - 1)
    assert out.loss.item() == pytest.approx(0.25, abs=1e-6)
    out.loss.backward()  # leaf tensor: must not raise


def test_zorro_load_balancer_opt_in():
    # Fixture has two uniform groups of 2 (seq0/seq1 share a prompt, seq2/seq3 share a prompt), so rollout_n=2 passes
    # the uniform-group fail-fast and the load_balancer flag reaches the wire.
    adapter = ArcticTrainingClient(
        client=_RecordingClient(torch.zeros(4, 7), torch.zeros(4, 7), 0.0),
        server_side_loss=True,
        zorro_train_enable=True,
        response_len=RESPONSE_LEN,
        zorro_load_balancer=True,
        rollout_n=2,
    )
    trainer = _FakeTrainer()
    loss_fn = _make_loss_fn(
        trainer,
        torch.zeros(1, 18),
        torch.zeros(1, 18),
        torch.zeros(1, 18),
        torch.tensor(9.0),
    )
    adapter.forward_backward(
        model=None,
        input_ids=PACKED_INPUT_IDS,
        position_ids=PACKED_POSITION_IDS,
        completion_mask=PACKED_COMPLETION_MASK,
        loss_fn=loss_fn,
    )
    assert adapter.client.last_payload["meta"]["load_balancer"] is True


# Ragged groups for the load-balancer fail-fast: group [5,6,7] has 2 rollouts but group [1,2] has only 1 (a stale
# mid-group drop). With rollout_n=2 the group sizes are {2, 1} -> the server bin-packer would raise the opaque
# "shouldn't reach here"; the client must reject it first with a clear error.
#   seq0: prompt [5,6,7] resp [8,9]   len 5
#   seq1: prompt [5,6,7] resp [8,10]  len 5   (same prompt as seq0)
#   seq2: prompt [1,2]   resp [3,4]   len 4   (lone survivor of a dropped group)
RAGGED_INPUT_IDS = torch.tensor([[5, 6, 7, 8, 9, 5, 6, 7, 8, 10, 1, 2, 3, 4]])
RAGGED_POSITION_IDS = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3]])
RAGGED_COMPLETION_MASK = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]])


def test_zorro_load_balancer_ragged_groups_raise():
    adapter = ArcticTrainingClient(
        client=_RecordingClient(torch.zeros(3, 6), torch.zeros(3, 6), 0.0),
        server_side_loss=True,
        zorro_train_enable=True,
        response_len=RESPONSE_LEN,
        zorro_load_balancer=True,
        rollout_n=2,
    )
    trainer = _FakeTrainer()
    n = RAGGED_INPUT_IDS.shape[1] - 1  # 13
    loss_fn = _make_loss_fn(trainer, torch.zeros(1, n), torch.zeros(1, n), torch.zeros(1, n), torch.tensor(6.0))
    with pytest.raises(ValueError, match="ragged prompt groups"):
        adapter.forward_backward(
            model=None,
            input_ids=RAGGED_INPUT_IDS,
            position_ids=RAGGED_POSITION_IDS,
            completion_mask=RAGGED_COMPLETION_MASK,
            loss_fn=loss_fn,
        )


def test_zorro_load_balancer_off_allows_ragged_groups():
    # Without the load balancer the server does not bin-pack groups, so ragged groups are fine (no fail-fast).
    adapter = ArcticTrainingClient(
        client=_RecordingClient(torch.zeros(3, 6), torch.zeros(3, 6), 0.0),
        server_side_loss=True,
        zorro_train_enable=True,
        response_len=RESPONSE_LEN,
        zorro_load_balancer=False,
        rollout_n=2,
    )
    trainer = _FakeTrainer()
    n = RAGGED_INPUT_IDS.shape[1] - 1
    loss_fn = _make_loss_fn(trainer, torch.zeros(1, n), torch.zeros(1, n), torch.zeros(1, n), torch.tensor(6.0))
    adapter.forward_backward(
        model=None,
        input_ids=RAGGED_INPUT_IDS,
        position_ids=RAGGED_POSITION_IDS,
        completion_mask=RAGGED_COMPLETION_MASK,
        loss_fn=loss_fn,
    )
    assert adapter.client.last_payload["meta"]["load_balancer"] is False


def test_zorro_requires_server_side_loss_and_response_len():
    with pytest.raises(ValueError, match="server_side_loss"):
        ArcticTrainingClient(client=object(), zorro_train_enable=True, response_len=4)
    with pytest.raises(ValueError, match="response_len"):
        ArcticTrainingClient(client=object(), zorro_train_enable=True, server_side_loss=True)
