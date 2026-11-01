"""Dispatch-side packed schedules."""

import torch

from arctic_platform.common.packing.pack import PackMetadata
from arctic_platform.common.payload import build_packed_schedule


def _meta():
    return PackMetadata(
        batch_size=1,
        sequence_length=2,
        valid_lengths=torch.tensor([2]),
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        padded_tokens=2,
    )


def test_padding_repeats_the_last_window_and_shares_its_payload():
    window = {
        "input_ids": torch.tensor([[1, 2]]),
        "position_ids": torch.tensor([[0, 1]]),
    }
    calls = build_packed_schedule(
        [window, window],
        [_meta(), _meta()],
        sp_size=1,
        sp_rank=0,
        request_call_count=3,
        processing={"loss_fn": "grpo"},
        loss_object=None,
    )

    assert [call.source_index for call in calls] == [0, 1, 1]
    assert [call.is_padding for call in calls] == [False, False, True]
    assert calls[2].payload is calls[1].payload
    assert calls[0].payload.batch_data["input_ids"].shape == (1, 2)
    assert calls[0].payload.meta_data["input_ids"].shape == (2,)
    assert calls[0].payload.meta_data["packed_loss_scale_factor"] == 2
    assert calls[0].payload.meta_data["cu_seqlens"].tolist() == [0, 2]
