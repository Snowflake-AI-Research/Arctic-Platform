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
"""CPU checks for the packed GDN boundary hook on the SFT path."""

from __future__ import annotations

import torch
from torch import nn

from arctic_platform.common.utils.packed_gdn import install_packed_gdn_boundaries
from arctic_platform.common.utils.packed_gdn import model_has_linear_attn
from arctic_platform.sft.processor import run_sft_pipeline


class _Loss:
    def __init__(self):
        self.loss = torch.tensor(1.0)


class _Capture(nn.Module):
    def __init__(self, *, linear_attn: bool):
        super().__init__()
        if linear_attn:
            self.linear_attn = nn.Identity()
        self.seen: dict = {}

    def forward(self, **kwargs):
        self.seen = kwargs
        return _Loss()


class _Engine:
    def __init__(self, module):
        self.module = module

    def train(self):
        return None

    def __call__(self, **kwargs):
        return self.module(**kwargs)

    def backward(self, loss, scale_wrt_gas=False):
        return None


def _packed_batch():
    # Two documents, lengths 3 and 4. position_ids reset to 0 at the boundary.
    return {
        "input_ids": torch.arange(7).unsqueeze(0),
        "labels": torch.tensor([[-100, 1, 2, -100, 3, 4, 5]]),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3]]),
    }


def test_dense_model_is_a_noop():
    model = _Capture(linear_attn=False)
    assert model_has_linear_attn(model) is False
    assert install_packed_gdn_boundaries(model) is False
    assert getattr(model, "_at_packed_gdn_boundaries_installed", False) is False
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    model(position_ids=position_ids)
    assert "cu_seq_lens_q" not in model.seen
    assert torch.equal(model.seen["position_ids"], position_ids)


def test_hook_derives_boundaries_from_position_ids():
    model = _Capture(linear_attn=True)
    assert install_packed_gdn_boundaries(model) is True
    assert install_packed_gdn_boundaries(model) is True
    assert len(model._forward_pre_hooks) == 1
    model(position_ids=torch.tensor([[0, 1, 2, 0, 1, 2, 3]]))
    cu = model.seen["cu_seq_lens_q"]
    assert torch.equal(cu, model.seen["cu_seq_lens_k"])
    assert cu.tolist() == [0, 3, 7]
    assert int(model.seen["max_length_q"]) == 4
    assert int(model.seen["max_length_k"]) == 4
    assert model.seen["seq_idx"].dtype == torch.int32
    assert model.seen["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 1, 1]]


def test_existing_cu_seq_lens_q_is_left_alone():
    model = _Capture(linear_attn=True)
    install_packed_gdn_boundaries(model)
    supplied = torch.tensor([0, 2, 4], dtype=torch.int32)
    model(position_ids=torch.tensor([[0, 1, 0, 1]]), cu_seq_lens_q=supplied)
    assert model.seen["cu_seq_lens_q"] is supplied
    assert "seq_idx" not in model.seen


def test_position_ids_rank_is_rejected():
    model = _Capture(linear_attn=True)
    install_packed_gdn_boundaries(model)
    try:
        model(position_ids=torch.tensor([[0, 1], [0, 1]]))
    except RuntimeError as exc:
        assert "position_ids of shape [1, seqlen]" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_run_sft_pipeline_installs_hook_for_packed_gdn():
    module = _Capture(linear_attn=True)
    run_sft_pipeline(
        _Engine(module),
        _packed_batch(),
        meta={"sample_packing": True},
        processing={"loss_fn": "sft"},
        device="cpu",
    )
    assert module.seen["cu_seq_lens_q"].tolist() == [0, 3, 7]
    assert module.seen["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 1, 1]]


def test_run_sft_pipeline_skips_hook_without_linear_attn():
    module = _Capture(linear_attn=False)
    run_sft_pipeline(
        _Engine(module),
        _packed_batch(),
        meta={"sample_packing": True},
        processing={"loss_fn": "sft"},
        device="cpu",
    )
    assert "cu_seq_lens_q" not in module.seen
    assert "position_ids" in module.seen
