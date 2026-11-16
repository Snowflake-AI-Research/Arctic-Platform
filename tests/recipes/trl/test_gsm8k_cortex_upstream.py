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

"""CPU-only contracts for the recipe-local upstream-TRL Cortex bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from arctic_platform.common.utils.batch import unpack_batch

RECIPE_ROOT = Path(__file__).resolve().parents[3] / "recipes" / "rl" / "trl" / "gsm8k_cortex"
sys.path.insert(0, str(RECIPE_ROOT))

from arctic_trl_cortex.client import engine_old_log_probs  # noqa: E402
from arctic_trl_cortex.upstream import CortexModelBridge  # noqa: E402
from arctic_trl_cortex.upstream import _flatten_row  # noqa: E402
from arctic_trl_cortex.upstream import _pad_rows  # noqa: E402


class _Row:
    samples = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "completion_mask": torch.tensor([0, 1, 1]),
            "old_log_probs": torch.tensor([0.0, -0.1, -0.2]),
            "advantage": 0.5,
        }
    ]


class _FakeCortexClient:
    def __init__(self) -> None:
        self.forward_payloads: list[dict] = []
        self.backward_payloads: list[dict] = []

    def fwd_no_grad(self, payload: dict) -> dict:
        self.forward_payloads.append(payload)
        input_ids = payload["kwargs"]["input_ids"]
        return {
            "batch": {
                "logprobs": input_ids.float() / 100,
                "entropy": torch.ones_like(input_ids, dtype=torch.float32),
            }
        }

    def fwd_bwd(self, payload: dict) -> dict:
        self.backward_payloads.append(payload)
        return {"batch": {}}


def test_flattened_rows_select_completion_predictors() -> None:
    batch = _pad_rows([_flatten_row(_Row(), pad_token_id=0)], num_processes=1, pad_token_id=0)

    assert batch["token_index"].tolist() == [[0, 1]]
    assert batch["shift_labels"].tolist() == [[2, 3, -100]]
    assert batch["sampling_mask_counts"].tolist() == [[0, 0, 0]]


def test_cortex_backward_runs_once_for_upstream_autograd() -> None:
    batch = _pad_rows([_flatten_row(_Row(), pad_token_id=0)], num_processes=1, pad_token_id=0)
    client = _FakeCortexClient()
    bridge = CortexModelBridge(client, temperature=1.0, pad_token_id=0, rollout_n=8, max_token_len_per_gpu=16)
    bridge.set_anchor(torch.nn.Parameter(torch.ones(())))
    bridge.stage(batch, epsilon=0.2)

    output = bridge.forward(
        batch["input_ids"],
        batch["position_ids"],
        batch["shift_labels"],
        fused_lm_head=True,
        tree_kv_index=batch["tree_kv_index"],
    )
    assert output.log_probs.shape == (1, 3)
    output.log_probs[:, :2].sum().backward()

    assert len(client.forward_payloads) == 1
    assert len(client.backward_payloads) == 1
    for payload in (*client.forward_payloads, *client.backward_payloads):
        _args, server_batch, server_meta, _processing = unpack_batch(payload)
        assert "attention_mask" in server_batch
        assert server_meta["pad_token_id"] == 0


def test_trainer_old_logprob_forward_carries_unpadding_metadata() -> None:
    client = _FakeCortexClient()

    engine_old_log_probs(
        client,
        [[1, 2, 3], [4, 5]],
        temperature=0.7,
        pad_token_id=42,
        rollout_n=8,
        max_token_len_per_gpu=16,
    )

    (payload,) = client.forward_payloads
    _args, server_batch, server_meta, _processing = unpack_batch(payload)
    assert server_batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert server_meta["pad_token_id"] == 42
