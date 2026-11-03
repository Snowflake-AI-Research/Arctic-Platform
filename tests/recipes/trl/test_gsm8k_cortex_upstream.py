"""CPU-only contracts for the recipe-local upstream-TRL Cortex bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


RECIPE_ROOT = Path(__file__).resolve().parents[3] / "recipes" / "rl" / "trl" / "gsm8k_cortex"
sys.path.insert(0, str(RECIPE_ROOT))

from arctic_trl_cortex.upstream import CortexModelBridge
from arctic_trl_cortex.upstream import _flatten_row
from arctic_trl_cortex.upstream import _pad_rows


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
        self.backward_payloads: list[dict] = []

    def fwd_no_grad(self, payload: dict) -> dict:
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
    bridge = CortexModelBridge(
        client, temperature=1.0, pad_token_id=0, rollout_n=8, max_token_len_per_gpu=16
    )
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

    assert len(client.backward_payloads) == 1
