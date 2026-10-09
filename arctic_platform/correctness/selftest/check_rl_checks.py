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

"""The two RL checks' verdicts, judged from fabricated zone responses, and the requests they build."""

from __future__ import annotations

import copy
import json

import pytest

from arctic_platform.correctness.checks import rl_router_replay
from arctic_platform.correctness.checks import rl_weight_sync
from arctic_platform.correctness.harness import router_replay_trace
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.rl_driver import rl_batch
from arctic_platform.correctness.harness.rl_driver import rl_gpu_count
from arctic_platform.correctness.harness.rl_driver import rl_sampling_payload
from arctic_platform.correctness.selftest.config_factory import native_config

from .check_rl_config import SAMPLING_CONFIG
from .check_rl_config import TRAINING_CONFIG


def _entry(numel, digest):
    return {"shape": [numel], "dtype": "bfloat16", "numel": numel, "sha256": digest}


def _sync(ranks=2):
    sent = {"a.weight": _entry(4, "aa"), "b.weight": _entry(6, "bb")}
    record = {
        "status": "ok",
        "received": {name: {**entry, "destinations": [name]} for name, entry in sent.items()},
        "destinations": {
            "a.weight": {"numel": 2, "uncovered": 0, "mismatched": 0, "equal": True},
            "b.weight": {"numel": 3, "uncovered": 0, "mismatched": 0, "equal": True},
        },
        "untouched_parameters": {},
        "non_parameter_destinations": [],
        "unverifiable_destinations": [],
    }
    return {
        "weight_format": "vllm",
        "targets": [
            {
                "send": {
                    "results": [
                        {"sent_manifest": sent, "trainer_parameters": {"count": 2, "numel": 10}},
                        {"sent_manifest": None},
                    ]
                },
                "recv": {
                    "recv": {
                        "workers": [
                            {"all_rank_weight_sync_verification": [copy.deepcopy(record) for _ in range(ranks)]}
                        ]
                    }
                },
            }
        ],
    }


def _rank(sync, index):
    return sync["targets"][0]["recv"]["recv"]["workers"][0]["all_rank_weight_sync_verification"][index]


def test_weight_sync_agreement_passes():
    failed, metrics = rl_weight_sync.judge(_sync())
    assert failed == []
    assert metrics["sent_elements"] == metrics["trainer_elements"] == 10
    assert metrics["sampler_ranks"] == 2


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda s: s["targets"][0]["send"]["results"][0]["trainer_parameters"].update(numel=12),
            "trainer elements sent",
        ),
        (lambda s: _rank(s, 1)["received"].pop("b.weight"), "sampler rank 1: b.weight not received"),
        (lambda s: _rank(s, 0)["received"]["a.weight"].update(sha256="xx"), "different bytes"),
        (lambda s: _rank(s, 0)["received"]["a.weight"].update(destinations=[]), "written to no parameter"),
        (lambda s: _rank(s, 0)["destinations"]["a.weight"].update(uncovered=1), "no tensor wrote"),
        (lambda s: _rank(s, 0)["destinations"]["b.weight"].update(mismatched=2), "differing from what was written"),
        (lambda s: _rank(s, 0)["untouched_parameters"].update({"visual.w": 5}), "written by no tensor"),
    ],
)
def test_weight_sync_disagreement_fails(mutate, expected):
    sync = _sync()
    mutate(sync)
    failed, _ = rl_weight_sync.judge(sync)
    assert any(expected in m.name for m in failed), [m.name for m in failed]


def test_weight_sync_without_sampler_verification_raises():
    sync = _sync()
    sync["targets"][0]["recv"] = {"recv": {"workers": [{}]}}
    with pytest.raises(RuntimeError, match=rl_weight_sync.VERIFY_ENV):
        rl_weight_sync.judge(sync)


# Two samples, two MoE layers, top-1. Sample 0 is prompt [5, 6] + answer [7]; sample 1 is prompt [8] + [9, 4].
# capture_len is one less than each row, so the last token of each row carries no routing.
def _samples():
    rows = [([5, 6, 7], 2, [[[1], [2]], [[3], [4]]]), ([8, 9, 4], 1, [[[5], [6]], [[7], [0]]])]
    samples = []
    for index, (tokens, prompt_len, routing) in enumerate(rows):
        marker = {
            "replay_id": f"rr1:{index}",
            "prompt_len": prompt_len,
            "generation_len": len(tokens) - prompt_len,
            "capture_len": len(routing),
            "routed_experts": routing,
        }
        samples.append({"tokens": tokens, "prompt_len": prompt_len, "marker": marker, "routing": routing})
    return samples


def _trace(samples):
    tokens = samples[0]["tokens"] + samples[1]["tokens"]
    positions = [0, 1, 2, 0, 1, 2]
    routed = copy.deepcopy(samples[0]["routing"] + [[[0], [0]]] + samples[1]["routing"] + [[[0], [0]]])
    router = [
        {"layer": layer, "phase": phase, "replayed": True, "indices": [row[layer] for row in routed]}
        for phase in ("forward", "recompute")
        for layer in (0, 1)
    ]
    return {
        "rank": 0,
        "received": {s["marker"]["replay_id"]: s["routing"] for s in samples},
        "calls": [
            {
                "padding": False,
                "input_ids": tokens,
                "position_ids": positions,
                "routed_experts": routed,
                "router": router,
            }
        ],
        "unattributed_router_calls": [],
    }


def test_router_replay_agreement_passes():
    samples = _samples()
    failed, metrics = rl_router_replay.judge(samples, [_trace(samples)], full_recompute=True)
    assert failed == []
    assert metrics["compared_positions"] == 4
    assert metrics["forward_invocations"] == metrics["recompute_invocations"] == 2


def test_router_trace_hooks_external_router_implementations():
    import torch

    class TokenChoiceTopKRouter(torch.nn.Module):
        def forward(self, hidden, *, routed_experts=None):
            indices = routed_experts if routed_experts is not None else torch.zeros_like(hidden, dtype=torch.long)
            return hidden, indices, torch.ones(1)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = TokenChoiceTopKRouter()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    model = Model()
    trace = router_replay_trace.start(model)
    trace.begin_call(
        {
            "input_ids": torch.tensor([[1]]),
            "position_ids": torch.tensor([[0]]),
            "routed_experts": torch.tensor([[[0]]]),
        },
        padding=False,
    )
    model.layers[0].router(torch.zeros(1, 1), routed_experts=torch.tensor([[3]]))
    exported = router_replay_trace.finish(0)
    assert exported["calls"][0]["router"] == [{"layer": 0, "phase": "forward", "replayed": True, "indices": [[3]]}]

    model.layers[0].router(torch.zeros(1, 1), routed_experts=torch.tensor([[2]]))
    assert len(exported["calls"][0]["router"]) == 1


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda t: t["received"].update({"rr1:1": [[[5], [6]], [[7], [1]]]}), "routing received differs"),
        (lambda t: t["calls"][0]["routed_experts"][1].__setitem__(0, [9]), "routed_experts differ"),
        (lambda t: t["calls"][0]["router"][3].update(indices=[[9]] * 6), "layer 1 recompute tokens"),
        (lambda t: t["calls"][0]["router"][0].update(replayed=False), "computed its own routing"),
        (lambda t: t["calls"][0]["router"].pop(2), "layer 0 has no recompute"),
        (lambda t: t["unattributed_router_calls"].append({"layer": 0}), "outside a traced call"),
        (lambda t: t["received"].pop("rr1:0"), "sample 0: no training rank received"),
    ],
)
def test_router_replay_disagreement_fails(mutate, expected):
    samples = _samples()
    trace = _trace(samples)
    mutate(trace)
    failed, _ = rl_router_replay.judge(samples, [trace], full_recompute=True)
    assert any(expected in m.name for m in failed), [m.name for m in failed]


def test_router_replay_recompute_not_required_without_full_checkpointing():
    samples = _samples()
    trace = _trace(samples)
    trace["calls"][0]["router"] = [r for r in trace["calls"][0]["router"] if r["phase"] == "forward"]
    failed, _ = rl_router_replay.judge(samples, [trace], full_recompute=False)
    assert failed == []
    assert rl_router_replay.expects_full_recompute({"ac_config": {"mode": "full", "freq": 1}})
    assert not rl_router_replay.expects_full_recompute({})


def test_rl_batch_scores_only_answer_predictions():
    batch = rl_batch([[5, 6, 7, 8], [9, 4]], [2, 1], pad_id=0)
    assert batch["kwargs"]["input_ids"].tolist() == [[5, 6, 7, 8], [9, 4, 0, 0]]
    assert batch["kwargs"]["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch["context"]["loss_mask"].tolist() == [[0, 1, 1, 0], [1, 0, 0, 0]]


def test_sampling_payload_keeps_the_config_and_swaps_only_the_model(tmp_path):
    path = tmp_path / "job.config"
    path.write_text(json.dumps(native_config(TRAINING_CONFIG, sampling=SAMPLING_CONFIG)))
    cfg = load_config(path)
    payload = rl_sampling_payload(cfg, "/models/reduced")
    assert payload == {
        "job_type": "sampling",
        "model_name": "/models/reduced",
        "dtype": "bfloat16",
        "seed": 42,
        "inference_config": {"n_gpus": 8, "max_seq_len": 4096, "vllm_config": {}},
    }
    assert rl_gpu_count(cfg) == 16
