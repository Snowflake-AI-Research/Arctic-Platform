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

"""The example ToyHybridMoE plugin against every CPU conformance tier, and broken variants each tier must reject.

Single-process tiers run in the test process. Sequence and expert parallelism spawn two gloo ranks on CPU and
compare logits and every gradient with the unsharded model.
"""

import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file
from toy_hybrid_moe import PLUGIN
from toy_hybrid_moe import modeling
from toy_hybrid_moe.config import ToyHybridMoEConfig
from toy_hybrid_moe.sampler_names import sampler_target

from arctic_platform.model_plugin import conformance
from arctic_platform.model_plugin.weights import CHECKPOINT_FILE
from arctic_platform.model_plugin.weights import iter_hf_weights
from arctic_platform.model_plugin.weights import save_hf_checkpoint
from arctic_platform.testing_utils import TestCasePlus

from .conftest import TOY_PLUGIN_ROOT

TARGET = "toy_hybrid_moe:PLUGIN"
SPEC = conformance.load_conformance_spec(TOY_PLUGIN_ROOT / "conformance.toml")


@pytest.mark.parametrize(
    "check",
    [
        conformance.check_static,
        conformance.check_reference_parity,
        conformance.check_packing,
        conformance.check_router_replay,
        conformance.check_checkpoint_and_sync,
        conformance.check_rl_step,
    ],
    ids=lambda check: check.__name__,
)
def test_single_process_tier(check):
    check(PLUGIN, SPEC)


@pytest.mark.parametrize("mode", ["sp", "ep", "sp+ep"])
def test_parallel_tier(mode):
    conformance.check_parallel(TARGET, SPEC, mode, 2)


def test_run_conformance_reports_every_tier():
    results = conformance.run_conformance(TARGET, SPEC.model_copy(update=dict(sp_sizes=[], ep_sizes=[])))
    assert [(result.tier, result.passed) for result in results] == [
        ("static", True),
        ("reference_parity", True),
        ("packing", True),
        ("checkpoint_and_sync", True),
        ("rl_step", True),
        ("router_replay", True),
    ]


def test_new_layer_count_and_pattern_needs_no_code_change():
    spec = SPEC.model_copy(
        update=dict(
            fixture=dict(
                SPEC.fixture,
                layer_types=[
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                ],
                mlp_types=["moe", "dense", "moe", "moe", "dense"],
                sliding_window=4,
            )
        )
    )
    for check in (conformance.check_static, conformance.check_reference_parity, conformance.check_packing):
        check(PLUGIN, spec)


def test_recurrence_that_ignores_row_boundaries_fails_packing():
    real_mix = modeling.ToyRecurrence.mix

    def one_row(self, u, g, *, cu_seqlens, shard=modeling.FULL_SHARD):
        return real_mix(self, u, g, cu_seqlens=torch.tensor([0, u.shape[1]], dtype=cu_seqlens.dtype), shard=shard)

    with patch.object(modeling.ToyRecurrence, "mix", one_row):
        with pytest.raises(AssertionError, match="packed vs alone"):
            conformance.check_packing(PLUGIN, SPEC)


def test_router_that_ignores_replay_fails_router_replay():
    real_forward = modeling.ToyMoE.forward
    with patch.object(modeling.ToyMoE, "forward", lambda self, x, routed=None: real_forward(self, x, None)):
        with pytest.raises(AssertionError, match="routed_experts is ignored"):
            conformance.check_router_replay(PLUGIN, SPEC)


def test_lossy_export_fails_checkpoint():
    real = PLUGIN.convert_layer_to_hf

    def lossy(state, index, config):
        return {name: tensor.to(torch.bfloat16).float() for name, tensor in real(state, index, config).items()}

    with pytest.raises(AssertionError, match="reloaded checkpoint"):
        conformance.check_checkpoint_and_sync(dataclasses.replace(PLUGIN, convert_layer_to_hf=lossy), SPEC)


@pytest.mark.parametrize(
    "target, mode",
    [
        ("tests.model_plugin.broken_toy:WRONG_HEAD_SLICE", "sp"),
        ("tests.model_plugin.broken_toy:NO_EXPERT_OFFSET", "ep"),
    ],
)
def test_parallelism_bug_fails_parallel_tier(target, mode):
    with pytest.raises(Exception, match=f"{mode} (logits|grad)"):
        conformance.check_parallel(target, SPEC, mode, 2)


def test_sampler_stream_maps_onto_the_vllm_model():
    torch.manual_seed(0)
    config = ToyHybridMoEConfig(**SPEC.fixture)
    model = PLUGIN.build_model(config)
    stream = dict(iter_hf_weights(model, PLUGIN, for_sampler=True))
    assert not any(".mlp.stats." in name for name in stream)

    shards: dict[str, set] = defaultdict(set)
    for name in stream:
        target = sampler_target(name)
        shards[target.param].add((target.shard, target.expert))
    experts = range(config.num_experts)
    assert shards["model.layers.0.recurrence.in_gate_proj.weight"] == {(0, None), (1, None)}
    assert shards["model.layers.1.self_attn.qkv_proj.weight"] == {("q", None), ("k", None), ("v", None)}
    assert shards["model.layers.1.mlp.experts.w13_weight"] == {(s, e) for s in ("w1", "w3") for e in experts}
    assert shards["model.layers.1.mlp.experts.w2_weight"] == {("w2", e) for e in experts}
    assert shards["model.layers.0.mlp.gate_up_proj.weight"] == {(0, None), (1, None)}
    with pytest.raises(ValueError, match="trainer-only"):
        sampler_target("model.layers.1.mlp.stats.tokens_per_expert")


class TestCheckpointLayout(TestCasePlus):
    def test_checkpoint_uses_hugging_face_names(self):
        torch.manual_seed(0)
        config = ToyHybridMoEConfig(**SPEC.fixture)
        model = PLUGIN.build_model(config)
        path = Path(self.get_auto_remove_tmp_dir_str())
        save_hf_checkpoint(model, PLUGIN, config, path)

        names = set(load_file(str(path / CHECKPOINT_FILE)))
        self.assertIn("model.layers.1.mlp.experts.3.gate_proj.weight", names)
        self.assertIn("model.layers.1.mlp.gate.weight", names)
        self.assertIn("model.layers.0.recurrence.conv1d.weight", names)
        self.assertIn("model.layers.1.mlp.stats.tokens_per_expert", names)
        self.assertNotIn("model.layers.1.mlp.experts.w1", names)
        saved = json.loads((path / "config.json").read_text())
        self.assertEqual(saved["model_type"], "toy_hybrid_moe")
        self.assertEqual(saved["layer_types"], SPEC.fixture["layer_types"])
