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
from arctic_platform.model_plugin.host import forward
from arctic_platform.model_plugin.host import pack_sequences
from arctic_platform.model_plugin.weights import CHECKPOINT_FILE
from arctic_platform.model_plugin.weights import iter_hf_weights
from arctic_platform.model_plugin.weights import load_hf_checkpoint
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
        conformance.check_activation_checkpointing,
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
    assert [(result.tier, result.status) for result in results] == [
        ("static", "PASS"),
        ("reference_parity", "PASS"),
        ("packing", "PASS"),
        ("activation_checkpointing", "PASS"),
        ("checkpoint_and_sync", "PASS"),
        ("rl_step", "PASS"),
        ("router_replay", "PASS"),
    ]


def test_rl_step_is_skipped_without_the_training_stack():
    # Importing the platform's GRPO loss needs arctic-platform[rl]; a plugin author's laptop may not have it.
    with patch.dict("sys.modules", {"arctic_platform.rl.processors.grpo": None}):
        results = conformance.run_conformance(TARGET, SPEC.model_copy(update=dict(sp_sizes=[], ep_sizes=[])))
    statuses = {result.tier: result.status for result in results}
    assert statuses["rl_step"] == "SKIP"
    assert all(status == "PASS" for tier, status in statuses.items() if tier != "rl_step")
    assert conformance.main(["--plugin", TARGET, "--spec", str(TOY_PLUGIN_ROOT / "conformance.toml")]) == 0


def test_new_layer_count_and_pattern_needs_no_code_change():
    spec = SPEC.model_copy(
        update=dict(
            fixture=dict(
                SPEC.fixture,
                layer_types=[
                    "full_attention",
                    "conv",
                    "linear_attention",
                    "conv",
                    "full_attention",
                    "linear_attention",
                ],
                mlp_types=["moe", "dense", "moe", "moe", "dense", "moe"],
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


def test_convolution_that_ignores_row_boundaries_fails_packing():
    real_mix = modeling.ToyShortConv.mix

    def one_row(self, u, *, cu_seqlens, shard=modeling.FULL_WINDOW):
        return real_mix(self, u, cu_seqlens=torch.tensor([0, 10**6], dtype=cu_seqlens.dtype), shard=shard)

    with patch.object(modeling.ToyShortConv, "mix", one_row):
        with pytest.raises(AssertionError, match="packed vs alone"):
            conformance.check_packing(PLUGIN, SPEC)


def test_router_that_ignores_replay_fails_router_replay():
    real_forward = modeling.ToyMoE.forward
    with patch.object(modeling.ToyMoE, "forward", lambda self, x, routed=None: real_forward(self, x, None)):
        with pytest.raises(AssertionError, match="routed_experts is ignored"):
            conformance.check_router_replay(PLUGIN, SPEC)


def test_side_effect_repeated_on_recompute_fails_activation_checkpointing():
    with patch.object(modeling, "in_recompute", lambda: False):
        with pytest.raises(AssertionError, match="tokens_per_expert after recompute"):
            conformance.check_activation_checkpointing(PLUGIN, SPEC)


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
        ("tests.model_plugin.broken_toy:NO_HALO", "sp"),
        ("tests.model_plugin.broken_toy:WRONG_EXPERT_SLICE", "ep"),
    ],
    ids=["head-parallel-slice", "halo-missing", "expert-slice"],
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
    assert shards["model.layers.2.conv.in_gate_proj.weight"] == {(0, None), (1, None)}
    assert shards["model.layers.1.self_attn.qkv_proj.weight"] == {("q", None), ("k", None), ("v", None)}
    assert shards["model.layers.1.mlp.experts.routed_experts.w13_weight"] == {
        (s, e) for s in ("w1", "w3") for e in experts
    }
    assert shards["model.layers.1.mlp.experts.routed_experts.w2_weight"] == {("w2", e) for e in experts}
    assert shards["model.layers.0.mlp.gate_up_proj.weight"] == {(0, None), (1, None)}
    with pytest.raises(ValueError, match="trainer-only"):
        sampler_target("model.layers.1.mlp.stats.tokens_per_expert")


def _save_tokenizer(path: Path, vocab_size: int) -> None:
    from tokenizers import Tokenizer
    from tokenizers import models
    from tokenizers import pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(vocab_size - 2)} | {"<unk>": vocab_size - 2, "</s>": vocab_size - 1}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="</s>").save_pretrained(path)


class TestCheckpointLayout(TestCasePlus):
    def _weights_directory(self, **config_overrides) -> Path:
        torch.manual_seed(0)
        config = ToyHybridMoEConfig(**(SPEC.fixture | config_overrides))
        path = Path(self.get_auto_remove_tmp_dir_str())
        save_hf_checkpoint(PLUGIN.build_model(config), PLUGIN, config, path, max_shard_bytes=8192)
        _save_tokenizer(path, config.vocab_size)
        return path

    def test_weights_directory_check_passes_on_a_complete_directory(self):
        PLUGIN.register()
        conformance.check_weights_directory(PLUGIN, self._weights_directory())

    def test_weights_directory_check_rejects(self):
        PLUGIN.register()
        path = self._weights_directory()
        (path / "tokenizer.json").unlink()
        with self.assertRaisesRegex(AssertionError, "tokenizer does not load"):
            conformance.check_weights_directory(PLUGIN, path)

        path = self._weights_directory()
        config = json.loads((path / "config.json").read_text())
        (path / "config.json").write_text(json.dumps(config | {"num_experts": 8}))
        with self.assertRaisesRegex(AssertionError, "tensor problems"):
            conformance.check_weights_directory(PLUGIN, path)

        path = self._weights_directory()
        (path / "config.json").write_text(json.dumps(config | {"model_type": "qwen3"}))
        with self.assertRaisesRegex(AssertionError, "model_type 'qwen3'"):
            conformance.check_weights_directory(PLUGIN, path)

    def test_checkpoint_from_another_plugin_is_refused(self):
        path = self._weights_directory()
        other = dataclasses.replace(
            PLUGIN, manifest=PLUGIN.manifest.model_copy(update=dict(name="custom/other/model"))
        )
        with self.assertRaisesRegex(ValueError, "was written by plugin custom/example/toy-hybrid-moe"):
            load_hf_checkpoint(other, path)

    def test_bf16_checkpoint_reloads_in_bf16_exactly(self):
        torch.manual_seed(0)
        config = ToyHybridMoEConfig(**SPEC.fixture)
        model = PLUGIN.build_model(config).to(torch.bfloat16)
        path = Path(self.get_auto_remove_tmp_dir_str())
        save_hf_checkpoint(model, PLUGIN, config, path)
        _, reloaded = load_hf_checkpoint(PLUGIN, path)
        self.assertEqual({p.dtype for p in reloaded.parameters()}, {torch.bfloat16})
        input_ids, position_ids = pack_sequences(conformance.fixture_sequences(config, SPEC))
        with torch.no_grad():
            expected = forward(model, input_ids, position_ids)
            actual = forward(reloaded, input_ids, position_ids)
        self.assertEqual((actual.float() - expected.float()).abs().max().item(), 0.0)

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
