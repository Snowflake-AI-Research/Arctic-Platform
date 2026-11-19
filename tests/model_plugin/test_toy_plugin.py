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

import copy
import dataclasses
import json
import shutil
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

from arctic_platform.model_plugin import cast_model
from arctic_platform.model_plugin import conformance
from arctic_platform.model_plugin import gather_sequence
from arctic_platform.model_plugin.host import forward
from arctic_platform.model_plugin.host import next_token_labels
from arctic_platform.model_plugin.host import pack_sequences
from arctic_platform.model_plugin.host import token_logprobs
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
        conformance.check_logprob_chunking,
        conformance.check_activation_checkpointing,
        conformance.check_meta_build,
        conformance.check_dtype_marks,
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
        ("logprob_chunking", "PASS"),
        ("activation_checkpointing", "PASS"),
        ("meta_build", "PASS"),
        ("dtype_marks", "PASS"),
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


def test_logprobs_go_through_the_models_softcapping_compute_logits():
    config, model = conformance.build_fixture(PLUGIN, SPEC)
    assert config.final_logit_softcapping == 2.0
    input_ids, position_ids = pack_sequences(conformance.fixture_sequences(config, SPEC))
    labels = next_token_labels(input_ids, position_ids)
    with torch.no_grad():
        hidden = forward(model, input_ids, position_ids)
        capped = 2.0 * torch.tanh(model.lm_head(hidden) / 2.0)
        expected = torch.log_softmax(capped, -1).gather(-1, labels.clamp(min=0)[..., None])[..., 0] * (labels >= 0)
        for chunk in (3, 1000):
            torch.testing.assert_close(
                token_logprobs(model, hidden, labels, chunk_size=chunk), expected, rtol=0, atol=1e-6
            )


def test_fused_kernel_option_matches_the_reference():
    spec = SPEC.model_copy(update=dict(options=dict(short_conv_kernel="fused")))
    for check in (conformance.check_reference_parity, conformance.check_packing):
        check(PLUGIN, spec)


def test_buffers_not_reinitialized_fail_meta_build():
    from .broken_toy import NO_BUFFER_INIT

    with pytest.raises(AssertionError, match="does its module implement ap_init_buffers"):
        conformance.check_meta_build(NO_BUFFER_INIT, SPEC)


def _gather_worker(rank, init_file):
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", world_size=2, rank=rank)
    try:
        torch.manual_seed(0)
        full = torch.randn(1, 6, 3, dtype=torch.bfloat16)
        weight = torch.randn(1, 6, 3)
        mine = slice(rank * 3, (rank + 1) * 3)
        # A core that needs the whole sequence (a cumulative sum); each rank keeps only its own outputs.
        local = full[:, mine].clone().requires_grad_()
        whole = gather_sequence(torch.distributed.group.WORLD, local).float().cumsum(dim=1)
        (whole[:, mine] * weight[:, mine]).sum().backward()
        reference = full.float().requires_grad_()
        (reference.cumsum(dim=1) * weight).sum().backward()
        # Summed in fp32 across ranks and rounded to bf16 once: within one bf16 rounding of the fp32 gradient.
        assert local.grad.dtype == torch.bfloat16
        torch.testing.assert_close(local.grad.float(), reference.grad[:, mine], rtol=2**-8, atol=0)
    finally:
        torch.distributed.destroy_process_group()


def test_gather_sequence_gives_each_rank_its_slice_of_the_whole_sequence_gradient(tmp_path):
    torch.multiprocessing.spawn(_gather_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2)


def test_choose_experts_replays_ids_and_routes_fresh_ones_itself():
    from arctic_platform.model_plugin import ROUTER_REPLAY_FRESH
    from arctic_platform.model_plugin import choose_experts

    scores = torch.tensor([[0.1, 0.5, 0.4], [0.7, 0.2, 0.1]])
    assert choose_experts(scores, 1).tolist() == [[1], [0]]
    assert choose_experts(scores, 1, torch.tensor([[2], [ROUTER_REPLAY_FRESH]])).tolist() == [[2], [0]]
    assert choose_experts(scores, 1, bias=torch.tensor([0.0, 0.0, 1.0])).tolist() == [[2], [2]]


class _JitteryRecomputeMoE(modeling.ToyMoE):
    """Picks other experts when recomputed, as last-bit noise from nondeterministic kernels can make it do."""

    def route(self, tokens, routed=None):
        scores = torch.softmax(self.router(tokens.float()), dim=-1)
        flip = torch.linspace(1.0, 0.0, self.num_experts) if modeling.in_recompute() else None
        selected = modeling.choose_experts(scores, self.top_k, routed, bias=flip)
        weights = scores.gather(1, selected)
        return selected, weights / weights.sum(-1, keepdim=True) * self.scale


def _jittery_plugin():
    def build(config, options):
        model = modeling.ToyHybridMoEForCausalLM(config, options)
        for module in model.modules():
            if type(module) is modeling.ToyMoE:
                module.__class__ = _JitteryRecomputeMoE
        return model

    return dataclasses.replace(PLUGIN, build_model=build)


def test_recompute_replays_its_forwards_routing_even_if_the_router_would_choose_otherwise():
    plugin = _jittery_plugin()
    conformance.check_activation_checkpointing(plugin, SPEC)  # the platform's routing is installed: passes

    # The same model without platform routing recomputes with other experts: other per-expert token counts, so
    # the recompute saves other tensors than its forward.
    torch.manual_seed(SPEC.seed)
    config = ToyHybridMoEConfig(**SPEC.fixture)
    model = plugin.build_model(config, None)
    conformance.apply_activation_checkpointing(model)
    input_ids, position_ids = pack_sequences(conformance.fixture_sequences(config, SPEC))
    loss = conformance._token_loss(
        model, forward(model, input_ids, position_ids), next_token_labels(input_ids, position_ids), SPEC
    )
    with pytest.raises(torch.utils.checkpoint.CheckpointError, match="different number of tensors"):
        loss.backward()


def test_pending_forwards_recompute_with_their_own_routing():
    config, checkpointed = conformance.build_fixture(PLUGIN, SPEC)
    plain = copy.deepcopy(checkpointed)
    conformance.apply_activation_checkpointing(checkpointed)
    input_ids, position_ids = pack_sequences(conformance.fixture_sequences(config, SPEC))
    labels = next_token_labels(input_ids, position_ids)
    with torch.no_grad():
        _, routing = forward(plain, input_ids, position_ids, return_routing=True)
    shifted = {layer: (ids + 1) % config.num_experts for layer, ids in routing.items()}
    # Two forwards with different routing, then one backward, as with pending forwards or gradient accumulation.
    for model in (plain, checkpointed):
        losses = [
            conformance._token_loss(
                model, forward(model, input_ids, position_ids, routed_experts=replay), labels, SPEC
            )
            for replay in (routing, shifted)
        ]
        # Back-propagate the first forward first: a per-layer stack instead of per-frame captures would replay the
        # second forward's routing into it.
        losses[0].backward()
        losses[1].backward()
    expected = conformance._grads(plain)
    for name, grad in conformance._grads(checkpointed).items():
        torch.testing.assert_close(grad, expected[name], rtol=0, atol=1e-6)


def test_moe_without_a_route_method_fails_binding():
    class NoRoute(modeling.ToyMoE):
        route = None

    config = ToyHybridMoEConfig(**SPEC.fixture)
    model = modeling.ToyHybridMoEForCausalLM(config)
    model.model.layers[1].mlp.__class__ = NoRoute
    with pytest.raises(ValueError, match=r"NoRoute has no route\(tokens, routed=None\); MoE modules must route"):
        conformance.prepare_model(model, PLUGIN, config)

    class NoRoutedParameter(modeling.ToyMoE):
        def route(self, tokens):
            return super().route(tokens)

    model.model.layers[1].mlp.__class__ = NoRoutedParameter
    with pytest.raises(ValueError, match="NoRoutedParameter has no route"):
        conformance.prepare_model(model, PLUGIN, config)


def test_preparing_twice_wraps_routing_once():
    config, model = conformance.build_fixture(PLUGIN, SPEC)  # prepared once already
    conformance.prepare_model(model, PLUGIN, config)
    input_ids, position_ids = pack_sequences(conformance.fixture_sequences(config, SPEC))
    with torch.no_grad():
        _, routing = forward(model, input_ids, position_ids, return_routing=True)  # a double wrap would raise
    assert sorted(routing) == [1, 2]


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
    real_route = modeling.ToyMoE.route
    with patch.object(modeling.ToyMoE, "route", lambda self, tokens, routed=None: real_route(self, tokens)):
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

    def test_passthrough_tensors_are_kept_out_of_the_model_and_copied_into_every_export(self):
        from safetensors.torch import load_file
        from safetensors.torch import save_file

        from arctic_platform.model_plugin.manifest import CheckpointSpec

        torch.manual_seed(0)
        config = ToyHybridMoEConfig(**SPEC.fixture)
        source = Path(self.get_auto_remove_tmp_dir_str())
        save_hf_checkpoint(
            cast_model(PLUGIN.build_model(config), torch.bfloat16), PLUGIN, config, source, max_shard_bytes=8192
        )
        _save_tokenizer(source, config.vocab_size)
        # An MTP head and a vision tower the trainer does not use; the tower is fp32 and outweighs the bf16 model.
        extra = {
            "mtp.0.proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "model.visual.patch.weight": torch.randn(1000, 1000),
        }
        save_file(extra, str(source / "extra.safetensors"))
        index = json.loads((source / "model.safetensors.index.json").read_text())
        index["weight_map"].update(dict.fromkeys(extra, "extra.safetensors"))
        (source / "model.safetensors.index.json").write_text(json.dumps(index))

        with self.assertRaisesRegex(ValueError, "match no trainer tensor"):
            load_hf_checkpoint(PLUGIN, source)
        plugin = dataclasses.replace(
            PLUGIN,
            manifest=PLUGIN.manifest.model_copy(
                update=dict(checkpoint=CheckpointSpec(passthrough=[r"mtp\.", r"model\.visual\."]))
            ),
        )
        _, model = load_hf_checkpoint(plugin, source)
        self.assertEqual(model.lm_head.weight.dtype, torch.bfloat16)  # the pass-through tower did not set the dtype
        self.assertFalse(set(extra) & set(model.state_dict()))
        self.assertFalse(set(extra) & {name for name, _ in iter_hf_weights(model, plugin, for_sampler=True)})

        exported = Path(self.get_auto_remove_tmp_dir_str())
        save_hf_checkpoint(model, plugin, config, exported, max_shard_bytes=8192)
        tensors = {}
        for file in exported.glob("*.safetensors"):
            tensors.update(load_file(str(file)))
        for name, tensor in extra.items():
            self.assertEqual(tensors[name].dtype, tensor.dtype)
            self.assertTrue(torch.equal(tensors[name], tensor))
        PLUGIN.register()
        conformance.check_weights_directory(plugin, source)
        # Later exports copy from the latest export: the source may be rotated away, or overwritten in place.
        shutil.rmtree(source)
        save_hf_checkpoint(model, plugin, config, exported)
        copied = load_file(str(exported / CHECKPOINT_FILE))["model.visual.patch.weight"]
        self.assertTrue(torch.equal(copied, extra["model.visual.patch.weight"]))

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
        model = cast_model(PLUGIN.build_model(config), torch.bfloat16)
        path = Path(self.get_auto_remove_tmp_dir_str())
        save_hf_checkpoint(model, PLUGIN, config, path)
        _, reloaded = load_hf_checkpoint(PLUGIN, path)
        dtypes = {name: p.dtype for name, p in reloaded.named_parameters()}
        self.assertEqual(dtypes, {name: p.dtype for name, p in model.named_parameters()})
        self.assertEqual(dtypes["model.layers.1.mlp.router.weight"], torch.float32)  # marked fp32
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
