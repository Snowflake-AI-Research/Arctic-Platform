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

"""Inkling loader guards and native forward/backward parity."""

import json
from types import SimpleNamespace

import pytest
import torch
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingForCausalLM

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model.implementations.inkling.deepspeed_integration import get_model
from arctic_platform.model.implementations.inkling.modeling_inkling import InklingForCausalLMPrimeRL
from arctic_platform.model.implementations.inkling.modeling_inkling import InklingMoE
from arctic_platform.model.implementations.inkling.weights import load_weights
from arctic_platform.model.implementations.inkling.weights import weight_targets
from arctic_platform.model.patches.peft import apply_peft
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal


def tiny_config():
    return InklingTextConfig(
        vocab_size=64,
        unpadded_vocab_size=61,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=8,
        sliding_window_size=4,
        local_layer_ids=[0, 1],
        d_rel=2,
        rel_extent=8,
        conv_kernel_size=3,
        intermediate_size=24,
        moe_intermediate_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=2,
        mlp_layer_types=["dense", "sparse", "sparse"],
        logits_mup_width_multiplier=2.0,
        use_cache=False,
        _attn_implementation="eager",
        use_grouped_mm=False,
        tie_word_embeddings=False,
    )


class TestInklingTraining(TestCasePlus):
    def checkpoint_config(self, **text):
        from pathlib import Path

        path = Path(self.get_auto_remove_tmp_dir())
        (path / "config.json").write_text(json.dumps({"model_type": "inkling_mm_model", "text_config": text}))
        return str(path)

    def test_loader_and_small_expert_width(self):
        path = self.checkpoint_config(intermediate_size=2048, dense_intermediate_size=16384)
        spec = ModelSpec(model_path_or_name=path, parallelism=ParallelismConfig(expert_parallel=8))
        self.assertEqual(spec.loader, "inkling")
        self.assertEqual(spec.attn_implementation, "eager")
        model = get_model(SimpleNamespace(name=path, ep=8, moe_use_grouped_mm=True), device=torch.device("meta"))
        self.assertEqual(model.config.intermediate_size, 16384)
        self.assertEqual(model.config.moe_intermediate_size, 2048)

    def test_rejects_attention_that_drops_relative_bias(self):
        path = self.checkpoint_config()
        with self.assertRaisesRegex(ValueError, "relative-position bias"):
            ModelSpec(model_path_or_name=path, attn_implementation="sdpa")

    def test_rejects_sequence_parallelism(self):
        path = self.checkpoint_config()
        with self.assertRaisesRegex(ValueError, "sequence parallelism"):
            ModelSpec(model_path_or_name=path, parallelism=ParallelismConfig(sequence_parallel=2))

    def test_worker_uses_native_head_by_default(self):
        from arctic_platform.model import ds_worker_moe_model_spec

        spec = ds_worker_moe_model_spec(self.checkpoint_config(), {"attn_implementation": "eager", "ep_size": 8})
        self.assertEqual(spec.loader, "inkling")
        self.assertNotIn("fused_cross_entropy", spec.loader_options)

    @require_deepspeed
    def test_bf16_keeps_fp32_state_and_separate_optimizer_groups(self):
        from deepspeed.runtime.engine import DeepSpeedEngine

        from arctic_platform.common.deepspeed_engine import build_training_optimizer
        from arctic_platform.common.optimizer import OptimizerSettings
        from arctic_platform.model import preserve_fp8_params_through_ds_cast

        path = self.checkpoint_config(**tiny_config().to_dict())
        model = get_model(SimpleNamespace(name=path, ep=1, moe_use_grouped_mm=False), dtype=torch.bfloat16)
        expected = {}
        for name, parameter in model.named_parameters():
            if model.keeps_fp32_parameter(name):
                self.assertEqual(parameter.dtype, torch.float32)
                parameter.data.fill_(0.1234567)
                expected[name] = parameter.detach().clone()
        self.assertTrue(expected)
        preserve_fp8_params_through_ds_cast(model)
        DeepSpeedEngine._cast_module_mixed_precision(SimpleNamespace(module=model), torch.bfloat16, None, False)
        for name, parameter in model.named_parameters():
            if name in expected:
                torch_assert_equal(parameter, expected[name])
            else:
                self.assertEqual(parameter.dtype, torch.bfloat16)
        optimizer = build_training_optimizer(model, OptimizerSettings(name="torch_adamw"), {}, fused=False)
        for group in optimizer.param_groups:
            self.assertEqual(len({parameter.dtype for parameter in group["params"]}), 1)
        self.assertEqual(
            {group["params"][0].dtype for group in optimizer.param_groups}, {torch.float32, torch.bfloat16}
        )

    @require_torch_gpu
    def test_bf16_full_and_lora_backward_with_fp32_routing(self):
        set_seed(42)
        path = self.checkpoint_config(**tiny_config().to_dict())
        runtime = SimpleNamespace(name=path, ep=1, moe_use_grouped_mm=False)
        initial = dict(InklingForCausalLM(tiny_config()).state_dict())
        for index in (1, 2):
            prefix = f"model.layers.{index}.mlp.experts."
            initial[prefix + "w1"], initial[prefix + "w3"] = initial.pop(prefix + "gate_up_proj").chunk(2, dim=1)
            initial[prefix + "w2"] = initial.pop(prefix + "down_proj")
        for lora in (False, True):
            model = get_model(runtime, dtype=torch.bfloat16).cuda()
            model.load_state_dict(initial, strict=True)
            if lora:
                model = apply_peft(
                    model, dict(peft_type="LORA", task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj"])
                )
            for module in model.modules():
                if isinstance(module, InklingMoE):
                    module.ep_comm_backend = "local"
                    hidden = torch.randn(2, 3, 16, dtype=torch.bfloat16, device="cuda")
                    logits, weights, _, gammas = module.gate(hidden)
                    torch_assert_equal(
                        logits,
                        torch.nn.functional.linear(hidden.flatten(0, 1).float(), module.gate.weight.float())[:, :4],
                    )
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertEqual(gammas.dtype, torch.float32)
            ids = torch.randint(0, 60, (2, 9), device="cuda")
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            norm = torch.stack([gradient.float().square().sum() for gradient in gradients]).sum().sqrt()
            self.assertGreater(norm.item(), 0)

    def test_lora_export_uses_sampling_projection_names(self):
        from pathlib import Path

        from safetensors.torch import load_file

        from arctic_platform.model import iter_model_weights
        from arctic_platform.model import save_peft_adapters

        model = apply_peft(
            InklingForCausalLMPrimeRL(tiny_config()).float(),
            dict(peft_type="LORA", task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
        )
        exported = dict(iter_model_weights(model, "lora", is_master=True, is_zero3=False))
        expected = {}
        for index, layer in enumerate(model.base_model.model.model.layers):
            for projection, served_name in ((layer.self_attn.q_proj, "wq_du"), (layer.self_attn.v_proj, "wv_dv")):
                for kind in ("A", "B"):
                    name = f"base_model.model.model.layers.{index}.attn.{served_name}.lora_{kind}.weight"
                    expected[name] = getattr(projection, f"lora_{kind}")["default"].weight
        self.assertEqual(exported.keys(), expected.keys())
        for name, tensor in expected.items():
            torch_assert_equal(exported[name], tensor)
        path = Path(self.get_auto_remove_tmp_dir())
        save_peft_adapters(model, str(path))
        saved = load_file(path / "default" / "adapter_model.safetensors")
        self.assertEqual(saved.keys(), expected.keys())
        for name, tensor in expected.items():
            torch_assert_equal(saved[name], tensor)
        config = json.loads((path / "default" / "adapter_config.json").read_text())
        self.assertEqual(config["target_modules"], ["wq_du", "wv_dv"])

    def test_lora_steps_match_native_loss_and_gradients(self):
        set_seed(42)
        reference = InklingForCausalLM(tiny_config()).float()
        inputs = torch.randint(0, 60, (2, 9))
        mask = torch.ones_like(inputs)
        mask[1, -2:] = 0
        labels = inputs.clone()
        labels[:, :3] = -100
        labels[mask == 0] = -100
        batch = dict(input_ids=inputs, attention_mask=mask, labels=labels)
        initial = {name: value.clone() for name, value in reference.state_dict().items()}
        peft_config = dict(
            peft_type="LORA", task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]
        )

        def train(model):
            set_seed(123)
            model = apply_peft(model, peft_config)
            params = [p for p in model.parameters() if p.requires_grad]
            before = [p.detach().clone() for p in params]
            optimizer = torch.optim.AdamW(params, lr=1e-3)
            steps = []
            for _ in range(3):
                optimizer.zero_grad()
                out = model(**batch)
                out.loss.backward()
                norm = torch.stack([p.grad.float().square().sum() for p in params]).sum().sqrt()
                steps.append((out.loss.detach(), norm, [p.grad.clone() for p in params]))
                optimizer.step()
            self.assertTrue(any(not torch.equal(p, old) for p, old in zip(params, before)))
            return steps

        baseline = train(reference)
        model = InklingForCausalLMPrimeRL(tiny_config()).float()
        adapted = dict(initial)
        for index in (1, 2):
            prefix = f"model.layers.{index}.mlp.experts."
            adapted[prefix + "w1"], adapted[prefix + "w3"] = adapted.pop(prefix + "gate_up_proj").chunk(2, dim=1)
            adapted[prefix + "w2"] = adapted.pop(prefix + "down_proj")
        model.load_state_dict(adapted, strict=True)
        for module in model.modules():
            if isinstance(module, InklingMoE):
                module.ep_comm_backend = "local"
        actual = train(model)
        for (loss_ref, norm_ref, grads_ref), (loss, norm, grads) in zip(baseline, actual):
            # Splitting the native fused gate/up GEMM changes FP32 accumulation order.
            torch_assert_close(loss, loss_ref, atol=1e-6, rtol=0)
            torch_assert_close(norm, norm_ref, atol=1e-6, rtol=0)
            for grad, grad_ref in zip(grads, grads_ref):
                torch_assert_close(grad, grad_ref, atol=1e-6, rtol=0)

    def test_full_vllm_export_matches_official_checkpoint(self):
        from unittest.mock import patch

        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import revert_weight_conversion

        from arctic_platform.model.implementations.inkling.weights import build_iter_vllm_weights

        set_seed(42)
        config = tiny_config()
        reference = InklingForCausalLM(config).float()
        native = reference.state_dict()
        reference._weight_conversions = get_checkpoint_conversion_mapping("inkling_mm_model")
        expected = revert_weight_conversion(
            reference,
            {
                name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name: value
                for name, value in native.items()
            },
        )
        adapted = dict(native)
        for index in (1, 2):
            prefix = f"model.layers.{index}.mlp.experts."
            adapted[prefix + "w1"], adapted[prefix + "w3"] = adapted.pop(prefix + "gate_up_proj").chunk(2, dim=1)
            adapted[prefix + "w2"] = adapted.pop(prefix + "down_proj")
        model = InklingForCausalLMPrimeRL(config).float()
        model.load_state_dict(adapted, strict=True)
        with patch("torch.distributed.get_rank", return_value=0):
            actual = dict(build_iter_vllm_weights(model)())
        self.assertEqual(actual.keys(), expected.keys())
        for name, tensor in expected.items():
            torch_assert_equal(actual[name], tensor)

    def test_full_steps_match_native_loss_and_gradients(self):
        set_seed(42)
        reference = InklingForCausalLM(tiny_config()).float()
        model = InklingForCausalLMPrimeRL(tiny_config()).float()
        initial = dict(reference.state_dict())
        for index in (1, 2):
            prefix = f"model.layers.{index}.mlp.experts."
            initial[prefix + "w1"], initial[prefix + "w3"] = initial.pop(prefix + "gate_up_proj").chunk(2, dim=1)
            initial[prefix + "w2"] = initial.pop(prefix + "down_proj")
        model.load_state_dict(initial, strict=True)
        for module in model.modules():
            if isinstance(module, InklingMoE):
                module.ep_comm_backend = "local"
        ids = torch.randint(0, 60, (2, 9))
        mask = torch.ones_like(ids)
        mask[1, -2:] = 0
        labels = ids.clone()
        labels[:, :3] = -100
        labels[mask == 0] = -100
        batch = dict(input_ids=ids, attention_mask=mask, labels=labels)

        def train(module):
            params = {name: p for name, p in module.named_parameters() if p.requires_grad}
            optimizer = torch.optim.AdamW(params.values(), lr=1e-4)
            before = {name: p.detach().clone() for name, p in params.items()}
            records = []
            for _ in range(3):
                optimizer.zero_grad()
                loss = module(**batch).loss
                loss.backward()
                gradients = {name: p.grad.clone() for name, p in params.items()}
                norm = torch.stack([value.square().sum() for value in gradients.values()]).sum().sqrt()
                if isinstance(module, InklingForCausalLMPrimeRL):
                    for index in (1, 2):
                        module.convert_layer_to_hf(gradients, index)
                records.append((loss.detach(), norm, gradients))
                optimizer.step()
            self.assertTrue(any(not torch.equal(p, before[name]) for name, p in params.items()))
            return records

        baseline = train(reference)
        actual = train(model)
        for (loss_ref, norm_ref, grads_ref), (loss, norm, grads) in zip(baseline, actual):
            # Splitting fused gate/up GEMMs changes FP32 accumulation order.
            torch_assert_close(loss, loss_ref, atol=1e-6, rtol=0)
            torch_assert_close(norm, norm_ref, atol=1e-6, rtol=0)
            self.assertEqual(grads.keys(), grads_ref.keys())
            for name, grad in grads.items():
                torch_assert_close(grad, grads_ref[name], atol=1e-6, rtol=0)

    @require_torch_gpu
    def test_official_checkpoint_loading_and_gpu_backward(self):
        from pathlib import Path

        from safetensors.torch import save_file
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import revert_weight_conversion

        set_seed(42)
        config = tiny_config()
        reference = InklingForCausalLM(config).cuda().float()
        batch = dict(input_ids=torch.randint(0, 60, (2, 9), device="cuda"))
        batch["labels"] = batch["input_ids"].clone()
        baseline = reference(**batch)
        baseline.loss.backward()
        baseline_norm = torch.stack([p.grad.square().sum() for p in reference.parameters()]).sum().sqrt()

        # Use Transformers' inverse checkpoint conversion as the independent source of the official format.
        reference._weight_conversions = get_checkpoint_conversion_mapping("inkling_mm_model")
        state = {
            name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name: value.cpu()
            for name, value in reference.state_dict().items()
        }
        state = revert_weight_conversion(reference, state)
        path = Path(self.checkpoint_config(**config.to_dict()))
        save_file({name: value.contiguous() for name, value in state.items()}, path / "model.safetensors")
        runtime = SimpleNamespace(name=str(path), ep=1, moe_use_grouped_mm=False)
        model = get_model(runtime, device=torch.device("meta"), dtype=torch.float32)
        load_weights(model, runtime, None)
        for module in model.modules():
            if isinstance(module, InklingMoE):
                module.ep_comm_backend = "local"
        actual = model(**batch)
        actual.loss.backward()
        norm = torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt()
        torch_assert_equal(model.model.embed_tokens.weight, reference.model.embed_tokens.weight)
        torch_assert_close(actual.logits, baseline.logits, atol=1e-6, rtol=0)
        torch_assert_close(actual.loss, baseline.loss, atol=1e-6, rtol=0)
        torch_assert_close(norm, baseline_norm, atol=1e-6, rtol=0)
        native_path = path / "native"
        native_path.mkdir()
        (native_path / "config.json").write_text(config.to_json_string())
        save_file(
            {name: value.cpu() for name, value in reference.state_dict().items()}, native_path / "model.safetensors"
        )
        runtime.name = str(native_path)
        reloaded = get_model(runtime, device=torch.device("meta"), dtype=torch.float32)
        load_weights(reloaded, runtime, None)
        for name, value in model.state_dict().items():
            torch_assert_equal(value, reloaded.state_dict()[name])


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.llm.embed.weight", [("model.embed_tokens.weight", None)]),
        (
            "model.llm.layers.2.mlp.experts.w13_weight",
            [("model.layers.2.mlp.experts.w1", 0), ("model.layers.2.mlp.experts.w3", 1)],
        ),
        ("model.llm.layers.2.mlp.gate.bias", [("model.layers.2.mlp.gate.e_score_correction_bias", None)]),
        ("model.llm.layers.0.attn.k_sconv.weight", [("model.layers.0.self_attn.k_sconv.conv1d.weight", None)]),
        (
            "model.llm.layers.2.mlp.shared_experts.shared_w2_weight",
            [("model.layers.2.mlp.shared_experts.down_proj", None)],
        ),
        ("model.audio.encoder.weight", []),
    ],
)
def test_official_weight_names(name, expected):
    assert weight_targets(name) == expected
