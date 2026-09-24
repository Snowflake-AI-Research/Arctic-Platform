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
"""Offline Qwen LoRA parity: real PEFT, optimizer updates, and adapter reload."""

from __future__ import annotations

import copy
import sys

import pytest
import torch
from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM

from arctic_platform.model import LoaderContext
from arctic_platform.model import ModelSpec
from arctic_platform.model import Patches
from arctic_platform.model import apply_patches
from arctic_platform.model import apply_peft
from arctic_platform.model import build_model
from arctic_platform.peft import cast_lora_adapters_off_fp8
from arctic_platform.peft import is_peft_lora_param
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal


def _tiny_qwen():
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            attention_dropout=0.0,
            use_cache=False,
        )
    )


class TestPeft(TestCasePlus):
    def test_helpers_import_without_training_dependencies(self):
        code = """
import importlib.abc
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'peft', 'transformers', 'deepspeed'}:
            raise AssertionError(f'unexpected training import: {fullname}')

sys.meta_path.insert(0, BlockTrainingImports())
from arctic_platform.peft import apply_peft, is_peft_lora_param
from types import SimpleNamespace
assert is_peft_lora_param('model.lora_A.default.weight', SimpleNamespace(requires_grad=True))
model = object()
assert apply_peft(model, None) is model
print('PEFT helpers imported without training dependencies')
"""
        execute_subprocess_async([sys.executable, "-c", code], env=self.get_env(), timeout=30)

    def _training_parity(self, device, dtype):
        base_dir = self.get_auto_remove_tmp_dir_str()
        adapter_dir = self.get_auto_remove_tmp_dir_str()
        set_seed(41)
        _tiny_qwen().to(dtype=dtype).save_pretrained(base_dir)
        config = {
            "peft_type": "Lora",
            "task_type": "CAUSAL_LM",
            "r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "gate_proj", "down_proj"],
        }
        spec = ModelSpec(
            model_path_or_name=base_dir,
            loader="huggingface",
            dtype=str(dtype).removeprefix("torch."),
            attn_implementation="eager",
            patches=Patches(gradient_checkpointing=True, peft=config),
        )
        self.assertEqual(ModelSpec.model_validate_json(spec.model_dump_json()), spec)

        set_seed(42)
        reference = AutoModelForCausalLM.from_pretrained(base_dir, dtype=dtype, attn_implementation="eager")
        reference.gradient_checkpointing_enable()
        reference = get_peft_model(reference, LoraConfig(**config))
        reference.enable_input_require_grads()
        reference.to(device)
        base_weights = {name: p.detach().clone() for name, p in reference.named_parameters() if not p.requires_grad}
        adapters = {name: p.detach().clone() for name, p in reference.named_parameters() if p.requires_grad}
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], device=device)

        def train(model):
            model.train()
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
            metrics = []
            for step in range(3):
                set_seed(50 + step)
                loss = model(input_ids=tokens, labels=tokens).loss
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                metrics.append((loss.detach().clone(), norm.detach().clone()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            return metrics

        expected = train(reference)
        set_seed(42)
        loaded = build_model(spec)
        self.assertEqual(loaded.applied_patches, frozenset({"gradient_checkpointing", "peft"}))
        wrapped = loaded.model
        apply_patches(loaded, LoaderContext(spec=spec))
        self.assertIs(loaded.model, wrapped)
        actual = loaded.model.to(device)
        observed = train(actual)
        for (loss, norm), (expected_loss, expected_norm) in zip(observed, expected, strict=True):
            torch_assert_equal(loss, expected_loss)
            torch_assert_equal(norm, expected_norm)
        for name, param in actual.named_parameters():
            torch_assert_equal(param, dict(reference.named_parameters())[name])
            if name in base_weights:
                torch_assert_equal(param, base_weights[name])
            else:
                self.assertTrue(is_peft_lora_param(name, param))
        self.assertTrue(any(not torch.equal(dict(actual.named_parameters())[name], p) for name, p in adapters.items()))

        actual.eval()
        with torch.no_grad():
            expected_logits = actual(input_ids=tokens).logits
        actual.save_pretrained(adapter_dir)
        reloaded_base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=dtype, attn_implementation="eager")
        reloaded = PeftModel.from_pretrained(reloaded_base, adapter_dir).to(device).eval()
        with torch.no_grad():
            torch_assert_equal(reloaded(input_ids=tokens).logits, expected_logits)

    def test_cpu_qwen_training_and_reload(self):
        self._training_parity("cpu", torch.float32)

    @require_torch_gpu
    def test_cuda_qwen_training_and_reload(self):
        self._training_parity("cuda", torch.bfloat16)

    def test_fp8_base_keeps_adapter_optimization_dtype(self):
        class Projections(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(16, 16, bias=False)
                self.proj.weight = nn.Parameter(self.proj.weight.to(torch.float8_e4m3fn), requires_grad=False)

            def forward(self, x):
                return self.proj(x)

        for dtype in (torch.bfloat16, torch.float32):
            model = Projections()
            original = model.proj.weight
            adapted = apply_peft(model, {"peft_type": "Lora", "target_modules": ["proj"]}, optimization_dtype=dtype)
            self.assertIs(adapted.base_model.model.proj.base_layer.weight, original)
            self.assertEqual(original.dtype, torch.float8_e4m3fn)
            self.assertFalse(original.requires_grad)
            trainable = [(name, p) for name, p in adapted.named_parameters() if p.requires_grad]
            self.assertEqual(len(trainable), 2)
            for name, param in trainable:
                self.assertTrue(is_peft_lora_param(name, param))
                self.assertEqual(param.dtype, dtype)
            self.assertEqual(cast_lora_adapters_off_fp8(adapted, dtype), 0)

    def test_disabled_peft_preserves_model(self):
        model = nn.Linear(2, 2)
        self.assertIs(apply_peft(model, None), model)
        self.assertIs(apply_peft(model, {}), model)

    def test_peft_does_not_mutate_config(self):
        config = {"peft_type": "Lora", "target_modules": ["q_proj"], "r": 4}
        expected = copy.deepcopy(config)
        apply_peft(_tiny_qwen(), config)
        self.assertEqual(config, expected)


@pytest.mark.parametrize("config", [{"r": 4}, {"peft_type": "Missing"}, {"peft_type": 5}])
def test_invalid_peft_type(config):
    with pytest.raises(ValueError, match="PEFT type|peft_type"):
        apply_peft(nn.Linear(2, 2), config)


def test_worker_bridge_forwards_peft():
    config = {"peft_type": "Lora", "target_modules": ["q_proj"]}
    spec = ModelSpec.from_ds_worker_config("unused", {"attn_implementation": "eager", "peft_config": config})
    assert spec.patches.peft == config


def test_custom_moe_peft_requires_expert_integration():
    with pytest.raises(ValueError, match="expert adapter integration"):
        ModelSpec(model_path_or_name="unused", loader="qwen3_5_moe", patches=Patches(peft={"peft_type": "Lora"}))
