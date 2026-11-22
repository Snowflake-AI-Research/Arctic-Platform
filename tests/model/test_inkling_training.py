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


def initialized_prime_model(config):
    native = InklingForCausalLM(config)
    state = dict(native.state_dict())
    for index, kind in enumerate(config.mlp_layer_types):
        if kind == "sparse":
            prefix = f"model.layers.{index}.mlp.experts."
            state[prefix + "w1"], state[prefix + "w3"] = state.pop(prefix + "gate_up_proj").chunk(2, dim=1)
            state[prefix + "w2"] = state.pop(prefix + "down_proj")
    model = InklingForCausalLMPrimeRL(config)
    model.load_state_dict(state, strict=True)
    return model


class TestInklingTraining(TestCasePlus):
    def test_nccl_rejects_combine_replay(self):
        from arctic_platform.model.implementations.moe.layers.moe import install_combine_replay

        model = initialized_prime_model(tiny_config())
        for module in model.modules():
            if isinstance(module, InklingMoE):
                module.set_ep_comm_backend("nccl")
        with self.assertRaisesRegex(ValueError, "combine_replay=False"):
            install_combine_replay(model)

    @require_torch_gpu
    def test_router_is_invariant_to_sequence_shard_length(self):
        from arctic_platform.model.implementations.inkling.modeling_inkling import InklingTopkRouter

        set_seed(947)
        config = tiny_config()
        config.hidden_size = 4096
        config.n_routed_experts = 256
        config.num_experts_per_tok = 6
        config.n_shared_experts = 4
        config.inkling_mlp_token_chunk_size = 128
        router = InklingTopkRouter(config).cuda().float()
        torch.nn.init.normal_(router.weight, std=0.02)
        hidden = torch.randn(2, 520, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        expected = [value.view(2, 520, -1) for value in router(hidden)]
        for chunk_size in (128, 256, 2048):
            config.inkling_mlp_token_chunk_size = chunk_size
            candidate = InklingTopkRouter(config).cuda().float()
            candidate.load_state_dict(router.state_dict())
            for size in (1, 2, 4):
                actual = [
                    torch.cat(parts, dim=1)
                    for parts in zip(
                        *[
                            [value.view(2, 520 // size, -1) for value in candidate(shard)]
                            for shard in hidden.chunk(size, dim=1)
                        ],
                        strict=True,
                    )
                ]
                for value, baseline in zip(actual, expected, strict=True):
                    torch_assert_equal(value, baseline)

    @require_torch_gpu
    def test_flex_chunked_full_and_lora_steps_with_activation_offload(self):
        import copy

        from arctic_platform.model import ActivationCheckpointConfig
        from arctic_platform.model.implementations.inkling.attention import enable_flex_attention
        from arctic_platform.model.implementations.inkling.deepspeed_integration import _configure_lm_head
        from arctic_platform.model.implementations.qwen35.model_builder import apply_ac

        set_seed(947)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = tiny_config()
        config.head_dim = config.swa_head_dim = 16
        initial = initialized_prime_model(config).cuda().float()
        for module in initial.modules():
            if isinstance(module, InklingMoE):
                module.ep_comm_backend = "local"
        ids = torch.randint(0, 60, (2, 259), device="cuda")
        labels = ids.clone()
        labels[:, :3] = -100
        batch = dict(input_ids=ids, labels=labels)
        for lora in (False, True):
            with self.subTest(lora=lora):
                reference = copy.deepcopy(initial)
                actual = copy.deepcopy(initial)
                if lora:
                    options = dict(
                        peft_type="LORA", task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]
                    )
                    set_seed(42)
                    reference = apply_peft(reference, options)
                    set_seed(42)
                    actual = apply_peft(actual, options)
                base = actual.base_model.model if lora else actual
                enable_flex_attention(base, query_chunk_size=128)
                _configure_lm_head(base, chunk_size=64, fused_cross_entropy=False, fp32_lm_head=False)
                for module in base.modules():
                    if hasattr(module, "token_chunk_size"):
                        module.token_chunk_size = 64
                apply_ac(
                    base,
                    ActivationCheckpointConfig(
                        router_replay_recompute=False,
                        offload_config=dict(enabled=True, tensor_size_threshold=0, keep_last_n=0),
                    ),
                )
                optimizers = [
                    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
                    for m in (reference, actual)
                ]
                for _ in range(3):
                    records = []
                    for model, optimizer in zip((reference, actual), optimizers):
                        optimizer.zero_grad()
                        loss = model(**batch).loss
                        loss.backward()
                        grads = [p.grad.clone() for p in model.parameters() if p.requires_grad]
                        norm = torch.stack([g.square().sum() for g in grads]).sum().sqrt()
                        records.append((loss.detach(), norm, grads))
                        optimizer.step()
                    expected, result = records
                    torch_assert_close(result[0], expected[0], atol=1e-6, rtol=0)
                    torch_assert_close(result[1], expected[1], atol=1e-6, rtol=0)
                    for grad, baseline in zip(result[2], expected[2]):
                        torch_assert_close(grad, baseline, atol=1e-6, rtol=0)

    @require_torch_gpu
    def test_flex_training_matches_eager_across_query_chunks(self):
        import copy

        from arctic_platform.model.implementations.inkling.attention import enable_flex_attention

        set_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = tiny_config()
        config.head_dim = config.swa_head_dim = 16
        config.log_scaling_n_floor = 128
        config.log_scaling_alpha = 0.1
        reference = initialized_prime_model(config).cuda().float()
        for module in reference.modules():
            if isinstance(module, InklingMoE):
                module.ep_comm_backend = "local"
        actual = copy.deepcopy(reference)
        enable_flex_attention(actual, query_chunk_size=128)
        ids = torch.randint(0, 60, (2, 259), device="cuda")
        mask = torch.ones_like(ids)
        mask[1, -3:] = 0
        labels = ids.clone()
        labels[:, :3] = -100
        labels[mask == 0] = -100
        batch = dict(input_ids=ids, attention_mask=mask, labels=labels)
        baseline = reference(**batch)
        baseline.loss.backward()
        result = actual(**batch)
        result.loss.backward()
        torch_assert_close(result.logits, baseline.logits, atol=1e-6, rtol=0)
        torch_assert_close(result.loss, baseline.loss, atol=1e-6, rtol=0)
        expected_norm = (
            torch.stack([p.grad.square().sum() for p in reference.parameters() if p.requires_grad]).sum().sqrt()
        )
        norm = torch.stack([p.grad.square().sum() for p in actual.parameters() if p.requires_grad]).sum().sqrt()
        torch_assert_close(norm, expected_norm, atol=1e-6, rtol=0)
        for (name, parameter), (expected_name, expected) in zip(
            actual.named_parameters(), reference.named_parameters()
        ):
            self.assertEqual(name, expected_name)
            if not parameter.requires_grad:
                continue
            torch_assert_close(parameter.grad, expected.grad, atol=1e-6, rtol=0, msg=name)

    @require_torch_gpu
    def test_flex_variable_lengths_preserve_parity_without_recompile_failure(self):
        import copy

        from arctic_platform.model.implementations.inkling.attention import enable_flex_attention

        set_seed(947)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = tiny_config()
        config.head_dim = config.swa_head_dim = 16
        reference = initialized_prime_model(config).cuda().float()
        for module in reference.modules():
            if isinstance(module, InklingMoE):
                module.ep_comm_backend = "local"
        actual = copy.deepcopy(reference)
        enable_flex_attention(actual, query_chunk_size=128)
        for length in range(17, 38):
            ids = torch.randint(0, 60, (1, length), device="cuda")
            records = []
            for model in (reference, actual):
                model.zero_grad(set_to_none=True)
                loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                grads = [p.grad.clone() for p in model.parameters() if p.requires_grad]
                norm = torch.stack([g.square().sum() for g in grads]).sum().sqrt()
                records.append((loss.detach(), norm, grads))
            for value, expected in zip(records[1][:2], records[0][:2]):
                torch_assert_close(value, expected, atol=1e-6, rtol=0)
            for value, expected in zip(records[1][2], records[0][2]):
                torch_assert_close(value, expected, atol=1e-6, rtol=0)

    @require_torch_gpu
    def test_flex_bf16_padding_and_packed_sequence_parity(self):
        import copy

        from arctic_platform.model.implementations.inkling.attention import enable_flex_attention
        from arctic_platform.model.implementations.inkling.deepspeed_integration import _configure_lm_head

        set_seed(947)
        config = tiny_config()
        config.head_dim = config.swa_head_dim = 16
        config.log_scaling_n_floor = 128
        config.log_scaling_alpha = 0.1
        initial = initialized_prime_model(config).cuda().bfloat16()
        for module in initial.modules():
            if isinstance(module, InklingMoE):
                module.ep_comm_backend = "local"
        ids = torch.randint(0, 60, (2, 259), device="cuda")
        for layout in ("right", "left", "packed"):
            with self.subTest(layout=layout):
                reference, actual = (copy.deepcopy(initial), copy.deepcopy(initial))
                enable_flex_attention(actual, query_chunk_size=128)
                _configure_lm_head(actual, chunk_size=64, fused_cross_entropy=False, fp32_lm_head=False)
                mask = torch.ones_like(ids)
                labels = ids.clone()
                positions = torch.arange(ids.shape[1], device="cuda")[None].expand_as(ids).clone()
                if layout == "right":
                    mask[1, -3:] = 0
                elif layout == "left":
                    mask[1, :3] = 0
                    labels[1, 3] = -100
                else:
                    positions[:, 131:] -= 131
                    labels[:, 131] = -100
                labels[mask == 0] = -100
                batch = dict(input_ids=ids, labels=labels, attention_mask=mask, position_ids=positions)
                baseline = reference(**batch)
                baseline.loss.backward()
                result = actual(**batch)
                result.loss.backward()
                # BF16 eager scores round between matmul and softmax; Flex fuses those operations.
                torch_assert_close(result.loss, baseline.loss, atol=1e-3, rtol=0)
                records = []
                for model in (reference, actual):
                    gradients = [p.grad.float() for p in model.parameters() if p.requires_grad]
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    records.append(torch.stack([g.square().sum() for g in gradients]).sum().sqrt())
                torch_assert_close(records[1], records[0], atol=1e-3, rtol=0)
                for (name, parameter), expected in zip(actual.named_parameters(), reference.parameters()):
                    if parameter.requires_grad:
                        torch_assert_close(parameter.grad, expected.grad, atol=1e-3, rtol=0, msg=name)

    @require_torch_gpu
    def test_chunked_head_preserves_sft_loss_gradients_and_logprobs(self):
        import copy

        from arctic_platform.model.implementations.inkling.deepspeed_integration import _configure_lm_head

        set_seed(947)
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = initialized_prime_model(tiny_config()).cuda().float()
        for module in reference.modules():
            if isinstance(module, InklingMoE):
                module.ep_comm_backend = "local"
        actual = copy.deepcopy(reference)
        _configure_lm_head(actual, chunk_size=4, fused_cross_entropy=False, fp32_lm_head=False)
        ids = torch.randint(0, 60, (2, 9), device="cuda")
        labels = ids.clone()
        labels[:, :3] = -100
        baseline = reference(input_ids=ids, labels=labels)
        baseline.loss.backward()
        result = actual(input_ids=ids, labels=labels)
        result.loss.backward()
        self.assertIsNone(result.logits)
        torch_assert_close(result.loss, baseline.loss, atol=1e-6, rtol=0)
        for (name, parameter), expected in zip(actual.named_parameters(), reference.parameters()):
            if not parameter.requires_grad:
                continue
            torch_assert_close(parameter.grad, expected.grad, atol=1e-6, rtol=0, msg=name)
        expected_logprobs = baseline.logits.float().log_softmax(-1).gather(-1, ids[:, :, None]).squeeze(-1)
        logprobs = actual(input_ids=ids, labels=ids, dss_compute_logprobs=True)["logprobs"]
        torch_assert_close(logprobs, expected_logprobs, atol=1e-6, rtol=0)
        last = actual(input_ids=ids, labels=ids, dss_compute_logprobs=True, logits_to_keep=3)["logprobs"]
        torch_assert_close(last, expected_logprobs[:, -3:], atol=1e-6, rtol=0)

    @require_torch_gpu
    def test_configured_chunked_head_preserves_grpo_logprobs_and_gradients(self):
        import copy

        from arctic_platform.model.implementations.inkling.deepspeed_integration import _adapter_for_inkling
        from arctic_platform.model.loaders.inkling import InklingOptions

        set_seed(947)
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = initialized_prime_model(tiny_config()).cuda().float()
        for module in reference.modules():
            if isinstance(module, InklingMoE):
                module.ep_comm_backend = "local"
        actual = copy.deepcopy(reference)
        options = InklingOptions(fused_lm_head_token_chunk_size=4, fp32_lm_head=True)
        _adapter_for_inkling(options).inject_lm_head(
            actual,
            chunk_size=options.fused_lm_head_token_chunk_size,
            fused_cross_entropy=False,
            fp32_lm_head=options.fp32_lm_head,
        )
        ids = torch.randint(0, 60, (2, 9), device="cuda")
        labels = ids.roll(-1, dims=1)
        temperature = 0.7
        logits = reference(input_ids=ids).logits.float() / temperature
        expected = logits.log_softmax(-1).gather(-1, labels[:, :, None]).squeeze(-1)
        result = actual(input_ids=ids, labels=labels, temperature=temperature, dss_compute_logprobs=True)["logprobs"]
        torch_assert_close(result, expected, atol=1e-6, rtol=0)
        (-expected.mean()).backward()
        (-result.mean()).backward()
        for (name, parameter), baseline in zip(actual.named_parameters(), reference.parameters()):
            torch_assert_close(parameter.grad, baseline.grad, atol=1e-6, rtol=0, msg=name)

    @require_torch_gpu
    def test_flex_checkpoint_resume_reproduces_next_full_and_lora_update(self):
        import copy

        from arctic_platform.model.implementations.inkling.attention import enable_flex_attention
        from arctic_platform.model.implementations.inkling.deepspeed_integration import _configure_lm_head

        set_seed(947)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = tiny_config()
        config.head_dim = config.swa_head_dim = 16
        initial = initialized_prime_model(config).cuda().float()
        for module in initial.modules():
            if isinstance(module, InklingMoE):
                module.ep_comm_backend = "local"
        ids = torch.randint(0, 60, (2, 259), device="cuda")
        for lora in (False, True):
            with self.subTest(lora=lora):

                def build():
                    model = copy.deepcopy(initial)
                    if lora:
                        set_seed(42)
                        model = apply_peft(
                            model,
                            dict(
                                peft_type="LORA",
                                task_type="CAUSAL_LM",
                                r=2,
                                lora_alpha=4,
                                target_modules=["q_proj", "v_proj"],
                            ),
                        )
                    base = model.base_model.model if lora else model
                    enable_flex_attention(base, query_chunk_size=128)
                    _configure_lm_head(base, chunk_size=64, fused_cross_entropy=False, fp32_lm_head=False)
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                    return model, optimizer

                def step(model, optimizer):
                    optimizer.zero_grad()
                    loss = model(input_ids=ids, labels=ids).loss
                    loss.backward()
                    norm = (
                        torch.stack([p.grad.square().sum() for p in model.parameters() if p.requires_grad])
                        .sum()
                        .sqrt()
                    )
                    optimizer.step()
                    return loss.detach(), norm

                model, optimizer = build()
                step(model, optimizer)
                checkpoint = dict(
                    model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict())
                )
                expected = step(model, optimizer)
                resumed, resumed_optimizer = build()
                resumed.load_state_dict(checkpoint["model"], strict=True)
                resumed_optimizer.load_state_dict(checkpoint["optimizer"])
                actual = step(resumed, resumed_optimizer)
                for value, baseline in zip(actual, expected):
                    torch_assert_close(value, baseline, atol=1e-6, rtol=0)
                for name, parameter in model.state_dict().items():
                    torch_assert_close(resumed.state_dict()[name], parameter, atol=1e-6, rtol=0, msg=name)

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

    def test_flex_loader_routes_chunking_options(self):
        from arctic_platform.model.implementations.inkling.attention import InklingFlexAttention
        from arctic_platform.model.implementations.inkling.deepspeed_integration import _adapter_for_inkling
        from arctic_platform.model.loaders.inkling import InklingOptions

        path = self.checkpoint_config(**tiny_config().to_dict())
        options = InklingOptions(
            attention_query_chunk_size=128,
            tiled_mlp_token_chunk_size=64,
            fused_lm_head_token_chunk_size=32,
            ep_comm_backend="nccl",
        )
        spec = ModelSpec(
            model_path_or_name=path, attn_implementation="flex_attention", loader_options=options.model_dump()
        )
        self.assertEqual(spec.attn_implementation, "flex_attention")
        runtime = SimpleNamespace(name=path, ep=1, moe_use_grouped_mm=False, attn="flex_attention")
        model = _adapter_for_inkling(options).get_model(runtime, device=torch.device("meta"), dtype=torch.float32)
        for layer in model.model.layers:
            self.assertIsInstance(layer.self_attn, InklingFlexAttention)
            self.assertEqual(layer.self_attn.config.inkling_attention_query_chunk_size, 128)
        self.assertEqual(model.model.layers[0].mlp.token_chunk_size, 64)
        self.assertEqual(model.model.layers[1].mlp.shared_experts.token_chunk_size, 64)
        with self.assertRaisesRegex(ValueError, "positive"):
            InklingOptions(fused_lm_head_token_chunk_size=0)

    def test_flex_rejects_torch_without_differentiable_score_buffers(self):
        from unittest.mock import patch

        from arctic_platform.model.implementations.inkling.attention import enable_flex_attention

        with patch.object(torch, "__version__", "2.12.0"):
            with self.assertRaisesRegex(ValueError, "PyTorch 2.13"):
                enable_flex_attention(None)

    def test_sequence_parallelism_requires_flex(self):
        path = self.checkpoint_config()
        with self.assertRaisesRegex(ValueError, "sequence parallelism"):
            ModelSpec(model_path_or_name=path, parallelism=ParallelismConfig(sequence_parallel=2))
        spec = ModelSpec(
            model_path_or_name=path,
            attn_implementation="flex_attention",
            parallelism=ParallelismConfig(sequence_parallel=2),
        )
        self.assertEqual(spec.parallelism.sequence_parallel, 2)

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

    @require_deepspeed
    def test_deepspeed_preserves_fp32_lora_parameters(self):
        from deepspeed.runtime.engine import DeepSpeedEngine

        from arctic_platform.common.deepspeed_engine import build_training_optimizer
        from arctic_platform.common.optimizer import OptimizerSettings
        from arctic_platform.model import preserve_fp8_params_through_ds_cast

        model = apply_peft(
            initialized_prime_model(tiny_config()).bfloat16(),
            dict(peft_type="LORA", task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
        )
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        expected = {name: p.detach().clone() for name, p in parameters.items()}
        preserve_fp8_params_through_ds_cast(model)
        DeepSpeedEngine._cast_module_mixed_precision(SimpleNamespace(module=model), torch.bfloat16, None, False)
        for name, parameter in parameters.items():
            self.assertEqual(parameter.dtype, torch.float32)
            torch_assert_equal(parameter, expected[name])
        optimizer = build_training_optimizer(model, OptimizerSettings(name="torch_adamw"), {}, fused=False)
        self.assertTrue(optimizer.param_groups)
        for group in optimizer.param_groups:
            self.assertEqual({p.dtype for p in group["params"]}, {torch.float32})

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
        ("model.llm.embed_norm.weight", [("model.embed_norm.weight", None)]),
        ("model.embed_tokens.embed_norm.weight", [("model.embed_norm.weight", None)]),
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


@pytest.mark.parametrize("full_checkpoint", [False, True])
def test_chunked_inkling_dispatch_bounds_pending_combines(monkeypatch, full_checkpoint):
    import copy

    from torch.utils.checkpoint import checkpoint

    from arctic_platform.model.implementations.moe.distributed import expert_parallel
    from arctic_platform.model.implementations.moe.layers import moe
    from tests.model.test_flash_trainer_rounding import _OneRankDispatch

    class Dispatch(_OneRankDispatch):
        def __init__(self):
            super().__init__()
            self.pending = []
            self.peak_pending = 0

        def combine_tokens(self, rows, state):
            output = super().combine_tokens(rows, state)
            self.pending.append(rows)
            self.peak_pending = max(self.peak_pending, len(self.pending))
            return output

        def sync_combine(self):
            self.pending.clear()

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(expert_parallel, "get_ep_group", lambda experts: None)
    set_seed(115)
    initial = initialized_prime_model(tiny_config()).model.layers[1].mlp.float()
    assert initial.sync_combine_per_chunk
    assert initial.checkpoint_dispatch_chunks
    initial.set_deepep_token_chunk_size(4)
    hidden = torch.randn(2, 9, 16)
    records = []
    for bounded in (False, True):
        layer = copy.deepcopy(initial)
        layer.sync_combine_per_chunk = bounded
        layer.checkpoint_dispatch_chunks = bounded
        dispatch = Dispatch()
        monkeypatch.setattr(moe, "get_ep_comm_module", lambda backend: dispatch)
        inputs = hidden.clone().requires_grad_(True)
        output = checkpoint(layer, inputs, use_reentrant=False, early_stop=False) if full_checkpoint else layer(inputs)
        loss = output.square().sum()
        loss.backward()
        grads = [p.grad for p in layer.parameters() if p.requires_grad]
        norm = torch.stack([grad.square().sum() for grad in grads]).sum().sqrt()
        records.append((output, loss, norm, inputs.grad, *grads))
        assert not dispatch.pending
        assert dispatch.peak_pending == (1 if bounded else 5)
    for actual, expected in zip(records[1], records[0], strict=True):
        torch_assert_close(actual, expected, atol=1e-6, rtol=0)
