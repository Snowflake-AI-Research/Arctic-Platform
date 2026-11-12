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

"""Inkling2 loader guards, checkpoint decoding and parity with a dense reference."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model.implementations.inkling2.configuration_inkling2 import Inkling2TextConfig
from arctic_platform.model.implementations.inkling2.deepspeed_integration import get_model
from arctic_platform.model.implementations.inkling2.modeling_inkling2 import Inkling2ForCausalLMPrimeRL
from arctic_platform.model.implementations.inkling2.modeling_inkling2 import Inkling2MoE
from arctic_platform.model.implementations.inkling2.weights import CheckpointReader
from arctic_platform.model.implementations.inkling2.weights import decode_mxfp8
from arctic_platform.model.implementations.inkling2.weights import load_weights
from arctic_platform.model.implementations.inkling2.weights import weight_targets
from arctic_platform.model.patches.peft import apply_peft
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

LORA = dict(peft_type="LORA", task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["wq_du", "wv_dv"])
# Rank 8: grouped GEMMs need 16-byte strides of the BF16 factors.
EXPERT_LORA = dict(LORA, r=8, lora_alpha=16, target_parameters=["experts.w1", "experts.w2", "experts.w3"])


def tiny_config(**overrides):
    values = dict(
        vocab_size=64,
        unpadded_vocab_size=61,
        hidden_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        sliding_window_size=8,
        local_layer_ids=[0, 1, 3, 4],
        d_rel=4,
        rel_extent=16,
        log_scaling_n_floor=4,
        qk_l2_clip_threshold=2.0,
        rel_l2_clip_threshold=1.0,
        sconv_kernel_size=4,
        n_layers_per_attn_res_block=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        intermediate_size=32,
        mlp_gate_softcap=1.0,
        mlp_up_softcap=2.0,
        route_scale=2.0,
        final_logit_softcapping=3.0,
        initializer_range=0.2,
        use_grouped_mm=False,
        inkling2_attention_query_chunk_size=16,
    )
    values.update(overrides)
    return Inkling2TextConfig(**values)


def local_moe(model):
    for module in model.modules():
        if isinstance(module, Inkling2MoE):
            module.ep_comm_backend = "local"
    return model


def tiny_model(**overrides):
    model = Inkling2ForCausalLMPrimeRL(tiny_config(**overrides))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, Inkling2MoE):
                module.gate.bias.normal_(std=0.1)
    return local_moe(model)


def set_query_chunk_size(model, size):
    model.config.inkling2_attention_query_chunk_size = size


def _rms(x, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _clip(x, threshold):
    norm = x.norm(dim=-1, keepdim=True)
    return torch.where(norm > threshold, x * threshold / norm, x)


def _softcap(x, cap):
    return cap * torch.tanh(x / cap)


def _static_conv(x, weight):
    width = weight.shape[-1]
    taps = F.conv1d(F.pad(x.T[None], (width - 1, 0)), weight, groups=x.shape[1])[0].T
    return (x + taps) * (width + 1) ** -0.5


def _dynamic_conv(x, inputs, conv):
    width, channels = conv.static_w.shape
    heads = conv.weight_proj.shape[0] // width
    lags = F.pad(x, (0, 0, width - 1, 0)).unfold(0, width, 1).flip(-1)
    taps = (inputs @ conv.weight_proj.T).view(-1, heads, width).repeat_interleave(channels // heads, dim=1)
    return (x + (lags * (conv.static_w.T + taps)).sum(-1)) * (2 * width + 1) ** -0.5


def _mix(values, query, norm):
    values = torch.stack(values)
    normalized = _rms(values)
    probs = (normalized @ query / math.sqrt(values.shape[-1])).softmax(0)
    return _rms((probs[..., None] * normalized).sum(0)) * norm.weight


def _attention(attn, x_q, x_k, x_v, tau, config):
    length = x_q.shape[0]
    heads, kv_heads, dim = attn.num_heads, attn.num_key_value_heads, attn.head_dim
    query = _clip(
        (x_q @ attn.wq_du.weight.T).view(length, heads, dim) * attn.q_l2_clip.scale, config.qk_l2_clip_threshold
    )
    relative = (x_q @ attn.wr_du.weight.T).view(length, heads, -1) * attn.rel_l2_clip.scale
    relative = _clip(relative, config.rel_l2_clip_threshold)
    key, value = x_k @ attn.wk_dv.weight.T, x_v @ attn.wv_dv.weight.T
    if attn.is_local:
        key, value = _static_conv(key, attn.k_sconv.weight), _static_conv(value, attn.v_sconv.weight)
    else:
        query, relative = query * tau[:, None, None], relative * tau[:, None, None]
        key, value = _dynamic_conv(key, x_k, attn.k_sconv), _dynamic_conv(value, x_v, attn.v_sconv)
    key = _clip(key.view(length, kv_heads, dim), config.qk_l2_clip_threshold).repeat_interleave(heads // kv_heads, 1)
    value = _rms(value.view(length, kv_heads, dim)).repeat_interleave(heads // kv_heads, 1)
    distance = torch.arange(length)[:, None] - torch.arange(length)[None]
    distance = distance.to(x_q.device).expand(heads, length, length)
    projection = attn.rel_logits_proj.proj
    extent = projection.shape[1]
    bias = torch.einsum("ihr,re->hie", relative, projection).gather(-1, distance.clamp(0, extent - 1))
    scores = torch.einsum("ihd,jhd->hij", query, key) / math.sqrt(dim) + torch.where(distance < extent, bias, 0)
    allowed = distance >= 0
    if attn.is_local:
        allowed &= distance < config.sliding_window_size
    scores = scores.masked_fill(~allowed, float("-inf"))
    output = torch.einsum("hij,jhd->ihd", scores.softmax(-1), value)
    output = output * torch.sigmoid(x_q @ attn.gate_attn.weight.T).view(length, heads, dim)
    output = output * torch.sigmoid(scores.logsumexp(-1).T - x_q @ attn.attn_sink_proj.weight.T)[..., None]
    return output.flatten(1) @ attn.wo_ud.weight.T


def _moe(moe, x, config):
    logits = x @ moe.gate.weight.T
    selected = (logits.sigmoid() + moe.gate.bias).topk(config.num_experts_per_tok).indices
    weights = F.logsigmoid(logits.gather(1, selected)).softmax(-1) * config.route_scale
    experts = moe.experts
    rows = []
    for token in range(x.shape[0]):
        row = 0
        for slot, expert in enumerate(selected[token].tolist()):
            gate, up = experts.w1[expert] @ x[token], experts.w3[expert] @ x[token]
            hidden = _softcap(gate, config.mlp_gate_softcap) * gate.sigmoid() * _softcap(up, config.mlp_up_softcap)
            row = row + weights[token, slot] * (experts.w2[expert] @ hidden)
        rows.append(row)
    return torch.stack(rows)


def reference_logits(model, ids):
    """Dense FP32 logits of one unpadded sequence ``ids``."""
    config, text = model.config, model.model
    assert config.attn_res_normalize_values
    length, hidden = ids.shape[0], config.hidden_size
    positions = torch.arange(length, device=ids.device).float()
    tau = 1 + config.log_scaling_alpha * torch.log(((positions + 1) / config.log_scaling_n_floor).clamp(min=1))
    blocks, partial = [], _rms(text.embed_tokens.weight[ids]) * text.embed_norm.weight
    for index, layer in enumerate(text.layers):
        norms = [layer.attn_norm] if layer.is_local else [layer.q_input_norm, layer.k_input_norm, layer.v_input_norm]
        if not blocks:
            inputs = [_rms(partial) * norm.weight for norm in norms]
        else:
            queries = text.attn_res.attn_queries[index - 1].view(len(norms), hidden)
            inputs = [_mix([*blocks, partial], query, norm) for query, norm in zip(queries, norms)]
        x_q, x_k, x_v = inputs * 3 if layer.is_local else inputs
        attn = _static_conv(_attention(layer.attn, x_q, x_k, x_v, tau, config), layer.attn_sconv.weight)
        if index % config.n_layers_per_attn_res_block == 0:
            blocks = [*blocks, partial]
            residual = attn
        else:
            residual = partial + attn
        mlp_in = _mix([*blocks, residual], text.attn_res.mlp_queries[index], layer.mlp_norm)
        partial = residual + _static_conv(_moe(layer.mlp, mlp_in, config), layer.mlp_sconv.weight)
    output = _mix([*blocks, partial], text.attn_res.output_query, text.norm)
    return _softcap(output @ model.lm_head.weight[: config.unpadded_vocab_size].T, config.final_logit_softcapping)


def gradients(model):
    return {name: p.grad for name, p in model.named_parameters() if p.requires_grad}


def write_checkpoint(path, config, tensors):
    from safetensors.torch import save_file

    path = Path(path)
    (path / "config.json").write_text(json.dumps({"model_type": "inkling2", "text_config": config.to_dict()}))
    save_file({name: value.contiguous() for name, value in tensors.items()}, path / "model.safetensors")
    return str(path)


class TestInkling2Training(TestCasePlus):
    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False

    @require_torch_gpu
    def test_matches_dense_reference_across_query_chunks(self):
        set_seed(42)
        initial = tiny_model().cuda()
        ids = torch.randint(0, 61, (2, 37), device="cuda")
        reference = copy.deepcopy(initial)
        expected = torch.stack([reference_logits(reference, row) for row in ids])
        loss = F.cross_entropy(expected[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
        loss.backward()
        expected_grads = gradients(reference)
        self.assertNotIn("model.layers.0.mlp.gate.bias", expected_grads)
        for chunk_size in (8, 13, 64):
            with self.subTest(chunk_size=chunk_size):
                model = copy.deepcopy(initial)
                set_query_chunk_size(model, chunk_size)
                result = model(input_ids=ids, labels=ids)
                result.loss.backward()
                torch_assert_close(result.logits, expected, atol=2e-5, rtol=0)
                torch_assert_close(result.loss, loss, atol=2e-6, rtol=0)
                actual_grads = gradients(model)
                self.assertEqual(actual_grads.keys(), expected_grads.keys())
                for name, grad in actual_grads.items():
                    self.assertIsNotNone(grad, name)
                    torch_assert_close(grad, expected_grads[name], atol=2e-5, rtol=1e-4, msg=name)

    @require_torch_gpu
    def test_variable_lengths_with_checkpointing_compile_without_failure(self):
        from arctic_platform.model import ActivationCheckpointConfig
        from arctic_platform.model.implementations.qwen35.model_builder import apply_ac

        # Earlier tests' batch sizes and grad modes would otherwise count against the recompile limit.
        torch._dynamo.reset()
        set_seed(11)
        reference = tiny_model().cuda()
        model = copy.deepcopy(reference)
        set_query_chunk_size(model, 8)
        apply_ac(model, ActivationCheckpointConfig(router_replay_recompute=False))
        for length in range(17, 38):
            ids = torch.randint(0, 61, (1, length), device="cuda")
            model.zero_grad(set_to_none=True)
            result = model(input_ids=ids, labels=ids)
            result.loss.backward()
            with torch.no_grad():
                expected = reference_logits(reference, ids[0])[None]
            torch_assert_close(result.logits, expected, atol=2e-5, rtol=0, msg=f"length {length}")
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    @require_torch_gpu
    def test_relative_attention_far_keys_match_dense_reference(self):
        from arctic_platform.model.implementations.inkling.attention import relative_attention

        set_seed(5)
        heads, kv_heads, dim, d_rel, extent, start, q_len = 4, 2, 16, 4, 64, 600, 96
        keys = start + q_len
        query = torch.randn(1, heads, q_len, dim, device="cuda", requires_grad=True)
        key = torch.randn(1, kv_heads, keys, dim, device="cuda", requires_grad=True)
        value = torch.randn(1, kv_heads, keys, dim, device="cuda", requires_grad=True)
        relative = torch.randn(1, q_len, heads, d_rel, device="cuda", requires_grad=True)
        projection = torch.randn(d_rel, extent, device="cuda", requires_grad=True)
        tensors = (query, key, value, relative, projection)
        sequence_ids = torch.zeros(1, keys, dtype=torch.int64, device="cuda")
        sequence_ids[:, 200:] = 1
        sequence_ids[:, start + 40 :] = 2
        padding = torch.ones(1, keys, dtype=torch.bool, device="cuda")
        padding[:, 300:330] = False
        padding[:, start + 40 : start + 42] = False
        q_pos = torch.arange(start, keys, device="cuda")[:, None]
        k_pos = torch.arange(keys, device="cuda")[None]
        valid = (k_pos <= q_pos) & (sequence_ids[0, q_pos] == sequence_ids[0, k_pos]) & padding[0, k_pos]
        upstream = torch.randn(1, q_len, heads, dim, device="cuda")
        lse_upstream = torch.randn(1, heads, q_len, device="cuda")

        def run(fn):
            output, lse = fn(*tensors)
            finite = torch.isfinite(lse)
            loss = (output.transpose(1, 2) * upstream).sum() + (torch.where(finite, lse, 0) * lse_upstream).sum()
            grads = torch.autograd.grad(loss, tensors)
            return output.detach(), lse.detach(), grads

        def dense(query, key, value, relative, projection):
            key, value = (tensor.repeat_interleave(heads // kv_heads, dim=1) for tensor in (key, value))
            distance = q_pos - k_pos
            bank = (relative @ projection).transpose(1, 2)
            bias = bank.gather(-1, distance.clamp(0, extent - 1).expand(1, heads, -1, -1))
            scores = query @ key.transpose(-1, -2) * dim**-0.5 + torch.where(distance < extent, bias, 0)
            scores = scores.masked_fill(~valid, float("-inf"))
            rows = valid.any(-1)
            lse = torch.where(rows[..., None], scores, 0).logsumexp(-1)
            weights = torch.where(rows[..., None], (scores - lse[..., None]).exp(), 0)
            return weights @ value, torch.where(rows, lse, float("-inf"))

        def flex(*args):
            return relative_attention(
                *args,
                query_offset=start,
                padding_mask=padding,
                sequence_ids=sequence_ids,
                scaling=dim**-0.5,
                return_lse=True,
            )

        expected, actual = run(dense), run(flex)
        self.assertTrue(torch.isneginf(expected[1][..., 40:42]).all())
        torch_assert_close(actual[0], expected[0], atol=2e-5, rtol=0)
        torch_assert_close(actual[1], expected[1], atol=2e-5, rtol=0)
        for name, grad, baseline in zip(("q", "k", "v", "relative", "projection"), actual[2], expected[2]):
            self.assertTrue(torch.isfinite(grad).all(), name)
            torch_assert_close(grad, baseline, atol=1e-4, rtol=1e-4, msg=name)

    @require_torch_gpu
    def test_packed_and_padded_rows_match_separate_sequences(self):
        set_seed(7)
        model = tiny_model().cuda()
        first = torch.randint(0, 61, (19,), device="cuda")
        second = torch.randint(0, 61, (23,), device="cuda")
        with torch.no_grad():
            expected = [model(input_ids=ids[None]).logits[0] for ids in (first, second)]
            packed = model(
                input_ids=torch.cat([first, second])[None],
                position_ids=torch.cat([torch.arange(19), torch.arange(23)]).cuda()[None],
            ).logits[0]
            torch_assert_close(packed[:19], expected[0], atol=1e-5, rtol=0)
            torch_assert_close(packed[19:], expected[1], atol=1e-5, rtol=0)

            pad = torch.zeros(4, dtype=first.dtype, device="cuda")
            mask = torch.ones(2, 23, dtype=torch.long, device="cuda")
            mask[0, 19:] = 0
            right = model(input_ids=torch.stack([torch.cat([first, pad]), second]), attention_mask=mask).logits
            torch_assert_close(right[0, :19], expected[0], atol=1e-5, rtol=0)
            torch_assert_close(right[1], expected[1], atol=1e-5, rtol=0)

            positions = torch.zeros(1, 23, dtype=torch.long, device="cuda")
            positions[0, 4:] = torch.arange(19)
            left = model(
                input_ids=torch.cat([pad, first])[None], attention_mask=mask.flip(-1)[:1], position_ids=positions
            ).logits[0]
            torch_assert_close(left[4:], expected[0], atol=1e-5, rtol=0)

    @require_torch_gpu
    def test_chunked_head_preserves_loss_gradients_and_logprobs(self):
        from arctic_platform.model.implementations.inkling2.deepspeed_integration import _configure_lm_head

        set_seed(947)
        reference = tiny_model().cuda()
        actual = copy.deepcopy(reference)
        _configure_lm_head(actual, chunk_size=8, fused_cross_entropy=False, fp32_lm_head=True)
        ids = torch.randint(0, 61, (2, 21), device="cuda")
        labels = ids.clone()
        labels[:, :3] = -100
        baseline = reference(input_ids=ids, labels=labels)
        baseline.loss.backward()
        result = actual(input_ids=ids, labels=labels)
        result.loss.backward()
        self.assertIsNone(result.logits)
        torch_assert_close(result.loss, baseline.loss, atol=1e-6, rtol=0)
        for name, grad in gradients(actual).items():
            torch_assert_close(grad, gradients(reference)[name], atol=1e-6, rtol=0, msg=name)
        with torch.no_grad():
            targets = ids.roll(-1, dims=1)
            expected = (reference(input_ids=ids).logits / 0.7).log_softmax(-1).gather(-1, targets[..., None])[..., 0]
            logprobs = actual(input_ids=ids, labels=targets, temperature=0.7, dss_compute_logprobs=True)["logprobs"]
            torch_assert_close(logprobs, expected, atol=1e-5, rtol=0)
            last = actual(input_ids=ids, labels=targets, temperature=0.7, dss_compute_logprobs=True, logits_to_keep=3)
            torch_assert_close(last["logprobs"], expected[:, -3:], atol=1e-5, rtol=0)
        with self.assertRaisesRegex(ValueError, "native LM head"):
            _configure_lm_head(actual, chunk_size=8, fused_cross_entropy=True, fp32_lm_head=False)

    @require_torch_gpu
    def test_lora_steps_with_activation_checkpointing(self):
        from arctic_platform.model import ActivationCheckpointConfig
        from arctic_platform.model.implementations.qwen35.model_builder import apply_ac

        set_seed(42)
        initial = tiny_model().cuda()
        ids = torch.randint(0, 61, (2, 29), device="cuda")
        records = []
        for checkpointed in (False, True):
            set_seed(123)
            model = apply_peft(copy.deepcopy(initial), LORA)
            if checkpointed:
                apply_ac(model.base_model.model, ActivationCheckpointConfig(router_replay_recompute=False))
            trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
            self.assertTrue(trainable)
            self.assertTrue(all("lora_" in name for name in trainable))
            self.assertTrue(all(p.dtype == torch.float32 for p in trainable.values()))
            before = {name: p.detach().clone() for name, p in trainable.items()}
            optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3)
            steps = []
            for _ in range(3):
                optimizer.zero_grad()
                loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                steps.append((loss.detach(), {name: p.grad.clone() for name, p in trainable.items()}))
                optimizer.step()
            self.assertTrue(any(not torch.equal(p, before[name]) for name, p in trainable.items()))
            records.append(steps)
        for (loss, grads), (expected_loss, expected_grads) in zip(records[1], records[0]):
            torch_assert_close(loss, expected_loss, atol=1e-6, rtol=0)
            for name, grad in grads.items():
                torch_assert_close(
                    grad, expected_grads[name.replace("._checkpoint_wrapped_module", "")], atol=1e-6, rtol=0
                )

    @require_torch_gpu
    def test_unfused_expert_lora_matches_merged_deltas(self):
        """Expert LoRA from the attached A/B factors matches PEFT's merged per-expert deltas."""
        from peft.tuners.lora.layer import ParamWrapper

        from arctic_platform.model.loader import LoaderContext
        from arctic_platform.model.patches.peft import apply_peft_patch

        original = ParamWrapper._activate_lora

        def restore_param_wrapper():
            ParamWrapper._activate_lora = original
            for attr in ("_ap_activate_lora_original", "_ap_skip_fused_expert_delta"):
                if hasattr(ParamWrapper, attr):
                    delattr(ParamWrapper, attr)

        self.addCleanup(restore_param_wrapper)
        spec = SimpleNamespace(
            dtype="auto",
            patches=SimpleNamespace(peft=EXPERT_LORA),
            parallelism=SimpleNamespace(expert_parallel=1),
        )
        set_seed(5)
        initial = tiny_model().cuda()
        hidden = torch.randn(40, initial.config.hidden_size, device="cuda")
        probe = torch.randn_like(hidden)

        def run(unfused, grouped=False, dtype=torch.float32):
            model = copy.deepcopy(initial)
            for name, parameter in model.named_parameters():
                if not model.keeps_fp32_parameter(name):
                    parameter.data = parameter.data.to(dtype)
            for module in model.modules():
                if isinstance(module, Inkling2MoE):
                    module.experts.use_grouped_mm = grouped
            set_seed(6)
            if unfused:
                model = apply_peft_patch(model, LoaderContext(spec=spec))
                experts = model.base_model.model.model.layers[0].mlp.experts
                while isinstance(experts, ParamWrapper):
                    experts = experts.get_base_layer()
                self.assertEqual(set(experts._ap_lora_ab), {"w1", "w2", "w3"})
            else:
                model = apply_peft(model, EXPERT_LORA)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if ".lora_B." in name and ".experts." in name:
                        parameter.normal_(std=0.5)
            moe = model.base_model.model.model.layers[0].mlp
            x = hidden.to(dtype, copy=True).requires_grad_()
            out = moe(x)
            (out.float() * probe).sum().backward()
            restore_param_wrapper()
            grads = [p.grad for p in moe.parameters() if p.requires_grad]
            self.assertEqual(len(grads), 6)
            return [t.detach().double() for t in (out, x.grad, *grads)]

        def error(actual, expected):
            return max((a - e).norm().item() / e.norm().item() for a, e in zip(actual, expected))

        merged = run(unfused=False)
        self.assertLess(error(run(unfused=True), merged), 1e-5)
        # BF16 weights with the deltas merged in are off by about 1% too.
        self.assertLess(error(run(unfused=True, grouped=True, dtype=torch.bfloat16), merged), 2e-2)

    def test_lora_sync_serves_expert_adapters_as_checkpoint_projections(self):
        """Synced expert adapters are the served ``w13_weight`` / ``w2_weight`` ones and carry PEFT's deltas."""
        from peft.tuners.lora.layer import ParamWrapper

        from arctic_platform.model.weight_export import iter_lora_weights

        set_seed(8)
        model = apply_peft(tiny_model(), EXPERT_LORA)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".lora_B." in name:
                    parameter.normal_()
        tensors = dict(iter_lora_weights(model, is_master=True, is_zero3=False))
        prefix = "base_model.model.model.layers.0.mlp.experts."
        self.assertEqual(
            {name.removeprefix(prefix) for name in tensors if name.startswith(prefix)},
            {f"{projection}_weight.lora_{kind}.weight" for projection in ("w13", "w2") for kind in "AB"},
        )
        self.assertIn("base_model.model.model.layers.0.attn.wq_du.lora_A.weight", tensors)
        expected = {
            module.parameter_name: module.get_delta_weight("default")
            for module in model.base_model.model.model.layers[0].mlp.modules()
            if isinstance(module, ParamWrapper)
        }
        scaling = EXPERT_LORA["lora_alpha"] / EXPERT_LORA["r"]

        def delta(projection):
            a, b = (tensors[f"{prefix}{projection}_weight.lora_{kind}.weight"] for kind in "AB")
            return scaling * b @ a

        w13 = delta("w13")
        self.assertEqual(w13.shape[0], tiny_config().n_routed_experts)
        torch_assert_close(w13[:, 0::2], expected["w1"], atol=1e-5, rtol=1e-5)
        torch_assert_close(w13[:, 1::2], expected["w3"], atol=1e-5, rtol=1e-5)
        torch_assert_close(delta("w2"), expected["w2"], atol=1e-5, rtol=1e-5)

    @require_torch_gpu
    def test_bf16_grouped_experts_backward_keeps_fp32_parameters(self):
        set_seed(3)
        model = tiny_model(use_grouped_mm=True)
        reference = local_moe(copy.deepcopy(model)).cuda()
        model = model.cuda()
        for name, parameter in model.named_parameters():
            if not model.keeps_fp32_parameter(name):
                parameter.data = parameter.data.bfloat16()
        fp32 = {name for name, p in model.named_parameters() if p.dtype == torch.float32}
        self.assertIn("model.layers.0.mlp.gate.bias", fp32)
        self.assertIn("model.attn_res.output_query", fp32)
        self.assertIn("model.layers.2.attn.k_sconv.static_w", fp32)
        self.assertNotIn("model.layers.2.attn.wq_du.weight", fp32)
        ids = torch.randint(0, 61, (2, 33), device="cuda")
        result = model(input_ids=ids, labels=ids)
        result.loss.backward()
        expected = reference(input_ids=ids, labels=ids).loss
        torch_assert_close(result.loss, expected, atol=5e-2, rtol=0)
        for name, grad in gradients(model).items():
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)

    @require_torch_gpu
    def test_mxfp8_checkpoint_loading(self):
        set_seed(11)
        config = tiny_config()
        source = Inkling2ForCausalLMPrimeRL(config)
        tensors, expected = {}, {}
        for name, value in source.state_dict().items():
            if ".mlp.experts." in name:
                continue
            expected[name] = value
            if name == "lm_head.weight":
                tensors["model.llm.unembed.weight"] = value
            else:
                tensors[name.replace("model.", "model.llm.", 1).replace("embed_tokens", "embed", 1)] = value
        experts, inter, hidden = config.n_routed_experts, config.intermediate_size, config.hidden_size
        for index in range(config.num_hidden_layers):
            prefix = f"model.layers.{index}.mlp.experts."
            for name, shape in (("w13_weight", (experts, 2 * inter, hidden)), ("w2_weight", (experts, hidden, inter))):
                values = (torch.randn(shape) * 4).to(torch.float8_e4m3fn)
                scales = torch.randint(120, 131, (*shape[:-1], shape[-1] // 32), dtype=torch.uint8)
                tensors["model.llm." + prefix.removeprefix("model.") + name] = values
                tensors["model.llm." + prefix.removeprefix("model.") + name + "_scale"] = scales
                decoded = decode_mxfp8(values, scales)
                torch_assert_equal(decoded[..., -1], values[..., -1].float() * 2.0 ** (scales[..., -1].float() - 127))
                if name == "w2_weight":
                    expected[prefix + "w2"] = decoded
                else:
                    expected[prefix + "w1"], expected[prefix + "w3"] = decoded[:, 0::2], decoded[:, 1::2]
                    w13 = decoded
            layer = {name: expected[prefix + name] for name in ("w1", "w2", "w3")}
            hf = Inkling2ForCausalLMPrimeRL.convert_layer_to_hf({prefix + k: v for k, v in layer.items()}, index)
            torch_assert_equal(hf[prefix + "w13_weight"], w13)
            torch_assert_equal(hf[prefix + "w2_weight"], layer["w2"])
        tensors["model.vision.patch_embed.weight"] = torch.ones(2, 2)
        tensors["model.mtp.layers.0.weight"] = torch.ones(2, 2)
        path = write_checkpoint(self.get_auto_remove_tmp_dir(), config, tensors)
        runtime = SimpleNamespace(name=path, ep=1, moe_use_grouped_mm=False)
        model = get_model(runtime, device=torch.device("meta"), dtype=torch.float32)
        load_weights(model, runtime, None)
        state = model.state_dict()
        self.assertEqual(state.keys(), expected.keys())
        for name, value in expected.items():
            torch_assert_equal(state[name].cpu(), value.float(), msg=name)

        del tensors["model.llm.norm.weight"]
        path = write_checkpoint(self.get_auto_remove_tmp_dir(), config, tensors)
        runtime = SimpleNamespace(name=path, ep=1, moe_use_grouped_mm=False)
        with self.assertRaisesRegex(ValueError, "Missing Inkling2 checkpoint weights"):
            load_weights(get_model(runtime, device=torch.device("meta")), runtime, None)

    def test_reader_rejects_nvfp4_experts(self):
        path = write_checkpoint(
            self.get_auto_remove_tmp_dir(),
            tiny_config(),
            {"model.llm.layers.0.mlp.experts.w2_weight": torch.zeros(2, 4, dtype=torch.uint8)},
        )
        with self.assertRaisesRegex(ValueError, "NVFP4"):
            CheckpointReader(path).read("model.llm.layers.0.mlp.experts.w2_weight")

    def checkpoint_path(self, **overrides):
        path = Path(self.get_auto_remove_tmp_dir())
        (path / "config.json").write_text(
            json.dumps({"model_type": "inkling2", "text_config": tiny_config(**overrides).to_dict()})
        )
        return str(path)

    def test_loader_selects_flex_and_rejects_unsupported_specs(self):
        from arctic_platform.model import ActivationCheckpointConfig
        from arctic_platform.model.loaders.inkling2 import Inkling2Options

        path = self.checkpoint_path()
        spec = ModelSpec(model_path_or_name=path, parallelism=ParallelismConfig(expert_parallel=8))
        self.assertEqual(spec.loader, "inkling2")
        self.assertEqual(spec.attn_implementation, "flex_attention")
        with self.assertRaisesRegex(ValueError, "flex_attention"):
            ModelSpec(model_path_or_name=path, attn_implementation="sdpa")
        spec = ModelSpec(model_path_or_name=path, parallelism=ParallelismConfig(sequence_parallel=2))
        self.assertEqual(spec.parallelism.sequence_parallel, 2)
        with self.assertRaisesRegex(ValueError, "full activation checkpointing"):
            ModelSpec(
                model_path_or_name=path,
                loader_options=Inkling2Options(ac_config=ActivationCheckpointConfig(mode="selective")).model_dump(),
            )
        with self.assertRaisesRegex(ValueError, "native LM head"):
            Inkling2Options(fused_cross_entropy=True)
        with self.assertRaisesRegex(ValueError, "positive"):
            Inkling2Options(fused_lm_head_token_chunk_size=0)

    def test_adapter_routes_options_and_validates_config(self):
        from arctic_platform.model.implementations.inkling2.deepspeed_integration import _adapter_for_inkling2
        from arctic_platform.model.loaders.inkling2 import Inkling2Options

        path = self.checkpoint_path()
        options = Inkling2Options(attention_query_chunk_size=256, fused_lm_head_token_chunk_size=32)
        runtime = SimpleNamespace(name=path, ep=4, moe_use_grouped_mm=True)
        adapter = _adapter_for_inkling2(options)
        model = adapter.get_model(runtime, device=torch.device("meta"), dtype=torch.bfloat16)
        self.assertEqual(model.config.inkling2_attention_query_chunk_size, 256)
        self.assertTrue(model.config.use_grouped_mm)
        self.assertIsNone(adapter.vllm_weight_export)
        adapter.inject_lm_head(model, chunk_size=32, fused_cross_entropy=False, fp32_lm_head=True)
        self.assertEqual(model._inkling2_lm_head_token_chunk_size, 32)
        self.assertTrue(model._inkling2_fp32_lm_head)
        with self.assertRaisesRegex(ValueError, "SP group size"):
            adapter.apply_sequence_parallelism(model, 2, None)
        with self.assertRaisesRegex(ValueError, "divisible"):
            get_model(SimpleNamespace(name=path, ep=3, moe_use_grouped_mm=False), device=torch.device("meta"))
        with self.assertRaisesRegex(ValueError, "n_shared_experts"):
            tiny_config(n_shared_experts=1)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.llm.embed.weight", [("model.embed_tokens.weight", None)]),
        ("model.llm.embed_norm.weight", [("model.embed_norm.weight", None)]),
        ("model.llm.unembed.weight", [("lm_head.weight", None)]),
        ("model.llm.attn_res.attn_queries.4", [("model.attn_res.attn_queries.4", None)]),
        (
            "model.llm.layers.5.mlp.experts.w13_weight",
            [("model.layers.5.mlp.experts.w1", "gate"), ("model.layers.5.mlp.experts.w3", "up")],
        ),
        ("model.llm.layers.5.mlp.experts.w2_weight", [("model.layers.5.mlp.experts.w2", None)]),
        ("model.llm.layers.5.mlp.experts.w2_weight_scale", []),
        ("model.llm.layers.5.attn.k_sconv.weight_proj", [("model.layers.5.attn.k_sconv.weight_proj", None)]),
        ("model.vision.blocks.0.attn.qkv.weight", []),
        ("model.mtp.layers.0.attn.wq_du.weight", []),
    ],
)
def test_checkpoint_weight_names(name, expected):
    assert weight_targets(name) == expected
