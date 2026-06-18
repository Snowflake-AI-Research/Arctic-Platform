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

"""GPU tests for the ZoRRO model patcher used in production: ``Qwen3ModelOncePatcher``.

This is the patcher the DeepSpeed worker installs (``deepspeed_worker.py``). It is patched onto a Qwen3 model once
and then computes deduplicated forward/backward internally: called with the full ``[B, S]`` batch it deduplicates
shared prompts, runs the model on the packed sequence, and returns per-response-token ``logprobs`` / ``entropy`` in
the original sample order (no padding). Driven here directly (single process, ``world_size=1``, so no collectives)
across all three ``logits_optimization`` dispatch modes (``none`` / ``memory`` / ``compute``)::

    pytest tests/zorro_train/test_once_patcher.py

Batches use the same layout as the other RL tests under ``tests/rl/``: prompts LEFT-padded and responses
RIGHT-padded to a fixed ``[left_pad][prompt][response][right_pad]`` row, with variable real lengths. The reference
runs each row's real tokens alone (positions ``0..len-1``, no left-pad offset to reconcile), which the deduplicated
forward must reproduce.

The bare ``Qwen3ModelPatcher`` context-manager path is intentionally not tested: it is stale against current
transformers (it applies rotary on reconstructed full-length Q/K while the model precomputes position embeddings on
the deduplicated length). The CPU-only algorithm round-trips live in ``test_dedup.py``.
"""

from __future__ import annotations

import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM

from arctic_platform.rl.zorro_train.qwen_model_patcher import Qwen3ModelOncePatcher
from arctic_platform.rl.zorro_train.tests import create_dummy_batch
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import torch_assert_close

device = "cuda"

batch_size = 6
num_unique_prompts = 2
prompt_len = 16
response_len = 8


def _valid_lengths(batch, prompt_len) -> tuple[list[int], list[int]]:
    """Per-row real prompt/response token counts from the attention mask (boundary fixed at column ``prompt_len``)."""
    mask = batch["attention_mask"].bool()
    prompt_lens = [int(mask[row, :prompt_len].sum()) for row in range(mask.shape[0])]
    response_lens = [int(mask[row, prompt_len:].sum()) for row in range(mask.shape[0])]
    return prompt_lens, response_lens


def _reference_response_logprobs(model, batch, prompt_lens, response_lens) -> torch.Tensor:
    """Per-row reference, packed in sample order: each row's real tokens are run alone (positions ``0..len-1``), and
    response token ``t`` is read from the logits at ``pl - 1 + t``. This is the value the deduplicated forward must
    reproduce."""
    rows = []
    with torch.no_grad():
        for row in range(batch["input_ids"].shape[0]):
            pl, rl = prompt_lens[row], response_lens[row]
            ids = batch["input_ids"][row][batch["attention_mask"][row].bool()].unsqueeze(0)
            log_probs = torch.log_softmax(model(input_ids=ids).logits.float(), dim=-1)[0]
            pred_idx = torch.arange(pl - 1, pl + rl - 1, device=ids.device)
            resp_tokens = ids[0, pl : pl + rl]
            rows.append(log_probs[pred_idx].gather(-1, resp_tokens.unsqueeze(-1)).squeeze(-1))
    return torch.cat(rows)


# The three logprob/entropy dispatch modes: "none" (full logits in one shot), "compute" (full logits, chunked
# follow-up) and "memory" (tiled compute under no_grad with a backward replay). All must produce the same result.
logits_optimization_modes = [("none",), ("memory",), ("compute",)]

def _build_case(model, prompt_length: int, add_padding: bool) -> dict:
    """Build a batch (deterministically) plus its per-row reference. Must run on the UNPATCHED model."""
    torch.manual_seed(0)
    case_batch = create_dummy_batch(
        batch_size=batch_size,
        num_unique_prompts=num_unique_prompts,
        prompt_len=prompt_length,
        response_len=response_len,
        device=device,
        include_training_fields=False,
        add_padding=add_padding,
    )
    case_prompt_lens, case_response_lens = _valid_lengths(case_batch, prompt_length)
    reference = _reference_response_logprobs(model, case_batch, case_prompt_lens, case_response_lens)
    return {
        "batch": case_batch,
        "prompt_lens": case_prompt_lens,
        "response_lens": case_response_lens,
        "reference": reference,
    }



class QwenOncePatcherTestMixin:
    model_name: str
    attn_implementation = "flash_attention_2"
    promptlen_padding_cases: list[tuple[int, bool]] = []

    @classmethod
    def setUpClass(cls):
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=cls.attn_implementation,
        )
        cls.model.eval()
        cls.length_cases = {
            (prompt_length, add_padding): _build_case(cls.model, prompt_length, add_padding)
            for prompt_length, add_padding in cls.promptlen_padding_cases
        }

    def _patch_and_forward(self, logits_optimization: str, calculate_entropy: bool, batch=None):
        Qwen3ModelOncePatcher(
            self.model,
            response_len=response_len,
            max_token_len=4096,
            rollout_n=batch_size // num_unique_prompts,
            temperature=1.0,
            logits_optimization=logits_optimization,
            world_size=1,
            use_unpad=True,
        ).patch_forward()
        return self.model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            calculate_entropy=calculate_entropy,
        )

    @parameterized.expand(logits_optimization_modes)
    def test_backward_produces_finite_gradients(self, logits_optimization):
        for prompt_length, add_padding in self.promptlen_padding_cases:
            with self.subTest(prompt_length=prompt_length, add_padding=add_padding):
                case = self.length_cases[(prompt_length, add_padding)]
                self.model.zero_grad(set_to_none=True)
                output = self._patch_and_forward(logits_optimization, calculate_entropy=False, batch=case["batch"])
                output.logprobs.mean().backward()
                grad = self.model.lm_head.weight.grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0.0)

    @parameterized.expand(logits_optimization_modes)
    def test_forward_matches_reference_across_prompt_lengths(self, logits_optimization):
        for prompt_length, add_padding in self.promptlen_padding_cases:
            with self.subTest(prompt_length=prompt_length, add_padding=add_padding):
                case = self.length_cases[(prompt_length, add_padding)]
                with torch.no_grad():
                    output = self._patch_and_forward(logits_optimization, calculate_entropy=True, batch=case["batch"])
                expected_response_tokens = sum(case["response_lens"])
                self.assertEqual(output.logprobs.shape, (expected_response_tokens,))
                self.assertTrue(torch.isfinite(output.logprobs).all())
                self.assertEqual(output.entropy.shape, (expected_response_tokens,))
                self.assertTrue(torch.isfinite(output.entropy).all())
                self.assertGreaterEqual(output.entropy.min().item(), 0.0)
                torch_assert_close(output.logprobs.float(), case["reference"], rtol=0, atol=1e-3)

@require_torch_gpu
class TestQwen3ModelOncePatcher(QwenOncePatcherTestMixin, TestCasePlus):
    model_name = "tiny-random/qwen3"
    # Full-attention-only model: the sweep just varies prompt length / padding (no gated-delta chunk boundary). The
    # tiny ``head_dim`` keeps the deduplicated/varlen forward bit-exact against the per-row reference, the same as for
    # the hybrid checkpoint below.
    promptlen_padding_cases = [
        (16, False),
    ]


@require_torch_gpu
class TestQwen3MoeModelOncePatcher(QwenOncePatcherTestMixin, TestCasePlus):
    model_name = "tiny-random/qwen3-moe"
    promptlen_padding_cases = [
        (16, False),
    ]

@require_torch_gpu
class TestQwen3NextModelOncePatcher(QwenOncePatcherTestMixin, TestCasePlus):
    model_name = "tiny-random/qwen3-next-moe"
    promptlen_padding_cases = [
        (16, False),
    ]

@require_torch_gpu
class TestQwen36ModelOncePatcher(QwenOncePatcherTestMixin, TestCasePlus):
    model_name = "tiny-random/qwen3.6"
    promptlen_padding_cases = [
        (16, False),  
        (64, False),  
        (160, False), 
        (160, True), 
    ]

@require_torch_gpu
class TestQwen36MoeModelOncePatcher(QwenOncePatcherTestMixin, TestCasePlus):
    model_name = "tiny-random/qwen3.6-moe"
    promptlen_padding_cases = [
        (16, False),
    ]
