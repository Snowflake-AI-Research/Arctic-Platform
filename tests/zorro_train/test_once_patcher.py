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
across a set of small ``tiny-random`` checkpoints (dense, MoE, and hybrid linear + full attention), both the
``flash_attention_2`` and ``eager`` attention implementations, padded and unpadded batches, and all three
``logits_optimization`` dispatch modes (``none`` / ``memory`` / ``compute``)::

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

import functools
import torch
import torch.distributed as dist
from parameterized import parameterized
from transformers import AutoModelForCausalLM

from arctic_platform.rl.zorro_train.qwen_model_patcher import Qwen3ModelOncePatcher
from arctic_platform.rl.zorro_train.tests import create_dummy_batch
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import torch_assert_close
import itertools

device = "cuda"

batch_size = 6
num_unique_prompts = 2
prompt_len = 16
response_len = 8


def _valid_lengths(batch) -> tuple[list[int], list[int]]:
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
model_names = [
    "tiny-random/qwen3",
    "tiny-random/qwen3-moe",
    "tiny-random/qwen3-next-moe",
    "tiny-random/qwen3.6",
    "tiny-random/qwen3.6-moe",
]
attn_implementations = ["flash_attention_2", "eager"]
add_padding_modes = [True, False]

model_attn_logits_cases = list(itertools.product(model_names, attn_implementations, add_padding_modes, logits_optimization_modes))


@functools.lru_cache(maxsize=None)
def _load_model_and_reference(model_name: str, attn_implementation: str, add_padding: bool):
    """Load a checkpoint and build its (deterministic) batch + per-row reference, cached per case.

    Reference must be computed before patching (the Once patcher mutates the model permanently). Re-patching with a
    different logits_optimization just swaps the forward closures, so a single cached model serves every mode (and
    both the forward and backward test) for a given (model, attn implementation, padding) case."""
    torch.manual_seed(0)
    batch = create_dummy_batch(
        batch_size=batch_size,
        num_unique_prompts=num_unique_prompts,
        prompt_len=prompt_len,
        response_len=response_len,
        device=device,
        include_training_fields=False,
        add_padding=add_padding,
    )
    prompt_lens, response_lens = _valid_lengths(batch)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device, attn_implementation=attn_implementation
    )
    model.eval()
    reference_logprobs = _reference_response_logprobs(model, batch, prompt_lens, response_lens)
    return model, batch, response_lens, reference_logprobs



@require_torch_gpu
class TestQwen3ModelOncePatcher(TestCasePlus):
    @classmethod
    def setUpClass(cls):
        # The "memory" logits-optimization path calls torch.distributed.get_rank(), so it needs a process group --
        # which the production DeepSpeed worker always has. A full pytest run can reach here after the GPU RL tests
        # have torn down the session group, so stand up a single-rank one if missing (and restore on teardown).
        cls._created_process_group = not dist.is_initialized()
        if cls._created_process_group:
            dist.init_process_group(backend="gloo", world_size=1, rank=0, store=dist.HashStore())

    @classmethod
    def tearDownClass(cls):
        if cls._created_process_group and dist.is_initialized():
            dist.destroy_process_group()

    def _patch_and_forward(self, model, batch, logits_optimization: str, calculate_entropy: bool):
        Qwen3ModelOncePatcher(
            model,
            response_len=response_len,
            max_token_len=4096,
            rollout_n=batch_size // num_unique_prompts,
            temperature=1.0,
            logits_optimization=logits_optimization,
            world_size=1,
            use_unpad=True,
        ).patch_forward()
        return model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            calculate_entropy=calculate_entropy,
        )

    @parameterized.expand(model_attn_logits_cases)
    def test_forward_matches_reference(self, model_name, attn_implementation, add_padding, logits_optimization):
        model, batch, response_lens, reference_logprobs = _load_model_and_reference(
            model_name, attn_implementation, add_padding
        )
        num_response_tokens = sum(response_lens)
        with torch.no_grad():
            output = self._patch_and_forward(model, batch, logits_optimization, calculate_entropy=True)

        self.assertEqual(output.logprobs.shape, (num_response_tokens,))
        self.assertTrue(torch.isfinite(output.logprobs).all())
        self.assertEqual(output.entropy.shape, (num_response_tokens,))
        self.assertTrue(torch.isfinite(output.entropy).all())
        self.assertGreaterEqual(output.entropy.min().item(), 0.0)
        # Against the per-row reference the deduplicated forward is numerically exact up to bf16 rounding, so a
        # tight atol catches any regression in the dedup alignment or math (e.g. the first-response-token error the
        # offset-aware extraction fixes).
        torch_assert_close(output.logprobs.float(), reference_logprobs, rtol=0, atol=1e-3)

    @parameterized.expand(model_attn_logits_cases)
    def test_backward_produces_finite_gradients(self, model_name, attn_implementation, add_padding, logits_optimization):
        model, batch, _, _ = _load_model_and_reference(model_name, attn_implementation, add_padding)
        model.zero_grad(set_to_none=True)
        output = self._patch_and_forward(model, batch, logits_optimization, calculate_entropy=False)
        output.logprobs.mean().backward()
        grad = model.lm_head.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)
