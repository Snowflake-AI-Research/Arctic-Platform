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

"""
Actor implementation with prompt deduplication.

Thin reference harness around :class:`Qwen3ModelOncePatcher` -- the production ZoRRO patcher installed by the
DeepSpeed worker. It loads a Qwen3 checkpoint, installs the patcher once, and exposes a deduplicated forward plus
a PPO train step.

The patched model takes the *full* ``[batch_size, seq_len]`` batch and returns per-response-token
``logprobs`` / ``entropy`` that are **packed into 1D in the original sample order** (padding removed). This differs
from a plain HF model, which returns ``[batch_size, seq_len, vocab]`` logits.

See ``tests/zorro_train/test_once_patcher.py`` for the same patcher driven directly.
"""

from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput

from .qwen_model_patcher import Qwen3ModelOncePatcher
from .zorro_train import ZoRRoTrain


def packed_ppo_policy_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    ref_log_prob: Optional[torch.Tensor] = None,
    clip_ratio: float = 0.2,
    kl_loss_coef: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """PPO clipped policy loss over packed (1D) response tokens.

    All inputs are 1D, aligned 1:1 over the same response tokens (token-mean reduction). Shared by
    :class:`DeduplicatedActor` and the non-deduplicated baseline in ``tests.py`` so both paths optimize an
    *identical* objective -- the basis for the gradient-equivalence check.

    Returns ``(policy_loss, metrics)``; ``policy_loss`` is *before* gradient-accumulation scaling (the caller
    divides and adds ``actor/loss`` after).
    """
    # PPO clipped objective (verl/trainer/ppo/core_algos.py::compute_policy_loss).
    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()

    policy_loss = pg_loss
    kl_loss = None
    if ref_log_prob is not None:
        kl_loss = (log_prob - ref_log_prob).mean()
        policy_loss = policy_loss + kl_loss * kl_loss_coef

    metrics = {
        "actor/pg_loss": pg_loss.detach().item(),
        "actor/policy_loss": policy_loss.detach().item(),
    }
    clipfrac = ((ratio - 1.0).abs() > clip_ratio).float().mean()
    metrics["actor/pg_clipfrac"] = clipfrac.detach().item()
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    metrics["actor/ppo_kl"] = approx_kl.detach().item()
    if kl_loss is not None:
        metrics["actor/kl_loss"] = kl_loss.detach().item()

    return policy_loss, metrics


class DeduplicatedActor:
    """Actor that runs deduplicated forward/backward via :class:`Qwen3ModelOncePatcher`."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        logits_optimization: str = "none",
        use_split_attention: bool = True,
        attn_implementation: str = "eager",
        world_size: int = 1,
        max_token_len: int = 4096,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize actor with prompt deduplication.

        Args:
            model_name_or_path: Hugging Face model identifier or local path (a Qwen3 family checkpoint).
            device: Device to load the model on.
            logits_optimization: logprob/entropy dispatch, one of ``"none"`` | ``"memory"`` | ``"compute"``.
                ``"memory"`` requires an initialized ``torch.distributed`` process group (the production
                DeepSpeed worker always has one); ``"none"`` / ``"compute"`` do not.
            use_split_attention: Use split attention (prompt-to-prompt + response-to-full) vs. a single
                reconstructed attention call.
            attn_implementation: Attention implementation (``eager``, ``flash_attention_2``, ...). ``flash_attention_2``
                requires a GPU + ``flash-attn``; ``eager`` works everywhere.
            world_size: Data-parallel world size; only used to sync shard counts in ``"memory"`` mode.
            max_token_len: Reserved (forwarded to the patcher; currently unused by it).
            dtype: Model dtype.
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(device)
        self.model.eval()

        self.logits_optimization = logits_optimization
        self.use_split_attention = use_split_attention
        self.world_size = world_size
        self.max_token_len = max_token_len

    def train(self):
        """Set model to training mode."""
        self.model.train()

    def eval(self):
        """Set model to eval mode."""
        self.model.eval()

    def patch(self, response_len: int, rollout_n: int, temperature: float) -> Qwen3ModelOncePatcher:
        """(Re)install :class:`Qwen3ModelOncePatcher` onto the model.

        The patcher is applied "once" in the sense that it permanently mutates the model's ``forward`` methods.
        Re-applying it just swaps the forward closures (e.g. for a new ``temperature``), which is exactly how
        ``tests/zorro_train/test_once_patcher.py`` re-patches a single cached model across cases.
        """
        patcher = Qwen3ModelOncePatcher(
            self.model,
            response_len=response_len,
            max_token_len=self.max_token_len,
            rollout_n=rollout_n,
            temperature=temperature,
            logits_optimization=self.logits_optimization,
            world_size=self.world_size,
            use_unpad=True,
            use_split_attention=self.use_split_attention,
        )
        patcher.patch_forward()
        return patcher

    def forward(
        self, micro_batch: Dict[str, torch.Tensor], temperature: float = 1.0, calculate_entropy: bool = False
    ) -> ModelOutput:
        """Deduplicated forward pass.

        Args:
            micro_batch: Dict with ``input_ids`` / ``position_ids`` / ``attention_mask`` ``[batch_size, seq_len]``
                and ``responses`` ``[batch_size, response_len]`` (e.g. from ``create_dummy_batch``).
            temperature: Temperature for scaling logits.
            calculate_entropy: Whether to also compute per-token entropy.

        Returns:
            ``ModelOutput`` with ``logprobs`` (and ``entropy`` if requested): 1D tensors of shape
            ``[num_valid_response_tokens]``, packed in the original sample order with padding removed.
        """
        input_ids = micro_batch["input_ids"].to(self.device)
        position_ids = micro_batch["position_ids"].to(self.device)
        attention_mask = micro_batch["attention_mask"].to(self.device)
        response_len = micro_batch["responses"].size(-1)

        # rollout_n is the number of responses sharing a prompt; the patcher only stores it, but we derive it from
        # the batch to keep this a faithful usage example.
        prompt_groups, _ = ZoRRoTrain.find_prompt_groups(input_ids, response_len)
        rollout_n = input_ids.size(0) // max(1, len(prompt_groups))

        self.patch(response_len=response_len, rollout_n=rollout_n, temperature=temperature)

        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            calculate_entropy=calculate_entropy,
        )

    def _forward_micro_batch(
        self, micro_batch: Dict[str, torch.Tensor], temperature: float = 1.0, calculate_entropy: bool = False
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Convenience wrapper returning ``(entropy, log_probs)`` (both packed 1D, see :meth:`forward`).

        ``entropy`` is ``None`` when ``calculate_entropy=False`` (``ModelOutput`` drops ``None`` fields, so we read
        it via membership rather than attribute access).
        """
        output = self.forward(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
        entropy = output["entropy"] if "entropy" in output else None
        return entropy, output["logprobs"]

    @staticmethod
    def _packed_response_validity(micro_batch: Dict[str, torch.Tensor], response_len: int) -> torch.Tensor:
        """Boolean ``[batch_size, response_len]`` mask of valid (non-pad) response tokens.

        Indexing a per-response field with this mask flattens it row-major, which matches the sample order the
        patcher returns ``logprobs`` / ``entropy`` in.
        """
        attention_mask = micro_batch["attention_mask"]
        prompt_len = micro_batch["input_ids"].shape[1] - response_len
        return attention_mask[:, prompt_len:].bool()

    def compute_policy_loss_and_backward(
        self,
        micro_batch: Dict[str, torch.Tensor],
        temperature: float = 1.0,
        gradient_accumulation: int = 1,
    ) -> Dict[str, float]:
        """
        Compute the PPO clipped policy loss with deduplication and run the backward pass.

        Works in the packed token space the patcher returns: the per-response training fields
        (``old_log_probs`` / ``advantages`` / optional ``ref_log_prob``, each ``[batch_size, response_len]``) are
        flattened to valid response tokens in sample order to align 1:1 with the returned ``log_probs``.

        Returns:
            metrics: Dict with loss values and PPO statistics.
        """
        self.train()

        _, log_prob = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=False)

        response_len = micro_batch["responses"].size(-1)
        valid = self._packed_response_validity(micro_batch, response_len).to(self.device)
        old_log_prob = micro_batch["old_log_probs"].to(self.device)[valid]
        advantages = micro_batch["advantages"].to(self.device)[valid]
        ref_log_prob = micro_batch["ref_log_prob"].to(self.device)[valid] if "ref_log_prob" in micro_batch else None

        policy_loss, metrics = packed_ppo_policy_loss(log_prob, old_log_prob, advantages, ref_log_prob)
        loss = policy_loss / gradient_accumulation

        # The model is already permanently patched, so the deduplicated graph was built during the forward above --
        # a plain backward suffices (no patching context manager, unlike the old Qwen3ModelPatcher path).
        loss.backward()
        metrics["actor/loss"] = loss.detach().item()

        return metrics
