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
Test utilities for prompt deduplication.

Includes correctness tests, gradient comparison, and performance benchmarks.
"""

import time
from typing import Dict

import torch
import torch.nn as nn

from .actor import DeduplicatedActor
from .actor import packed_ppo_policy_loss


def create_dummy_batch(
    batch_size: int = 8,
    num_unique_prompts: int = 2,
    prompt_len: int = 8,
    response_len: int = 8,
    vocab_size: int = 32000,
    device: str = "cuda",
    include_training_fields: bool = False,
    add_padding: bool = False,
    min_valid_prompt_len: int = 5,
    min_valid_response_len: int = 5,
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Create a dummy batch with shared prompts for testing.

    Args:
        batch_size: Number of samples in the batch
        num_unique_prompts: Number of unique prompts (samples will be distributed across them)
        prompt_len: Prompt length (max length, actual may be less with padding)
        response_len: Response length (max length, actual may be less with padding)
        vocab_size: Vocabulary size for token generation
        device: Device to create tensors on
        include_training_fields: Whether to include fields needed for training (advantages, old_log_probs, etc.)
        add_padding: If True, add random padding (left for prompts, right for responses)
        min_valid_prompt_len: Minimum valid prompt tokens when padding is enabled
        min_valid_response_len: Minimum valid response tokens when padding is enabled
        pad_token_id: Token ID to use for padding

    Padding structure (matching verl's convention):
        [left_pad][valid_prompt][valid_response][right_pad]
        - Prompts: LEFT-padded (padding before prompt tokens)
        - Responses: RIGHT-padded (padding after response tokens)
    """

    # set seed
    # torch.manual_seed(42)

    # Create unique prompts (valid tokens, without left padding yet)
    unique_prompts_list = []

    for _ in range(num_unique_prompts):
        # Create valid prompt tokens (will be padded later if needed)
        valid_prompt_len = (
            prompt_len if not add_padding else torch.randint(min_valid_prompt_len, prompt_len + 1, (1,)).item()
        )
        prompt_tokens = torch.randint(1, vocab_size, (valid_prompt_len,), device=device, dtype=torch.long)
        unique_prompts_list.append(prompt_tokens)

    # Assign samples to prompts (some will share)
    samples_per_prompt = batch_size // num_unique_prompts

    input_ids_list = []
    responses_list = []
    attention_mask_list = []
    position_ids_list = []

    seq_len = prompt_len + response_len

    for prompt_idx in range(num_unique_prompts):
        for _ in range(samples_per_prompt):
            # Create valid response tokens
            valid_response_len = (
                response_len
                if not add_padding
                else torch.randint(min_valid_response_len, response_len + 1, (1,)).item()
            )
            response_tokens = torch.randint(1, vocab_size, (valid_response_len,), device=device, dtype=torch.long)

            # Get the prompt (shared across samples)
            prompt_tokens = unique_prompts_list[prompt_idx]
            valid_prompt_len = len(prompt_tokens)

            if add_padding:
                # Calculate padding amounts
                left_pad_len = prompt_len - valid_prompt_len
                right_pad_len = response_len - valid_response_len

                # Create padded sequence: [left_pad][prompt][response][right_pad]
                left_pad = torch.full((left_pad_len,), pad_token_id, device=device, dtype=torch.long)
                right_pad = torch.full((right_pad_len,), pad_token_id, device=device, dtype=torch.long)

                input_ids = torch.cat([left_pad, prompt_tokens, response_tokens, right_pad])

                # Create attention mask: 0 for padding, 1 for valid tokens
                attention_mask = torch.cat(
                    [
                        torch.zeros(left_pad_len, device=device, dtype=torch.long),
                        torch.ones(valid_prompt_len, device=device, dtype=torch.long),
                        torch.ones(valid_response_len, device=device, dtype=torch.long),
                        torch.zeros(right_pad_len, device=device, dtype=torch.long),
                    ]
                )

                # Create position IDs: 0 for left padding, incremental for valid, repeat last for right padding
                position_ids = torch.cat(
                    [
                        torch.zeros(left_pad_len, device=device, dtype=torch.long),
                        torch.arange(valid_prompt_len + valid_response_len, device=device, dtype=torch.long),
                        torch.full(
                            (right_pad_len,),
                            valid_prompt_len + valid_response_len - 1,
                            device=device,
                            dtype=torch.long,
                        ),
                    ]
                )

                # Create response tensor (padded)
                response = torch.cat([response_tokens, right_pad])
            else:
                # No padding - concatenate directly
                input_ids = torch.cat([prompt_tokens, response_tokens])
                attention_mask = torch.ones(seq_len, device=device, dtype=torch.long)
                position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
                response = response_tokens

            input_ids_list.append(input_ids)
            responses_list.append(response)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)

    # Handle remainder
    remainder = batch_size - len(input_ids_list)
    for i in range(remainder):
        prompt_idx = i % num_unique_prompts

        # Create valid response tokens
        valid_response_len = (
            response_len if not add_padding else torch.randint(min_valid_response_len, response_len + 1, (1,)).item()
        )
        response_tokens = torch.randint(1, vocab_size, (valid_response_len,), device=device, dtype=torch.long)

        # Get the prompt (shared)
        prompt_tokens = unique_prompts_list[prompt_idx]
        valid_prompt_len = len(prompt_tokens)

        if add_padding:
            left_pad_len = prompt_len - valid_prompt_len
            right_pad_len = response_len - valid_response_len

            left_pad = torch.full((left_pad_len,), pad_token_id, device=device, dtype=torch.long)
            right_pad = torch.full((right_pad_len,), pad_token_id, device=device, dtype=torch.long)

            input_ids = torch.cat([left_pad, prompt_tokens, response_tokens, right_pad])

            attention_mask = torch.cat(
                [
                    torch.zeros(left_pad_len, device=device, dtype=torch.long),
                    torch.ones(valid_prompt_len, device=device, dtype=torch.long),
                    torch.ones(valid_response_len, device=device, dtype=torch.long),
                    torch.zeros(right_pad_len, device=device, dtype=torch.long),
                ]
            )

            position_ids = torch.cat(
                [
                    torch.zeros(left_pad_len, device=device, dtype=torch.long),
                    torch.arange(valid_prompt_len + valid_response_len, device=device, dtype=torch.long),
                    torch.full(
                        (right_pad_len,), valid_prompt_len + valid_response_len - 1, device=device, dtype=torch.long
                    ),
                ]
            )

            response = torch.cat([response_tokens, right_pad])
        else:
            input_ids = torch.cat([prompt_tokens, response_tokens])
            attention_mask = torch.ones(seq_len, device=device, dtype=torch.long)
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
            response = response_tokens

        input_ids_list.append(input_ids)
        responses_list.append(response)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)

    input_ids = torch.stack(input_ids_list)
    responses = torch.stack(responses_list)
    position_ids = torch.stack(position_ids_list)
    attention_mask = torch.stack(attention_mask_list) if add_padding else torch.ones_like(input_ids)

    batch = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "responses": responses,
        "attention_mask": attention_mask,
    }

    # Add training-specific fields for backward pass
    if include_training_fields:

        # Old log probs (random for demo, only for valid tokens)
        batch["old_log_probs"] = torch.randn(batch_size, response_len, device=device) * 0.5 - 2.0

        # Advantages (random, some positive, some negative)
        batch["advantages"] = torch.randn(batch_size, response_len, device=device) * 0.5

        # Reference log probs (optional, slightly different from old_log_probs)
        batch["ref_log_prob"] = batch["old_log_probs"] + torch.randn(batch_size, response_len, device=device) * 0.1

    return batch


def baseline_response_logprobs(
    model: nn.Module,
    micro_batch: Dict[str, torch.Tensor],
    temperature: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Non-deduplicated response logprobs, packed 1D in original sample order (valid response tokens only).

    Runs a single full-batch forward on the **unpatched** model and extracts, per row, the logprob of each
    response token from the logits at the preceding position. The layout (sample order, padding removed) matches
    what :meth:`DeduplicatedActor.forward` returns, so the two can be compared 1:1.

    Must be called before the model is patched: :class:`Qwen3ModelOncePatcher` permanently rewrites the forward to
    return logprobs instead of ``logits``.
    """
    input_ids = micro_batch["input_ids"].to(device)
    attention_mask = micro_batch["attention_mask"].to(device)
    position_ids = micro_batch["position_ids"].to(device)
    response_len = micro_batch["responses"].size(-1)
    prompt_len = input_ids.shape[1] - response_len

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = outputs.logits  # [batch_size, seq_len, vocab]

    valid = attention_mask[:, prompt_len:].bool()
    rows = []
    for row in range(input_ids.shape[0]):
        valid_response_len = int(valid[row].sum())
        # response token t is predicted by the logits at column (prompt_len - 1 + t)
        pred_idx = torch.arange(prompt_len - 1, prompt_len - 1 + valid_response_len, device=device)
        # log_softmax only over the needed positions to avoid a full [batch, seq, vocab] intermediate.
        log_probs = torch.log_softmax(logits[row, pred_idx].float() / temperature, dim=-1)
        resp_tokens = input_ids[row, prompt_len : prompt_len + valid_response_len]
        rows.append(log_probs.gather(-1, resp_tokens.unsqueeze(-1)).squeeze(-1))
    return torch.cat(rows)


def compute_baseline_loss_and_backward(
    model: nn.Module,
    micro_batch: Dict[str, torch.Tensor],
    temperature: float = 1.0,
    gradient_accumulation: int = 1,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Baseline (no-deduplication) forward+backward, used for comparison/testing.

    Mirrors :meth:`DeduplicatedActor.compute_policy_loss_and_backward` but on the unpatched model and via a plain
    full-batch forward, sharing the same packed PPO objective so the two paths' gradients can be compared.

    Returns ``(metrics, baseline_logprobs)``; must run before the model is patched.
    """
    log_prob = baseline_response_logprobs(model, micro_batch, temperature=temperature, device=device)

    response_len = micro_batch["responses"].size(-1)
    valid = DeduplicatedActor._packed_response_validity(micro_batch, response_len).to(device)
    old_log_prob = micro_batch["old_log_probs"].to(device)[valid].float()
    advantages = micro_batch["advantages"].to(device)[valid].float()
    ref_log_prob = micro_batch["ref_log_prob"].to(device)[valid].float() if "ref_log_prob" in micro_batch else None

    policy_loss, metrics = packed_ppo_policy_loss(log_prob, old_log_prob, advantages, ref_log_prob)
    loss = policy_loss / gradient_accumulation
    loss.backward()
    metrics["actor/loss"] = loss.detach().item()

    return metrics, log_prob.detach()


def compare_gradients(model1: nn.Module, model2: nn.Module, name: str = "param") -> Dict[str, float]:
    """
    Compare gradients between two models.

    Returns:
        stats: Dict with comparison statistics
    """
    stats = {
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "mean_rel_diff": 0.0,
        "num_params": 0,
        "cosine_sim": 0.0,
    }

    total_abs_diff = 0.0
    total_rel_diff = 0.0
    num_params = 0

    all_grad1 = []
    all_grad2 = []

    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert n1 == n2, f"Parameter names don't match: {n1} vs {n2}"

        if p1.grad is None or p2.grad is None:
            continue

        grad1 = p1.grad
        grad2 = p2.grad

        # Absolute difference
        abs_diff = (grad1 - grad2).abs()
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()

        stats["max_abs_diff"] = max(stats["max_abs_diff"], max_abs)
        total_abs_diff += mean_abs

        # Relative difference
        denom = torch.maximum(grad1.abs(), grad2.abs())
        rel_diff = abs_diff / (denom + 1e-8)
        max_rel = rel_diff.max().item()
        mean_rel = rel_diff.mean().item()

        stats["max_rel_diff"] = max(stats["max_rel_diff"], max_rel)
        total_rel_diff += mean_rel

        num_params += 1

        # Collect for cosine similarity
        all_grad1.append(grad1.flatten())
        all_grad2.append(grad2.flatten())

    stats["num_params"] = num_params
    stats["mean_abs_diff"] = total_abs_diff / max(num_params, 1)
    stats["mean_rel_diff"] = total_rel_diff / max(num_params, 1)

    # Compute cosine similarity
    if all_grad1 and all_grad2:
        flat_grad1 = torch.cat(all_grad1)
        flat_grad2 = torch.cat(all_grad2)
        cosine_sim = torch.nn.functional.cosine_similarity(flat_grad1.unsqueeze(0), flat_grad2.unsqueeze(0)).item()
        stats["cosine_sim"] = cosine_sim

    return stats


def test_gradient_correctness(
    actor: DeduplicatedActor,
    batch_size: int = 8,
    num_unique_prompts: int = 2,
    prompt_len: int = 32,
    response_len: int = 16,
    device: str = "cuda",
    temperature: float = 1.0,
):
    """Test that the deduplicated forward/backward matches a non-deduplicated baseline (logprobs + gradients).

    NOTE: ``actor`` must be FRESH -- its model not yet patched. The baseline is computed first on the unpatched
    model; the deduplicated path then patches it permanently (Qwen3ModelOncePatcher).
    """
    print("=" * 80)
    print("Gradient Correctness Test")
    print("=" * 80)

    # Create training batch
    train_batch = create_dummy_batch(
        batch_size=batch_size,
        num_unique_prompts=num_unique_prompts,
        prompt_len=prompt_len,
        response_len=response_len,
        device=device,
        include_training_fields=True,
    )

    print("\nTest setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num unique prompts: {num_unique_prompts}")
    print(f"  Samples per prompt: {batch_size // num_unique_prompts}")

    # Per-response training fields, flattened to valid response tokens in sample order (shared by both paths).
    valid = DeduplicatedActor._packed_response_validity(train_batch, response_len).to(device)
    old_log_prob = train_batch["old_log_probs"].to(device)[valid].float()
    advantages = train_batch["advantages"].to(device)[valid].float()
    ref_log_prob = train_batch["ref_log_prob"].to(device)[valid].float() if "ref_log_prob" in train_batch else None

    # Baseline FIRST, on the still-unpatched model.
    print("\n[1/3] Running baseline forward+backward (unpatched model)...")
    actor.train()
    actor.model.zero_grad(set_to_none=True)

    baseline_logprobs = baseline_response_logprobs(actor.model, train_batch, temperature=temperature, device=device)
    policy_loss_baseline, metrics_baseline = packed_ppo_policy_loss(
        baseline_logprobs, old_log_prob, advantages, ref_log_prob
    )
    policy_loss_baseline.backward()
    metrics_baseline["actor/loss"] = policy_loss_baseline.detach().item()

    grads_baseline = {}
    grad_norm_baseline = 0.0
    for name, param in actor.model.named_parameters():
        if param.grad is not None:
            grads_baseline[name] = param.grad.clone()
            grad_norm_baseline += param.grad.data.norm(2).item() ** 2
    grad_norm_baseline = grad_norm_baseline**0.5
    baseline_logprobs = baseline_logprobs.detach()

    print(f"  Loss: {metrics_baseline['actor/loss']:.6f}")
    print(f"  Gradient norm: {grad_norm_baseline:.6f}")

    # Deduplicated path (this patches the model permanently).
    print("\n[2/3] Running deduplicated forward+backward...")
    actor.model.zero_grad(set_to_none=True)

    dedup_output = actor.forward(train_batch, temperature=temperature, calculate_entropy=False)
    dedup_logprobs = dedup_output.logprobs.float()
    policy_loss_dedup, metrics_dedup = packed_ppo_policy_loss(dedup_logprobs, old_log_prob, advantages, ref_log_prob)
    policy_loss_dedup.backward()
    metrics_dedup["actor/loss"] = policy_loss_dedup.detach().item()

    grads_dedup = {}
    grad_norm_dedup = 0.0
    for name, param in actor.model.named_parameters():
        if param.grad is not None:
            grads_dedup[name] = param.grad.clone()
            grad_norm_dedup += param.grad.data.norm(2).item() ** 2
    grad_norm_dedup = grad_norm_dedup**0.5
    dedup_logprobs = dedup_logprobs.detach()

    print(f"  Loss: {metrics_dedup['actor/loss']:.6f}")
    print(f"  Num params with grad: {len(grads_dedup)}")
    print(f"  Gradient norm: {grad_norm_dedup:.6f}")

    # Compare deduplicated vs baseline response logprobs (both packed 1D in sample order).
    print("\n[3/3] Comparing deduplicated logprobs with baseline...")
    print(f"  Dedup logprobs shape:    {tuple(dedup_logprobs.shape)}")
    print(f"  Baseline logprobs shape: {tuple(baseline_logprobs.shape)}")

    if dedup_logprobs.shape == baseline_logprobs.shape:
        logprobs_diff = (dedup_logprobs - baseline_logprobs).abs()
        max_logprobs_diff = logprobs_diff.max().item()
        mean_logprobs_diff = logprobs_diff.mean().item()

        baseline_magnitude = baseline_logprobs.abs().mean().item()
        rel_logprobs_diff = mean_logprobs_diff / (baseline_magnitude + 1e-8) * 100

        flat_dedup = dedup_logprobs.flatten()
        flat_baseline = baseline_logprobs.flatten()
        logprobs_cosine_sim = torch.nn.functional.cosine_similarity(
            flat_dedup.unsqueeze(0), flat_baseline.unsqueeze(0)
        ).item()

        print(f"  Max absolute diff:   {max_logprobs_diff:.2e}")
        print(f"  Mean absolute diff:  {mean_logprobs_diff:.2e}")
        print(f"  Mean relative diff:  {rel_logprobs_diff:.2f}%")
        print(f"  Cosine similarity:   {logprobs_cosine_sim:.6f}")

        if mean_logprobs_diff < 1e-4 and logprobs_cosine_sim > 0.9999:
            print("  ✓ Deduplicated logprobs match baseline perfectly!")
        elif mean_logprobs_diff < 1e-3 and logprobs_cosine_sim > 0.999:
            print("  ~ Deduplicated logprobs are very close to baseline (minor numerical differences)")
        else:
            print("  ✗ Significant differences in deduplicated logprobs!")
    else:
        print("  ✗ Shape mismatch! Cannot compare.")

    # Compare metrics
    print("\n" + "-" * 80)
    print("Metric Comparison:")
    print("-" * 80)
    for key in metrics_dedup.keys():
        dedup_val = metrics_dedup[key]
        baseline_val = metrics_baseline[key]
        diff = abs(dedup_val - baseline_val)
        rel_diff = diff / (abs(baseline_val) + 1e-8) * 100
        match = "✓" if diff < 1e-4 else "✗"
        print(
            f"  {match} {key:30s}: dedup={dedup_val:.6f}, baseline={baseline_val:.6f},"
            f" diff={diff:.2e} ({rel_diff:.2f}%)"
        )

    # Compare gradients
    print("\n" + "-" * 80)
    print("Gradient Comparison:")
    print("-" * 80)

    # Compare gradient norms
    grad_norm_diff = abs(grad_norm_dedup - grad_norm_baseline)
    grad_norm_rel_diff = grad_norm_diff / (grad_norm_baseline + 1e-8) * 100
    print(f"  Gradient norm (dedup):    {grad_norm_dedup:.6f}")
    print(f"  Gradient norm (baseline): {grad_norm_baseline:.6f}")
    print(f"  Norm difference:          {grad_norm_diff:.2e} ({grad_norm_rel_diff:.2f}%)")
    print()

    # Compare saved gradients element-wise
    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    max_rel_diff = 0.0
    mean_rel_diff = 0.0
    total_elements = 0
    dot_product = 0.0
    norm_dedup_sq = 0.0
    norm_baseline_sq = 0.0

    for name in grads_dedup.keys():
        if name in grads_baseline:
            grad_d = grads_dedup[name]
            grad_b = grads_baseline[name]

            abs_diff = (grad_d - grad_b).abs()
            max_abs_diff = max(max_abs_diff, abs_diff.max().item())
            mean_abs_diff += abs_diff.sum().item()

            rel_diff = abs_diff / (grad_b.abs() + 1e-8)
            max_rel_diff = max(max_rel_diff, rel_diff.max().item())
            mean_rel_diff += rel_diff.sum().item()

            total_elements += grad_d.numel()

            # For cosine similarity
            dot_product += (grad_d * grad_b).sum().item()
            norm_dedup_sq += (grad_d**2).sum().item()
            norm_baseline_sq += (grad_b**2).sum().item()

    mean_abs_diff /= total_elements
    mean_rel_diff /= total_elements
    cosine_sim = dot_product / (torch.sqrt(torch.tensor(norm_dedup_sq * norm_baseline_sq)).item() + 1e-8)

    print(f"  Params compared: {len(grads_dedup)}")
    print(f"  Max absolute diff: {max_abs_diff:.2e}")
    print(f"  Mean absolute diff: {mean_abs_diff:.2e}")
    print(f"  Max relative diff: {max_rel_diff:.2%}")
    print(f"  Mean relative diff: {mean_rel_diff:.2%}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")

    # Verdict.
    #
    # Element-wise cosine over *all* gradients is brittle in low precision: most elements are ~0 and bf16 rounding
    # flips their sign, which drags cosine down even when the gradients agree (e.g. on a tiny-random model the grad
    # norms match to <0.1% yet cosine sits at ~0.998). So we accept either a tight element-wise match OR a tight
    # relative gradient-norm match. The separate logprob comparison above (atol-based) is the precise forward check.
    print("\n" + "=" * 80)
    if mean_abs_diff < 1e-4 and cosine_sim > 0.9999:
        print("✓ PASS: Gradients match between deduplicated and baseline!")
        return True
    elif cosine_sim > 0.99 and (mean_abs_diff < 1e-3 or grad_norm_rel_diff < 1.0):
        print("~ CLOSE: Gradients are very similar (minor numerical differences)")
        return True
    else:
        print("✗ FAIL: Significant gradient differences detected")
        return False


def benchmark_performance(
    actor: DeduplicatedActor,
    batch_size: int = 10,
    prompt_len: int = 10000,
    response_len: int = 1000,
    num_unique_prompts: int = 1,
    num_warmup: int = 2,
    num_runs: int = 5,
    device: str = "cuda",
):
    """
    Benchmark forward+backward time for deduplicated vs baseline.

    NOTE: ``actor`` must be FRESH -- the baseline is timed first on the unpatched model, then the deduplicated path
    patches it permanently (Qwen3ModelOncePatcher rewrites the forward to return logprobs instead of logits).
    """
    print("=" * 80)
    print("Performance Benchmark")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Prompt length: {prompt_len}")
    print(f"  Response length: {response_len}")
    print(f"  Total sequence length: {prompt_len + response_len}")
    print(f"  Unique prompts: {num_unique_prompts}")
    print(f"  Samples per prompt: {batch_size // num_unique_prompts}")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Timed iterations: {num_runs}")

    # Create large batch
    print("\nCreating benchmark batch...")
    batch = create_dummy_batch(
        batch_size=batch_size,
        num_unique_prompts=num_unique_prompts,
        prompt_len=prompt_len,
        response_len=response_len,
        device=device,
        include_training_fields=True,
    )

    total_tokens = batch_size * (prompt_len + response_len)
    print(f"  Total tokens in batch: {total_tokens:,}")

    actor.train()

    # Enable gradient checkpointing for memory efficiency
    print("\nEnabling gradient checkpointing...")
    actor.model.gradient_checkpointing_enable()

    # Benchmark baseline approach FIRST -- it needs the unpatched model (the dedup path patches it permanently).
    print("\n[1/2] Benchmarking BASELINE forward+backward (unpatched model)...")

    # Warmup
    for i in range(num_warmup):
        actor.model.zero_grad()
        _ = compute_baseline_loss_and_backward(
            actor.model, batch, temperature=1.0, gradient_accumulation=1, device=device
        )
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    baseline_times = []
    for i in range(num_runs):
        actor.model.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        _ = compute_baseline_loss_and_backward(
            actor.model, batch, temperature=1.0, gradient_accumulation=1, device=device
        )

        if device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0
        else:
            elapsed = time.perf_counter() - start_time

        baseline_times.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")

    baseline_mean = sum(baseline_times) / len(baseline_times)
    baseline_std = (sum((t - baseline_mean) ** 2 for t in baseline_times) / len(baseline_times)) ** 0.5

    # Benchmark deduplicated approach (patches the model on the first call).
    print("\n[2/2] Benchmarking DEDUPLICATED forward+backward...")

    # Warmup
    for i in range(num_warmup):
        actor.model.zero_grad()
        _ = actor.compute_policy_loss_and_backward(batch, temperature=1.0, gradient_accumulation=1)
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    dedup_times = []
    for i in range(num_runs):
        actor.model.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        _ = actor.compute_policy_loss_and_backward(batch, temperature=1.0, gradient_accumulation=1)

        if device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        else:
            elapsed = time.perf_counter() - start_time

        dedup_times.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")

    dedup_mean = sum(dedup_times) / len(dedup_times)
    dedup_std = (sum((t - dedup_mean) ** 2 for t in dedup_times) / len(dedup_times)) ** 0.5

    # Summary
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    print("\nDeduplicated approach:")
    print(f"  Mean time: {dedup_mean:.3f}s (± {dedup_std:.3f}s)")
    print(f"  Throughput: {total_tokens / dedup_mean:,.0f} tokens/sec")

    print("\nBaseline approach:")
    print(f"  Mean time: {baseline_mean:.3f}s (± {baseline_std:.3f}s)")
    print(f"  Throughput: {total_tokens / baseline_mean:,.0f} tokens/sec")

    speedup = baseline_mean / dedup_mean
    time_saved = baseline_mean - dedup_mean
    time_saved_pct = (time_saved / baseline_mean) * 100

    print("\nSpeedup:")
    print(f"  {speedup:.2f}x faster")
    print(f"  Time saved: {time_saved:.3f}s ({time_saved_pct:.1f}%)")

    # Expected speedup (theoretical)
    unique_prompt_tokens = num_unique_prompts * prompt_len
    all_prompt_tokens = batch_size * prompt_len
    tokens_saved = all_prompt_tokens - unique_prompt_tokens
    tokens_saved_pct = (tokens_saved / total_tokens) * 100

    print("\nTheoretical analysis:")
    print(f"  Tokens saved by deduplication: {tokens_saved:,} / {total_tokens:,} ({tokens_saved_pct:.1f}%)")
    print(f"  Expected speedup (approximate): {1 / (1 - tokens_saved_pct/100):.2f}x")

    if speedup > 1.1:
        print(f"\n✓ Deduplication provides {speedup:.2f}x speedup!")
    elif speedup > 0.95:
        print("\n~ Deduplication has similar performance (overhead may offset savings for small batches)")
    else:
        print("\n✗ Deduplication is slower (likely due to implementation overhead)")

    print("=" * 80)
