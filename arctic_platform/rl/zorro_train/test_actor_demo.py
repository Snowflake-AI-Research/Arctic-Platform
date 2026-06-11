#!/usr/bin/env python3

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
Quick test to verify actor.py and demo.py work correctly.
"""

import os
import sys

import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

if True:  # deal with sys.path adjustment
    from arctic_platform.rl.zorro_train.actor import DeduplicatedActor
    from arctic_platform.rl.zorro_train.tests import create_dummy_batch


def test_actor():
    """Test that actor can be instantiated and run a forward pass."""
    print("Testing DeduplicatedActor...")

    # Use a small model
    model_name = "Qwen/Qwen2.5-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model: {model_name}")

    # Create actor
    actor = DeduplicatedActor(model_name, device=device, use_split_attention=True, attn_implementation="sdpa")

    print("✓ Actor created successfully")

    # Create test batch
    batch = create_dummy_batch(
        batch_size=4, num_unique_prompts=2, prompt_len=16, response_len=8, device=device, include_training_fields=False
    )

    print("✓ Dummy batch created")

    # Run forward pass
    entropy, log_probs = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=True)

    print("✓ Forward pass completed")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Entropy shape: {entropy.shape if entropy is not None else None}")

    # Check shapes
    assert log_probs.shape == (4, 8), f"Expected log_probs shape (4, 8), got {log_probs.shape}"
    if entropy is not None:
        assert entropy.shape == (4, 8), f"Expected entropy shape (4, 8), got {entropy.shape}"

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_actor()
