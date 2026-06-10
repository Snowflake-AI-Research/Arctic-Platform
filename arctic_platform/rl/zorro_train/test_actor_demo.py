#!/usr/bin/env python3
"""
Quick test to verify actor.py and demo.py work correctly.
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
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
    actor = DeduplicatedActor(
        model_name,
        device=device,
        use_split_attention=True,
        attn_implementation="sdpa"
    )
    
    print("✓ Actor created successfully")
    
    # Create test batch
    batch = create_dummy_batch(
        batch_size=4,
        num_unique_prompts=2,
        prompt_len=16,
        response_len=8,
        device=device,
        include_training_fields=False
    )
    
    print("✓ Dummy batch created")
    
    # Run forward pass
    entropy, log_probs = actor._forward_micro_batch(
        batch,
        temperature=1.0,
        calculate_entropy=True
    )
    
    print(f"✓ Forward pass completed")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Entropy shape: {entropy.shape if entropy is not None else None}")
    
    # Check shapes
    assert log_probs.shape == (4, 8), f"Expected log_probs shape (4, 8), got {log_probs.shape}"
    if entropy is not None:
        assert entropy.shape == (4, 8), f"Expected entropy shape (4, 8), got {entropy.shape}"
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_actor()

