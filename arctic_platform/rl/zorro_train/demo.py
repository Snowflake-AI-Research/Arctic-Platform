"""
Demo script for prompt deduplication optimization.

Run this script to test the implementation.
"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
from arctic_platform.rl.zorro_train.actor import DeduplicatedActor
from arctic_platform.rl.zorro_train.tests import (
    create_dummy_batch,
    test_gradient_correctness,
    benchmark_performance
)


def main():
    """Run all demos."""
    print("=" * 80)
    print("Prompt Deduplication Optimization Demo")
    print("=" * 80)
    
    # Setup
    model_name = "Qwen/Qwen3-4B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "sdpa"  # Use SDPA for compatibility (flash_attention_2 or flash_attention_3 also work)
    
    print(f"\nDevice: {device}")
    print(f"Model: {model_name}")
    print(f"Attention: {attn_impl}")
    
    try:
        actor = DeduplicatedActor(
            model_name, 
            device=device,
            use_split_attention=True,  # Enable split attention optimization
            attn_implementation=attn_impl
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nPlease check that the model is available and your environment is set up correctly")
        import traceback
        traceback.print_exc()
        return
    
    # Demo 1: Simple forward pass
    print("\n" + "=" * 80)
    print("Demo 1: Forward Pass with Deduplication")
    print("=" * 80)
    
    batch = create_dummy_batch(
        batch_size=8,
        num_unique_prompts=2,
        prompt_len=32,
        response_len=16,
        device=device,
        include_training_fields=False
    )
    
    print("\nRunning forward pass...")
    entropy, log_probs = actor._forward_micro_batch(
        batch,
        temperature=1.0,
        calculate_entropy=True
    )
    
    print(f"\nResults:")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Log probs mean: {log_probs.mean().item():.4f}")
    if entropy is not None:
        print(f"  Entropy shape: {entropy.shape}")
        print(f"  Entropy mean: {entropy.mean().item():.4f}")
    
    # Demo 2: Gradient correctness test
    print("\n" + "=" * 80)
    print("Demo 2: Gradient Correctness Test")
    print("=" * 80)
    
    passed = test_gradient_correctness(
        actor=actor,
        batch_size=8,
        num_unique_prompts=2,
        prompt_len=32,
        response_len=16,
        device=device,
    )
    
    if not passed:
        print("\nWarning: Gradient test did not pass. Implementation may have issues.")
        return
    
    # Demo 3: Performance benchmark (optional)
    print("\n" + "=" * 80)
    print("Demo 3: Performance Benchmark")
    print("=" * 80)
    print("\nThis benchmark tests:")
    print("  - 10 samples with the SAME 10K-token prompt")
    print("  - Each has a different 1K-token response")
    print("  - Total: 110K tokens")
    print("\nDeduplication should save ~81% of computation")
    
    user_input = input("\nRun performance benchmark? (y/n): ")
    if user_input.lower() == 'y':
        benchmark_performance(
            actor=actor,
            batch_size=4,
            prompt_len=8192,
            response_len=1000,
            num_unique_prompts=1,
            num_warmup=1,
            num_runs=1,
            device=device,
        )
    else:
        print("Skipping benchmark.")
    
    print("\n" + "=" * 80)
    print("All demos completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

