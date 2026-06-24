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
Demo script for ZoRRO prompt deduplication via :class:`Qwen3ModelOncePatcher`.

Loads a Qwen3 checkpoint and exercises the deduplicated forward/backward through :class:`DeduplicatedActor` and the
helpers in ``tests.py``:

1. ``test_gradient_correctness`` -- deduplicated forward/backward vs a non-deduplicated baseline (logprobs + grads).
2. ``benchmark_performance`` -- optional dedup-vs-baseline timing.

The Once patcher mutates the model permanently, and the baseline must run on the *unpatched* model, so each helper
needs its own fresh actor (hence the second model load for the benchmark).

Run::

    python arctic_platform/rl/zorro_train/demo.py
"""

import os
import sys

import torch

# Add the repo root to sys.path so this file runs directly (python .../demo.py).
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

if True:  # deal with sys.path adjustment
    from arctic_platform.rl.zorro_train.actor import DeduplicatedActor
    from arctic_platform.rl.zorro_train.tests import benchmark_performance
    from arctic_platform.rl.zorro_train.tests import test_gradient_correctness


def build_actor(model_name, device, attn_impl, dtype):
    """Load a fresh (unpatched) DeduplicatedActor."""
    return DeduplicatedActor(
        model_name,
        device=device,
        logits_optimization="none",  # "none"/"compute" need no process group; "memory" would.
        use_split_attention=True,
        attn_implementation=attn_impl,
        dtype=dtype,
    )


def main():
    """Run the ZoRRO deduplication demos."""
    print("=" * 80)
    print("ZoRRO Prompt Deduplication Demo (Qwen3ModelOncePatcher)")
    print("=" * 80)

    model_name = os.environ.get("ZORRO_DEMO_MODEL", "Qwen/Qwen3-0.6B")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # eager runs everywhere (CPU/GPU, no flash-attn dependency). The Once patcher also supports
    # flash_attention_2 (the production GPU path); sdpa is not supported by its attention patcher.
    attn_impl = os.environ.get("ZORRO_DEMO_ATTN", "eager")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"\nDevice: {device}")
    print(f"Model: {model_name}")
    print(f"Attention: {attn_impl}")

    # Demo 1: gradient/logprob correctness vs a non-deduplicated baseline (needs a fresh actor).
    print("\n" + "=" * 80)
    print("Demo 1: Gradient/Logprob Correctness vs Baseline")
    print("=" * 80)
    try:
        actor = build_actor(model_name, device, attn_impl, dtype)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nPlease check that the model is available and your environment is set up correctly.")
        import traceback

        traceback.print_exc()
        return

    passed = test_gradient_correctness(
        actor=actor,
        batch_size=6,
        num_unique_prompts=2,
        prompt_len=32,
        response_len=16,
        device=device,
    )
    if not passed:
        print("\nWarning: correctness check did not pass. Implementation may have issues.")

    # Demo 2: optional performance benchmark (needs a second fresh actor -- the first is now patched).
    print("\n" + "=" * 80)
    print("Demo 2: Performance Benchmark (optional)")
    print("=" * 80)
    print("\nBenchmarks deduplicated vs baseline forward+backward on a batch of shared prompts.")
    if input("\nRun performance benchmark (loads a second copy of the model)? (y/n): ").lower() == "y":
        actor2 = build_actor(model_name, device, attn_impl, dtype)
        benchmark_performance(
            actor=actor2,
            batch_size=4,
            prompt_len=2048,
            response_len=256,
            num_unique_prompts=1,
            num_warmup=1,
            num_runs=3,
            device=device,
        )
    else:
        print("Skipping benchmark.")

    print("\n" + "=" * 80)
    print("All demos completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
