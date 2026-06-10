"""
Simple test: Compare first layer attention output between dedup and baseline.
"""
import sys
import os

# Add parent directory to path to enable package imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arctic_platform.rl.zorro_train.qwen_attention_patcher import QwenAttentionPatcher, compare_debug_tensors, debug_object
from arctic_platform.rl.zorro_train import ZoRRoTrain
from arctic_platform.rl.zorro_train.tests import create_dummy_batch
from arctic_platform.rl.zorro_train.qwen_model_patcher import Qwen3ModelPatcher


def get_all_gradients(model):
    """
    Collect all model parameter gradients into a list.
    
    Args:
        model: The model to extract gradients from
    
    Returns:
        Tuple of (gradient list, parameter names list)
    """
    gradients = []
    param_names = []
    for name, param in model.named_parameters():
        param_names.append(name)
        if param.grad is not None:
            gradients.append(param.grad.clone().detach())
        else:
            gradients.append(None)
    return gradients, param_names


def compare_gradients(grad_list1, grad_list2, param_names=None, name1="baseline", name2="patched", threshold=0.05):
    """
    Compare global gradient norm. Test passes if (global_norm(g2) - global_norm(g1)) / global_norm(g1) < threshold.
    
    Args:
        param_names: Optional list of parameter names
    
    Returns:
        bool: True if global gradient norms match within threshold
    """
    assert len(grad_list1) == len(grad_list2), f"Gradient lists have different lengths"
    
    # Compute global norms (L2 norm across all parameters)
    grad_tensors1 = [g.float() for g in grad_list1 if g is not None]
    grad_tensors2 = [g.float() for g in grad_list2 if g is not None]
    
    global_norm1 = torch.norm(torch.stack([torch.norm(g) for g in grad_tensors1]))
    global_norm2 = torch.norm(torch.stack([torch.norm(g) for g in grad_tensors2]))
    
    global_rel_diff = (global_norm2 - global_norm1) / (global_norm1 + 1e-10)
    passed = abs(global_rel_diff) < threshold
    
    # Also compute per-parameter stats for telemetry
    diffs = []
    for idx, (g1, g2) in enumerate(zip(grad_list1, grad_list2)):
        if g1 is None and g2 is None:
            continue
        if g1 is None or g2 is None:
            continue
        
        norm1 = torch.norm(g1.float()).item()
        norm2 = torch.norm(g2.float()).item()
        rel_diff = (norm2 - norm1) / (norm1 + 1e-10)
        diffs.append((idx, abs(rel_diff), g1.shape, norm1, norm2, rel_diff))
    
    # Show global norm comparison
    print(f"\nGlobal gradient norm comparison:")
    print(f"  {name1} global norm: {global_norm1.item():.6e}")
    print(f"  {name2} global norm: {global_norm2.item():.6e}")
    print(f"  Relative difference: {global_rel_diff.item()*100:+.2f}%")
    
    # Sort by absolute relative difference for telemetry
    diffs.sort(key=lambda x: x[1], reverse=True)
    
    # Show top 5 largest per-parameter differences
    print(f"\nTop 5 largest per-parameter gradient norm differences:")
    for item in diffs[:5]:
        idx, abs_rel, shape, n1, n2, rel = item
        pname = param_names[idx] if param_names else f"param_{idx}"
        print(f"  [{idx:3d}] {pname:60s} rel_diff={(rel*100):+6.2f}%, norm({name1})={n1:.3e}, norm({name2})={n2:.3e}")
    
    # Show top 5 smallest per-parameter differences
    print(f"\nTop 5 smallest per-parameter gradient norm differences:")
    for item in reversed(diffs[-5:]):
        idx, abs_rel, shape, n1, n2, rel = item
        pname = param_names[idx] if param_names else f"param_{idx}"
        print(f"  [{idx:3d}] {pname:60s} rel_diff={(rel*100):+6.2f}%, norm({name1})={n1:.3e}, norm({name2})={n2:.3e}")
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status}: Global gradient norm relative diff = {abs(global_rel_diff.item())*100:.2f}% (threshold = {threshold*100:.1f}%)")
    return passed


def test_forward_no_grad(
    model_name: str = "Qwen/Qwen3-4B",
    batch_size: int = 8,
    num_unique_prompts: int = 2,
    prompt_len: int = 32,
    response_len: int = 16,
    device: str = "cuda",
    attn_implementation: str = "flash_attention_2",
):
    """Test that first layer inputs match after reconstruction."""
    print("=" * 80)
    print("First Layer Input Test")
    print("=" * 80)
    print(f"  attn_implementation: {attn_implementation}")
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    
    # Create test batch
    print("\nCreating test batch...")
    batch = create_dummy_batch(
        batch_size=batch_size,
        num_unique_prompts=num_unique_prompts,
        prompt_len=prompt_len,
        response_len=response_len,
        device=device,
        include_training_fields=False
    )
    
    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]
    
    print(f"  Batch size: {batch_size}")
    print(f"  Num unique prompts: {num_unique_prompts}")
    print(f"  Input shape: {input_ids.shape}")
    
    # Identify prompt groups
    deduplicator = ZoRRoTrain()
    prompt_groups, unique_prompts = deduplicator.find_prompt_groups(
        input_ids=input_ids,
        response_length=response_len
    )
    
    # Create deduplicated batch
    dedup_input_ids, dedup_position_ids, reconstruction_info = \
        deduplicator.create_deduplicated_batch(
            input_ids=input_ids,
            position_ids=position_ids,
            response_length=response_len,
            prompt_groups=prompt_groups,
            unique_prompts=unique_prompts
        )
    
    print(f"  Deduplicated shape: {dedup_input_ids.shape}")
    
   
    # Test 1: Baseline forward pass
    print("\n[1/3] Running baseline forward pass with local patching...")
    with torch.no_grad():
        with Qwen3ModelPatcher(
            model=model,
            reconstruction_info=reconstruction_info, 
            patch_with_local=True  
        ):
            output_baseline = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            )

    # Test 2: Deduplicated forward pass with reconstruction
    print("\n[2/3] Running deduplicated forward pass...")
    with torch.no_grad():
        
        # Apply monkey patching
        with Qwen3ModelPatcher(
            model=model,
            reconstruction_info=reconstruction_info
        ):
            output_dedup = model(
                input_ids=dedup_input_ids,
                position_ids=position_ids,
                use_cache=False,
            )
    # Test 3: Deduplicated forward pass with no patching    
    print("\n[3/3] Running deduplicated forward pass with no patching...")
    with torch.no_grad():
        output_dedup_no_patching = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        )

    # Compare outputs logits norm
    print(f"Baseline logits norm: {output_baseline.logits.norm()}, Deduplicated logits norm: {output_dedup.logits.norm()}, Deduplicated logits norm no patching: {output_dedup_no_patching.logits.norm()}")
    

def test_forward_backward(
    model_name: str = "Qwen/Qwen3-4B",
    batch_size: int = 8,
    num_unique_prompts: int = 2,
    prompt_len: int =32,
    response_len: int = 16,
    device: str = "cuda",
    add_padding: bool = False,
    use_unpad: bool = False,
    use_split_attention: bool = False,
    attn_implementation: str = "sdpa",
):
    """Test forward and backward passes with deduplication.
    
    Args:
        add_padding: If True, create batch with padding (left for prompts, right for responses)
        use_unpad: If True, use unpadded deduplication (requires add_padding=True)
        use_split_attention: If True, use split attention optimization
    """
    print("=" * 80)
    print("Forward and Backward Test")
    print("=" * 80)
    print(f"\nTest configuration:")
    print(f"  add_padding: {add_padding}")
    print(f"  use_unpad: {use_unpad}")
    print(f"  use_split_attention: {use_split_attention}")
    print(f"  attn_implementation: {attn_implementation}")
    
    if use_unpad and not add_padding:
        print("Warning: use_unpad requires add_padding=True. Enabling add_padding.")
        add_padding = True
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    
    #set print width to 200 
    import pandas as pd
    pd.set_option('display.width', 200)
    
    # Create test batch
    print("\nCreating test batch...")
    batch = create_dummy_batch(
        batch_size=batch_size,
        num_unique_prompts=num_unique_prompts,
        prompt_len=prompt_len,
        response_len=response_len,
        device=device,
        include_training_fields=False,
        add_padding=add_padding,
    )
    
    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]
    attention_mask = batch["attention_mask"]
    
    print(f"  Batch size: {batch_size}")
    print(f"  Num unique prompts: {num_unique_prompts}")
    print(f"  Input shape: {input_ids.shape}")
    
    #print attention mask in a matrix form
    print(f" Attention mask shape: \n {attention_mask}")
    
    if add_padding:
        total_tokens = attention_mask.sum().item()
        total_possible = attention_mask.numel()
        padding_pct = 100 * (1 - total_tokens / total_possible)
        print(f"  Valid tokens: {total_tokens} / {total_possible} ({padding_pct:.1f}% padding)")
    
    # Identify prompt groups
    deduplicator = ZoRRoTrain()
    prompt_groups, unique_prompts = deduplicator.find_prompt_groups(
        input_ids=input_ids,
        response_length=response_len
    )
    
    # Create deduplicated batch
    print(f"\nCreating deduplicated batch (use_unpad={use_unpad})...")
    dedup_input_ids, adapted_position_ids, reconstruction_info = \
        deduplicator.create_deduplicated_batch(
            input_ids=input_ids,
            position_ids=position_ids,
            response_length=response_len,
            prompt_groups=prompt_groups,
            unique_prompts=unique_prompts,
            attention_mask=attention_mask if add_padding else None,
            use_unpad=use_unpad,
        )
            
    # For baseline: use packed format (if unpadded) but keep all sequences (no deduplication)
    if use_unpad:
        # Unpad the full batch for baseline (without deduplication)
        baseline_input_ids = []
        baseline_position_ids = []
        for i in range(batch_size):
            if attention_mask is not None:
                valid_mask = attention_mask[i].bool()
                baseline_input_ids.append(input_ids[i, valid_mask])
                baseline_position_ids.append(position_ids[i, valid_mask])
            else:
                baseline_input_ids.append(input_ids[i])
                baseline_position_ids.append(position_ids[i])
        
        # Concatenate into packed format [1, total_valid_tokens]
        baseline_input_ids = torch.cat(baseline_input_ids).unsqueeze(0)
        baseline_position_ids = torch.cat(baseline_position_ids).unsqueeze(0)
    else:
        # Use original padded format
        baseline_input_ids = input_ids
        baseline_position_ids = position_ids
    
    # Print token statistics
    print(f"  Total tokens in replicated batch: {input_ids.numel()}")
    if use_unpad:
        print(f"  Total tokens in baseline (unpadded, with duplication): {baseline_input_ids.numel()}")
    print(f" Total tokens after unpadding but before deduplication: {adapted_position_ids.numel()}")
    print(f" Total tokens after deduplication: {dedup_input_ids.numel()}")
    print(f"  reconstruction_info['is_unpadded']: {reconstruction_info.get('is_unpadded', False)}")
    
    
    # Reset debug_object before testing
    debug_object['baseline'] = None
    debug_object['patched'] = None
   
    # Test 1: Baseline forward pass and backward pass
    print("\n[1/3] Running baseline forward pass with local patching...")
    with Qwen3ModelPatcher(
        model=model,
        reconstruction_info=reconstruction_info, 
        patch_with_local=True,
        use_split_attention=use_split_attention,
    ):
        output_baseline = model(
            input_ids=baseline_input_ids,
            position_ids=baseline_position_ids,
            use_cache=False,
        )
        loss_baseline  = output_baseline.logits.sum() / output_baseline.logits.numel()
        loss_baseline.backward()
        
        # Store gradients 
        gradient_baseline, param_names = get_all_gradients(model)
        
        # Clear the gradients
        model.zero_grad()

    # Test 2: Deduplicated forward pass with reconstruction
    print("\n[2/3] Running deduplicated forward pass...")
        
    # Apply dedup
    with Qwen3ModelPatcher(
        model=model,
        reconstruction_info=reconstruction_info,
        use_split_attention=use_split_attention,
    ):
        output_dedup = model(
            input_ids=dedup_input_ids,
            position_ids=adapted_position_ids,
            use_cache=False,
        )
        
        output_replicated_logits = ZoRRoTrain.reconstruct_sequences(output_dedup.logits, reconstruction_info)
        
        loss_dedup = output_replicated_logits.sum() / output_replicated_logits.numel()
        
        loss_dedup.backward()
        
        # Store gradients 
        gradient_dedup, _ = get_all_gradients(model)
        
        # Clear the gradients
        model.zero_grad()
        
    #Compare baseline and dedup loss
    print(f"Baseline loss: {loss_baseline}, Deduplicated loss: {loss_dedup}")
    
    #compare debug_objects for baseline and dedup
    compare_debug_tensors(debug_object)
    
    #exit(0)
    
    # Test 3: Deduplicated forward pass with no patching    
    print("\n[3/3] Running deduplicated forward pass with no patching...")
    output_dedup_no_patching = model(
        input_ids=input_ids,
        position_ids=position_ids,
        use_cache=False,
    )
    loss_no_patching = output_dedup_no_patching.logits.sum() / output_dedup_no_patching.logits.numel()
    loss_no_patching.backward()
    
    # Store gradients 
    gradient_dedup_no_patching, _ = get_all_gradients(model)
    
    # Clear the gradients
    model.zero_grad()
    
    # Compare gradients
    print("\n" + "=" * 80)
    print("Gradient Comparison")
    print("=" * 80)
    compare_gradients(gradient_baseline, gradient_dedup, param_names, "baseline", "patched")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments for test mode and attention implementation
    test_mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    attn_impl = sys.argv[2] if len(sys.argv) > 2 else "sdpa"
    
    print(f"\nUsing attention implementation: {attn_impl}\n")
    
    if test_mode == "baseline":
        print("\n" + "="*80)
        print("TEST MODE: Baseline (no padding, no unpad, no split attention)")
        print("="*80 + "\n")
        test_forward_backward(
            add_padding=False,
            use_unpad=False,
            use_split_attention=True,
            attn_implementation=attn_impl,
        )
    elif test_mode == "padded":
        print("\n" + "="*80)
        print("TEST MODE: Padded (with padding, no unpad, no split attention)")
        print("="*80 + "\n")
        test_forward_backward(
            add_padding=True,
            use_unpad=False,
            use_split_attention=True,
            attn_implementation=attn_impl,
        )

    elif test_mode == "unpadded_split":
        print("\n" + "="*80)
        print("TEST MODE: Unpadded with Split Attention (with padding, with unpad, with split attention)")
        print("="*80 + "\n")
        test_forward_backward(
            add_padding=True,
            use_unpad=True,
            use_split_attention=True,
            attn_implementation=attn_impl,
        )
    elif test_mode == "all":
        print("\n" + "="*80)
        print("RUNNING ALL TEST MODES")
        print("="*80 + "\n")
        
        modes = [
            ("baseline", False, False, True),
            ("padded", True, False, True),
            ("unpadded", True, True, True),
            ("unpadded_split", True, True, True),
        ]
        
        for mode_name, add_pad, use_unp, use_split in modes:
            print("\n\n" + "="*80)
            print(f"TEST MODE: {mode_name}")
            print("="*80 + "\n")
            try:
                test_forward_backward(
                    add_padding=add_pad,
                    use_unpad=use_unp,
                    use_split_attention=use_split,
                    attn_implementation=attn_impl,
                )
                print(f"\n✅ {mode_name} test PASSED")
            except Exception as e:
                print(f"\n❌ {mode_name} test FAILED: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"Unknown test mode: {test_mode}")
        print("Available modes: baseline, padded, unpadded, unpadded_split, all")
        print("Usage: python test_forward_and_backward.py <mode> [attn_impl]")
        print("  attn_impl: sdpa (default), flash_attention_3, eager, sdpa")
        sys.exit(1)
