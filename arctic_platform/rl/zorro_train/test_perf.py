"""
Performance test: Compare forward/backward timing between dedup and baseline with 32K tokens.
"""
import sys
import os
import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from arctic_platform.rl.zorro_train import ZoRRoTrain
from arctic_platform.rl.zorro_train.qwen_attention_patcher import reset_debug_object
from arctic_platform.rl.zorro_train.qwen_model_patcher import Qwen3ModelPatcher
from arctic_platform.rl.zorro_train.tests import create_dummy_batch
from arctic_platform.rl.zorro_train.zorro_train import analyze_normal_batch_via_attention_mask
#from deepspeed.profiling.flops_profiler import FlopsProfiler
from liger_kernel.transformers import AutoLigerKernelForCausalLM as AutoLigerModelForCausalLM
#from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM

#import deepspeed
import torch
import torch.distributed as dist
import torch.distributed as dist

import random, torch, numpy as np
def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    pr(f"Using seed: {seed}")

    random.seed(seed)    # python RNG
    np.random.seed(seed) # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)          # cpu + cuda
    torch.cuda.manual_seed_all(seed) # multi-gpu - can be called without gpus
    if use_seed: # slower speed! https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    return seed

import builtins
import fcntl

def printflock(*args, **kwargs):
    """ prevents rank output interleaving """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
print = printflock

def pr(msg):
    """ print with rank prefix if distributed """
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 0))
    if world_size > 1:
        print(f"{rank}: {msg}")
    else:
        print(msg)

def pr0(msg):
    """ print only on rank 0 """
    rank = int(os.getenv("RANK", 0))
    if rank == 0:
        pr(msg)

local_rank = int(os.getenv("LOCAL_RANK", 0))
rank = int(os.getenv("RANK", 0))
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(local_rank)
#pr(f"{rank=} {local_rank=}")
world_size = int(os.getenv("WORLD_SIZE", 0))
if world_size == 0:
    os.environ.update(dict(
        LOCAL_RANK="0",
        RANK="0",
        WORLD_SIZE="1",
        MASTER_ADDR="localhost",
        MASTER_PORT="8889",
    ))
dist.init_process_group("nccl")

# same seed per rank
seed = 42 + rank
enforce_reproducibility(seed)

def test_perf(
    model_name: str = "Qwen/Qwen3-32B",
    batch_size: int = 6,
    num_unique_prompts: int = 1,
    prompt_len: int = 8192,
    response_len: int = 1024,
    #device: str = "cuda",
    liger_kernels: bool = True,
    add_padding: bool = True,
    use_unpad: bool = True,
    min_valid_prompt_len: int = 6144,
    min_valid_response_len: int = 768,
    max_token_len: int = 65536,
    use_load_balancing = False,
    use_model_builtin_deduplicator = False

):
    """Performance test with ~32K tokens.

    Args:
        add_padding: If True, add random padding to sequences
        use_unpad: If True, enable unpadding optimization (packed sequences)
        min_valid_prompt_len: Minimum valid prompt tokens when padding is enabled
        min_valid_response_len: Minimum valid response tokens when padding is enabled
    """
    total_tokens = batch_size * (prompt_len + response_len)
    pr0("=" * 80)
    pr0(f"Performance Test (~{total_tokens//1000}K tokens)")
    pr0("=" * 80)

    # Load model
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    #config.num_hidden_layers = 10
    #attn_implementation = "flash_attention_2"
    attn_implementation = "flash_attention_3"

    pr0(f"Model: {model_name} ({config.num_hidden_layers=}, {attn_implementation=})")

    # Verify Flash Attention 3 availability
    try:
        import flash_attn_3
        #pr(f"  flash-attn version: {flash_attn.__version__}")
        if hasattr(flash_attn_3, 'flash_attn_func'):
            pr0(f"  Flash Attention 3 available: Yes")
        # Check CUDA architecture
        if torch.cuda.is_available():
            cuda_arch = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
            pr0(f"  CUDA architecture: sm{cuda_arch} ({torch.cuda.get_device_name(0)})")
            if cuda_arch < 90:
                pr0(f"  Note: Flash Attention 3 optimizations require sm90+ (H100). Will use FA2 kernels on sm{cuda_arch}.")
    except ImportError:
        pr0(f"  Warning: flash-attn-3 not installed. Install with: pip install flash-attn --no-build-isolation")

    with torch.device(device):
        automodel = AutoLigerModelForCausalLM if liger_kernels else AutoModelForCausalLM
        base_model = automodel.from_config(
            config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

    model = base_model
    model.gradient_checkpointing_enable()
    model.train()

    # Create test batch
    batch = create_dummy_batch(
        batch_size=batch_size,
        num_unique_prompts=num_unique_prompts,
        prompt_len=prompt_len,
        response_len=response_len,
        device=device,
        include_training_fields=False,
        add_padding=add_padding,
        min_valid_prompt_len=min_valid_prompt_len,
        min_valid_response_len=min_valid_response_len,
    )

    print(f"non-pad tokens: {int(batch['attention_mask'].sum()/1000)}K")
    analyze_normal_batch_via_attention_mask(batch["input_ids"], batch["attention_mask"], response_len)

    rollout_n = int(batch_size / num_unique_prompts)

    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]
    attention_mask = batch.get("attention_mask", None)

    pr(f"  Input: {input_ids.shape}, Total tokens: {input_ids.numel()}")
    if attention_mask is not None:
        valid_tokens = attention_mask.sum().item()
        padding_pct = 100 * (1 - valid_tokens / input_ids.numel())
        pr(f"  Padding: {padding_pct:.1f}% ({input_ids.numel() - valid_tokens} / {input_ids.numel()} tokens)")

    # Setup deduplication
    if use_model_builtin_deduplicator:
        micro_batches=[
            dict(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        ]
    elif use_load_balancing:
        from tensordict import TensorDict
        from arctic_platform.rl.zorro_train.seqlen_balancing import rearrange_micro_batches_with_dedup
        batch = TensorDict(batch, batch_size=batch_size) # batch["input_ids"].shape[0])

        TIME_ME = True
        if TIME_ME:
            from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimple
            timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
            #timers.wall_clock_breakdown = True
            dist.barrier() # align all ranks
            timers.start("rearrange_micro_batches")

        micro_batches, _ = rearrange_micro_batches_with_dedup(
            batch=batch,
            response_length=response_len,
            max_token_len=max_token_len,
            max_group_length_threshold=rollout_n,
        )
        if TIME_ME:
            timers.stop("rearrange_micro_batches")
            print(f"rearrange_micro_batches elapsed {timers.times['rearrange_micro_batches']:.2f}msec")
        #exit()
        pr("MicroBatches:")
        pr(list(micro_batches[i]["dedup_input_ids"].shape for i in range(len(micro_batches))))

        # this is just for stats below
        dedup_input_ids = torch.cat(list(micro_batches[i]["dedup_input_ids"].squeeze() for i in range(len(micro_batches)))).unsqueeze(0)
        #pr(dedup_input_ids)

    else:
        deduplicator = ZoRRoTrain()
        prompt_groups, unique_prompts = deduplicator.find_prompt_groups(input_ids=input_ids, response_length=response_len)
        dedup_input_ids, adapted_position_ids, reconstruction_info = deduplicator.create_deduplicated_batch(
            input_ids=input_ids, position_ids=position_ids, response_length=response_len,
            prompt_groups=prompt_groups, unique_prompts=unique_prompts,
            attention_mask=attention_mask,
            use_unpad=use_unpad,
        )
        pr(dedup_input_ids.shape)
        micro_batches=[
            dict(
                dedup_input_ids=dedup_input_ids,
                adapted_position_ids=adapted_position_ids,
                reconstruction_info=reconstruction_info,
                # ignoring the other fields for now as only above 2 are used below
            )
        ]
    if not use_model_builtin_deduplicator:
        pr(f"  Dedup: {dedup_input_ids.shape}, Dedup tokens: {dedup_input_ids.numel()}")

        if use_unpad:
            dedup_pct = 100 * dedup_input_ids.numel() / valid_tokens if attention_mask is not None else 0
            pr(f"  Unpad optimization: ENABLED (packed format, {dedup_pct:.1f}% of valid tokens)")
        else:
            pr(f"  Unpad optimization: DISABLED")

        if use_unpad:
            position_ids = ZoRRoTrain._unpad_replicated_ids(
                position_ids, attention_mask
            )
            input_ids = ZoRRoTrain._unpad_replicated_ids(
                input_ids, attention_mask
            )

    #dist.destroy_process_group(); exit()

    # Warmup
    pr0(f"\nWarmup...")
    for _ in range(1):
        output = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        loss = output.logits.sum() / output.logits.numel()
        loss.backward()
        model.zero_grad()

        # just one mb is enough to warmup
        micro_batch = micro_batches[0]

        if use_model_builtin_deduplicator:
            reconstruction_info = {}
        else:
            reconstruction_info = micro_batch["reconstruction_info"]
            # with Qwen3ModelPatcher(model=model, reconstruction_info=reconstruction_info):
            #     dedup_input_ids = micro_batch["dedup_input_ids"]
            #     adapted_position_ids = micro_batch["adapted_position_ids"]
            #     output = model(input_ids=dedup_input_ids, position_ids=adapted_position_ids, use_cache=False)
            #     loss = output.logits.sum() / output.logits.numel()
            #     loss.backward()

        model.zero_grad()
    torch.cuda.synchronize()

    reset_debug_object()

    # Baseline
    pr("\n[1/2] Baseline...")
    torch.cuda.synchronize()
    fwd_start = time.time()

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof_fwd:
        with record_function("baseline_forward"):
            with Qwen3ModelPatcher(model=model, reconstruction_info=reconstruction_info, patch_with_local=True):
                output = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
                loss = output.logits.sum() / output.logits.numel()

    torch.cuda.synchronize()
    baseline_fwd_time = time.time() - fwd_start
    pr("***FWD***\n" + prof_fwd.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    bwd_start = time.time()
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=True,
    ) as prof_bwd:
        with record_function("baseline_backward"):
            loss.backward()

    torch.cuda.synchronize()
    baseline_bwd_time = time.time() - bwd_start
    pr("***BWD***\n" + prof_bwd.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    baseline_total_time = baseline_fwd_time + baseline_bwd_time
    model.zero_grad()

    # Deduplicated
    pr("\n[2/2] Deduplicated...")

    if use_model_builtin_deduplicator:

        from arctic_platform.rl.zorro_train.qwen_model_patcher import Qwen3ModelOncePatcher
        rollout_n = batch_size // num_unique_prompts
        dedup_actor_model_once_patcher = Qwen3ModelOncePatcher(model, response_len=response_len, max_token_len=max_token_len, rollout_n=rollout_n, temperature=1, use_unpad=use_unpad, logits_optimization="memory", world_size=world_size)
        dedup_actor_model_once_patcher.patch_forward()

    torch.cuda.synchronize()
    fwd_start = time.time()

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof_fwd:
        with record_function("dedup_forward"):
            losses = []
            if use_model_builtin_deduplicator:
                outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=False)
                entropy = outputs.entropy
                log_probs = outputs.logprobs
                loss = log_probs.sum() / log_probs.numel()
                losses.append(loss.unsqueeze(0))
            else:
                print("FWD w/ dynamic dedup")
                for i, micro_batch in enumerate(micro_batches):
                    reconstruction_info = micro_batch["reconstruction_info"]
                    with Qwen3ModelPatcher(model=model, reconstruction_info=reconstruction_info):
                        dedup_input_ids = micro_batch["dedup_input_ids"]
                        adapted_position_ids = micro_batch["adapted_position_ids"]
                        output = model(input_ids=dedup_input_ids, position_ids=adapted_position_ids, use_cache=False)
                        loss = output.logits.sum() / output.logits.numel()
                        losses.append(loss.unsqueeze(0))

    #exit()

    torch.cuda.synchronize()
    dedup_fwd_time = time.time() - fwd_start
    pr("***FWD***\n" + prof_fwd.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    bwd_start = time.time()
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=True,
    ) as prof_bwd:
        with record_function("dedup_backward"):
            if use_model_builtin_deduplicator:
                losses[0].backward()
            else:
                for i, micro_batch in enumerate(micro_batches):
                    reconstruction_info = micro_batch["reconstruction_info"]
                    with Qwen3ModelPatcher(model=model, reconstruction_info=reconstruction_info):
                        losses[i].backward()
    #            with Qwen3ModelPatcher(model=model, reconstruction_info=reconstruction_info):
    #                loss.backward()

    torch.cuda.synchronize()
    dedup_bwd_time = time.time() - bwd_start
    pr("***BWD***\n" + prof_bwd.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    dedup_total_time = dedup_fwd_time + dedup_bwd_time
    model.zero_grad()

    # Results
    pr("\n" + "=" * 80)
    pr("RESULTS")
    pr("=" * 80)
    pr(f"{'':20} {'Baseline':>12} {'Dedup':>12} {'Speedup':>12}")
    pr("-" * 80)
    pr(f"{'Forward:':20} {baseline_fwd_time:>11.3f}s {dedup_fwd_time:>11.3f}s {baseline_fwd_time/dedup_fwd_time:>11.2f}x")
    pr(f"{'Backward:':20} {baseline_bwd_time:>11.3f}s {dedup_bwd_time:>11.3f}s {baseline_bwd_time/dedup_bwd_time:>11.2f}x")
    pr(f"{'Total:':20} {baseline_total_time:>11.3f}s {dedup_total_time:>11.3f}s {baseline_total_time/dedup_total_time:>11.2f}x")
    pr("=" * 80)

    dist.destroy_process_group();

if __name__ == "__main__":
    # Test with padding and unpadding optimization enabled by default

    #model_name = "Qwen/Qwen3-0.6B"
    #model_name = "Qwen/Qwen3-1.7B"
    model_name = "Qwen/Qwen3-8B"
    #model_name = "Qwen/Qwen3-32B"

    kwargs = dict(
        model_name=model_name,
        #batch_size=30,
        num_unique_prompts=5,
        use_load_balancing=False,
    )

    # mutually exclusive flags
    kwargs["use_model_builtin_deduplicator"] = not kwargs["use_load_balancing"]

    # these are tuned up for running w/o OOM on "Qwen/Qwen3-32B" on H200
    # each next run is progressively bigger and slower
    # additionally using a smaller model will make things faster
    if 0:
        # quick functional test
        kwargs.update(
            batch_size=20,
            prompt_len=64,
            response_len=16,
            min_valid_prompt_len=48,  # ~75% of prompt_len
            min_valid_response_len=12,  # ~75% of response_len
            max_token_len=512,
        )
    elif 0:
        kwargs.update(
            batch_size=12,
            prompt_len=16,
            response_len=4,
            min_valid_prompt_len=12,  # ~75% of prompt_len
            min_valid_response_len=2,  # ~75% of response_len
            max_token_len=400,
            num_unique_prompts=4,
        )
    elif 0:
        kwargs.update(
            batch_size=40,
            prompt_len=1024,
            response_len=128,
            min_valid_prompt_len=758,  # ~75% of prompt_len
            min_valid_response_len=96,  # ~75% of response_len
            max_token_len=10_000,
        )
    elif 0:
        kwargs.update(
            batch_size=20,
            prompt_len=4096,
            response_len=512,
            min_valid_prompt_len=3072,  # ~75% of prompt_len
            min_valid_response_len=384,  # ~75% of response_len
            max_token_len=60_000,
        )
    elif 0:
        prompts = 2
        rollouts = 2
        prompt_len = 4096
        response_len = 4096
        max_token_len = prompt_len + response_len*rollouts + 10
        kwargs.update(
            batch_size=prompts*rollouts,
            prompt_len=prompt_len,
            response_len=response_len,
            min_valid_prompt_len=int(0.75*prompt_len),
            min_valid_response_len=int(0.75*response_len),
            max_token_len=max_token_len,
            num_unique_prompts=prompts,
        )
    else:
        prompts = 1
        rollouts = 16
        # prompt_len = 3072
        # response_len = 1024
        prompt_len = 3000
        response_len = 3619
        max_token_len = prompt_len + response_len*rollouts + 10
        variance_pct = 0.99
        kwargs.update(
            batch_size=prompts*rollouts,
            prompt_len=prompt_len,
            response_len=response_len,
            min_valid_prompt_len=int(variance_pct*prompt_len),
            min_valid_response_len=int(variance_pct*response_len),
            max_token_len=max_token_len,
            num_unique_prompts=prompts,
        )

    test_perf(
        add_padding=True,
        use_unpad=True,
        **kwargs,
    )

