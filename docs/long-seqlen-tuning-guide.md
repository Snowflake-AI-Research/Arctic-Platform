# Tuning for long sequence length guide

Reaching a long sequence length is a memory problem, not a throughput problem. Every knob in this guide trades GPU memory for something else -- recompute, compute sharding, host RAM, etc. -- and the purpose of tuning is to use first the knobs that impact performance the least and stop as soon as the target length fits.

This guide covers the knobs `dss-platform` exposes, what each one costs, and the order to turn them on. The techniques behind the three biggest ones -- token-tiled MLP, activation CPU offload, and sequence parallelism -- are [Arctic Long Sequence Training](https://arxiv.org/abs/2506.13996); the per-subsystem docs linked below give the implementation detail this guide summarizes.

## What sets the wall

A rank runs out of one of two resources, and which one decides everything about what to do next.

**GPU memory** is the usual wall. On an H200 the card exposes ~141 GiB, of which for some models up to ~13 GiB is not PyTorch's to allocate: the CUDA primary context, NCCL buffers, DeepEP/NVSHMEM heaps, and cuBLAS and attention workspaces. That leaves ~126-130 GiB of torch-reserved memory, split between (1) terms that are constant (parameters, optimizer state, comm buffers) and (2) terms that scale with sequence length (activations, recompute transients, the logits tensor). Only the second group moves when a knob changes.

**Host memory** becomes the wall as soon as optimizer and activation offload are both on, because both move their bytes to the same place. Ray kills workers at 0.95 of node RAM, so on a 2000 GB node the ceiling arrives while the cards still have room: a dense 27B run at 1.67M tokens holds 109.96 GiB of a card's ~130 GiB and still dies at the next length up, on host RSS of 175.52 GiB per rank across eight ranks.

Knowing which wall is binding is the whole diagnosis. `training_config.step_peak_memory_log` returns both peaks in the `/step` response for one MAX all-reduce per step; `DSS_MEM_BREAKDOWN=1` splits a rank's GPU memory into parameters, gradients, optimizer state, activations, and allocator slack. `dss-platform` [`docs/memory_diagnostics.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/memory_diagnostics.md) covers the rest.

## The knobs

### Already on, and worth leaving on

| Knob | Default | What it does |
| --- | --- | --- |
| `cuda_allocator_conf` | `"expandable_segments:True"` | Lets the caching allocator grow and reuse segments instead of stranding gaps between variable-sized activation buffers. Long-context runs fragment badly without it and OOM on a later step at a footprint that fit earlier. `dss-platform` [`docs/cuda_allocator.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/cuda_allocator.md) |
| `mb_spec.max_tokens_per_mb` | `10240` | Packs rows into one `[1, packed_tokens]` model call so no compute is spent on padding. It removes work rather than trading for memory. `dss-platform` [`docs/packing.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/packing.md) |
| `activation_checkpointing` | `True` | Whole-block recompute: the largest memory lever and the largest step-time cost here, about +29% FLOPs at long context. Also the precondition for activation offload. |
| `attn_implementation` | `"flash_attention_3"` (or 2/4) | An attention kernel with an O(S) memory footprint. An eager or math fallback materializes the `S x S` score matrix and no other knob will rescue that. |
| `ac_config.router_replay_recompute` | `True` | MoE only. Replays the router's `topk` on the backward recompute so tensor shapes inside a checkpointed block match the forward. Whole-block AC on an MoE model fails with `CheckpointError: recomputed values have different metadata` without it. |

### Activation-transient shavers

These cap a tensor whose size grows with tokens, without changing what is computed.

| Knob | Default | Effect |
| --- | --- | --- |
| `fused_lm_head_token_chunk_size` | unset | Tiles the `hidden @ lm_head.T` projection over tokens, so peak logits are `O(token_chunk x vocab_chunk)` instead of `O(tokens x vocab)`. |
| `fused_lm_head_vocab_chunk_size` | `8192` | The vocab extent of that tile. |
| `tiled_mlp_token_chunk_size` | unset | Shards a dense FFN over tokens and recomputes it shard by shard, holding ~`1/shards` of its intermediate. On MoE it reaches only the shared expert, which is not worth the recompute -- see the tiled-MLP section. `dss-platform` [`docs/tiled_mlp.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/tiled_mlp.md) |
| `prime_rl.fused_cross_entropy` | `"liger"` | Fuses head and loss so full-vocabulary logits never materialize at all. `"liger"` (or `True`) selects the Liger kernel and `False` opts out. The AP-owned GLM loader also supports `"quack"`; the AP-owned Qwen3.5 MoE loader rejects it when the model spec is built. Mutually exclusive with `fused_lm_head_token_chunk_size`: both select an lm_head implementation, and `load_ap_moe` rejects the pair before model construction. |
| `prime_rl.deepep_token_chunk_size` | `None` (unchunked) | MoE only. Chunks the expert all-to-all so a rank's per-chunk received tokens are capped at `ep x chunk`, independent of sequence length and of routing skew. |

The head is the largest of these on a large-vocabulary model. Hugging Face cross-entropy materializes `[tokens, vocab]` logits in bf16, keeps the gradient, and upcasts to fp32 inside the loss. On a 248320-token vocabulary that is about 2.53 GiB per 1,000 rank-local tokens -- roughly 44 GiB of a card at 24,000 tokens per rank, spent on the head alone. Chunking replaces it with a constant tile: at `16384 x 8192` in fp32, 0.5 GiB regardless of sequence length.

Three routes reach that state and they are not interchangeable. `model_provider="liger"` installs the Liger fused head for a dense model with nothing further to set. On the Prime-RL path `prime_rl.fused_cross_entropy` is already `"liger"` and `False` opts out; `"quack"` is a CuTe-DSL alternative only for AP-owned GLM. `model_provider="huggingface"` has no fused head at all, so it needs the explicit chunk sizes.

A fused head returns a scalar loss instead of logits once labels are present, which two request shapes cannot use: a non-binary SFT `loss_mask` routes through an external cross-entropy, and `loss_fn="causal_cross_entropy"` needs logits or per-token logprobs. Both raise rather than degrading quietly, and both are fixed by setting `fused_cross_entropy=False` and chunking the head instead.

On the Prime-RL MoE path the chunk size is read from `prime_rl.fused_lm_head_token_chunk_size` first and falls back to the top-level field, so a value set only at the top level still reaches the carved loaders.

### Recompute

`ac_config.mode` selects what the backward pass recomputes.

- **`"full"`** (default) wraps each whole transformer block, so autograd saves only the block input. Largest saving, one extra forward per block, and the only mode with a single per-block boundary that offload can stream.
- **`"selective"`** with `targets` (default `["norm"]`) checkpoints only the named submodules and keeps everything else resident. It recomputes less and saves less, and it cannot be combined with offload -- `offload_config.enabled=True` with `mode="selective"` raises a `ValueError`.

`ac_config.freq` applies checkpointing to every Nth block, which trades recompute against residency in between the two modes. `dss-platform` [`docs/activation_offload.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/activation_offload.md) has the full field list.

### Offload

Both offloads move bytes to host RAM. Together they are what turns a GPU-bound run into a host-bound one.

**Optimizer state**, through `ds_config.zero_optimization.offload_optimizer`. ZeRO stage 2 is the default; adding the offload key moves Adam state to CPU and switches the optimizer to `DeepSpeedCPUAdam` automatically (`_resolve_optimizer_name`). This is the single largest static saving and the precondition for everything below it.

```python
"training_config": {
    "ds_config": {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        }
    }
}
```

**Activation-checkpoint boundaries**, through `ac_config.offload_config.enabled`. Full-mode checkpointing drops each block's internals but cannot drop the block *input*: autograd has to save it, because that is what the recomputation replays from. A completed forward therefore leaves one boundary hidden state per layer resident, `[tokens_per_rank, hidden]` each. For a 64-layer model at hidden 5120 in bf16 that is `64 x 5120 x 2` bytes = 640 KB per token, and it grows linearly with sequence length after everything else checkpointing saved has been freed. Offload streams those boundaries to pinned host memory on a dedicated stream and prefetches them back one step ahead in backward, leaving `keep_last_n` resident.

```python
"training_config": {
    "activation_checkpointing": True,
    "ac_config": {
        "mode": "full",                 # offload requires "full"; "selective" keeps intermediates on GPU
        "freq": 1,                      # checkpoint every Nth block
        "router_replay_recompute": True,  # MoE only; keeps recompute shapes equal to the forward
        "offload_config": {
            "enabled": True,            # off by default
            "keep_last_n": 1,           # boundaries left resident; backward needs the newest first
            "use_streams": True,        # overlap the copies with compute
            # omit tensor_size_threshold (or 1048576) for the 1 MiB floor; smaller saves pass through
            "pin_memory_enabled": True,        # False falls back to pageable buffers and slower copies
            "pin_memory_max_size_gib": "auto", # retained pinned-cache cap, or a hard GiB number
            "pin_memory_bucket_size_mib": 64,  # capacity rounding for contiguous pinned buffers
        },
    },
}
```

Every field is shown at its default except `enabled`, so a job that only wants the offload sets that one key:

```python
"training_config": {
    "ac_config": {"offload_config": {"enabled": True}}
}
```

`keep_last_n` and the three `pin_memory_*` fields are the ones to reach for when this offload is what pushed the run into the host-RAM wall: `keep_last_n` decides how many boundaries stay on the card instead of going to host (see below), `pin_memory_max_size_gib` caps what the staging cache retains, and `pin_memory_enabled=False` gives up copy speed to stop pinning host pages altogether. `dss-platform` [`docs/activation_offload.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/activation_offload.md) has the cache's retention and eviction rules.

The saved-tensor hooks that do this are *dynamically scoped*: they cover only tensors saved by code running inside the wrapped forward. Both the causal LM and its backbone are wrapped under one manager, because the chunked LM head calls the backbone directly to skip the full-vocabulary projection. A future rebinding of `forward` that runs the decoder blocks outside both wrappers would silently keep every boundary on the card -- measurably, that costs 640 KB per token and drops a dense 27B ceiling from 1.67M tokens to 680K, with no error to explain it.

### Parallelism

| Knob | Effect on the wall | Cost |
| --- | --- | --- |
| `sp_size` | The largest lever. Every token-shaped tensor in the backbone shrinks ~`1/sp_size`, attention included. | Eight all-to-alls per full-attention layer per step, on the critical path. About +11% forward/backward per doubling inside a node; a step change when the group crosses the node boundary. |
| `ep_size` | MoE only. Higher `ep` shards expert weights further but grows the dispatch/combine transient, and the transient wins. | Roughly neutral in step time. |

**The two degrees are independent axes over the same ranks, not a factorization of the world.** Nothing requires `sp_size * ep_size == world_size`. Each divides `world_size` on its own and the two are free to overlap completely, so `sp == ep == world_size` is a supported shape -- and it is one of the points measured below: `sp8·ep8` on a single 8xH200 node is `sp == ep == world == 8`, one SP group spanning every rank while the experts are sharded eight ways across those same eight GPUs. Every rank belongs to one SP group and one EP group and runs both families of collectives every step.

What the pair decides is placement rather than legality. `sp_size == world_size` leaves `dp = world_size // sp_size` at 1, so there is no data-parallel axis for EP to be a strided factor of; that shape takes the decoupled intra-node-EP layout, where SP keeps its contiguous rank blocks and EP ignores them in favour of NVLink blocks inside a node. The constraints that do bind are divisibility and node placement -- both degrees divide `world_size`, and an expert group has to fit inside a node because DeepEP faults on a cross-node one -- and `validate_parallelism_config` rejects a pair it cannot place rather than mis-grouping it silently.

`sp_size` must divide `num_attention_heads`, so the degree cannot exceed the model's head count however much memory a longer sequence needs. For a 16-head model that caps `sp` at 16, which two 8-GPU nodes already reach: a third and fourth node cannot raise the maximum sequence length, only throughput. `dss-platform` [`docs/sequence_parallelism.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/sequence_parallelism.md) covers the topology rules and which layout is chosen for a given shape.

The `ep` direction is counter-intuitive and holds at every `sp` tier measured. On Qwen3.6-35B-A3B, one 8xH200 node, sp8: `ep2` reaches 2.28M tokens, `ep4` 2.18M, and `ep8` 1.70M. Static expert weights fall as `ep` rises, but the DeepEP dispatch/combine buffers and the token-permutation activation rise faster. Keep `ep` at the smallest value that fits the expert weights.

**Measured utilization rises with sequence length, not with `sp_size`.** Every hardware FLOPs utilization figure here is taken at that configuration's own ceiling -- about 24% at `sp1`, 38% at `sp4`, 44.6% at `sp8` on one node, 47.4% at `sp16` on two -- and each ceiling is a longer sequence than the one before it, so the column tracks length and the degree only comes along with it. The mechanism is the `O(S^2)` term: on Qwen3.6-35B-A3B attention is about 89% of FLOPs at 512K and 95% at 1.15M, FlashAttention runs at high arithmetic intensity, and its quadratic work amortizes the fixed per-step overheads. Utilization plateaus around 48%.

**Rising utilization is not rising throughput.** The same `sp8·ep2` configuration at DeepEP chunk 1024 takes 12.68 secs per 100K tokens at a 1000K sequence and 24.45 secs per 100K at its 2.28M ceiling, while utilization goes from 39.0% to 44.6%. The GPU is busier and every token costs nearly twice the wall-clock, because the work per token itself grows with the sequence. Raising `sp` at a *fixed* length costs about 11% per doubling in forward/backward; what it buys is the ability to hold the longer sequence at all, not a cheaper token.

### Diagnostics

The first is a request config field; the rest are environment variables.

`step_peak_memory_log` is a request field, set in `training_config` beside every other knob in this guide:

```python
"training_config": {
    "step_peak_memory_log": True
}
```

It returns the job-wide maximum across ranks of peak GPU and host memory in the `/step` response, at a cost of one MAX all-reduce per step -- cheap enough to leave on for the whole tuning session, and the number a binary search on sequence length reads.

The remaining four are **environment variables on the trainer processes**. They do not appear in the request, and setting them requires reaching the worker environment rather than editing a payload:

| Environment variable | Use |
| --- | --- |
| `DSS_MEM_BREAKDOWN=1` | Per-rank split into parameters, gradients, optimizer state, activations, allocator slack, and non-PyTorch allocations. This is what says which term is binding. |
| `DSS_MEM_STAGED=1` | Attributes the fixed non-PyTorch tax to CUDA context, NCCL, DeepSpeed init, and first forward/backward. |
| `DSS_MEM_SNAPSHOT=<prefix>` | Records the CUDA allocator's full allocation history with Python stacks and writes a per-rank pickle. This is the one that names the module responsible for a peak -- see [Reading the memory allocator profiler](#reading-the-memory-allocator-profiler). |
| `DSS_ACT_OFFLOAD_LOG=1` | Offload counters per rank: boundaries offloaded, pinned allocations, reuses, evictions, cache size. A count of zero when offload is configured means the hooks are not covering the forward that runs the blocks. |
| `DSS_MOE_IMBALANCE=1` | MoE only. Per-rank `max_chunk_recv`: the largest received-token count of any DeepEP dispatch, which is the quantity that sizes the fp32 combine transient. Its spread across ranks is routing skew, and that spread is the headroom a ceiling measured on synthetic data does not account for -- it sizes the reserve described under `deepep_token_chunk_size` below. |

### Not levers

Each of these looks like a memory lever and measures as flat.

- **`train_batch_size` / batch size.** With `sp_size >= 2` the forward runs a `[1, tokens_per_rank]` shard per rank, so batch size only adds accumulation steps. Peak is a function of sequence length and the parallel degrees, not of batch size.
- **ZeRO comm buckets** (`DSS_ZERO_BUCKET_SIZE`). Shrinking the reduce/allgather buckets from 500M to 50M elements on Qwen3.6-35B-A3B at sp8·ep2 and 1000K tokens moves peak reserved from 79.7 GiB to 79.5 GiB and throughput from 441 TFLOPS to 445. The buckets are live during gradient reduction, which is disjoint from the instant that sets the peak.
- **The DeepEP `Config` buffer.** Fixed per-rank token capacity, so it does not grow with sequence length: `<=0.12 GiB` at `ep <= 8`. Shrinking it reclaims ~0.1 GiB of a ~13 GiB non-PyTorch tax.
- **`fp32_lm_head`.** This one costs memory rather than saving it -- it widens head arithmetic proportional to `vocab x hidden`. It exists so gradients do not depend on how tokens were grouped into model calls, which matters for parity tests and layout comparisons, not for reaching a length. `dss-platform` [`docs/lm_head_gradient_precision.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/lm_head_gradient_precision.md)
- **`compile.fullgraph`.** Incompatible with `tiled_mlp_token_chunk_size`; request validation rejects the combination.

## Tiled MLP pays off on dense models and not on MoE

The MLP tiling saving is proportional to the intermediate width of the FFN being tiled, and that width differs by more than an order of magnitude between the two architectures.

A dense SwiGLU FFN holds two `[tokens, intermediate]` tensors at once, and `intermediate` is typically three to five times `hidden`. Tiling it over tokens leaves ~`1/shards` of that resident, so on a dense model it caps the largest token-linear activation in the block after attention.

On an MoE model it saves almost nothing, for two reasons that compound:

- **The intermediate is small.** Qwen3.6-35B-A3B has `hidden_size=2048` with `shared_expert_intermediate_size=512` and `moe_intermediate_size=512`. The tiled tensor is a quarter of hidden width, where a dense model's is several times it.
- **Routed experts are deliberately excluded.** `apply_dense_tiled_mlp` skips any module with `experts` as a component of its qualified name. Tiling couples the ranks to one another, and routing decouples them. Under ZeRO-3 the tiled forward all-reduces its local shard count with `MAX` over the whole world before it tiles, so every rank executes the same number of tiles and the parameter all-gathers stay in one order; under ZeRO-1/2 `TiledMLP` instead defers the FFN weights' gradient reduction to the last shard. Both mechanisms assume the ranks agree on how many shards there are. A dense MLP runs exactly once per rank per layer per step, so they do. A routed expert runs only when the router sent it tokens on that rank, with a token count no peer shares -- so ranks can disagree about whether the collective is reached at all, which is a hang rather than a wrong shard count. The Prime-RL loaders therefore tile only the `FeedForward` / `BCFeedForward` shared expert, which every rank runs unconditionally. The routed path -- where an MoE model's token-linear FFN activation actually lives, at `topk` copies of every token -- is not tiled at all. It is bounded by `deepep_token_chunk_size` instead.

On Qwen3.6-35B-A3B, one 8xH200 node, sequence held at 1000K so memory is not the limiter, DeepEP chunk 4096, comparing tiling at 16k against tiling off entirely: `sp8·ep2` gives 79.7 GiB and 441 TFLOPS tiled against 79.1 GiB and 443 TFLOPS untiled, and `sp8·ep8` gives 82.2 GiB and 423 TFLOPS tiled against 82.2 GiB and 421 TFLOPS untiled. Tiling loses 0.6 GiB and 0.5% at `ep2` and gains 0.0 GiB and 0.5% at `ep8`. The two arms disagree in sign, which is what a difference below the noise floor looks like; 32k and 64k chunks fall inside the same 437-446 TFLOPS band. The sweep establishes that the *saving* is zero, not that the cost is.

The cost is not zero in principle, and it points the wrong way. `TiledMLP` runs the FFN on one token shard at a time and recomputes each shard in backward, so tiling adds a forward pass of the module it wraps. Full activation checkpointing is already on, so the FFN is computed once in the forward and again in the checkpoint recompute; tiling makes the tiled module pay a third. On a dense model that third pass buys back a large intermediate. Here it applies only to the 512-wide shared expert, which carries about a ninth of the FFN work -- one shared expert against `topk=8` routed experts of the same width -- so the extra pass is real and lands under the noise floor rather than being free.

Pushed to its own ceiling instead of held at 1000K, a finer 1k chunk moves the wall from 2.28M tokens to 2.31M -- 1.3% -- for a 1.6% slower step, and 4.64M to 4.70M for 2.3% slower on two nodes. That is the cost becoming visible once the chunk is small enough to multiply the launches.

**What to take from it.**

1. On a dense model, set `tiled_mlp_token_chunk_size` early: it is cheap and it caps a large tensor.
2. On an MoE model, leave it unset. It has nothing to cap -- the routed path is untiled by construction and the shared expert is a ninth of the FFN -- so enabling it pays a recompute pass for a saving that measures as zero. The MoE equivalent of this knob is `deepep_token_chunk_size`, and that one is neither free nor optional near the ceiling.

## The order to enable them

Ordered by step-time cost at the length you are trying to reach, cheapest first. Stop at the first row where the target fits -- every row below it is throughput given away for headroom that is not needed.

| # | Knob | Step-time cost | What it buys |
| --- | --- | --- | --- |
| 1 | `cuda_allocator_conf` | none; default `"expandable_segments:True"` | the allocator grows and reuses segments instead of stranding gaps between variable-sized activation buffers. Without it a footprint that fit on one step OOMs on a later one |
| 2 | `mb_spec.max_tokens_per_mb` | none; default `10240` | rows pack into one `[1, packed_tokens]` call, so no compute is spent on padding. It removes work rather than trading for memory |
| 3 | `attn_implementation` | none; default FlashAttention | attention with an O(S) footprint. An eager or math fallback materializes the `S x S` score matrix, and no row below can rescue that |
| 4 | `ep_size` **down**, on MoE | roughly neutral | 1.70M at ep8 to 2.28M at ep2, sp8 on one node |
| 5 | `ac_config.offload_config.enabled` | ~0%; the copies overlap compute | ~790K to ~1.15M at sp4·ep2. Host RAM becomes the new wall. Needs optimizer state off the card first -- see below |
| 6 | `fused_lm_head_token_chunk_size` + `fused_lm_head_vocab_chunk_size`, or a fused head: `prime_rl.fused_cross_entropy` is Liger by default, and `model_provider="liger"` fuses head and loss with no setting at all | ~1.5% | removes the `[tokens, vocab]` logits tensor. On a 248320-vocabulary dense 27B this alone moves the ceiling from 195K to 680K tokens |
| 7 | `ds_config.zero_optimization.offload_optimizer` | ~3% at the ceiling -- a 14-20 secs CPU Adam step against ~0.3 secs on GPU, inside a ~560 secs step at 2.28M tokens -- but dominant at short lengths | the gate: ~32K to ~790K at sp4·ep2, a factor of 24. Usually already on |
| 8 | `tiled_mlp_token_chunk_size` | ~3.6% | caps the dense FFN intermediate. Dense models only -- skip it on MoE, where it caps nothing and adds a recompute pass (previous section) |
| 9 | `sp_size` | ~+11% forward/backward per doubling within a node; a step change when the group spans nodes | the largest lever, ~2x context per doubling |
| 10 | `activation_checkpointing` | ~+29%. Full-mode recompute adds one forward pass, taking the causal FLOPs coefficient on the `S^2 d` attention term from 7 to 9, and attention is about 95% of FLOPs at 1.15M tokens. Derived from the FLOPs accounting rather than an A/B run -- see below | the largest memory lever in the document: drops every block's internal activation and leaves only the per-layer boundary, 640 KB per token on the dense 27B. On by default, and the precondition for activation offload |
| 11 | `prime_rl.deepep_token_chunk_size` | ~+54% forward/backward at 128K; -13% to -17% throughput going from 4096 to 1024 at 1000K | bounds the fp32 combine transient against routing skew. Not a length lever -- see below |

**Those numbers came off two specific models: treat them as a reference point rather than a prediction.** The measurements are 8xH200 nodes in bf16 with ZeRO-2, FlashAttention-3 and full activation checkpointing -- the MoE figures on Qwen3.6-35B-A3B at the configurations named under [Measured ceilings](#measured-ceilings), the vocabulary figure on a dense 27B with a 248320-token vocabulary.

The ranking is the part that holds up best: DeepEP chunking really is an order of magnitude dearer than the head, and activation offload really is close to free. Magnitudes usually land in the same band on a model of similar shape. The exact percentages will not reproduce, and one entry can change category outright rather than drift -- tiled MLP is a large saving on a wide dense FFN and too small to measure on this MoE, where the difference it makes sits inside the run-to-run spread.

What each cost tracks is what lets you carry a figure to a model of a different shape:

| Knob | Its cost scales with |
| --- | --- |
| lm-head chunking / fused CE | vocabulary against hidden size. Large vocabulary, large saving; a small one makes the knob pointless |
| `offload_optimizer` | the CPU Adam step as a fraction of the whole step, so it falls as sequence length grows and rises on a host with slower memory or fewer cores |
| `ac_config.offload_config.enabled` | whether the boundary copies still fit under compute. It stays near zero until the PCIe link saturates, then it stops hiding |
| `tiled_mlp_token_chunk_size` | the intermediate width of the FFN being tiled, and the number of shards it is cut into |
| `sp_size` | the share of time in full attention, and whether the SP group crosses the node boundary |
| `deepep_token_chunk_size` | the MoE layer count and how many dispatches the chunk size implies |
| `activation_checkpointing` | one extra forward pass, so it lands near +30% for any transformer. The 29% figure is the attention-dominated long-context case |

**Optimizer offload sits at its cost because it is usually already on.** When it is not, it comes before everything else regardless of cost, because activation offload cannot move a wall that Adam state is setting. At sp4·ep2 on Qwen3.6-35B-A3B with neither offload the card fills at about 32K tokens; that term is constant in sequence length, so lowering `ep_size`, streaming the checkpoint boundaries and chunking the head do not touch it. Optimizer offload takes it to ~790K, and only then does streaming the checkpoint boundaries buy the further ~1.15M. Read `DSS_MEM_BREAKDOWN` first and attack whichever term is binding; the ladder breaks ties.

**Activation checkpointing is the knob a job is most likely to be paying without having chosen it,** since it defaults to `True`. Its 29% is the one figure in the table that is not a subtraction between two runs: every measurement in this document was taken with full checkpointing on, so the cost comes from the FLOPs accounting instead -- full-mode recompute is one extra forward pass, and at 1.15M tokens attention dominates enough that the causal `S^2 d` coefficient going from 7 to 9 is the whole of it. The recompute never appears in the TFLOPS column, because hardware FLOPs utilization counts it as work the GPU genuinely did; it appears in wall-clock seconds per 100K tokens. Turning it **down** is a graded option whenever memory is not the binding term: `activation_checkpointing` takes `False` to stop paying the recompute outright, or a positive integer to checkpoint every Nth block (`activation_checkpointing_freq`, `gradient_checkpointing_freq` and `gradient_checkpointing` are aliases of the same field), and `ac_config.mode="selective"` with `targets` recomputes only the named submodules. Each recomputes less and saves less, and none of them combines with activation offload, which needs the single per-block boundary that only full mode produces -- so a run that streams its boundaries to host pays the recompute as well, whatever the ordering suggests.

### Notes on `sp_size` and `deepep_token_chunk_size`

**Take the smallest `sp_size` that fits, and try to keep it at or below the GPUs per node.** Under the node size the group stays on NVLink and the cost is the ~11% slope; above it the same traffic leaves the node every layer, on a fabric with far lower per-GPU bandwidth. A full-attention layer exchanges twice on the forward: sequence shards are traded for head shards before attention and traded back after. Query, key and value cross at the first point and the attention output crosses at the second, which is four `all_to_all` collectives as the code issues them -- three calls to `sequence_head_all_to_all` for query, key and value and a fourth for the output, all in AP's `_ulysses_attn_forward` (`arctic-platform` [`arctic_platform/model/implementations/qwen35/sequence_parallel.py`](../arctic_platform/model/implementations/qwen35/sequence_parallel.py)), and every one is differentiable, so the backward repeats them transposed. A GatedDeltaNet layer exchanges three times, packing its inputs into one; `dss-platform` [`docs/sequence_parallelism.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/sequence_parallelism.md) has both patterns with what each one moves. That is a step change, not a continuation of the slope, so `sp16` is a different regime from `sp8` rather than one more doubling of it, and it is worth timing on the target cluster rather than assuming a ratio. Degrees that do not divide the node size are legal but land some contiguous groups across the boundary and some inside it, which makes step time depend on which group a rank is in.

Before taking SP across nodes, verify the EFA network plugin itself. aws-ofi-nccl 1.17.2 retained about 10 KiB of host memory per cross-node collective and grew without a plateau; 1.21.1 was flat at 0.0002 MiB/step over 5000 steps of the same bare collective probe. Use the tested 1.21.1 plugin and confirm the run selected `NET/OFI` with `Using network Libfabric` rather than silently falling back to sockets. `dss-platform` [`docs/sequence_parallelism_memory_map.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/sequence_parallelism_memory_map.md#upgrading-the-plugin-removes-it) records the version and transport controls.

Smallest-that-fits is a throughput rule, and per-token time is where it shows: on Qwen3.6-35B-A3B each configuration at its own ceiling gives 14.63 secs/100K at `sp4*ep2` (1.15M) against 24.45 secs/100K at `sp8*ep2` (2.28M). The TFLOPS column moves the other way, 381 against 441, because both length and utilization rise together -- which is why the decision is made on seconds per 100K tokens at the length you actually need, not on utilization.

**`deepep_token_chunk_size` is a safety and throughput dial, not a way to reach a longer sequence.** The wall is set by the `O(tokens/sp)` resident activation, which chunking does not touch; if anything the extra collectives lower the ceiling slightly. What chunking does is bound the *data-dependent* spike. Unchunked, the fp32 combine transient scales with `tokens/sp` and reaches roughly 16.8 GiB at 2.2M tokens on Qwen3.6-35B-A3B with balanced routing -- it rides at the wall, and a skewed batch OOMs there. Chunked, a rank's per-chunk received tokens are capped at `ep x chunk` whatever the routing does.

Because the default is `None` -- unchunked, the whole local shard in one dispatch -- a ceiling measured without the flag does not carry that bound. Real data routes with more skew than uniform-random probe data does, so a config tuned to its wall on synthetic tokens needs headroom left for it. Reserve roughly `(ep - 1) x chunk x topk x (2 x hidden + 3 x moe_intermediate) x dtype_bytes`, which at `ep4` and chunk 8192 on this model is ~4 GiB and matches the measured ~4 GiB inter-rank peak spread. Or shrink the chunk until the budget fits the headroom already there: 1024 cuts it about eightfold.

At the wall the two settings trade cleanly rather than one dominating: sp8·ep2 on one node reaches 2.28M tokens at 441 TFLOPS with chunk 1024, and 2.18M at 482 TFLOPS with chunk 4096. Take 1024 when context-bound and 4096 when not.

## Measured ceilings

Two reference points, both on 8xH200 nodes in bf16 with ZeRO-2, optimizer CPU offload, full activation checkpointing with CPU offload at `keep_last_n=1`, `tiled_mlp_token_chunk_size=16384`, and FlashAttention-3. Every figure is a binary search: the longest length that completes two steps, adjacent to the first that OOMs.

**Dense, Qwen3.8-27B** (64 layers, hidden 5120, 24 query / 4 KV heads, vocabulary 248320), one node, `sp8·ep1`. The three rows differ only in how the head computes the loss.

| LM head | Max sequence | GPU peak | Host per rank | Binding wall |
| --- | --- | --- | --- | --- |
| Hugging Face cross-entropy, full logits | 195K | 131.69 GiB | 97.08 GiB | GPU |
| Chunked logprobs, `16384 x 8192` | 1.67M | 116.07 GiB | 179.59 GiB | host RAM |
| Liger fused cross-entropy | 1.67M | 109.96 GiB | 175.52 GiB | host RAM |

Per 1,000 rank-local tokens the unchunked head costs about 2.80 GiB of GPU total, of which 2.53 GiB is the logits tensor and its gradient. The other two rows cost 0.30 GiB and 0.27 GiB, so the card is no longer the limiter for either -- both stop at the same 10K-wide bracket against node RAM, with about 20 GiB of the card unused.

**MoE, Qwen3.6-35B-A3B** (40 layers: 10 full-attention, 30 gated-delta; hidden 2048; 16 query / 2 KV heads; 256 experts, top-8), Liger fused cross-entropy, DeepEP chunk 1024.

| Nodes | Config | Max sequence | TFLOPS/GPU (HFU) | GPU peak |
| --- | --- | --- | --- | --- |
| 1 | sp1·ep2 | 300K | 238 (24.0%) | 132.1 GiB |
| 1 | sp4·ep2 | 1.15M | 381 (38.5%) | 131.3 GiB |
| 1 | sp8·ep8 | 1.70M | 411 (41.6%) | 129.6 GiB |
| 1 | sp8·ep4 | 2.18M | 443 (44.7%) | 127.4 GiB |
| 1 | **sp8·ep2** | **2.28M** | 441 (44.6%) | 129.3 GiB |
| 2 | sp16·ep8 | 3.44M | 453 (45.8%) | 130.2 GiB |
| 2 | sp16·ep4 | 4.59M | 470 (47.5%) | 131.3 GiB |
| 2 | **sp16·ep2** | **4.64M** | 469 (47.4%) | 132.0 GiB |

At fixed `ep`, one node to two scales the context about twofold: 2.28M to 4.64M at ep2 (2.04x) and 2.18M to 4.59M at ep4 (2.11x). It then stops, because `sp` cannot exceed the model's 16 attention heads and two nodes already reach that. Comparing across different `ep` values hides this -- 1-node ep4 against 2-node ep8 reads as 1.56x because raising `ep` enlarges the cross-node combine transient, which is a second variable and not a scaling limit.

## What the two offloads move to host

Both offload knobs buy GPU memory with host RAM, and host RAM is a wall of its own rather than free space: every rank on the node draws from one budget, with a cgroup limit near 1900 GiB against 2 TB physical. Exceeding it kills the job as surely as filling the card, which is what "host RAM becomes the new wall" in the ladder means.

The two terms behave differently, and that difference is why they are worth separating. Optimizer state is fixed by the parameter count and the sharding, so it does not move with sequence length. Activation boundaries scale with tokens per rank until the pinned staging cache hits its retention cap, and then they plateau.

### Qwen3.8-27B dense, one 8xH200 node, sp8·ep1

Per-rank figures are measured; the per-node column is eight ranks of them.

| Term | Host per rank | Host per node | Measured at |
| --- | --- | --- | --- |
| Optimizer offload -- ZeRO-2 Adam state on CPU | 43.8 GiB | 350 GiB | flat across 500K to 680K: 43.55, 43.63, 43.72, 43.81 GiB |
| Activation offload -- 64 block boundaries at `keep_last_n=1` | 135.8 GiB | 1086 GiB | the difference at 1.0M to 1.67M, where the pinned cache has plateaued |
| **Both** | **179.6 GiB** | **1437 GiB** | 179.48 at 1.0M, 179.58 at 1.5M, 179.59 at 1.67M; 1.68M is a host OOM |

The optimizer figure is close to what the arithmetic predicts: Adam keeps an fp32 master weight plus two fp32 moments, 12 bytes per parameter, and ZeRO-2 shards that over the 8-rank group, giving `27e9 x 12 / 8` = 37.7 GiB. The measured 43.8 GiB is that plus pinned staging buffers and the process itself.

The activation term is the one that stops growing. It rises at about 5.3 GiB per 1,000 rank-local tokens and then flattens at ~179.6 GiB per rank, unchanged from 1.0M to 1.67M, because `pin_memory_max_size_gib` caps what the staging cache retains rather than letting it track the sequence.

### Qwen3.6-35B-A3B MoE, one 8xH200 node, sp4·ep2

Per-node figures are measured here; the per-rank column divides by eight.

| Term | Host per rank | Host per node | Measured at |
| --- | --- | --- | --- |
| Neither offload | negligible | negligible | ceiling is ~32K tokens |
| Optimizer offload -- ZeRO-2 Adam state on CPU | 62.5 GiB | 500 GiB | at its ~790K ceiling; constant in sequence length |
| Activation offload -- per-layer boundaries at `keep_last_n=1` | 94.5 GiB | 756 GiB | the difference at 1.17M |
| **Both** | **157 GiB** | **1256 GiB** | against a ~1900 GiB node limit |

The optimizer term is larger per rank than the dense model's despite a similar parameter count, because expert parameters do not shard over all eight ranks. `ep_size=2` halves the expert weights a rank holds, but their optimizer state is then sharded only across the expert-data-parallel replica group, which at `sp4·ep2` is four ranks rather than eight. Raising `ep` does not reduce this in the way the parameter count suggests.

### Reading these against the node limit

Two cautions before using the totals as a budget. The per-node column sums the ranks' resident sets and is a **lower bound** on what the node is using: at the dense model's 1.67M ceiling the sum of ranks is ~1437 GiB while the node itself measured ~1902 GB of 2000 GB, the remainder being the Ray object store, page cache, and everything else on the box. And the two terms are not independent budgets -- they compete for the same cgroup allowance, so the question at the wall is whether their sum plus that overhead clears the limit, not whether either fits alone.

The practical consequence is that on a long dense run the host term is dominated by activation offload by roughly three to one, so `pin_memory_max_size_gib` and `keep_last_n` are the knobs that move it. On the MoE model the split is closer to two to one, and the optimizer half of it cannot be reduced by any setting in this guide short of a different ZeRO stage.

### Splitting the boundaries between the two walls

The choice is not all 64 boundaries on the card or 63 of them on host. `keep_last_n` is where the set divides: a boundary is offloaded only once `keep_last_n` newer ones exist, so the N most recent stay resident and everything older streams out. Raising it hands bytes back from the host budget to the card one boundary at a time, and the offload keeps working on the rest.

One boundary is `tokens_per_rank x hidden x 2` bytes. On the dense 27B at its 1.67M ceiling a rank holds 208.75K tokens, so a boundary is `208750 x 5120 x 2` = 1.99 GiB on the card against 135.8 GiB per rank on host for the 63 that leave. That run's GPU peak is 109.96 GiB of a card's ~130, so about ten boundaries fit in the headroom -- 19.9 GiB per rank back onto the cards, 159 GiB per node off a budget whose limit is ~1900 GiB. Take fewer than the headroom allows: the resident boundaries add to the peak that measured 109.96, so claiming all of it puts the card back at the wall it was moved away from.

That arithmetic bounds the trade rather than predicting it. The card side is exact -- the boundary shape is known. The host side falls by at most the bytes no longer offloaded, and by less where the pinned staging cache is retaining to its capacity cap rather than tracking live boundaries, which is what the flat 135.8 GiB from 1.0M to 1.67M shows.

Step time does not pay for it. The resident boundaries are the ones backward reaches first, whose offload-then-immediate-reload is pure latency; keeping them removes a copy pair instead of adding one.

Which way to move it follows from which peak is closer to its limit: raise `keep_last_n` when the host figure times the ranks approaches the node limit and the card has room, leave it at 1 when the card is binding.

## Finding max seqlen for a given model

Follow these steps:

1. Run the target length once with the `step_peak_memory_log` field on and read both peaks. If the GPU peak is near 126-130 GiB the card is binding; if the host figure times the ranks approaches the node limit, host RAM is.
2. Split the binding peak with the `DSS_MEM_BREAKDOWN=1` environment variable. A large optimizer-state term means optimizer CPU offload; a large activation term means `ep_size`, activation offload, tiled MLP and `sp_size`.
3. Binary search the length rather than stepping toward the wall. Two passing points give a slope in GiB per 100K tokens; project the wall from it, overshoot, and bracket. Stopping once a pass and a failure are 10K apart is enough -- halving below that costs a model load and a training step to move the answer by under 2% of one rank's tokens.
4. Pin every knob explicitly in the recipe, including the ones at their defaults. `deepep_token_chunk_size` defaults to unchunked and `fused_lm_head_token_chunk_size` to unset, so a recipe that omits them is not the configuration it looks like.

## Reference recipes

Dense Hugging Face or Liger, one node, targeting the longest sequence:

```python
"training_config": {
    "model_provider": "liger",
    "sp_size": 8,
    "ep_size": 1,
    "attn_implementation": "flash_attention_3",
    "tiled_mlp_token_chunk_size": 16384,
    "activation_checkpointing": True,
    "ac_config": {
        "mode": "full",
        "freq": 1,
        "offload_config": {"enabled": True, "keep_last_n": 1},
    },
    "ds_config": {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        }
    },
    "step_peak_memory_log": True,
}
```

With `model_provider="huggingface"` there is no fused cross-entropy, so add `"fused_lm_head_token_chunk_size": 16384` and `"fused_lm_head_vocab_chunk_size": 8192` to keep the logits tensor bounded.

Prime-RL MoE, one node, same target:

```python
"training_config": {
    "model_provider": "prime_rl",
    "sp_size": 8,
    "ep_size": 2,
    "tiled_mlp_token_chunk_size": 16384,  # what the 2.28M ceiling was measured with; saves nothing here
    "activation_checkpointing": True,
    "ac_config": {
        "mode": "full",
        "freq": 1,
        "router_replay_recompute": True,
        "offload_config": {"enabled": True, "keep_last_n": 1},
    },
    "prime_rl": {
        "fused_cross_entropy": "liger",
        "deepep_token_chunk_size": 1024,
    },
    "ds_config": {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        }
    },
    "step_peak_memory_log": True,
}
```

Raise `deepep_token_chunk_size` to 4096 when the run is throughput-bound rather than context-bound: it recovers about 14% of throughput for about 100K tokens of context on this model.

## If the knobs aren't enough to meet the required seqlen

What happens if all the knobs are on and you're still struggling with reaching the target seqlen.

Often times this could be an indication of a bug where one or more of the knobs are configured but misfire.

But more likely there is some inefficient code that needs a very large gpu peak memory. For example, manifesting full logits is one such example and we already have knobs to overcome it. The next thing to do to dianose the issue is to run your code under torch cuda memory profiler while keeping your seqlen short enough so that you don't get OOM, generate the memory profiler trace and then study it and fix the code in question.

The following sections will guide you through the process.

### Reading the memory allocator profiler

DSS is already instrumented to run the memory profiler, so it just needs to be enabled.

- `step_peak_memory_log` reports the peak GPU and host figures, which settles which wall is binding.
- `DSS_MEM_BREAKDOWN=1` splits the GPU peak into parameters, gradients, optimizer state, activations and allocator slack, which settles which category is large. Neither names a line of code, so when the binding category is activations and every knob left on the ladder costs step time, the open question is which tensor. That is the one the allocator profiler answers, and it is how a submodule that allocates a single oversized buffer gets found and fixed.

Reading its output takes the allocator's own vocabulary, because the quantity the card limits is not the one the model asks for. PyTorch does not go to the driver per tensor: it reserves large **segments** and carves **blocks** out of them, reusing a freed block for the next request that fits. **Allocated** is the bytes that live tensors hold. **Reserved** is the bytes those segments hold from the card, and reserved is what runs out -- an OOM is the allocator failing to obtain another segment, which can happen while allocated still has room. The gap between the two is slack: freed blocks not yet reused, plus holes the wrong size for what is being asked for next. Keeping that gap small is what `expandable_segments:True` is for.

A snapshot records that machinery instead of summarizing it. With history enabled the allocator keeps a ring buffer of every allocation and free, each tagged with the Python stack responsible, and the dump writes that history together with the current segment layout into a pickle. Both halves are load-bearing. The history is a timeline, which locates a peak that a short-lived transient sets -- a per-step maximum reports only its height, never where in the step it happened. The layout is a map, which shows fragmentation directly rather than leaving it inferred from a reserved-minus-allocated gap.

The worker wires all of this to `DSS_MEM_SNAPSHOT`, and the rest of this section is the workflow: turn recording on, collect the pickle from the one rank that writes it, and read the plots it supports.

### Turning it on

Set these on the gateway or launcher process; `worker_env()` forwards the whole family to every worker.

```bash
DSS_MEM_SNAPSHOT=/some/shared/path   # path prefix; setting it is what enables recording
DSS_MEM_SNAPSHOT_RANK=0              # default 0 -- one rank records, not all of them
DSS_MEM_SNAPSHOT_ITERS=3             # default 3 -- dump after this many completed fwd/bwd
DSS_MEM_SNAPSHOT_ENTRIES=200000      # default 200000 -- ring buffer of allocation events
DSS_MEM_SNAPSHOT_STACKS=python       # default python -- C++ frames are noise at this level
```

Recording starts at the first `/fwd-bwd`. After `DSS_MEM_SNAPSHOT_ITERS` completed iterations the worker writes `<prefix>-rank<N>.pickle`, stops recording, and logs `[mem-snapshot] dumped`. Run it **below** the wall: the dump is triggered by a completed iteration, so a job that OOMs never produces one. For the OOM itself the failure path emits its own allocator snapshot without any flag, which `dss-platform` [`docs/memory_diagnostics.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/memory_diagnostics.md) describes.

### Viewing the trace

Drop the pickle on <https://pytorch.org/memory_viz>, or render it locally:

```bash
python -m torch.cuda._memory_viz trace_plot   snap-rank0.pickle -o trace.html
python -m torch.cuda._memory_viz segment_plot snap-rank0.pickle -o segments.html
python -m torch.cuda._memory_viz compare      before.pickle after.pickle -o diff.html
```

`trace_plot` draws allocated memory over time and attributes every block to the Python stack that allocated it, which is the view that names the offending module. `segment_plot` shows how those blocks pack into allocator segments, which is where fragmentation is visible rather than inferred. `compare` flamegraphs what changed between two snapshots -- the direct way to confirm a knob did what it claimed instead of trusting a peak number that moved for another reason.

If <https://pytorch.org/memory_viz> crashes, when the trace is 1GB or higher, use this tool instead https://github.com/Da1sypetals/SnapViewer

### What to look for

- **A tall narrow spike.** One allocation, briefly held, that sets the high-water mark by itself. The stack names the module, and a shape of `[tokens, something-wide]` marks it as a chunking candidate. The lm-head logits tensor and the fp32 MoE combine transient both look like this.
- **A term that does not shrink when you shard.** Snapshot at two `sp_size` values and `compare` them. Whatever is unchanged is not token-parallel -- parameters, optimizer state, or a buffer sized by the model rather than the batch -- and no activation knob will move it.
- **A wide plateau instead of a spike.** Memory resident across the whole backward is saved state, not a transient. That is the activation-offload case; `DSS_ACT_OFFLOAD_LOG=1` then confirms whether the hooks are actually covering it.
- **Reserved sitting far above allocated.** Fragmentation rather than demand. Confirm `expandable_segments` is on before spending a knob on it.

Recording timestamps every allocation and walks a Python stack for each one, so it costs step time. Treat throughput measured with a snapshot running as indicative, and re-measure with it off.

## See also

- `dss-platform` [`docs/sequence_parallelism_memory_map.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/sequence_parallelism_memory_map.md) -- detailed GPU and host measurements behind the sequence-parallel memory guidance, including the token-independent floor and cross-node transport behavior.
- `dss-platform` [`docs/sequence_parallelism.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/sequence_parallelism.md) -- what an SP degree costs, the topology rules, and the supported layouts.
- `dss-platform` [`docs/activation_offload.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/activation_offload.md) -- every `ac_config` field, the pinned staging cache, and why MoE needs router replay.
- `dss-platform` [`docs/tiled_mlp.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/tiled_mlp.md) -- which modules are tiled on each provider, and the ZeRO-3 shard-count agreement.
- `dss-platform` [`docs/memory_diagnostics.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/memory_diagnostics.md) -- the three observability modes and what each answers.
- `dss-platform` [`docs/cuda_allocator.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/cuda_allocator.md) -- allocator configuration and its precedence.
- `dss-platform` [`docs/packing.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/packing.md) -- the token-budget packing contract shared by every training path.
- `dss-platform` [`docs/lm_head_gradient_precision.md`](https://github.com/snowflake-eng/dss-platform/blob/main/docs/lm_head_gradient_precision.md) -- why `fp32_lm_head` exists and what it costs.
