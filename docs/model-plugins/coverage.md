# Model plugin coverage of the built-in implementations

How much of each built-in model implementation in `arctic_platform/model/implementations` (main @ `75cd04e`, 13 implementations) could be delivered as a customer model plugin under contract draft 1.1 ([spec](spec.md)), and what each one is missing. Median coverage: **75%**.

## What the survey shows

1. **The model math fits.** Attention variants, linear attention, short convolutions, hyper-connections, qk-norms, partial and NoPE rotary, sigmoid and biased routing, shared experts, muP and softcapping all land in plugin code or existing hooks. The gaps are mostly platform performance and scale features.
2. **One cheap extension unblocks 7 of 13.** The loader only recognises `model.layers.<i>.` and has no converter for tensors outside decoder layers. A declared layer-name pattern plus a top-level converter removes the first blocker for every family with `model.language_model.*`, `model.llm.*`, `backbone.*` or prefix-less checkpoints.
3. **The platform must expose its fast paths.** Fused attention behind the platform core, DeepEP with a grouped-GEMM experts core, and sharded checkpoint loading decide whether a plugin is competitive, and whether models above about 60B load at all.
4. **Nothing runs in production yet.** No loader or training-service path uses the plugin API. Three families pass as plugins functionally at small sizes, but only through the reference host.

## Per implementation

Coverage = (supported + half of partial) / capabilities needed, rounded to 5%. S / P / M: supported, partial (works through the contract's reference path, or with a limit), missing.

| Implementation | Coverage | S / P / M | As a plugin today | Biggest blocker |
|---|---|---|---|---|
| `inkling` | 85% `████████▌` | 11 / 3 / 1 | No | Checkpoint uses a model.llm.* prefix the loader does not recognise |
| `glm52` | 80% `████████` | 8 / 3 / 1 | Functionally, bf16 at SP=1 | Mixture-of-experts performance: all-gather EP over 256 experts |
| `qwen38` | 80% `████████` | 10 / 5 / 1 | No | Checkpoint uses a model.language_model.* prefix |
| `afmoe` | 75% `███████▌` | 11 / 8 / 1 | Functionally, Nano and Mini | Reference attention core: no fused sliding-window kernel, dense T x T masks |
| `deepseek_v4` | 75% `███████▌` | 10 / 4 / 2 | No | Published checkpoint has no model. prefix and top-level tensors the loader cannot convert |
| `nemotron_h` | 75% `███████▌` | 9 / 5 / 1 | No | backbone.* checkpoint prefix and renames outside decoder layers |
| `qwen35` | 75% `███████▌` | 9 / 4 / 2 | No | VLM checkpoint uses a model.language_model.* prefix |
| `glm4_moe` | 70% `███████` | 10 / 6 / 2 | No | Smallest real checkpoint (Air, ~212 GB bf16) exceeds one GPU under load-then-shard |
| `minimax_m2` | 70% `███████` | 8 / 6 / 2 | No | Only real size (~460 GB bf16) does not load under load-then-shard |
| `qwen3_moe` | 70% `███████` | 7 / 8 / 1 | Functionally, 30B at SP up to 4 | No fused attention kernel behind the platform core; 235B does not load |
| `deepseek_v41` | 65% `██████▌` | 8 / 4 / 4 | No | Engram: ~200 GB of frozen FP8 tables with no sharded-table primitive |
| `glm53` | 65% `██████▌` | 6 / 7 / 2 | No | Checkpoint uses a model.language_model.layers.N prefix |
| `dflash2` | 45% `████▌` | 4 / 0 / 5 | Does not fit | Custom objective over a frozen target model |

## What is missing, and who should own it

A **plugin extension** is a hook worth adding to the contract; a **platform capability** stays platform-owned and is exposed to plugins.

### Plugin extensions

| Gap | What | Needed by | Note |
|---|---|---|---|
| Checkpoint naming | A plugin-declared layer-name pattern and a converter for tensors outside decoder layers. Today the loader only recognises model.layers.<i>. | `qwen35`, `qwen38`, `nemotron_h`, `inkling`, `deepseek_v4`, `deepseek_v41`, `glm53` | The first blocker for 7 of 13; small change |
| Cross-layer state under SP | Let a mixer hand tensors to later layers (compressed KV, top-k) under sequence parallelism, beyond mix's single output. | `deepseek_v41`, `glm53` |  |
| Config-derived pass-through | Pass-through patterns computed from the config, for MTP layers stored as model.layers.N. | `glm52`, `glm53` |  |
| Quantized source checkpoints | Declare a quantized source (FP8, MXFP4) that the converter dequantizes, with bf16 exports. | `deepseek_v4`, `deepseek_v41` |  |
| Layers without a sequence mixer | Allow token-wise layer types, such as MoE-only blocks. | `nemotron_h` |  |
| Recompute replay beyond MoE | Generalise the routing tape to other discrete choices, such as a sparse-attention indexer's top-k. | `qwen38` |  |
| State-passing sequence parallelism | A collective that carries recurrent state between ranks, for linear-attention layers. | `qwen38` |  |
| Sharded frozen tables, build-time tokenizer | A row-sharded frozen lookup table with id exchange, and build-time access to the tokenizer. | `deepseek_v41` |  |
| Post-weight-sync hook | Let the sampler model refresh caches derived from its weights after each sync. | `glm53` |  |
| Module roles | Declare roles (attention projection, experts, token-wise FFN) for selective checkpointing and tiled MLP. | all | Performance |
| Per-token context for routing | Hash-routed layers need input_ids inside route. | `deepseek_v4` | Addressed: route now forwards extra arguments |

### Platform capabilities

| Gap | What | Needed by | Note |
|---|---|---|---|
| Production wiring | No loader or training-service path uses the plugin API yet. Every verdict here is about the contract, not a running job. | all | Comes first |
| Fused attention behind the platform core | AttentionModule.mix is a reference SDPA with a dense T x T mask (4 GiB per layer at 64K tokens); FlashAttention varlen and sliding window are not swapped in. | all attention families | Promised by spec §7.3 |
| Fast expert parallelism | DeepEP/UCCL dispatch and a grouped-GEMM or SonicMoE experts core behind StackedExpertLayer. Today EP is all-gather plus all-reduce. | all MoE families | Spec §16 Q1 |
| Sharded checkpoint load | Each rank reads only its expert slice. Today every rank allocates the full model first, which rules out models above about 60B. | `glm4_moe`, `minimax_m2`, qwen3_moe 235B, afmoe Large |  |
| KV-head replication in SP | Allow sp above the KV-head count in the platform attention mixer. | `qwen3_moe`, `afmoe`, `qwen35` |  |
| Platform attention variants | Sparse MLA, DeepSeek V4 sparse attention and FP8 indexers as platform-owned kernels a plugin can call. | `deepseek_v4`, `deepseek_v41`, `glm52`, `glm53` |  |
| Triton kernels for recurrent layers | Causal convolution and Mamba2 SSD without precompiled wheels. | `qwen35`, `nemotron_h`, `glm53` |  |
| Sampler router replay for hybrids | vLLM 0.33 captures routing only with prefix caching on, so stateful layers need align-mode state caching. | `qwen35`, `qwen38`, `nemotron_h`, `inkling`, `glm53` | Next in this PR |
| LM head, weight sync, offload | Vocabulary tiling and fused cross-entropy, packed vLLM weight transport, activation CPU offload. | all | Performance |
| FP8 storage and GEMMs | An fp8 dtype mark and platform fp8 linear for frozen bases. | `deepseek_v4`, `deepseek_v41`, `glm52`, `glm53`, `qwen35` | Mostly with LoRA |

### Out of scope for 1.x

| Gap | What | Needed by | Note |
|---|---|---|---|
| LoRA | Including expert LoRA; plugin LoRA targets are recorded with mark(lora_target=True) but not used. | `qwen3_moe`, `glm4_moe`, `minimax_m2`, `afmoe`, `qwen35`, `inkling`, `glm52`, `glm53`, `deepseek_v4`, `deepseek_v41` |  |
| Custom objectives and composite models | Draft-model training over a frozen target. | `dflash2` |  |
| Multimodal training, TP, PP | Vision towers ride along as pass-through tensors; they are not trained. | `qwen35`, `qwen38`, `deepseek_v41`, `glm53` |  |

## Evidence by implementation

Partial and missing capabilities, with the code that needs them. Paths are relative to `arctic_platform/model/implementations/` unless they start with another package.

<details><summary><code>inkling</code>: 85% (11 supported, 3 partial, 1 missing)</summary>

Full and sliding attention with a learned relative-position bias; short convolutions on K and V and around attention and MLP; sigmoid MoE with router-gated shared experts; muP logits.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Missing | Checkpoint layout | Tensors under model.llm.layers.*, a renamed attention block and interleaved w13 never reach the per-layer converter. | extension | `inkling/weights.py:42-63` |
| Partial | Expert-parallel performance | Works through the contract's all-gather EP; chunked DeepEP/UCCL dispatch is not exposed. | platform | `inkling/modeling_inkling.py:127-161` |
| Partial | Tokenizer without remote code | A custom vLLM tokenizer mode suggests the HF tokenizer may not pass the weights-directory check. Not verified. | extension | `inference/server/config.py:186` |
| Partial | Sampler router replay | Short-convolution state on the sampler needs align-mode prefix caching for vLLM to capture routing. | platform | `spec §7.7` |

</details>

<details><summary><code>glm52</code>: 80% (8 supported, 3 partial, 1 missing)</summary>

MLA with DSA sparse attention (q/k 576 wide, v 512), IndexShare between layers, dense first layers, sigmoid MoE with correction bias and shared experts.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Partial | Sparse-MLA kernels | The built-in defaults to the pure-PyTorch reference, which a plugin can match; the opt-in tilelang/FlashMLA kernels are not exposed. | platform | `loaders/glm_moe_dsa.py:60` |
| Partial | Expert-parallel performance | DeepEP, SonicMoE and grouped GEMM are not exposed. | platform | `glm52/model_builder.py:66-79` |
| Partial | MTP layer at model.layers.N | Pass-through patterns are static, so the MTP layer index must be hard-coded per model size. | extension | `correctness/harness/inference.py:611-620` |
| Missing | Frozen FP8 base weights | No fp8 storage or fp8 linear; only needed for the LoRA mode. | out of scope | `glm52/.../sparse_mla_attention.py:212-238` |

</details>

<details><summary><code>qwen38</code>: 80% (10 supported, 5 partial, 1 missing)</summary>

GatedDeltaNet with QSA gated sparse attention, hyper-connections, per-layer n-gram embeddings with a dilated short convolution, 512-expert MoE.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Missing | Checkpoint layout | model.language_model.layers.* and PLE shard concatenation are outside the fixed model.layers.<i>. pattern. | extension | `qwen38/converting_qwen4_exp.py:24-55` |
| Partial | Sparse-attention top-k on recompute | Only MoE routing is replayed on recompute; the QSA indexer's discrete choice can flip under nondeterministic kernels. | extension | `qwen38/qsa_flex.py:640` |
| Partial | GatedDeltaNet sequence parallelism | Head-parallel works; FLA's state-passing policy needs a collective that carries recurrent state. | extension | `qwen38/context_parallel.py:94` |
| Partial | Frozen n-gram table | Works, but replicated on every rank: no vocabulary-parallel embedding. | platform | `qwen38/modeling_qwen4_exp.py:50-122` |
| Partial | Expert-parallel performance | DeepEP, grouped GEMM and SonicMoE are not exposed. | platform | `moe/layers/moe.py:543-644` |
| Partial | Sampler router replay | GatedDeltaNet and convolution state need align-mode prefix caching. | platform | `spec §7.7` |

</details>

<details><summary><code>afmoe</code>: 75% (11 supported, 8 partial, 1 missing)</summary>

Sliding and global attention 3:1 with NoPE on global layers, per-head qk-norm and a sigmoid output gate, sandwich fp32 norms, muP embeddings, sigmoid MoE with shared experts.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Partial | Attention kernel | Runs on the reference core, which builds a dense T x T mask per layer. | platform | `afmoe/modeling_afmoe.py:207-230` |
| Partial | Expert-parallel dispatch | DeepEP/UCCL not exposed. | platform | `loaders/generic_moe.py:46` |
| Partial | Grouped GEMM / SonicMoE | No platform experts core. | platform | `moe/layers/moe.py:365-384` |
| Partial | Sequence parallelism | 4 KV heads cap SP at 4: no KV-head replication in the platform attention mixer. | platform | `moe/sequence_parallel.py:41-50` |
| Partial | Activation checkpointing modes | Full per-layer checkpointing only; selective targets and CPU offload are missing. | platform / extension | `model/config.py:81-103` |
| Partial | Checkpoint load at scale | Every rank allocates the full model before experts are sharded: Trinity-Large does not load. | platform | `moe/hf_loader.py:123-264` |
| Partial | Weight-sync transport | Functional with per-expert HF names; the packed vLLM layout is not used. | platform | `moe/vllm_weights.py` |
| Partial | LM head | Token chunking only; no vocabulary tiling or fused cross-entropy. | platform | `gpu/lm_head.py` |
| Missing | LoRA | Plugin LoRA is not supported in 1.x. | out of scope | `model/patches/peft.py:172-190` |

</details>

<details><summary><code>deepseek_v4</code>: 75% (10 supported, 4 partial, 2 missing)</summary>

One shared K=V head with sinks, sliding-window plus compressed sparse attention with an FP8 indexer, hyper-connection residual streams, hash-routed first layers, sqrtsoftplus MoE; FP8/MXFP4 checkpoint.

The hash-routed layers row moved from Partial to Supported after `route` started forwarding extra arguments; this raised the score from 70% to 75%.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Missing | Checkpoint names | 72k keys with no model. prefix; top-level tensors (embed, head, hc_head) have no converter. | extension | `deepseek_v4/converting_deepseek_v4.py:4-6` |
| Partial | Sparse-attention kernel | Plugin must provide an eager or Triton path; the platform's tilelang/FlashMLA/cuDNN kernels are not exposed. | platform | `kernels/deepseek_v4/dsv4_sparse_attn.py:22` |
| Partial | Expert-parallel dispatch, MXFP4 experts | Functional through all-gather EP; DeepEP and grouped GEMM not exposed. | platform | `deepseek_v4/moe.py:117-139` |
| Partial | FP8/MXFP4 source checkpoint | Per-layer dequantization in the converter works; the contract has no notion of a quantized source exported as bf16. | extension | `models/.../modeling_deepseek_v4.py:177-195` |
| Partial | Layer-type names | The built-in adds names to transformers' allowed list at import time; legal only if the runtime ships transformers 5.15 or later. | platform | `configuration_deepseek_v4.py:13-19` |
| Missing | Frozen FP8/MXFP4 bases | LoRA-only mode. | out of scope | `fp8.py:96-147` |

</details>

<details><summary><code>nemotron_h</code>: 75% (9 supported, 5 partial, 1 missing)</summary>

Hybrid of Mamba2, GQA attention without RoPE and MoE-only blocks; LatentMoE with relu^2 experts and sigmoid group-limited routing.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Missing | Checkpoint layout | backbone.layers.*, backbone.embeddings and norm_f are outside the loader's pattern. | extension | `nemotron_h/converting_nemotron_h.py:13-16` |
| Partial | Mamba2 SSD kernel | The Triton SSD ships inside mamba_ssm, a precompiled wheel plugins cannot depend on. | platform | `nemotron_h/modeling_nemotron_h.py:66-67` |
| Partial | Attention kernel | Reference core only. | platform | `layers/attn.py` |
| Partial | Decoder layers with no sequence mixer | Every layer_types entry must bind a mixing module; MoE-only blocks need a no-op mixer. | extension | `nemotron_h/modeling_nemotron_h.py:324-337` |
| Partial | Expert-parallel performance | DeepEP on latent activations not exposed. | platform | `moe/layers/moe.py:1429-1484` |
| Partial | Sampler router replay | Mamba state needs align-mode prefix caching. | platform | `spec §7.7` |

</details>

<details><summary><code>qwen35</code>: 75% (9 supported, 4 partial, 2 missing)</summary>

GatedDeltaNet linear attention with gated full attention (partial RoPE, sigmoid output gate), MoE with a sigmoid-gated shared expert; VLM checkpoints with vision and MTP towers.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Missing | VLM checkpoint layout | model.language_model.layers.* does not match the loader's layer pattern. | extension | `qwen35/.../modeling_qwen3_5_moe.py:999-1011` |
| Missing | Fine-grained FP8 checkpoints | No FP8 storage or matmul capability. | platform | `fp8.py:156-195` |
| Partial | Attention kernel | Reference SDPA is O(T^2) memory at long context. | platform | `qwen35/.../modeling_qwen3_5_moe.py:505-551` |
| Partial | Attention sequence parallelism | 2 KV heads cap SP at 2 without KV-head replication. | platform | `qwen35/sequence_parallel.py:49-62` |
| Partial | Expert-parallel performance | Platform MoE kernels not exposed. | platform | `moe/layers/moe.py:543-644` |
| Partial | Sampler router replay | GatedDeltaNet state needs align-mode prefix caching. | platform | `spec §7.7` |

</details>

<details><summary><code>glm4_moe</code>: 70% (10 supported, 6 partial, 2 missing)</summary>

GQA with qkv bias and partial rotary; dense first layers then sigmoid MoE with correction bias and one shared expert; MTP layer ignored.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Missing | Checkpoint load at real size | Every rank allocates the full model before sharding experts. | platform | `moe/hf_loader.py:123-264` |
| Partial | Attention kernel | Reference core only. | platform | `layers/attn.py:95-111` |
| Partial | Expert-parallel dispatch | DeepEP/UCCL not exposed. | platform | `loaders/generic_moe.py:46` |
| Partial | Grouped GEMM / SonicMoE | No platform experts core. | platform | `moe/layers/moe.py:365-384` |
| Partial | Activation checkpointing modes | Full per-layer only. | platform / extension | `model/config.py:81-103` |
| Partial | Weight-sync transport | Functional, per-expert names. | platform | `moe/vllm_weights.py` |
| Partial | LM head | Token chunking only. | platform | `loaders/generic_moe.py:51-53` |
| Missing | LoRA | Not supported in 1.x. | out of scope | `model/patches/peft.py:172-190` |

</details>

<details><summary><code>minimax_m2</code>: 70% (8 supported, 6 partial, 2 missing)</summary>

GQA (48/8 heads) with per-layer qk-norm and partial rotary; every layer MoE with 256 experts, sigmoid routing with bias.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Missing | Checkpoint load at real size | Every rank allocates the full model before sharding experts. | platform | `moe/hf_loader.py:123-264` |
| Partial | Attention kernel | Reference core only. | platform | `layers/attn.py:95-111` |
| Partial | Expert-parallel dispatch | DeepEP/UCCL not exposed. | platform | `loaders/generic_moe.py:46` |
| Partial | Grouped GEMM / SonicMoE | No platform experts core. | platform | `moe/layers/moe.py:365-384` |
| Partial | Activation checkpointing modes | Full per-layer only. | platform / extension | `model/config.py:81-103` |
| Partial | Weight-sync transport | Functional; avoids the built-in's ArcticInference alias. | platform | `minimax_m2/vllm_weights.py:15-57` |
| Partial | LM head | Token chunking only. | platform | `loaders/generic_moe.py:51-53` |
| Missing | LoRA | Not supported in 1.x. | out of scope | `model/patches/peft.py:172-190` |

</details>

<details><summary><code>qwen3_moe</code>: 70% (7 supported, 8 partial, 1 missing)</summary>

GQA transformer with per-head qk-norm and RoPE scaling; every layer MoE with softmax top-k and no shared expert.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Partial | Attention kernel | FA2/3/4 varlen not behind AttentionModule.mix. | platform | `layers/attn.py:95-111` |
| Partial | Expert-parallel dispatch | The generic loader requires DeepEP/UCCL; the contract's EP is all-gather plus all-reduce. | platform | `loaders/generic_moe.py:46` |
| Partial | Grouped GEMM / SonicMoE | No platform experts core. | platform | `moe/layers/moe.py:365-384` |
| Partial | Sequence parallelism | 4 KV heads cap SP at 4. | platform | `moe/sequence_parallel.py:41-50` |
| Partial | Activation checkpointing modes | Full per-layer only. | platform / extension | `model/config.py:81-103` |
| Partial | Checkpoint load at scale | 30B fits; 235B does not, since every rank allocates the full model. | platform | `moe/hf_loader.py:123-264` |
| Partial | Weight-sync transport | Per-expert HF names; functional. | platform | `moe/vllm_weights.py` |
| Partial | LM head | Token chunking only. | platform | `gpu/lm_head.py` |
| Missing | LoRA, including expert LoRA | Not supported in 1.x. | out of scope | `model/patches/peft.py:172-190` |

</details>

<details><summary><code>deepseek_v41</code>: 65% (8 supported, 4 partial, 4 missing)</summary>

The V4 attention lineage with cross-layer shared compressed KV and top-k, shifted hyper-connections, Engram hashed n-gram FP8 tables, all-routed MoE; VLM wrapper checkpoint.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Missing | Engram tables | Two ~98 GB FP8 tables, row-sharded across EP ranks with an id all-to-all in the built-in; the contract can only shard stacked experts. | extension + platform | `deepseek_v41/engram.py:389-495` |
| Missing | Engram token map at load | Built from the tokenizer with remote code; plugins get no build-time access to the weights directory or tokenizer. | extension | `deepseek_v41/deepspeed_integration.py:110-136` |
| Missing | Checkpoint names | embed.weight, head.weight and layers.N carry no model. prefix. | extension | `deepseek_v41/converting_deepseek_v41.py:164-169` |
| Partial | Cross-layer shared attention state | mix returns one tensor; a source layer that emits compressed KV and per-query top-k for later layers only fits outside the stated mix signature. | extension | `deepseek_v41/attention.py:77-99` |
| Partial | Sparse-attention kernel | Eager path only. | platform | `deepseek_v41/attention.py:49-54` |
| Partial | Expert-parallel dispatch, MXFP4 experts | Not exposed. | platform | `deepseek_v41/moe.py:70-77` |
| Partial | FP8 source checkpoint | Dequantize-in-converter works; no contract notion of a quantized source. | extension | `deepseek_v41/dequantize.py:35-41` |
| Missing | Frozen MXFP4 experts | LoRA mode. | out of scope | `deepseek_v41/moe.py:70-77` |

</details>

<details><summary><code>glm53</code>: 65% (6 supported, 7 partial, 2 missing)</summary>

Hybrid of KDA linear attention and MLA with DSA, hyper-connections with Triton kernels, sigmoid MoE; composite VLM with MTP layers and an optional frozen FP8 checkpoint.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Missing | Checkpoint names | model.language_model.layers.N is never routed to the per-layer converter. | extension | `glm53/converting_glm5_next.py:26-27` |
| Partial | KDA kernels | fla Triton is fine; causal_conv1d is a precompiled wheel plugins cannot use. | platform | `glm53/context_parallel.py:782,869` |
| Partial | Sparse-MLA kernels | flashinfer/FlashMLA are precompiled and not exposed; a tiled reference exists. | platform | `glm53/context_parallel.py:370-540` |
| Partial | IndexShare under SP | mix returns one tensor; previous-layer top-k cannot be handed on under sequence parallelism. | extension | `glm53/context_parallel.py:611-614` |
| Partial | Expert-parallel performance | Not exposed. | platform | `moe/layers/moe.py:913` |
| Partial | Sampler router replay | KDA state needs align-mode prefix caching; the built-in relies on platform vLLM patches. | platform | `inference/server/weight_sync/receiver.py:419-421` |
| Partial | MTP layers | Static pass-through patterns. | extension | `correctness/harness/inference.py:611-620` |
| Partial | Sampler weight-sync caches | vLLM keeps merged conv and fp32 indexer copies the platform refreshes after each sync; plugins have no post-sync hook. | extension | `inference/server/weight_sync/receiver.py:385-411` |
| Missing | Frozen FP8 linears | LoRA mode. | out of scope | `glm53/modeling_glm5_next.py:42-82` |

</details>

<details><summary><code>dflash2</code>: 45% (4 supported, 0 partial, 5 missing)</summary>

Not a model family: speculative-decoding draft training over a frozen target model, with a custom objective.

| Status | Capability | What is missing | Verdict | Evidence |
|---|---|---|---|---|
| Missing | Frozen target model and hidden-state capture | The contract describes one model family; a composite teacher plus draft is outside it. | out of scope | `loaders/dflash2.py:104-108` |
| Missing | Custom loss | Custom losses are out of scope for 1.x. | out of scope | `dflash2/online.py:124-181` |
| Missing | Stochastic anchor sampling | Conformance requires deterministic forwards. | out of scope | `dflash2/online.py:103-112` |
| Missing | Draft-only export | Exports are full-model HF directories. | out of scope | `dflash2/online.py:184-207` |
| Missing | Serving as a vLLM speculative draft | The contract has no draft-model concept. | out of scope | `inference/vllm/config.py:134` |

</details>

## How this was measured

- Three reviewers read the implementations independently against the draft 1.1 spec and the `arctic_platform.model_plugin` code, with one shared rubric, and cited file and line for every capability.
- A capability counts as supported when a plugin can express it with the contract as written; partial when it works through the reference path or with a limit (sequence-parallel degree, model size); missing when the contract has no way to express it.
- Row granularity differs a little between reviewers, so treat differences under 10 points as noise. LoRA, ZoRRo, tensor and pipeline parallelism and multimodal training are excluded as out of scope for 1.x, except where a row records LoRA as missing.
- Spot checks against the code confirmed the main blockers: the loader's fixed `^model\.layers\.(\d+)\.` pattern, DeepSeek V4's prefix-less checkpoint keys, Inkling's `model.llm.*` rename and GLM-5.2's reference sparse-MLA default. Nothing was run end to end as a plugin.
