# DeepCompile / torch.compile coverage in the ZoRRO training path

Progress notes for enabling and widening graph-compilation coverage of the
DeepSpeed training engine used by the ZoRRO RL path
(`verl/examples/arctic_rl/run_gsm8k_grpo_arl_zorro_yes.sh`,
Qwen3-0.6B, GSM8K GRPO). Measured on 1 GPU.

## TL;DR

- A config-gated, name-safe compile capability is wired in two independent forms:
  - **DeepCompile** (`train.deepspeed.compile.deepcompile`) — DeepSpeed's
    `engine.compile()`, schedules ZeRO all-gather/reduce-scatter into the graph.
  - **Regional compile** (`train.regional_compile`) — in-place `layer.compile()`
    on each decoder layer, DeepCompile off.
- The untraceable ZoRRO host-side glue is fenced with `@torch.compiler.disable`
  so dynamo runs it eagerly and compiles the real per-layer compute around it.
- Both toggles are **default-off**; the committed run script leaves them off.
- At this scale (0.6B / 1 GPU) **neither gives a steady-state speedup**; compile
  payoff is blocked by dynamic sequence lengths (per-step recompiles). DeepCompile's
  real win (in-graph grad-reduce) is expected at multi-GPU / ZeRO-3.

## Changes

- `verl/verl/trainer/config/remote_backend/arctic.yaml`
  - `train.deepspeed.compile.deepcompile: False` (passed through to `deepspeed.initialize`).
  - `train.regional_compile: False` (new arctic-side knob, under `train` NOT
    `train.deepspeed`, to avoid DeepSpeed `CompileConfig` strict validation).
- `verl/verl/workers/remote_client/arctic_rl.py`
  - Reads `regional_compile` and surfaces it into `ds_worker_config`.
- `Arctic-Platform/arctic_platform/rl/deepspeed_worker.py`
  - After `deepspeed.initialize`: gated `self.engine.compile()` (DeepCompile) and a
    gated in-place per-decoder-layer `layer.compile()` block (regional). Both gated on
    `job_type != "log_prob"`.
- Glue fences (`@torch.compiler.disable`; no-op in eager and for the un-compiled
  log_prob engine):
  - `zorro_train.py` (10 staticmethods): `find_prompt_groups`,
    `create_deduplicated_batch`, `deduplicate_sequences`, `_extract_cu_seqlens`,
    `_get_sequences_packed_from_dedup_tensor`, `_get_responses_packed_from_dedup_tensor`,
    `_get_sequences_reconstructed_from_dedup_tensors`,
    `_replicate_and_concat_prompt_responses`,
    `extract_unpadded_responses_from_deduped_packed_ids`, `responses_in_orig_sample_order`.
  - `qwen_attention_patcher.py`: `_prepare_attention_kwargs_and_masks`,
    `Dedup_Cosine_Sine_Coeff`.
  - `qwen_model_patcher.py`: `tiled_`/`chunked_entropy_and_logprobs_with_temperature_from_logits`.
- `qwen_attention_patcher.py` (earlier fix): FA3 `max_seqlen_q/k` coerced to Python
  `int` (`_as_int`) so dynamo can pass a SymInt/int instead of a 0-d FakeTensor.

## Weight-sync constraint (hard)

Weight-sync validates strict name-set equality between
`engine.module.named_parameters()` and `compute_expected_hf_param_names`, and
`_orig_mod.` is never stripped. So compilation MUST be name-safe:

- `engine.compile()` and in-place `nn.Module.compile()` do NOT add `_orig_mod.` — safe.
- Assignment-style `module.layers[i] = torch.compile(...)` injects `_orig_mod.` — breaks sync.

Every run below logged `[weight-sync names validated] ... sender=310 expected=310`.

## A/B results (ZeRO-1, `grad_accum_dtype=bf16`, 4-step runs; both `rc=0`, weight-sync 310=310)

| Metric                    | Baseline (eager) | A: DeepCompile + fences | B: regional + fences |
|---------------------------|------------------|-------------------------|----------------------|
| Graph-break events        | 402¹             | 233                     | 186                  |
| Remaining break sites     | data-dependent (`.item()`, `torch.equal`, `nonzero`, dyn slicing) | clean `disable` boundaries only | per-layer attention boundaries only |
| Step-1 compile warmup     | ~269s            | 177s                    | 59s                  |
| Steps 2-4 (steady)        | ~33-35s          | ~33-35s                 | ~36s                 |
| Dynamic-shape recompiles  | -                | 76                      | 65                   |

¹ Baseline break count was a 1-step probe; A/B are 4-step, so their lower counts
understate the reduction. The key win is that all *failed-compile* breaks are gone;
what remains are clean `disable` handoffs.

## Recommendation

- Keep both toggles **off** by default for the current single-GPU / small-model config.
- Fences are always safe and widen coverage + cut warmup; keep them.
- DeepCompile (A) is the option to carry to the **multi-GPU / ZeRO-3 target** (in-graph
  grad-reduce). Regional (B) is the simpler, cheaper-warmup alternative.
- Highest-leverage follow-up: **dynamic sequence lengths** cause 65-76 recompiles/run and
  block compile payoff — address via `dynamic=True` or seqlen bucketing (out of scope so far).

## ZeRO-3 (resolved)

ZeRO-3 works on the ZoRRO path **without DeepCompile**. DeepCompile is the specific
blocker. 2-step smokes (`grad_accum_dtype=bf16`, fences in place):

| ZeRO-3 with…                | Result | Notes |
|-----------------------------|--------|-------|
| DeepCompile                 | FAILS  | `o_proj.weight` released (`vec (0)`) in post-break resume subgraph |
| eager (no compile)          | works  | rc=0, weight-sync 310=310, ~38s/step |
| regional per-layer compile  | works  | rc=0, weight-sync 310=310, warmup ~62s, ~37s/step |

Root cause: DeepCompile moves ZeRO-3 all-gather/release **into the compiled graph**,
but the data-dependent ZoRRO forward graph-breaks (now at the clean `torch.compiler.disable`
fence boundaries). DeepCompile does not re-gather the sharded params in the post-break
resume subgraphs, so a projection weight (`o_proj`) is used while still partitioned:

```
engine.py:2509 forward -> eval_frame compile_wrapper
  qwen_model_patcher.py:421/422 (resume after fenced deduplicate_sequences)
    qwen_attention_patcher.py:1156/1159/1260 (resume after fenced attention helpers)
      o_proj_dedup = module.o_proj(attn_output_dedup)
RuntimeError: size mismatch, got input (16831), mat (16831x2048), vec (0)
```

Eager and regional compile both keep DeepSpeed's **native** ZeRO-3 param coordinator
(eager forward hooks), which gathers each submodule's params on demand and is robust to
the fragmented forward. (This is the log-prob/`forward_no_grad` path in the colocated
setup, which reuses the training engine's compiled module.)

Optimizer note: bf16 with `grad_accum_dtype=fp32` instantiates `BF16_Optimizer`, which
DeepCompile's `init_z1` does not support (`bit16_groups`/`get_param_id`); use
`grad_accum_dtype=bf16` (→ `DeepSpeedZeroOptimizer`) if DeepCompile is ever used at ZeRO-1.

Direction (current): DeepCompile + ZeRO-3 IS a supported DeepSpeed combination, so the fix
is NOT to disable DeepCompile. The real blocker is that the ZoRRO forward graph-breaks so
heavily that projection weights are used in post-break resume subgraphs where DeepCompile has
not (re-)gathered them. The path forward is to make the ZoRRO forward **compiler-friendly**
(remove/relocate the data-dependent host logic out of the compiled compute so params are
gathered and used within the same graph). See the graph-break inventory below.

### How to run ZeRO-3 today (until the forward is made compiler-friendly)
- Eager: `zero_optimization.stage=3`, `compile.deepcompile=False`, `regional_compile=False`.
- Per-layer compile: `zero_optimization.stage=3`, `compile.deepcompile=False`,
  `regional_compile=True`.
- `compile.deepcompile=True` + stage 3 currently fails (see inventory) — the target to fix.

## ZoRRO graph-break inventory (what blocks DeepCompile under ZeRO-3)

The whole ZoRRO dedup/reconstruction runs **inside** the compiled `engine.forward`
(`engine.compile()` compiles `engine.module`, the Qwen3 causal-LM whose `forward` is patched).
So all the data-dependent host logic below is interleaved with the real per-layer compute.
Under ZeRO-3, DeepCompile schedules param all-gather/release into the graph; each break splits
the forward into a new resume subgraph, and the sharded projection weights are not re-gathered
there — so e.g. `o_proj.weight` is used while partitioned (`vec (0)` → size mismatch).

Observed failing frame chain (DeepCompile + ZeRO-3):
```
compile_wrapper
  qwen_model_patcher.py:560 patched_forward            (causal-LM: calls backbone)
    qwen_model_patcher.py:421 patched_forward          (backbone)
      :422 resume_at_421      (after deduplicate_sequences #1)
        :443 resume_at_422    (after deduplicate_sequences #2 -> decoder-layer loop)
          qwen_attention_patcher.py:1156 patched_forward_split_unpadded
            :1159 resume_at_1156   (after _extract_cu_seqlens)
              :1260 resume_at_1159 (after Dedup_Cosine_Sine_Coeff/helpers)
                o_proj_dedup = module.o_proj(attn_output_dedup)   # weight partitioned
RuntimeError: size mismatch, got input (16831), mat (16831x2048), vec (0)
```

### Zone 0 - top of causal-LM forward (`qwen_model_patcher.py` `patched_forward`, ~497-558)
- `ZoRRoTrain.find_prompt_groups(...)` (`:528`) - O(batch^2) Python loop with a data-dependent
  branch `if torch.equal(prompts[i], prompts[representative])` (`zorro_train.py:268-274`).
  `torch.equal` returns a Python bool used in control flow -> hard break.
- `ZoRRoTrain.create_deduplicated_batch(...)` (`:531`) - the dedup builder
  (`zorro_train.py:565+`): `torch.nonzero` (`:637`, data-dependent shape), pervasive
  `.item()`/`.tolist()`/`.cpu()`/`.numpy()` (`:687,:710,:546-547,...`), Python loops over
  `prompt_groups`/`segment_info` building variable-length lists.
- Python side effects: `sys.modules[...]` (`:500`), `os.environ.get` (`:516`),
  `timers.start` (`:497`), and `reconstruction_info.update(**...)` (`:544`) mutating the
  closure dict shared by the patched forwards.

### Zone 1 - backbone forward (`qwen_model_patcher.py` `patched_forward`, ~419-454)
- `ZoRRoTrain.deduplicate_sequences(position_ids, ...)` and `(hidden_states, ...)`
  (`:421,:422`) - reconstruction-info-driven gather/index; each is a break (the `:422`/`:443`
  resume frames above).
- `causal_mask_mapping[attention_type]` (`:441`) - dict indexing keyed by a per-layer runtime
  attribute. (The `for decoder_layer in module.layers[:num_hidden_layers]` loop itself is
  fixed-length/traceable; the breaks are inside each layer.)

### Zone 2 - patched attention (`qwen_attention_patcher.py` `patched_forward_split_unpadded`, ~1156-1260)
This zone dominates (per-layer, ~28x) and is where `o_proj` dies.
- `ZoRRoTrain._extract_cu_seqlens(...)` (`:1156`) - `.diff().max().item()` and **mutates**
  `reconstruction_info` every layer (side effect).
- `Dedup_Cosine_Sine_Coeff(...)` (`:1159`) - `.tolist()` (`:1085-1086`) and variable-length
  Python list building over `group_sizes` (`:1095-1121`).
- `debug_object["patched_counter"] == debug_object["capture_at_invocation"]` branches
  (`:1164,:1243,:1262`) and `debug_object[...] = ...` mutations - data-dependent Python control
  flow + dict side effects on every layer.
- `ZoRRoTrain._get_sequences_packed_from_dedup_tensor` (`:1181-1184`),
  `_get_responses_packed_from_dedup_tensor` (`:1186`),
  `_get_sequences_reconstructed_from_dedup_tensors` (`:1220,:1223`),
  `_replicate_and_concat_prompt_responses` (`:1238`) - all use `.item()`/`.tolist()` and
  Python loops over `prompt_groups` (`zorro_train.py:832-899,:953+`).
- `self._prepare_attention_kwargs_and_masks(...)` (`:1195`) -> `_as_int` = `int(v.item())`
  (`:366-367`) plus per-sequence `.item()` loops (`:207-215,:267-277`).
- `attention_interface(...)` (`:1207,:1227`) - opaque FA3 custom op; a call boundary, not a
  Python break (kept clean by the `_as_int` fix that feeds it Python ints, not a 0-d FakeTensor).
- `module.o_proj(attn_output_dedup)` (`:1260`) - the victim: executes in the resume subgraph
  after all the above breaks, so its ZeRO-3-sharded weight is still partitioned.

### Zone 3 - post-backbone (causal-LM forward, ~571+)
- `ZoRRoTrain.extract_unpadded_responses_from_deduped_packed_ids(...)` (`:573`) - `.item()`,
  `torch.nonzero`, loops.
- Log-prob/entropy path `tiled_`/`chunked_entropy_and_logprobs_with_temperature_from_logits`
  (`qwen_model_patcher.py:1081/1117`) - flash-attn `cross_entropy` does pointer arithmetic
  (`%` on a data pointer) that dynamo cannot trace.
- `ZoRRoTrain.responses_in_orig_sample_order(...)` - reconstruction gather.

### Break-cause categories (summary)
1. Data-dependent Python control flow on tensor values: `torch.equal` in `if`
   (`zorro_train.py:274`); `debug_object` equality branches
   (`qwen_attention_patcher.py:1164/1243/1262`); `if len(cu_seqlens_dedup) > 1` (`:710`).
2. Tensor -> Python scalar/list materialization: `.item()`, `.tolist()`, `.cpu()`, `.numpy()`
   (throughout `create_deduplicated_batch` and the per-layer packing/reconstruct helpers).
3. Data-dependent output shapes: `torch.nonzero` (`zorro_train.py:637`).
4. Python loops over data-dependent structures (`prompt_groups`, `segment_info`, ranges from
   `.item()`) that build variable-length lists then `torch.stack`/`torch.cat`.
5. Python-object side effects: `reconstruction_info.update`/mutation, `debug_object` mutation,
   `sys.modules`, `os.environ`, `timers`.
6. Dict indexing by runtime values: `causal_mask_mapping[attention_type]`,
   `reconstruction_info[...]`.
7. Opaque custom op boundary: FA3 `attention_interface` (not a Python break; stays a call).

Note: the current `@torch.compiler.disable` fences convert most of (1)-(6) into *clean* break
boundaries (fewer, cheaper breaks) but do not remove the breaks — so under DeepCompile + ZeRO-3
the projection weights still land in resume subgraphs. Making DeepCompile + ZeRO-3 work requires
either moving this host logic out of the compiled compute (e.g. compute `reconstruction_info`
before the forward and pass concrete ints/tensors in) or ensuring each parametrized submodule's
gather and use sit in the same graph (e.g. `set_z3_leaf_modules` on the decoder layers so the
whole layer's params stay gathered across internal breaks).

---

## Fullgraph attention refactor (DeepCompile + ZeRO-3) — status & findings

Key realization: the Zone 1/2 breaks that run **before** the decoder-layer loop (find_prompt_groups,
create_deduplicated_batch, deduplicate_sequences) touch **no layer params**, so they do NOT strand
layer params. The stranding (`o_proj` `vec(0)`) came purely from the **intra-layer** breaks in the
split-attention forward. So the fix is to make each decoder layer break-free; then all its params
(input_layernorm, q/k/v, o_proj, mlp) land in one compiled graph and DeepCompile gathers them.

### What was implemented (works)
- `ZoRRoTrain.build_attention_gather_indices(reconstruction_info, total_dedup_tokens, device)`
  (`zorro_train.py`): precomputes, ONCE per forward (eagerly, in Zone 0), the token gather indices
  used by split attention: `prompt_gather_idx`, `response_gather_idx`, `recon_gather_idx`,
  `combine_gather_idx`, `group_sizes`. Each index is derived by running the ORIGINAL extraction
  helper on an `arange` "position" tensor (the helper's slice bounds come from cu_seqlens metadata,
  not the data), so the index-based path is provably identical to the legacy logic.
- Zone-0 precompute call + constant `max_seqlen_bound` in `qwen_model_patcher.py` causal-LM forward.
- `qwen_attention_patcher.py` `patched_forward_split_unpadded`: fullgraph branch (flag
  `ARCTIC_ZORRO_FULLGRAPH`, default on) that replaces the Python-loop/`.item()` extraction helpers
  with pure `index_select`, uses cu_seqlens tensors + constant int max_seqlen (no per-batch `.item()`
  / recompile), and recombines with a single `index_select` over `cat([prompt, response])`. Legacy
  branch retained for `ARCTIC_ZORRO_FULLGRAPH=0`.
- **Verified**: under DeepCompile + ZeRO-3 the model now passes through ALL decoder layers — the
  `o_proj` `vec(0)` stranding is gone. Weight-sync validated (310=310), DeepCompile z3 pass runs.

### Remaining blocker: `lm_head` (tied with `embed_tokens`)
The failure moved to the final projection. `lm_head.weight` **is** `embed_tokens.weight` (tied),
and `embed_tokens` is used inside the compiled backbone (DeepCompile manages it). Two dead ends,
both DeepCompile-internal:
1. **In-graph lm_head** (compute logits in the compiled region so DeepCompile gathers it): crashes
   DeepCompile's backward scheduler — `KeyError: 'view_869'` in `fast_free_schedule`
   (`deepspeed/compile/list_schedule.py`, via `add_z3_gather_release_bw`). The huge in-graph logits
   backward graph is mis-scheduled.
2. **Eager gather** (keep logits fenced; force `memory` mode = `TiledLogProbEntropy`, add
   `GatheredParameters(compute_params)` to its fwd/bwd): fwd log-prob path fixed by grad-gating the
   gather (so it doesn't nest inside `_z3_eager_inference` and re-partition the tied weight → the
   `'weight' must be 2-D` embed error). But the training backward then hits a dynamo guard:
   `tensor 'self.params[0]' rank mismatch. expected 1, actual 2` — because `GatheredParameters`
   flips the tied weight rank 1↔2 in place while DeepCompile's compiled graphs guard it as rank-1.
   `torch._dynamo.config.force_parameter_static_shapes = False` did NOT clear this assertion.

### Likely next steps for lm_head
- Custom autograd fn that all-gathers the weight shards into a SEPARATE tensor for the lm_head
  matmul (leaving the Parameter object rank-1 so DeepCompile's guards never see rank-2), and
  reduce-scatters the grad back into the tied param's grad buffer (needs to match DeepCompile's grad
  handling for the tied embed path — the fragile part), or
- Patch/avoid DeepCompile's `fast_free_schedule` bug so the in-graph lm_head path (option 1) works.

Flags/config added: `deepspeed_worker.py` forces `logits_optimization=memory` under DeepCompile+z3
and sets `torch._dynamo.config.force_parameter_static_shapes = False` at import.

---

## FINAL RESOLUTION — DeepCompile + ZeRO-3 + ZoRRO works end-to-end

Status: **WORKING.** A 2-step GSM8K GRPO run (Qwen3-0.6B, 1 GPU, `zero_optimization.stage=3`,
`compile.deepcompile=True`, `zorro_train.enable=True`) completes `rc=0` with healthy grads
(step:1 grad_norm≈0.28, step:2 grad_norm≈0.33). The fix is the sum of the following.

### 1. `lm_head` (tied with `embed_tokens`) — fresh-leaf grad deposit
`memory` mode (`TiledLogProbEntropy`) is forced under DeepCompile+z3. Its `apply` is wrapped in
`apply_tiled_logprob_entropy` (`@torch.compiler.disable`) so the `GatheredParameters` on the tied
weight runs **eager/untraced** (no dynamo rank-1↔2 guard). In `TiledLogProbEntropy`:
- `forward`: the inner `GatheredParameters` is gated on `hidden_states.requires_grad` so inference
  (`forward_no_grad`, already inside `_z3_eager_inference`) does not re-partition the tied weight
  (fixes the `'weight' must be 2-D` embed error).
- `backward`: **fresh-leaf trick** — inside `GatheredParameters`, temporarily swap
  `lm_head.weight` for `nn.Parameter(real_weight.detach())`, backprop into that fresh leaf, then
  add its full-shape grad into the real tied param's `.grad` (which is full-shape inside the gather).
  This bypasses the stale rank-0/`[0]` `AccumulateGrad` metadata DeepCompile registered for the
  partitioned param (fixes `TBackward0 ... expected shape compatible with [0]`).

### 2. DeepCompile scheduler bugs (editable `DeepSpeed/` fork)
- `deepspeed/compile/list_schedule.py` `get_node_requirements`: also traverse `node.kwargs` (not just
  `node.args`) when building topological deps. Fixes `KeyError: 'view_XXXX'` in
  `make_graph_from_schedule`/`fast_free_schedule` (some backward ops carry tensor operands as kwargs).
- `deepspeed/compile/util.py` `register_last_uses`: guard the no-copy-op passthrough with
  `user in node_to_last_use`. Fixes `KeyError: wait_allgather_ds_param__...` in the forward scheduler.

### 3. ROOT CAUSE of the multi-step `vec (0)` — dynamo recompile budget
With the above, step 1 passed but **step 2 failed** with `size mismatch ... vec (0)` at `q_proj`
(first projection), running **eagerly** (`is_compiling=False`) even though the eager Z3 fallback was
active (`ZeROOrderedDict._in_forward=True`). Diagnosis:
- ZoRRO produces a **different total-token count almost every step**, so each new shape triggers a
  dynamo recompile. The log showed `torch._dynamo hit config.recompile_limit (8)`.
- Once the recompile budget is exhausted, dynamo stops compiling that frame and runs it **eagerly**.
  DeepCompile's eager Z3 fallback (`deepcompile_z3_forward_context` + `ZeROOrderedDict.__getitem__`
  auto-gather) does **not** reliably gather params: after z3 registration the params are reachable via
  normal attribute lookup, so `__getitem__` is bypassed and nothing gathers. (Confirmed: at the
  strand the weight is a real ds-param — `has_all_gather=True` — and an explicit
  `param.all_gather(param_list=[param])` materializes it to `AVAILABLE`; forcing q_proj just moved
  the strand to k_proj, i.e. **every** projection strands in the eager region.)
- Attempts to gather from ZoRRO code (top of causal-LM / backbone forward, guarded by
  `is_compiling()`) did NOT help: dynamo compiles the forward **prologue** (the gather traces away to
  a no-op) and only graph-break/**resumes eagerly deep inside the decoder layers**, past the gather.

**Fix (`deepspeed_worker.py`, module scope):** raise the dynamo recompile budget so every
sequence-length shape gets its own **compiled** graph (with DeepCompile's in-graph gather) instead of
falling back to the broken eager path:
```python
_ZORRO_DYNAMO_RECOMPILE_LIMIT = int(os.environ.get("ZORRO_DYNAMO_RECOMPILE_LIMIT", "1024"))
# sets torch._dynamo.config.recompile_limit / cache_size_limit (+ accumulated_* ) 
```

### Perf caveat / follow-up
Because each distinct token count recompiles, steady-state is slow (~14 min/step in the 2-step smoke,
dominated by per-step recompilation). The highest-leverage follow-up is **dynamic sequence lengths**
(`dynamic=True` / `mark_dynamic` on the token dim, or seqlen bucketing) so a single compiled graph
serves all shapes — this both removes the recompiles and makes the recompile-budget bump unnecessary.

### Files changed for the final resolution
- `Arctic-Platform/arctic_platform/rl/deepspeed_worker.py`: raise dynamo recompile limit; force
  `logits_optimization=memory` under DeepCompile+z3; `force_parameter_static_shapes=False` (module scope).
- `Arctic-Platform/arctic_platform/rl/zorro_train/qwen_model_patcher.py`: `apply_tiled_logprob_entropy`
  disable-wrapper; grad-gated forward gather; fresh-leaf backward grad deposit.
- `Arctic-Platform/arctic_platform/rl/zorro_train/qwen_attention_patcher.py` + `zorro_train.py`:
  fullgraph `index_select` attention + `build_attention_gather_indices` (from the section above).
- `DeepSpeed/deepspeed/compile/list_schedule.py` and `.../util.py`: scheduler kwargs/no-copy fixes
  (editable install required).

---

## 3-way compile-strategy comparison (2026-07-11)

**Setup:** ZoRRO + ZeRO-3, Qwen3-0.6B, 1×H200, 4 training steps, colocated vLLM.
Per-step shapes vary run-to-run (ZoRRO dedup/reconstruct). Runs:
`bench/results/20260711T143939Z` (eager, 1a_bucket), `20260711T202111Z` (1b_mark),
`20260703T100349Z` (regional, wholegraph). Harness: `bench/run_matrix.sh` + `bench/parse_logs.py`.

| Strategy                     | works | warmup (step1) | steady s/step (2–4) | vs eager | steady tok/s | recompiles | peak mem GiB | grad_norm(1) |
|------------------------------|:-----:|---------------:|--------------------:|---------:|-------------:|-----------:|-------------:|-------------:|
| **Eager (no compile)**       |  yes  |         35.6 s |            **32.4** |    1.0×  |     **2058** |          0 |         99.2 |        0.743 |
| DeepCompile + mark_dynamic   |  yes  |        762.5 s |              1072.4 |    33×   |           65 |        114 |         97.6 |        0.254 |
| DeepCompile + seqlen bucket  |  yes  |        831.3 s |              1220.2 |    38×   |           57 |        152 |         97.7 |        0.272 |
| torch.compile — regional     |  yes  |       5141.0 s |               156.5 |   4.8×   |          463 |       4803 |         98.4 |        0.727 |
| torch.compile — whole-model  |  yes  |       1885.6 s |              1603.9 |   50×    |          388 |       3754 |        103.4 |        0.618 |

Per-step wall clock (s): eager `35.6 / 33.0 / 31.9 / 32.1` (flat); DeepCompile+mark
`762 / 932 / 1092 / 1193`; DeepCompile+bucket `831 / 1049 / 1230 / 1382` (both climb every
step — recompiles accumulate, never amortize); regional `5141 / 208 / 133 / 128` (storm then eager
fallback); whole-model `1886 / 4582 / 115 / 115` (two storms then eager fallback).

### Findings
- **Eager wins outright** (32 s/step, 2058 tok/s). Every compile strategy is net-negative on this
  workload.
- **DeepCompile dynamic-shape fixes are correct but do not restore perf.** `maybe_mark_dynamic`
  (mark) and seqlen bucketing both run end-to-end and are numerically correct — grad_norm 0.254/0.272
  matches the original DeepCompile run (0.272). They cut recompile *count* ~30× vs native
  torch.compile (114–152 vs 4803), but only constrain the dedup token dim; `reconstruct_sequences`
  emits several other data-dependent dims that DeepCompile re-specializes and recompiles every step.
  Per-step cost therefore *grows* (763→1193 s, 831→1382 s).
- **Regional / whole-model torch.compile** hit recompile storms (3754–4803 recompiles) under native
  ZeRO-3, then fall back to eager (~115–156 s/step) after exhausting the recompile budget — so their
  "steady" is mostly eager + guard overhead, and the multi-thousand-second warmup is never repaid.
- **grad_norm caveat:** DeepCompile family ~0.27 vs eager/regional/whole-model ~0.62–0.73 is a
  logprob-path artifact (`logits_optimization=memory` forced only under DeepCompile), not a
  correctness gap. Within-family parity confirms the mark/bucket fixes.

### `1b_mark` fix
`torch._dynamo.mark_dynamic` raised `ConstraintViolationError` because DeepCompile specializes some
gather-index dims to constants. Switched `_mark_dynamic_dedup` to `torch._dynamo.maybe_mark_dynamic`
(best-effort; no error on specialization) in `qwen_model_patcher.py`.

### Recommendation
Run ZoRRO + ZeRO-3 **eager** for now. To make compilation viable, bucket **every** reconstructed
data-dependent dim (not just the dedup token dim) so one compiled graph serves all steps; otherwise
recompilation dominates. Visual comparison: `canvases/compile-strategy-comparison.canvas.tsx`.

---

## Same comparison under ZeRO-1 (2026-07-13)

**Setup:** identical to the z3 matrix but `remote_backend.train.deepspeed.zero_optimization.stage=1`.
Run `bench/results/20260713T024958Z`, harness `BENCH_ZERO_STAGE=1`. Under z1 there is no parameter
sharding, so the z3-only workarounds are inactive: `logits_optimization=memory` is **not** forced
(gated on `_deepcompile_on and _z3`), and the `GatheredParameters` lm_head/tied-weight paths are
skipped (gated on `stage == 3`). DeepCompile itself supports z1 (routes to `init_z1`).

| Strategy                     | works | warmup (step1) | steady s/step (2–4) | vs z1 eager | steady tok/s | recompiles | peak mem GiB | gn(1) |
|------------------------------|:-----:|---------------:|--------------------:|------------:|-------------:|-----------:|-------------:|------:|
| **Eager (no compile)**       |  yes  |         32.6 s |            **31.0** |       1.0×  |     **2190** |          0 |         93.9 | 0.627 |
| torch.compile — regional     |  yes  |        381.3 s |                35.5 |       1.1×  |         1900 |        334 |        101.1 | 0.798 |
| DeepCompile + mark_dynamic   |  yes  |        416.0 s |                41.6 |       1.3×  |         1631 |         50 |        110.3 | 0.679 |
| DeepCompile + seqlen bucket  |  yes  |        453.0 s |                45.4 |       1.5×  |         1503 |         58 |        106.0 | 0.540 |
| torch.compile — whole-model  |  yes  |        389.8 s |                58.2 |       1.9×  |         1508 |         60 |        101.2 | 0.727 |

Per-step wall clock (s): eager `32.6/31.2/31.1/30.6`; regional `381/36.1/35.5/35.1`;
mark `416/51.4/37.1/36.4`; bucket `453/52.9/45.8/37.7`; whole-model `390/104/35.5/35.0`.
Note steps *decrease* toward steady (early recompiles amortize) — the opposite of z3.

### Findings — ZeRO-1 rescues compilation
- **The z3 recompile catastrophe is a ZeRO-3 phenomenon.** Under z1 every strategy converges after a
  ~6–8 min warmup. DeepCompile steady drops from **1072–1220 s/step (z3) to 42–45 s/step (z1)** (~27×);
  regional warmup collapses from **5141 s (85 min) to 381 s**; recompiles fall from thousands
  (3754–4803) to 50–334. Root cause confirmed: the z3 runaway came from DeepCompile's in-graph
  parameter gather/release passes re-specializing on ZoRRO's data-dependent shapes. z1 keeps params
  replicated, so the compiled graph is stable and reused.
- **But eager still wins.** Even converged, all compiled configs are 1.1–1.9× slower than z1 eager
  (31 s) and add a multi-minute warmup. ZoRRO's dynamic shapes still cause 50–334 recompiles, so steps
  never go fully recompile-free. Compilation is *viable* under z1, not *beneficial*.
- **grad_norm is a loose parity check here.** z1 uses the default logprob path for all configs, yet
  gn(1) spreads 0.54–0.80, and z1 eager (0.627) differs from z3 eager (0.743). This is rollout
  non-determinism between runs (vLLM sampling), not a compile-correctness gap.

### Why is DeepCompile slower than eager even on z1? (phase breakdown)
The compiled step is not the problem — the surrounding phases are. Per-step wall clock (s), from
`bench/results/20260713T024958Z`:

```
phase                        eager (s1..s4)         DeepCompile+mark (s1..s4)
update_actor (compiled f/b)  2.3 2.3 2.3 1.8        271  1.5  1.2  0.94   <- compiled, and FASTER than eager
old_log_prob                 1.6 0.75 0.68 0.66     110  14.4 1.05 1.0    <- recompiles, decays to eager by s4
update_weights (sync->vLLM)  5.2 4.9 4.9 4.8        9.8  12.1 10.5 10.5   <- ~2x eager, EVERY step (persistent)
total step                   32.6 31.2 31.1 30.6    416  51.4 37.1 36.4
```

Two costs, one transient and one persistent:
- **Transient — recompilation.** ZoRRO reconstructs a different total token count every step, so the
  forward re-traces for each new shape. This is the ~7-min warmup (`update_actor` 271–298 s on step 1)
  plus elevated `old_log_prob` (110 → 14 → 1 s) as the log-prob forward recompiles per rollout shape;
  the 50–58 recompiles are almost all `Recompiling function patched_forward`. By step 4 `old_log_prob`
  is back to ~1 s (≈ eager) — this tax amortizes.
- **Persistent — weight sync.** `update_weights` (copying the trained actor params into the colocated
  vLLM engine) ~doubles under DeepCompile: ~4.8 → ~10.5 s **every** step and never decays. At the
  converged step 4 this +5.7 s is essentially the *entire* gap (eager 30.6 s vs mark 36.4 s). The
  compiled module's parameter management (extra hooks, higher resident memory: 106–110 vs 94 GiB, and
  gradient checkpointing disabled) makes state-dict extraction for the sync costlier than for an eager
  module.
- **Meanwhile the compiled fwd/bwd (`update_actor`) is actually faster than eager** (≤2.6 s vs 2.3 s).
  DeepCompile does speed up the step it targets — but Qwen3-0.6B's fwd/bwd is only ~2 s, so that win
  is dwarfed by the weight-sync tax + early recompiles. Compilation pays off on large models with
  heavy fwd/bwd and static shapes; this workload has neither.

### Harness change
`bench/run_matrix.sh` gained `BENCH_ZERO_STAGE` (default 3; sed-rewrites
`zero_optimization.stage=3`) and now prepends the repo `verl/` dir to `PYTHONPATH` when launching
each generated script — the script is copied to `/tmp`, so its `BASH_SOURCE`-derived `REPO_ROOT`
(and thus `PYTHONPATH`) was wrong, which broke `import verl` whenever verl was not pip-installed
(e.g. after `install-arctic-rl.sh`, which does not install verl).
