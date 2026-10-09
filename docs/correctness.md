# Correctness harness

The harness validates a Arctic Platform training config against a reduced single-GPU Hugging Face golden reference. Both engines load the same checkpoint weights and consume the same generated batch. The gradient check compares every aligned trainable parameter after Arctic Platform gradient reduction and before clipping. The optimizer check runs one AdamW step and compares every aligned first moment, second moment, and moment-independent parameter-update residual as full tensors. The checkpoint-and-resume check is the one check with no reference engine in it: it compares twenty Arctic Platform iterations run straight through against the same twenty run across a checkpoint boundary.

There are 2 steps to this process

1. Run onboarding of a new model config on a given gpu arch - once per config
2. Run the regression test that validates this config - many times as the code changes

If you're not onboarding and need to test things, skip to [Run regression testing](#part-2-run-regression-testing).

## Progress Status

Each cell is one invocation of one check against one config, so a runtime belongs to that pair alone rather than being a share of a combined run.

### AP outcome by config and check

This table records only results produced after the port. Each paired-check slot contains `gas1` then `gas4`; `?` means that exact case and placement has not reached an Arctic Platform verdict. The inherited table below remains the DSS baseline for comparison.

P+**F**+? 114(55+**24**+35)

| config                                        | single-step-grads       | single-step-optimizer | checkpoint-resume-loss | inference-checkpoint-loss | e2e-sft-train-validate | rl-weight-sync | rl-router-replay |
| --------------------------------------------- | ----------------------- | --------------------- | ---------------------- | ------------------------- | ---------------------- | -------------- | ---------------- |
| qwen3-8b-h200-train-sft-full-4gpus-64k        | P**F**(4) \| P**F**(8)      | PP(4) \| PP(8)       | P**F**(4) \| P**F**(8)           | P(4) \| P(8)               | P(4) \| P(8)          | -              | -                |
| qwen3-8b-h200-train-sft-lora-4gpus-64k        | PP(4) \| PP(8)            | PP(4) \| PP(8)           | P**F**(4) \| P**F**(8) | -                         | P(4) \| P(8)                | -              | -                |
| qwen3.6-35b-a3b-h200-train-rl-16gpus-64k      | -                       | -                     | ?? \| ??               | ? \| ?                   | -                      | ? \| ?         | ? \| ?           |
| qwen3.6-35b-a3b-h200-train-sft-8gpus-64k      | **FF**(8) \| **F**?(16) | PP(8) \| PP(16) | PP(8) \| ?? | **F**(8) \| **F**(16) | **F**(8) \| ? | - | - |
| qwen3.6-35b-a3b-h200-train-sft-full-8gpus-sp8-64k | **FF**(8) \| ??     | PP(8) \| PP(16)              | PP(8) \| ??            | **F**(8) \| ?                   | P(8) \| ?             | -              | -                |
| qwen3.6-35b-a3b-h200-train-sft-lora-8gpus-sp8-64k | **FF**(8) \| ??    | **FF**(8) \| ??       | **FF**(8) \| ??        | -                         | P(8) \| ?              | -              | -                |
| qwen3.8-27b-h200-train-sft-8gpus-2k           | PP(8) \| ??            | PP(8) \| ??            | ?? \| ??               | -                         | -                      | -              | -                |
| qwen3.8-27b-b200-train-sft-8gpus-2k           | **FF**(8) \| P**F**(16)        | PP(8) \| PP(16)           | PP(8) \| PP(16)        | P(8) \| P(16)            | P(8) \| P(16)         | -              | -                |

### AP failures

The nine issues below account for all 24 AP `F` markers: 8 + 2 + 1 + 2 + 1 + 2 + 1 + 6 + 1.

1. Qwen3.6 H200 `single-step-grads` has a common scalar gradient-normalization regression across all three supervised configs. For `qwen3.6-35b-a3b-h200-train-sft-8gpus-64k` on 8 H200 GPUs, job `20261008T023607Z-3279840-000-67244775` `gas1` fails 46 of 72 norms against the `25.000e-03` gate. The median Arctic Platform/reference ratio is `1.092928`, and the largest disagreement is `2.304e+00` on `layers.2.mlp.experts.gate_up_proj` (ratio `4.645322`) against a gate of `0.025e+00`; the loss difference is `1.163e-04`. The 16-GPU `gas1` marker remains the earlier measurement, 61 of 72 norms with median ratio `2.077012`. For `qwen3.6-35b-a3b-h200-train-sft-full-8gpus-sp8-64k` on 8 GPUs, 69 of 72 norms exceed the `24.000e-03` gate in both cases: the median ratio is `8.004090` for `gas1` and `8.002192` for `gas4`; the `gas4` loss differs by only `0.165e-03`, and all 216 `gas4` optimizer moment and update-residual comparisons pass the `1.000e-03` gate. The retained optimizer cell is `PP(8)`. Guardrail job `20261008T232110Z-758682-000-214332718` later reran the same full-SP8 `single-step-optimizer` slot against the active worktree and failed both cases: `gas1` misses 33 of 216 optimizer tensors, led by `9.856e-03` on `layers.0.linear_attn.out_proj.weight::exp_avg`, and `gas4` misses 32 of 216, led by `9.922e-03` on the same first moment; the wall clock is 13m24.8s. That guardrail evidence is retained as diagnostic evidence for the active patch under review and does not add an AP outcome marker. For `qwen3.6-35b-a3b-h200-train-sft-lora-8gpus-sp8-64k` `gas1` on 8 GPUs, 4 of 8 LoRA gradient norms exceed the `1.000e-03` gate, all on `lora_B` attention adapters: `v_proj` is `2.95454` against `0.369822` with a `2.585e+00` difference and `7.989089` ratio; `q_proj` is `1.16321` against `0.145514` with a `1.018e+00` difference and `7.993836` ratio; `o_proj` is `0.554686` against `0.0693682` with a `4.853e-01` difference and `7.996263` ratio; and `k_proj` is `0.289155` against `0.0361168` with a `2.530e-01` difference and `8.006109` ratio. The corresponding DSS gradient cells pass. The ratio changes with request shape, isolating the port regression to request-loss or gradient normalization before the optimizer rather than to a model module; all three Qwen3.6 supervised expansion lanes remain blocked while that API boundary is isolated. A logical-DP loss-scaling candidate made the Qwen3.6 LoRA SP8 `single-step-grads` `gas1` reproducer pass all 8 gradient comparisons in 168 seconds, but regressed the required `single-step-optimizer` `gas1` no-regression path. Job `20261007T200354Z-2629035-000-2281919556` failed 4 of 24 comparisons after 4 whole minutes, led by the `v_proj` LoRA-B first moment at `32.110e-03` against the `1.000e-03` gate. The isolated corrected baseline passes all 24 comparisons in 161 seconds with a largest disagreement of `0.199e-03`. The candidate is reverted; its failure is retained only as diagnostic and accumulated-runtime evidence rather than entered as an outcome. Job `20261007T210954Z-2780720-000-58062499` reproduces the retained LoRA SP8 `gas1` gradient failure after the Ray token-authentication repair: 4 of 8 LoRA gradient norms fail, led by the `v_proj` LoRA-B difference of `2.757e+00` against the `1.000e-03` gate, in 158 seconds. Job `20261007T211755Z-2805066-000-2008623112` reproduces the corresponding `gas4` failure: 4 of 8 LoRA gradient norms fail, led by the `v_proj` LoRA-B difference of `2.810e+00` against the same gate, in 169 seconds. Job `20261008T010455Z-3177499-000-2323523798` verifies the rank-local ZeRO-fragment telemetry without a collective stall and reproduces the retained `gas1` semantic failure: 4 of 8 norms fail, led by the `v_proj` LoRA-B difference of `2.662e+00` against the `1.000e-03` gate, in 156 seconds. Job `20261007T232039Z-3036955-000-66513342` compares `single-step-optimizer` `gas1` on 8 H200 GPUs and fails 4 of 24 tensors against the `1.000e-03` gate. The failures are layer-3 attention `lora_B` first moments: `v_proj` `103.5e-03`, `q_proj` `35.54e-03`, `o_proj` `16.55e-03`, and `k_proj` `10.11e-03`. Reference loss is `14.285037` and Arctic Platform loss is `14.265423`. The invocation takes 167 seconds, recorded as 3 whole minutes. That invocation establishes the `gas1` failure; the current retained-tree `gas4` result is recorded below, so the optimizer cell is `**FF**(8)`. Job `20261008T024032Z-3289703-000-52377310` verifies the rank-local gradient telemetry on the optimizer path and reproduces the same `gas1` result: 4 of 24 comparisons fail, led by the `v_proj` LoRA-B first moment at `103.5e-03`, in 151 seconds. Job `20261008T025742Z-3319684-000-676220737` runs `gas4` under the same current candidate and regresses the retained pass: 4 of 24 comparisons fail, led by the same first moment at `103.5e-03`, in 159 seconds. The candidate result is diagnostic; the retained-tree rerun below determines the current `gas4` marker. The earlier 8-GPU `gas1` failure on `qwen3-8b-h200-train-sft-full-4gpus-64k`, job `20261008T030509Z-3330919-000-167498565`, is superseded by the exact retained pass recorded below. Jobs `20261008T033025Z-3361351-000-259419947`, `20261008T034827Z-3379258-000-301569188`, `20261008T035458Z-3385466-000-2192031031`, and `20261008T040004Z-3389755-000-803919300` test four rejected Qwen3.6 LoRA SP8 gradient-reduction candidates on optimizer `gas1`; each still fails the same four layer-3 attention LoRA-B first moments in 2m29.3s to 2m37.5s. The local-fragment division candidate reduces the leading disagreement from `103.5e-03` to `57.52e-03`, while the three collective candidates leave it at `103.5e-03`; all four candidates are reverted.

The exact `gas1` markers on `qwen3-8b-h200-train-sft-full-4gpus-64k` pass in retained reruns. The 4-GPU marker passes in job `20261009T032932Z-866675-000-442319848`: 47 of 47 gradient norms are inside the `5.000e-03` gate, the median Arctic Platform/reference ratio is `1.000000`, the largest disagreement is `2.851e-03` on `layers.1.mlp.gate_proj.weight`, and the 1m04.3s wall clock is recorded as 2 whole minutes. The 8-GPU marker passes in job `20261009T035130Z-884587-000-2768425066`: 47 of 47 gradient norms are inside the same gate, the median ratio is `0.999989`, the largest disagreement is `2.813e-03` on `layers.1.input_layernorm.weight`, and the 1m22.6s wall clock is recorded as 2 whole minutes. Earlier shape-mismatched or failing attempts are superseded for these markers.

2. `qwen3.8-27b-b200-train-sft-8gpus-2k`, `single-step-grads`, 8 B200 GPUs: both retained cases now fail. Job `20261009T032932Z-866577-000-66314962` compares the `gas1` case and fails 22 of 56 gradient norms against the `4.000e-03` gate. The median Arctic Platform/reference ratio is `0.983620`, the largest disagreement is `71.65e-03` on `layers.0.linear_attn.out_proj.weight`, the loss differs by `0.011e-03`, and the 1m53.8s wall clock is recorded as 2 whole minutes. The retained `gas4` case differs on `norm.weight` by `13.670e-03` against the same gate; Arctic Platform reads `3.14086`, the reference reads `3.12719`, the loss differs by `0.010e-03`, and the median Arctic Platform/reference norm ratio is `0.999968`.

3. `qwen3.8-27b-b200-train-sft-8gpus-2k`, `single-step-grads` `gas4`, 16 B200 GPUs: job `20261008T023552Z-3278630-000-1151025444` compares 32 sequences of 8,192 tokens, 262,144 token slots and 138,065 active tokens. The loss differs by `5.835e-06`. 26 of 56 gradient norms miss the `4.000e-03` gate. The median Arctic Platform/reference ratio is `0.982882`. The largest disagreement is `76.49e-03` on `layers.0.linear_attn.out_proj.weight`, where Arctic Platform reads `4.15855` and the reference reads `4.23504`. The wall clock is 2m57.9s, recorded as 3 whole minutes and added to the placement's existing 2, so the runtime cell is `8(8) | 5(16)`. The 16-GPU `gas1` comparison stays inside its gate, so the cell is `P**F**(16)`.

4. Qwen3.6 DP `inference-checkpoint-loss` fails at both measured placements while the corresponding DSS cells pass. The retained 16-GPU result predates the native-output adapter and raised `KeyError: 'logits'` because the custom Qwen3.5 LM head returns target-token `logprobs` in a dictionary. Job `20261007T222255Z-2931308-000-239731895` reaches the semantic comparison with the retained adapter on 8 H200 GPUs: the live and reloaded AP training engines agree exactly at `22.511743`, while ArcticInference reads `22.520628`, a difference of `8.885e-03` against the `1.000e-03` gate. The adapter execution failure is resolved at the declared width, exposing a numerical disagreement isolated to the sampling path; the 16-GPU marker remains failed until rerun against the retained adapter. Jobs `20261007T221332Z-2908295-000-65888128` and `20261007T221844Z-2923770-000-816612839` fail before a comparison because the reloaded training engine's converted checkpoint lacks `model.embed_tokens.weight`, in 214 and 218 seconds, recorded as 4 whole minutes each. Job `20261007T222357Z-2933354-000-699522268` compares after composite `language_model` parameter names are rewritten to text names: live and reloaded training both score `21.863531126`, sampling scores `21.861551788`, and the absolute difference is `1.979e-03` against the `1.000e-03` gate, in 231 seconds, recorded as 4 whole minutes. The reload matches the live training engine, and the remaining disagreement is the sampling score. Job `20261007T225926Z-2995295-000-764313474` compares after the sampling job selects the Triton/FLA GDN prefill: live and reloaded training both score `2.893903177`, sampling scores `2.895371969`, and the absolute difference is `1.469e-03` against the `1.000e-03` gate, over 4088 scored answer tokens, in 299 seconds, recorded as 5 whole minutes. Job `20261008T005828Z-3164127-000-2170230394` reproduces the sampling-path failure after preserving the activation dtype through the GDN prefill: live and reloaded training both score `2.899124830`, sampling scores `2.900869853`, and the absolute difference is `1.745e-03` against the `1.000e-03` gate, in 236 seconds, recorded as 4 whole minutes.

5. Qwen3.6 DP `e2e-sft-train-validate` has a retained-code trajectory disagreement on 8 H200 GPUs while the corresponding DSS cell passes. Job `20261007T213028Z-2831754-000-395412394` agrees at step 1 within `1.917e-03`, first exceeds the `30.000e-03` gate at step 4, and finishes with a training-loss difference of `62.245e-03`; held-out validation remains inside the gate at `8.002e-03`. The widest recorded training-step difference is `254.230e-03` at step 68, isolating the marker to accumulated optimizer, gradient-reduction, or reduction-order behavior rather than the repaired forward-only adapter.

6. `qwen3.6-35b-a3b-h200-train-sft-lora-8gpus-sp8-64k`, `checkpoint-resume-loss`, 8 H200 GPUs: both cases fail 1 of 2 recorded loss comparisons against the `2.000e-03` gate. For `gas1`, job `20261007T203947Z-2726252-000-83326703` records a resumed iteration-20 loss of `12.6131` against the uninterrupted `12.6194`, a difference of `6.346e-03`; jobs `20261007T201153Z-2665703-000-1959820434` and `20261007T195906Z-2616765-000-1073728311` independently reproduce the same case with differences of `6.313e-03` and `2.719e-03`. For `gas4`, job `20261007T210139Z-2766653-000-240668939` records `12.6042` after resume against `12.6113` uninterrupted, a difference of `7.096e-03`. Job `20261007T202526Z-2693439-000-178969835` reproduces that `gas4` failure: resumed iteration-20 loss `12.6042` against uninterrupted `12.6113`, a difference of `7.080e-03`. Every iteration-10 comparison is inside tolerance. The corresponding DSS cell is `PP(8)`, so both markers belong to one AP checkpoint-resume regression.

7. `qwen3.6-35b-a3b-h200-train-sft-8gpus-64k`, `single-step-grads` `gas4`, 8 H200 GPUs: job `20261008T023607Z-3279840-000-67244775` compares 32 sequences of 8,192 tokens, 262,144 token slots, and 138,065 real tokens. 28 of 72 gradient norms exceed the `25.000e-03` gate. The median Arctic Platform/reference ratio is `1.028233`. The largest disagreement is `1.875e+00` on `layers.2.mlp.experts.gate_up_proj` (Arctic Platform `2.49502`, reference `0.61954`, ratio `4.027213`) against a gate of `0.025e+00`. Reference loss is `14.285983` and Arctic Platform loss is `14.285818`, a difference of `1.650e-04`. The same invocation's wall clock, including gateway startup and model load, is 4m44.8s, recorded as 5 whole minutes for the completed `gas1` and `gas4` pair. Job `20261008T024704Z-3304898-000-84828377` then reruns `gas4` alone on the same 8-GPU allocation: 32 sequences of 8,192 tokens, 262,144 token slots, and 138,065 real tokens. Only 8 of 72 gradient-norm names match, so the `25.000e-03` gate is not applied. Arctic Platform contributes no unmatched names. The unmatched reference names begin with `lm_head.weight`, `model.language_model.embed_tokens.weight`, `model.language_model.layers.0.input_layernorm.weight`, and `model.language_model.layers.0.linear_attn.A_log`. That invocation's wall clock is 2m56.2s, recorded as 3 whole minutes. The placement accumulates from 5 to `8(8)`, and the runtime cell is `8(8) | 2(16)`. The 16-GPU `gas4` case stays unknown.

8. Qwen3-8B long-`gas4` cases account for six AP `F` markers. For `qwen3-8b-h200-train-sft-lora-4gpus-64k`, `checkpoint-resume-loss` `gas4` job `20261008T153241Z-578066-000-2222617733` fails both placements with `RayTaskError(OutOfMemoryError)` while trying to allocate `37.09 GiB`; the 4-GPU arm has `36.94 GiB` free at the reported failure and the 8-GPU arm reports the same free memory. The declared-width case runs 16 sequences of 16,384 tokens, 262,144 token slots, and 133,124 real tokens; the 8-GPU placement runs 32 sequences, 524,288 token slots, and 320,164 real tokens. The wall clocks are 1m18.6s and 1m21.6s, recorded as 2 whole minutes each and added to the placement accumulations. The LoRA runtime cell is `10(4)‡ | 51(8)‡`. For `qwen3-8b-h200-train-sft-full-4gpus-64k`, `checkpoint-resume-loss` `gas4` job `20261008T153855Z-585171-000-2762931087` reaches the same check on both placements and fails during `DeepSpeedWorker.forward_backward` at `engine.backward(loss, scale_wrt_gas=False)`. The 4-GPU arm tries to allocate `37.09 GiB` with `34.79 GiB` free and the 8-GPU arm tries to allocate `37.09 GiB` with `35.67 GiB` free. The wall clocks are 39.7s and 39.9s, recorded as 1 whole minute each and added to the placement accumulations. The full-model checkpoint runtime cell becomes `5(4)‡ | 37(8)‡`. The full-model `single-step-grads` `gas4` case no longer fails at the earlier full-vocab allocation after the retained AP payload is routed through the memory-CE path. Job `20261009T170545Z-1110424-000-26057958` reaches gradient comparison at both placements and fails only `lm_head.weight`. The 4-GPU arm compares 16 sequences of 16,384 tokens with loss delta `4.938e-07`; 1 of 47 gradient norms fails, `lm_head.weight` reads Arctic Platform `0.367213` against reference `0.175442`, the absolute difference is `1.918e-01`, and the ratio is `2.093078`. The 1m57.6s wall clock is recorded as 2 whole minutes. The 8-GPU arm compares 32 sequences of 16,384 tokens with loss delta `9.805e-08`; 1 of 47 norms fails, `lm_head.weight` reads Arctic Platform `0.31858` against reference `0.113393`, the absolute difference is `2.052e-01`, and the ratio is `2.809531`. The 3m01.3s wall clock is recorded as 3 whole minutes. Earlier `single-step-grads gas4` OOM attempts remain accumulated runtime but are superseded as the failure explanation; the full-model gradient runtime cell becomes `20(4)‡ | 15(8)‡`, and the outcome cell remains `P**F**(4) | P**F**(8)`. Full-model `single-step-optimizer` `gas4` is no longer an F marker: job `20261009T170549Z-1110685-000-136535625` passes both placements after the same memory-CE routing. The 4-GPU arm passes 141 of 141 optimizer tensors with largest disagreement `5.971e-03` on `lm_head.weight::exp_avg` and wall clock 3m46.8s, recorded as 4 whole minutes. The 8-GPU arm passes 141 of 141 tensors with largest disagreement `5.653e-03` on `lm_head.weight::exp_avg` and wall clock 4m53.4s, recorded as 5 whole minutes. Earlier optimizer OOM attempts remain accumulated runtime; the full-model optimizer runtime cell becomes `13(4)‡ | 11(8)‡`, and the outcome cell is `PP(4) | PP(8)`.

The retained `single-step-grads gas1` reruns on the Qwen3-8B full 4-GPU and 8-GPU placements are passing evidence. The corresponding `gas4` failures are now semantic `lm_head.weight` gradient mismatches rather than OOM-only evidence, and the earlier OOM attempts remain only in accumulated timing. The full-model gradient outcome reads `P**F**(4) | P**F**(8)` and the runtime reads `20(4)‡ | 15(8)‡`; this Qwen3-8B long-`gas4` bucket now accounts for 6 `F` markers.

9. `qwen3.6-35b-a3b-h200-train-sft-full-8gpus-sp8-64k`, `inference-checkpoint-loss`, 8 H200 GPUs: job `20261008T153333Z-578767-000-28545089` reaches the training path and fails with `RayTaskError(ActorDiedError)` after NCCL watchdog timeouts take down DeepSpeed workers. The log includes a rank-local `ALLREDUCE` timeout with `NumelIn=508559360`, an `ALLTOALL` timeout, and an upstream `AssertionError: loss must be a scalar tensor` at `engine.backward(loss, scale_wrt_gas=False)`. The harness emits `Overall: 0 passed, 1 failed, 0 inapplicable` and report `.agent-work/wave2b-stas-dev-7-qwen36-fullsp8-inference/report-2026-10-08-15-46.md`. The correctness wall clock is 12m13.9s, recorded as 13 whole minutes and added to the placement's prior 25 no-verdict minutes, so the runtime cell becomes `38(8)‡ | 11(16)‡`.

### AP execution errors and accumulated runtimes

Job `20261009T041249Z-904044-000-1779919187` supersedes the earlier Qwen3-8B full-model `inference-checkpoint-loss` failure on 4 H200 GPUs. The live and reloaded training engines both score `6.194458974`, the sampling zone scores `6.194930076`, and the absolute difference is `0.471e-03` against the `1.000e-03` gate. The semantic runtime is 2m22.2s and the total harness wall clock is 2m29.0s; the placement remains recorded in the existing `6(4)` accumulated runtime. With the earlier 8-GPU pass from job `20261008T023844Z-3285990-000-1184024854`, this inference cell is now `P(4) | P(8)`.

Jobs `20261007T200524Z-2635378-000-2447019397` and `20261007T212419Z-2821304-000-300030712` passed Qwen3-8B full-model `e2e-sft-train-validate` on 4 and 8 H200 GPUs after clean preflight. The 4-GPU run took 5m54.9s and the 8-GPU run took 5m56.4s total, recording 6 whole minutes for each placement. The 8-GPU run matched both gated losses: final training loss differed by `0.166e-03` and validation loss by `0.036e-03` against the `30.000e-03` gate. The cell records `P(4) | P(8)` with `6(4) | 6(8)`; the new right placement adds one timed slot.

Job `20261007T214043Z-2857930-000-327548712` passed Qwen3-8B LoRA `e2e-sft-train-validate` on 8 H200 GPUs after allocation-wide Ray and distributed cleanup. Final training loss differed by `1.016e-03` and held-out validation by `0.715e-03` against the `30.000e-03` gate; step 1 was exact and the widest recorded step differed by `19.280e-03` at step 9. The passing invocation took 2m48s, recorded as 3 whole minutes. Job `20261007T213755Z-2851783-000-2408123073` reached no semantic verdict after 55 seconds because its Ray driver started without token authentication, adding 1 whole minute. The existing `P(8)` outcome remains unchanged and the placement runtime accumulates from `2(8)` to `6(8)‡` without adding a timed slot.

The Qwen3-8B full-model `single-step-optimizer` placements retain their `gas1` passes and now record `gas4` as passed. Earlier `gas4` attempts ended before semantic comparison with CUDA OOM while allocating 37.09 GiB, so their elapsed time remains accumulated but no longer defines the outcome. Job `20261009T170549Z-1110685-000-136535625` passes the 4-GPU `gas4` arm in 3m46.8s and the 8-GPU arm in 4m53.4s, with 141 of 141 optimizer tensors inside the `8.804e+00` gate in both placements. The complete cell is `PP(4) | PP(8)` and the runtime is `13(4)‡ | 11(8)‡`.

The Qwen3-8B full-model `single-step-grads` 4-GPU placement records `gas1` as passed by the retained reruns described above and `gas4` as failed. Job `20261009T025404Z-837655-000-1156419491` is superseded for outcome but remains accumulated runtime: it reran `gas1` through the semantic gradient comparison and failed 30 of 47 gradient norms, led by `0.183359` on `embed_tokens.weight`, with median ratio `0.988966` and wall clock 2m27.9s, recorded as 3 whole minutes. The `gas4` evidence remains the AP-backward OOM series: job `20261007T202854Z-2701271-000-1200613482` reached backward after clean preflight but ended before comparison with CUDA OOM while requesting 37.09 GiB, its 159-second semantic runtime records 3 whole minutes, cleanup job `20261007T203712Z-2722816-000-869719745` verified distributed processes, GPU processes, and port 29500 were clear, rerun `20261007T203735Z-2723432-000-2371317043` reproduced the same allocation failure after 158 seconds, job `20261007T205429Z-2745395-000-2931017122` lost its Ray actor before execution and adds 2 whole minutes, and job `20261008T154511Z-590456-000-268062983` reaches AP backward again and fails with `RayTaskError(OutOfMemoryError)` while trying to allocate `37.09 GiB`, with `34.79 GiB` free, after 1m58s. The placement therefore records `P**F**(4)` and `18(4)‡` as one timed slot.

The Qwen3-8B full-model `single-step-grads` 8-GPU placement records `gas1` as passed by the retained rerun described above and `gas4` as failed after reaching AP backward but not a gradient comparison. Job `20261007T210112Z-2765727-000-1007329035` established the 8-slot hostfile but failed Ray token authentication after 183 seconds, adding 4 whole minutes. Job `20261007T210638Z-2775271-000-1010379` then stopped all prior Ray and gateway processes, verified an empty process set, read 8 slots from the allocation hostfile, and reproduced CUDA OOM while requesting 37.09 GiB after 231 seconds, adding 4 whole minutes. Together with the superseded earlier 2-minute 8-GPU `gas1` attempt and the retained 2-minute `gas1` pass, the right placement records `P**F**(8)` and `12(8)‡`.

Job `20261007T202152Z-2679838-000-55889377` passed Qwen3-8B full-model `checkpoint-resume-loss` `gas1` on the 4-GPU placement in 158 seconds, recorded as 3 whole minutes. Job `20261007T195924Z-2617598-000-2208615629` ran `gas4` on the same placement; CUDA OOM ended it before a verdict after 1m22.7s of semantic runtime and 1m25.1s total harness time, recorded as 1 whole minute. The prior 8-GPU slot already held a `gas1` pass plus non-verdict accumulated time, including job `20261007T192256Z-2518902-000-495019474`, whose first timestamped log event is 19:22:59.833 UTC, whose cancellation record is timestamped 19:55:31 UTC, and whose log ends with exit 143 after weight load. Job `20261008T153855Z-585171-000-2762931087` reruns `gas4` on both placements and reaches semantic failures: the 4-GPU case fails during backward with CUDA OOM after 39.7s, and the 8-GPU case fails the same way after 39.9s. Both are recorded as 1 whole minute. The complete outcome cell is `P**F**(4) | P**F**(8)`, and the placement runtimes are `5(4)‡ | 37(8)‡`.

Qwen3-8B LoRA `checkpoint-resume-loss` `gas1` passes at both placements after checkpoint-save worker arguments are forwarded by keyword. Job `20261007T220807Z-2898409-000-7983168` passes on 4 H200 GPUs in 2m57.2s and on 8 H200 GPUs in 3m28.0s, recorded as 3 and 4 whole minutes. Job `20261008T153241Z-578066-000-2222617733` reruns `gas4` on both placements and reaches semantic failures during backward with CUDA OOM while allocating 37.09 GiB. The 4-GPU case is 16 sequences of 16,384 tokens, 262,144 token slots, and 133,124 real tokens, with wall clock 1m18.6s, recorded as 2 whole minutes. The 8-GPU case is 32 sequences of 16,384 tokens, 524,288 token slots, and 320,164 real tokens, with wall clock 1m21.6s, recorded as 2 whole minutes. The outcome cell is `P**F**(4) | P**F**(8)`, and the runtime cell is `10(4)‡ | 51(8)‡`.

Job `20261007T202214Z-2682747-000-3215812659` passed Qwen3.8-27B B200 `e2e-sft-train-validate` on the 8-GPU allocation in 41m51.1s, recorded as 41 whole minutes. The declared-width slot records `P(8)` and `41(8)`.

The Qwen3.8-27B B200 `checkpoint-resume-loss` placement on 8 GPUs passes both cases in token-authentication mode. Job `20261007T211304Z-2790790-000-1169618181` records both loss comparisons inside the `23.685` gate in 8m00.2s wall clock, recorded as 8 whole minutes. Job `20261007T210842Z-2778961-000-2181430429` ended before comparison after 1m05.3s because Ray was imported with authentication disabled before the harness enabled token mode, recording 1 whole minute without a verdict. Together with the prior 85-minute canceled attempt, the placement records `PP(8)` and `94(8)‡` as one timed slot.

The Qwen3.8-27B B200 `checkpoint-resume-loss` 16-GPU placement passes both cases. Job `20261007T194938Z-2586159-000-2893315720` records `gas4` as 32 sequences of 8,192 tokens on a report that lists 8 GPUs, with 0 of 2 recorded losses outside the gate, in 5m35.3s, recorded as 6 whole minutes; that case shape is the declared 8-GPU placement. Job `20261007T182610Z-2458295-000-82019780` ran `gas4` for 61 whole minutes before external cancellation terminated it with exit 143 and no semantic verdict. Jobs `20261007T194156Z-2566591-000-978031267` and `20261007T200334Z-2628423-000-81418919` ran `gas1` for 4m49.7s and 3m23.1s, recorded as 5 and 3 whole minutes, before the same `TypeError: too many positional arguments` prevented a semantic comparison. Job `20261008T003223Z-3102152-000-841027377` passes `gas1` on the 16-GPU allocation: 16 sequences of 8,192 tokens, 131,072 token slots, and 62,573 real tokens. Iteration 10 scores 14.199740 after resume against the uninterrupted 13.842164, a difference of 0.358; iteration 20 scores 14.328584 against 13.830166, a difference of 0.498. Both are inside the 23.685 gate. The wall clock is 5m13.0s, recorded as 5 whole minutes. Job `20261007T231904Z-3033020-000-3210313223` ended after 1m26.3s, recorded as 1 whole minute, when the merged step response wrapped the rank-owned gradient-norm mapping in a list and no loss was compared. Jobs `20261008T005648Z-3159832-000-1231231809` and `20261008T010703Z-3182721-000-1251426108` each ran `gas4` as 64 sequences of 8,192 tokens, 524,288 token slots, and 295,012 real tokens, and ended while loading the resume checkpoint because `bf16_zero_pp_rank_8_mp_rank_00_optim_states.pt` was absent. Neither attempt compared a loss. Their wall clocks are 4m53.2s and 4m46.1s, recorded as 5 whole minutes each. Job `20261008T024516Z-3300676-000-268728817` passes `gas4` on the 16-GPU allocation: 64 sequences of 8,192 tokens, 524,288 token slots, and 295,012 real tokens. Iteration 10 scores 13.535466 after resume against the uninterrupted 13.426383, a difference of 0.109; iteration 20 scores 12.641196 against 12.674189, a difference of 0.033. Both are inside the 23.685 gate. The wall clock is 8m11.0s, recorded as 8 whole minutes. The placement records `PP(16)` and `99(16)‡` as one timed slot: 75 prior minutes, plus that 1 non-verdict minute, plus the 5-minute `gas1` comparison, plus these 5 and 5, plus these 8.

Qwen3.8-27B B200 `inference-checkpoint-loss` passes on the 16-GPU allocation in job `20261007T225357Z-2987715-000-144278933`: the live and reloaded AP training engines both score `4.246202258`, sampling scores `4.245831488`, and the `0.371e-03` difference is inside the `1.000e-03` gate. The passing invocation takes 6m01.0s, recorded as 6 whole minutes. The path is unblocked by keeping the ephemeral Ray server-state actor alive, removing the redundant checkpoint-export collective, accepting standard Hugging Face models without `is_prime_state_dict`, and disabling unused full-gradient telemetry for this inference-only check. Six prior non-verdict attempts add 54 whole minutes: 13 from job `20261007T210103Z-2765199-000-24741398`, 23 from the 8-GPU and cancelled 16-GPU phases of job `20261007T213040Z-2832118-000-2670012998`, 6 from job `20261007T215607Z-2884092-000-2792222138`, and 12 from job `20261007T223700Z-2956789-000-131688894`; two sub-minute authentication attempts add zero whole minutes. With the prior 7 minutes, the right placement records `P(16)` and `67(16)‡` as one timed slot.

Qwen3.6 full-model SP8 `single-step-grads` retains `FF(8)`. Job `20261007T163046Z-2362531-000-2521417398` fails `gas1`: 69 of 72 gradient norms exceed the `24.000e-03` gate, the median Arctic Platform/reference ratio is `8.004090`, and the wall clock including gateway startup and model load is 1m47.3s, recorded as 2 whole minutes. Job `20261007T174653Z-2420963-000-1587921807` fails `gas4`: 69 of 72 gradient norms exceed the same gate, the median ratio is `8.002192`, and the wall clock is 5m19.1s, recorded as 5 whole minutes. The pair records `7(8)` as one timed slot. Job `20261007T075215Z-1949264-000-169920395` also fails `gas1` at the same ratio, but that invocation runs the optimizer check in the same wall clock, so its 8m37.2s is not assigned to this slot.

Qwen3.6 LoRA SP8 `single-step-grads` retains `FF(8)`. Job `20261008T010455Z-3177499-000-2323523798` reproduces the `gas1` failure with rank-local ZeRO-fragment telemetry in 2m36.1s wall clock, recorded as 3 whole minutes. The placement accumulates to `11(8)` while remaining one timed slot.

Qwen3.6 full-model SP8 `single-step-optimizer` retains `PP(8)` from the clean baseline. Job `20261007T215437Z-2881685-000-140076621` ran the later-rejected masked-rank and pre-shard-label candidate: `gas1` failed 34 of 216 comparisons and `gas4` failed 32 of 216, led by `9.910e-03` against the `1.000e-03` gate. The candidate was reverted immediately; its failure is diagnostic rather than a semantic outcome. The retained `PP(8)` wall clock is job `20261007T180148Z-2435474-000-631219340`, which passes all 216 `gas1` comparisons in 4m44.3s, recorded as 5 whole minutes, and job `20261007T180836Z-2444076-000-2310613231`, which passes all 216 `gas4` comparisons in 6m18.3s, recorded as 6 whole minutes. The placement records `11(8)`. The rejected candidate's 13 whole minutes are diagnostic and are not part of that slot. Jobs `20261007T221559Z-2916122-000-1652131908`, `20261007T223156Z-2948739-000-2937018269`, `20261007T230912Z-3011307-000-578321889`, `20261008T002947Z-3094561-000-111721290`, and `20261008T005047Z-3146702-000-1129823533` add 14, 17, 14, 14, and 14 whole minutes from later candidate and isolation attempts; those minutes stay out of the retained slot.

The Qwen3.6 LoRA SP8 `single-step-optimizer` placement on 8 GPUs currently fails both cases. Its prior 13 whole minutes remain accumulated. Job `20261007T194135Z-2564891-000-2504432234` ended after `VALIDATION_RUNTIME_SECONDS=24` when allocation cleanup failed, adding 1 whole minute. The overlapping job `20261007T194139Z-2565150-000-2078515039` ended after a Ray authentication mismatch, but its local log has no validation-runtime field or timestamped exit, so no duration is recoverable locally and none is added. Job `20261007T194328Z-2571071-000-1585617810` reached the optimizer call before `TypeError: too many positional arguments` and records `VALIDATION_RUNTIME_SECONDS=133`, adding 3 whole minutes. Job `20261007T194557Z-2576998-000-261416` ran for 6 whole minutes but produced no verdict because its optimizer manifest was node-local and inaccessible, so those minutes are retained under `‡`. Retained job `20261007T194727Z-2580262-000-1595011329` passes all 24 `gas1` comparisons and records 161 seconds, adding 3 whole minutes; job `20261007T195536Z-2604398-000-303979151` passes all 24 `gas4` comparisons and records 167 seconds, adding 3 whole minutes. Job `20261007T200354Z-2629035-000-2281919556` ran for 4 whole minutes under the reverted logical-DP candidate and failed 4 of 24 `gas1` comparisons, led by the `v_proj` LoRA-B `exp_avg` difference at `32.110e-03` against the `1.000e-03` gate; its failure is diagnostic evidence and does not replace the retained baseline pass. Job `20261008T024032Z-3289703-000-52377310` reproduces the semantic `gas1` failure in 2m31.0s wall clock, recorded as 3 whole minutes. Jobs `20261008T033025Z-3361351-000-259419947`, `20261008T034827Z-3379258-000-301569188`, `20261008T035458Z-3385466-000-2192031031`, and `20261008T040004Z-3389755-000-803919300` add 12 whole minutes from rejected `gas1` diagnostics. Job `20261008T040538Z-3394027-000-299021509` reruns `gas4` after every candidate is reverted and fails the same 4 of 24 comparisons in 2m32.1s, so the current outcome is `**FF**(8)`. The placement records `57(8)‡`: 42 prior whole minutes plus these 15 diagnostic and current-baseline minutes, with the unrecoverable collision omitted. These attempts add runtime to the existing timed placement rather than another timed slot.

The Qwen3.6 DP `checkpoint-resume-loss` placement on 8 GPUs passes both cases after clean preflight: `gas1` ran for 3m38.3s and recovered `gas4` job `20261007T194410Z-2573277-000-2348728851` ran for 4m11.2s, recorded as 4 whole minutes each. The invalidated `gas4` attempt began its clean preflight at 19:41:51 and was terminated at 19:43:40 after an overlapping cleanup killed its Ray runtime, so its recoverable 1m49s is recorded as 2 whole minutes under non-verdict timing. The placement therefore records `10(8)‡` while retaining one timed slot: 4 gas1 minutes + 4 recovered gas4 minutes + 2 invalidated-attempt minutes.

Job `20261007T195642Z-2608519-000-1144523046` ran Qwen3.6 DP `inference-checkpoint-loss` on 8 H200 GPUs for 2m18.4s, recorded as 2 whole minutes, before vLLM rejected an exported parameter path containing `layers.0._checkpoint_wrapped_module`. Job `20261007T222255Z-2931308-000-239731895` then reaches a semantic failure in 4m56.1s total harness time after the composite export repair, recording 5 whole minutes: live and reloaded training score `22.511743`, sampling scores `22.520628`, and the `8.885e-03` difference exceeds the `1.000e-03` gate. The cell becomes `F(8) | F(16)`. Jobs `20261007T221332Z-2908295-000-65888128` and `20261007T221844Z-2923770-000-816612839` add 4 whole minutes each before comparison, and job `20261007T222357Z-2933354-000-699522268` adds 4 whole minutes for the `1.979e-03` comparison. Job `20261007T225926Z-2995295-000-764313474` compares with the Triton/FLA GDN prefill in 4m43.5s wall clock and 299 seconds, recorded as 5 whole minutes: live and reloaded training both score `2.893903177`, sampling scores `2.895371969`, and the absolute difference is `1.469e-03` against the `1.000e-03` gate. Job `20261008T005828Z-3164127-000-2170230394` reproduces the failure after preserving the activation dtype: live and reloaded training both score `2.899124830`, sampling scores `2.900869853`, and the absolute difference is `1.745e-03` against the `1.000e-03` gate, in 236 seconds, recorded as 4 whole minutes. The declared-width runtime accumulates to `28(8)‡`: 2 + 5 + 4 + 4 + 4 + 5 + 4, while remaining one timed slot.

Job `20261007T213028Z-2831754-000-395412394` fails Qwen3.6 DP `e2e-sft-train-validate` on 8 H200 GPUs in 50m08.6s total harness time, recording 51 whole minutes. Its final training loss differs by `62.245e-03` against the `30.000e-03` gate after first exceeding that gate at step 4; validation differs by `8.002e-03` and passes. Three retained-code repeat attempts were externally cancelled before a verdict: job `20261007T224219Z-2967944-000-1158022426` after 5m11s, job `20261007T224800Z-2978128-000-2005028494` after 51 seconds, and job `20261007T230400Z-3001748-000-36446756` after 50 seconds, recording 6, 1, and 1 whole minutes. The retained placement remains `F(8)` and accumulates from 53 to `112(8)‡` while remaining one timed slot.

Job `20261007T202957Z-2703781-000-303928418` attempted Qwen3.6 full-model SP8 `inference-checkpoint-loss` on the 16-GPU placement. The training workers loaded on the peer node, but the job process disappeared after its last output at 20:31:11 UTC without producing a comparison or report; the stale runner claim was cancelled at 20:41:23 UTC with exit 143. The 11m26s allocation occupancy records 11 whole minutes as `11(16)‡`, while the outcome remains unknown.

Job `20261007T222856Z-2943219-000-2154214424` attempted Qwen3.6 full-model SP8 `inference-checkpoint-loss` on 8 H200 GPUs after clean Ray preflight. The training path reached its first SP forward but rank 5 waited in an all-to-all while rank 4 waited in a gradient all-reduce; both collectives timed out after 600 seconds, so no inference comparison or semantic verdict exists. Its 12m14.9s total harness time records 13 whole minutes. Job `20261008T022826Z-3252420-000-270943738` reproduces the initialization stall after rank-local gradient telemetry removes per-parameter collectives; it emits no event after model initialization and is cancelled after 11m18s, recorded as 12 whole minutes. Job `20261008T153333Z-578767-000-28545089` reaches a harness verdict and fails with `RayTaskError(ActorDiedError)` after NCCL watchdog timeouts and worker death; its 12m13.9s correctness wall clock is recorded as 13 whole minutes. The placement outcome becomes `F(8)` and accumulates to `38(8)‡` while remaining one timed slot.

The Qwen3.6 full-model SP8 `checkpoint-resume-loss` placement on 8 H200 GPUs passes both cases after the single-node gateway placement and Ray step-call ports. Job `20261007T195713Z-2610481-000-2646118581` passes `gas1` in 4m50.6s semantic runtime and 4m54.1s total harness time, recorded as 5 whole minutes. Job `20261007T202337Z-2687987-000-838918214` passes `gas4` in 5m20.5s semantic runtime and 5m27.9s total harness time, recorded as 5 whole minutes. Five non-verdict or invalid attempts contribute another 20 whole minutes: jobs `20261007T182010Z-2453535-000-1213928253` and `20261007T192259Z-2519015-000-272162858` were cancelled after 3 whole minutes each; job `20261007T194846Z-2584136-000-1852813392` ran 6 whole minutes but resumed from a node-local path with no checkpoint on the head node; jobs `20261007T200230Z-2626439-000-2554230329` and `20261007T201050Z-2660465-000-264123493` ended after 5 and 3 whole minutes when the Ray server passed the new learning-rate argument positionally to an actor whose live signature required keyword forwarding. Job `20261008T034039Z-3372416-000-501124073` reconfirms `gas1` with 0 of 2 losses outside the gate in 4m35.6s, recorded as 5 whole minutes. The placement therefore records `PP(8)` and `35(8)‡` as one timed slot.

The Qwen3.6 LoRA SP8 `checkpoint-resume-loss` placement on 8 H200 GPUs fails both cases. The prior two `gas1` failures contribute 14 whole minutes. Job `20261007T203947Z-2726252-000-83326703` reproduces the `gas1` failure in 5m26.9s total harness time, recording 6 whole minutes; job `20261007T210139Z-2766653-000-240668939` fails `gas4` in 5m54.8s, recording 6 whole minutes. Job `20261007T202526Z-2693439-000-178969835` fails `gas4` in 8m03.2s total harness time, recording 9 whole minutes. The placement therefore records `FF(8)` and accumulates to `35(8)` while remaining one timed slot.

Job `20261007T211309Z-2791116-000-830718819` fails Qwen3.6 LoRA SP8 `e2e-sft-train-validate` on 8 H200 GPUs before the multi-row sequence-parallel log-probability repair. The final training loss differs by `0.083e-03` against the `30.000e-03` gate, while the held-out validation cross entropy differs by `1530.349e-03`; its 4m03.5s total harness time records 5 whole minutes. Job `20261007T212212Z-2814606-000-1258122406` then passes with final-training and held-out-validation differences of `3.129e-03` and `0.488e-03` against the same gate in 3m56.1s, recording 4 whole minutes. The placement records `P(8)` and accumulates to `9(8)` while remaining one timed slot.

Qwen3.6 full-model SP8 `e2e-sft-train-validate` records `P(8)` after two pre-fix attempts and two passing verifications. Job `20261007T204140Z-2728548-000-1855319063` ended before a semantic verdict after 8m59.5s when Ray token authentication had no matching client token, recording 9 whole minutes. Job `20261007T205513Z-2749405-000-2385330502` then reached a semantic failure in 10m48.0s total: final training loss differed by `6.059e-03`, inside the `30.000e-03` gate, while held-out validation differed by `2157.896e-03`, recording 11 whole minutes. After padded validation rows were packed before SP sharding, job `20261007T212148Z-2813704-000-181230179` passed in 11m28.1s total with final-training and held-out-validation differences of `24.871e-03` and `1.577e-03`, both inside the gate, recording 11 whole minutes. Job `20261007T212804Z-2827515-000-545122846` also passed in 11m29.5s under the later-rejected masked-rank and pre-shard-label candidate; its runtime remains diagnostic evidence and does not establish a retained-fix verdict. The retained `P(8)` comes from job `20261007T212148Z-2813704-000-181230179`, and the placement accumulates to `42(8)‡` as one timed slot.

Job `20261007T215824Z-2887216-000-2394717984` passes Qwen3.8-27B B200 `single-step-optimizer` on the 8-GPU allocation. `gas1` and `gas4` each pass 168 of 168 optimizer moment and update-residual tensors inside the frozen `2.707e+01` gate, with largest disagreements `1.598e-04` and `1.580e-04`. The wall clock including gateway startup and model load is 13m33.1s, recorded as 14 whole minutes. The placement records `PP(8)` and `14(8)` as one timed slot.

Job `20261008T025834Z-3320986-000-24695932` passes Qwen3.8-27B B200 `inference-checkpoint-loss` on the 8-GPU allocation. The live and reloaded training engines both score `4.402008576`, sampling scores `4.401837257`, and the absolute difference is `0.171e-03` against the `3.000e-03` gate. The wall clock is 6m25.7s, recorded as 6 whole minutes and added to the prior non-verdict 3, so the placement records `P(8)` and `9(8)‡`. Job `20261008T005457Z-3155609-000-247957229` ended before a score comparison and contributes those 3 whole minutes.

### AP runtime by config and check, whole minutes

| config                                        | single-step-grads | single-step-optimizer | checkpoint-resume-loss | inference-checkpoint-loss | e2e-sft-train-validate | rl-weight-sync | rl-router-replay |
| --------------------------------------------- | ----------------- | --------------------- | ---------------------- | ------------------------- | ---------------------- | -------------- | ---------------- |
| qwen3-8b-h200-train-sft-full-4gpus-64k        | 20(4)‡ \| 15(8)‡   | 13(4)‡ \| 11(8)‡         | 5(4)‡ \| 37(8)‡       | 6(4) \| 3(8)               | 6(4) \| 6(8)          | -              | -                |
| qwen3-8b-h200-train-sft-lora-4gpus-64k        | 3(4) \| 2(8)     | 1(4) \| 1(8)            | 10(4)‡ \| 51(8)‡     | -                         | 2(4) \| 6(8)‡               | -              | -                |
| qwen3.6-35b-a3b-h200-train-rl-16gpus-64k      | -                 | -                     | ? \| ?                | ? \| ?                   | -                      | ? \| ?         | ? \| ?           |
| qwen3.6-35b-a3b-h200-train-sft-8gpus-64k      | 8(8) \| 2(16)        | 11(8) \| 11(16)        | 10(8)‡ \| ?           | 28(8)‡ \| 1(16)           | 112(8)‡ \| ?              | -              | -                |
| qwen3.6-35b-a3b-h200-train-sft-full-8gpus-sp8-64k | 7(8) \| ?        | 11(8) \| 11(16)        | 35(8)‡ \| ?           | 38(8)‡ \| 11(16)‡          | 42(8)‡ \| ?          | -              | -                |
| qwen3.6-35b-a3b-h200-train-sft-lora-8gpus-sp8-64k | 11(8) \| ?        | 57(8)‡ \| ?          | 35(8) \| ?            | -                         | 9(8) \| ?             | -              | -                |
| qwen3.8-27b-h200-train-sft-8gpus-2k           | 5(8) \| ?        | 9(8) \| ?            | ? \| ?                | -                         | -                      | -              | -                |
| qwen3.8-27b-b200-train-sft-8gpus-2k           | 10(8) \| 5(16)        | 14(8) \| 9(16)            | 94(8)‡ \| 99(16)‡          | 9(8)‡ \| 67(16)‡             | 41(8) \| 39(16)       | -              | -                |

- Each numeric slot is the known wall clock of the AP regression run in the same slot of the AP outcome table, in whole minutes, including gateway startup and model load, whether or not the run reached a semantic verdict. For a completed pair, one runtime covers `gas1` plus `gas4`; a partially measured pair records the completed case's runtime. `‡` marks a slot whose known runtime includes elapsed time from a terminated attempt that emitted no semantic verdict.

49 timed AP slots sum to 1049 machine-minutes.

### AP GPU-node time tally

| nodes | time in hours |
| ----- | ------------- |
| 1 | 13.23 |
| 2 | 4.25 |
| 4 | 0.00 |
| 8 | 0.00 |
| Total | 21.73 node-hours |

Maintain this tally from every measured slot in the AP runtime table above, regardless of whether the matching outcome is `P`, `F`, or mixed `P/F`; exclude only `?` and `-`. Group each counted slot by its allocation: 4 or 8 GPUs count as 1 node, 16 GPUs as 2 nodes, 32 GPUs as 4 nodes, and 64 GPUs as 8 nodes. Sum each group's whole-minute runtimes and divide by 60 for the four intermediary rows. Compute the Total row as the sum of each intermediary time multiplied by its node count, so Total is measured-run node-hours rather than elapsed wall time. Values are rounded to two decimal places only after the minute sums and weighted total are computed.

### DSS outcome by config and check

P+**F**+? 114(107+**7**+0)

| config                                        | single-step-grads       | single-step-optimizer   | checkpoint-resume-loss  | inference-checkpoint-loss | e2e-sft-train-validate | rl-weight-sync | rl-router-replay |
| --------------------------------------------- | ----------------------- | ----------------------- | ----------------------- | ------------------------- | ---------------------- | -------------- | ---------------- |
| qwen3-8b-h200-train-sft-full-4gpus-64k        | PP(4) \| PP(16)         | PP(4) \| PP(16)         | PP(4) \| PP(16)         | P(4) \| P(16)             | P(4) \| P(16)          | -              | -                |
| qwen3-8b-h200-train-sft-lora-4gpus-64k        | PP(4) \| PP(16)         | PP(4) \| PP(16)         | PP(4) \| PP(16)         | -                         | P(4) \| P(16)          | -              | -                |
| qwen3.6-35b-a3b-h200-train-rl-16gpus-64k      | -                       | -                       | PP(8) \| PP(32)         | P(16) \| P(32)            | -                      | P(16) \| P(32) | P(16) \| P(32)       |
| qwen3.6-35b-a3b-h200-train-sft-8gpus-64k      | P**F**(8) \| PP(16)     | PP(8) \| PP(16)         | PP(8) \| PP(16)           | P(8) \| P(16)             | P(8) \| P(16)          | -              | -                |
| qwen3.6-35b-a3b-h200-train-sft-full-8gpus-sp8-64k | P**F**(8) \| PP(16)     | PP(8) \| PP(16)         | PP(8) \| PP(16)          | P(8) \| P(16) | P(8) \| P(16)          | -              | -                |
| qwen3.6-35b-a3b-h200-train-sft-lora-8gpus-sp8-64k | PP(8) \| PP(16)         | PP(8) \| PP(16)          | PP(8) \| PP(16)           | -                         | P(8) \| P(16)          | -              | -                |
| qwen3.8-27b-h200-train-sft-8gpus-2k           | PP(8) \| PP(16)         | PP(8) \| PP(16)         | **FF**(8)† \| **FF**(16)† | -                         | -                      | -              | -                |
| qwen3.8-27b-b200-train-sft-8gpus-2k           | PP(8) \| P**F**(16)      | PP(8) \| PP(16)         | PP(8) \| PP(16)     | P(8) \| P(16)             | P(8) \| P(16)          | -              | -                |

Legend:
- Each cell holds two slots, written `verdict(gpus)`, where the number is how many GPUs were available to that run, the sum of the slots in the allocation it was started with. The left slot is the width the config declares; a reinforcement-learning config places its training and sampling sub-jobs on disjoint GPUs, so its width is the sum of the two. The right slot is the widest multiple of that width which has been measured -- 2 or 4 times it, depending on how many GPUs were available for that run. A wider measurement replaces a narrower one, so the right slot only ever moves wider and a run on a larger fleet does not invalidate the rows already in the table. A config is measured at more than its declared width because the extra GPUs widen data parallelism, which the declared width cannot exercise.
- `P` -- the case agreed within the tolerance that gates it.
- **`F`** -- the case ran and disagreed by more than that tolerance. Only each `F` marker is bold, so a row that needs attention is visible without visually marking a passing case or its GPU count as failed.
- `-` -- the check does not apply to this config, so no comparison exists to make. A reinforcement-learning config reads this way for every check needing the single-GPU reference, which runs one supervised step and cannot generate rollouts. A supervised config reads this way for `rl-weight-sync` and `rl-router-replay`, which need a sampling sub-job.
- `??` -- outcome unknown although the pair did run: it ended without reaching a verdict, so nothing was measured. A config whose contents have drifted from its reviewed spec reads this way, because the harness refuses it rather than running it against a stale gate.
- `?` -- outcome unknown because the pair has not run. A check the config's reviewed spec does not list among its applicable tests reads this way, and so does a pair whose run has not landed yet. `checkpoint-resume-loss` appears in no spec's applicable list until `onboard --test-id checkpoint-resume-loss` has run it against that config.
- `†` -- pending the production update. The hosted worker image predates `debug.full_determinism_must_comply` and its first backward raises at head dimension 256 -- `Deterministic backward not supported for hdim 256` from FlashAttention 3 on H200, `SM100 backward with head_dim=256 does not support deterministic mode` from FlashAttention 4 on B200 -- so these `checkpoint-resume-loss` jobs never compared a loss; the failure belongs to the image, not to the check.
- Each letter is one case. Only `single-step-grads`, `single-step-optimizer`, and `checkpoint-resume-loss` run both a `gas1` and a `gas4` case; their slots carry two letters, `gas1` then `gas4`. Every other check runs a single case and its slot carries one letter. A `?` in one position of a two-letter slot is a case that has not run.


### DSS runtime by config and check, whole minutes

| config                                        | single-step-grads | single-step-optimizer | checkpoint-resume-loss | inference-checkpoint-loss | e2e-sft-train-validate | rl-weight-sync | rl-router-replay |
| --------------------------------------------- | ----------------- | --------------------- | ---------------------- | ------------------------- | ---------------------- | -------------- | ---------------- |
| qwen3-8b-h200-train-sft-full-4gpus-64k        | 3(4) \| ?(16)     | 6(4) \| ?(16)         | 37(4) \| 37(16)        | 4(4) \| 3(16)             | 6(4) \| 8(16)          | -              | -                |
| qwen3-8b-h200-train-sft-lora-4gpus-64k        | 4(4) \| ?(16)     | 4(4) \| ?(16)         | 35(4) \| 44(16)        | -                         | 4(4) \| 5(16)          | -              | -                |
| qwen3.6-35b-a3b-h200-train-rl-16gpus-64k      | -                 | -                     | 7(16) \| 24(32)       | 4(16) \| 12(32)           | -                      | 5(16) \| 12(32) | 6(16) \| 12(32)  |
| qwen3.6-35b-a3b-h200-train-sft-8gpus-64k      | 6(8) \| 4(16)     | 12(8) \| 15(16)       | 13(8) \| 16(16)        | 4(8) \| 4(16)             | 51(8) \| 97(16)        | -              | -                |
| qwen3.6-35b-a3b-h200-train-sft-full-8gpus-sp8-64k | 7(8) \| 4(16)     | 13(8) \| 16(16)       | 19(8) \| ?(16)         | 4(8) \| 4(16) | 11(8) \| 12(16)         | -              | -                |
| qwen3.6-35b-a3b-h200-train-sft-lora-8gpus-sp8-64k | 6(8) \| 8(16)     | 6(8) \| 6(16)       | 17(8) \| 19(16)        | -                         | 5(8) \| 6(16)          | -              | -                |
| qwen3.8-27b-h200-train-sft-8gpus-2k           | 5(8) \| ?(16)     | 11(8) \| ?(16)        | 8(8) \| 9(16)          | -                         | -                      | -              | -                |
| qwen3.8-27b-b200-train-sft-8gpus-2k           | 5(8) \| ?(16)     | 11(8) \| ?(16)        | 99(8) \| ?(16)        | 5(8) \| 4(16)             | 10(8) \| ?(16)         | -              | -                |

- Each slot is the wall clock of the regression run that produced the verdict in the same slot of the outcome table, in whole minutes, including gateway startup and model load. For `single-step-grads`, `single-step-optimizer`, and `checkpoint-resume-loss`, that one runtime covers the combined `gas1` and `gas4` run; it is not a per-case runtime. The number in parentheses is that slot's GPU count.
- `-` and `?` read as in the outcome table. `?(gpus)` beneath a measured verdict means the retained verdict-producing artifacts do not isolate that check's wall clock; combined per-config time is not assigned to an individual check.

59 timed slots sum to 834 machine-minutes. Several hosts run concurrently, so that total is machine time and not elapsed time.

### DSS GPU-node time tally

| nodes | time in hours |
| ----- | ------------- |
| 1 | 7.18 |
| 2 | 5.72 |
| 4 | 1.00 |
| 8 | 0 |
| Total | 22.62 node-hours |

Maintain this tally from every measured slot in the DSS runtime table above, regardless of verdict, using the same allocation grouping, unknown-slot exclusion, minute-to-hour conversion, node weighting, and final rounding specified for the AP tally. The intermediary rows sum all recorded runtime on allocations of that node count; Total is measured-run node-hours.

### Node-hours by model

| model | AP node-hours | DSS node-hours |
| ----- | ------------- | -------------- |
| qwen3-8b | 3.30 | 4.95 |
| qwen3.6-35b-a3b | 8.10 | 14.67 |
| qwen3.8-27b | 10.33 | 3.00 |
| Total | 21.73 | 22.62 |

Maintain each platform column from every measured slot in its corresponding runtime table, regardless of verdict. Assign each config to its model family, multiply each slot's whole-minute runtime by its allocation node count, sum those node-minutes by model, divide by 60, and round to two decimal places only after summation. Each platform Total must equal the Total in that platform's GPU-node time tally.

### Production code changes

Enumerated product edits made while driving remaining outcome-table `F` cells to `P`. Config knobs under test are not changed.

#### DSS (old)

1. **Liger fused-CE weight-grad accumulator always uses `accum_dtype=torch.float32`** (`arctic_platform/model/implementations/gpu/lm_head.py`). Liger otherwise accumulates the output projection's gradient in the weight dtype, so its result depends on how tokens are split across chunks and ranks. `fp32_lm_head` already selected this accumulator; the product default now matches without requiring a config knob change. Qwen3.6's input and output embeddings are distinct, so this change did not address the former `embed_tokens.weight` `gas4` failures. Needs `liger_kernel >= 0.8.1` for per-chunk `addmm(out_dtype=float32)`.

#### AP (new)

2. On one H200 with torch 2.11.0+cu130, CUDA 13.0, and liger-kernel 0.8.4, a per-rank fused forward-backward at `BT=2048`, `V=248320`, and `H=5120` uses 32 chunks of 64 tokens with the default `chunk_mem_const=1`. The persistent `V x H` accumulator has 1,271,398,400 elements: 2,542,796,800 bytes (2.368 GiB) in bf16 and 5,085,593,600 bytes (4.736 GiB) in fp32. The bounded bf16 logits buffer is 31,784,960 bytes (30.312 MiB). A no-weight-gradient control at the same shape isolates the non-accumulator allocations at 86,321,152 peak allocated bytes (82.322 MiB) and 88,080,384 peak reserved bytes (84.000 MiB), retaining 54,526,464 allocated bytes (52.000 MiB) after the fused forward. Fresh-process allocator measurements from the same 2,564,833,280-byte allocated baseline show a bf16/default forward peak delta of 2,630,166,528 allocated and 2,631,925,760 reserved bytes (2.450 and 2.451 GiB), versus 7,683,975,680 and 7,717,519,360 bytes (7.156 and 7.188 GiB) for fp32 accumulation. The production default therefore adds 2.368 GiB throughout chunk accumulation and raises the observed per-rank peak by 5,053,809,152 allocated bytes (4.707 GiB) and 5,085,593,600 reserved bytes (4.736 GiB): at the peak, Liger's fp32 `V x H` accumulator overlaps the bf16 `V x H` cast returned to autograd. After the fused forward both modes retain the same 2,598,371,840-byte delta in bf16 gradients; backward adds only 2,048 allocated bytes.

3. **Sequence-parallel batches are cut by logical data-parallel row, then along the sequence** (`arctic_platform/common/utils/batch.py`). `_split_batch` divides the workers by `sp_size`, stamps `dp_size` with the logical replica count, pads each row shard, and splits it on the sequence dimension. `merge_sp_dict_shards` concatenates those sequence shards before the data-parallel merge. `flatten_sequence_rows` packs a padded multi-row batch into the single row sequence parallelism requires.

4. **The client wire keeps `ds_worker_config.sp_size`** (`arctic_platform/common/utils/server_models.py`). Sequence-parallel degree is read from `ds_worker_config.sp_size` and from the nested training-config worker paths in addition to the existing sequence-parallel keys.

5. **Ray forward and step follow that sequence-parallel shape and forward the client learning rate** (`arctic_platform/common/ray_server.py`). A multi-row log-probability request with `sp_size > 1` is flattened before sharding and restored to its original row shape after the sequence shards are merged. `step` passes `learning_rate` to each worker by keyword. `create_arctic_rl_ray_server_state` waits on the server-state actor's `__ray_ready__` before returning the handle. Prompt log probabilities are converted to the platform log-prob contract.

6. **A single-node Ray start can omit peer workers** (`arctic_platform/common/ray_cluster.py`). `init_ray_cluster(..., include_peers=False)` does not start hostfile peers, so a job that fits on the local node keeps node-local checkpoints on the workers that wrote them.

7. **The DeepSpeed worker keeps full-group SFT loss scaling, forwards the client learning rate, and exports a complete Hugging Face checkpoint** (`arctic_platform/common/deepspeed_worker.py`). `_inject_sft_global_token_meta` all-reduces the valid-target count and sets `dp_size` to `world_size` even when sequence parallelism leaves one logical data-parallel replica. `step` writes a client learning rate into every optimizer group. Before DeepSpeed partitions the model, rank zero exports the live PEFT adapter and its config. Checkpoint export gathers expert-parallel tensors into one full state, restores source tensor names and source-only tensors, rewrites the safetensors index, and copies non-weight sidecars. An export saved as `Qwen3_5TextConfig` is rewritten to `Qwen3_5Config`, using the source checkpoint's vision config when that checkpoint's model type is `qwen3_5`. Correctness gradient norms call `safe_get_full_grad`. The logical-DP loss-scaling candidate is not in this diff, and neither is a `get_full_hp_grad` reader.

8. **The flat worker config maps into the native model spec** (`arctic_platform/model/config.py`). `from_ds_worker_config` carries `sp_size`. When `ep_size > 1` it selects the `qwen3_5_moe` loader with expert and sequence parallelism, loader options, and PEFT. Otherwise it carries the worker dtype, Liger selection, activation checkpointing and offload, compile, tiled MLP, and LM-head fp32 and token-chunk settings.

9. **FlashAttention determinism can withdraw only the refused backward** (`arctic_platform/model/implementations/debug/determinism.py`). `full_determinism_must_comply` false clears the FlashAttention deterministic backward the kernel refuses and leaves the rest of the determinism set. The first FLA gated-delta autotune configuration is pinned.

10. **DeepEP combine sends bf16** (`arctic_platform/model/implementations/moe/distributed/deepep.py`). The post-unpermute payload is cast to bfloat16 before combine, the dtype used to size the buffer. A combine failure reports the tensor shape, dtype, hidden bytes, allocated NVLink bytes, and the current size hint.

11. **The fused LM head can return per-token log probabilities** (`arctic_platform/model/implementations/moe/layers/lm_head.py`). `dss_compute_logprobs` runs fused cross-entropy with `reduction="none"` and returns those values as `logprobs`. The patched forward sets `_dss_native_lm_head_logprobs`.

12. **A `ForCausalLM` config is not treated as a VLM** (`arctic_platform/model/implementations/moe/vlm.py`). `is_vlm_architecture` returns false when every declared architecture name ends with `ForCausalLM`.

13. **Qwen3.5 MoE export converts router and expert tensors independently** (`arctic_platform/model/implementations/qwen35/models/qwen3_5_moe/converting_qwen3_5_moe.py`). An internal `mlp.router.gate.weight` is written as `mlp.gate.weight` even when the expert tensors are already in the Hugging Face layout, and `w1`/`w2`/`w3` are packed only when those keys are present.

14. **Qwen3.5 flash attention builds segment lengths from position-id boundaries** (`arctic_platform/model/implementations/qwen35/models/qwen3_5_moe/modeling_qwen3_5_moe.py`). For FlashAttention 2, 3, and 4, `cu_seqlens` comes from indices where `position_ids` reset to 0, including a sequence-parallel shard whose positions are global offsets. A multi-row input is packed to one row on that path. Deterministic-backward resolution receives `must_comply`.

15. **The Qwen3.5 MoE loader accepts PEFT** (`arctic_platform/model/loaders/qwen3_5_moe.py`). The loader no longer rejects `spec.patches.peft`.

16. **SFT model outputs may be result objects or mappings** (`arctic_platform/sft/processor.py`). The loss and logits adapters preserve the existing SFT loss path while accepting native model output mappings. The prediction-aligned-label and model-owned chunked training-loss candidate is not retained.

17. **Correctness sampling selects the FLA GDN prefill** (`arctic_platform/correctness/harness/dss_driver.py`). Qwen GDN training calls FLA `chunk_gated_delta_rule`. vLLM's auto prefill selects FlashInfer on Hopper. The sampling job sets `gdn_prefill_backend` to `triton`, which is vLLM's FLA prefill.


### Request accumulation meets ZeRO gradient storage

Qwen3.6 declares `tie_word_embeddings=false`: `embed_tokens.weight` and `lm_head.weight` are distinct parameter objects with distinct storage. The former `single-step-grads` failure was the input embedding, so the fp32 fused-head accumulator changed a different parameter. A request accumulator guarded by input/output parameter identity never attaches in this model and cannot affect the failure.

Each model call still produces the best input-embedding gradient available from bf16 operands: a bf16 tensor. Converting each available tensor to fp32 before summation preserves every bit it contains; recomputing the call in a wider dtype is not required. The required path for `embed_tokens.weight` is:

```
bf16 model-call gradients -> fp32 request sum -> fp32 reduction and gradient storage
```

The measured Qwen3.6 path through DeepSpeed 0.19.7 is instead:

```
bf16 model-call gradients -> bf16 ZeRO accumulation and partition -> fp32 communication -> bf16 partition
```

A parameter hook cannot repair that path by returning fp32: autograd presents the bf16 leaf gradient to the hook, and the parameter dtype constrains its output. Promoting a bf16 partition to fp32 for communication cannot restore discarded significand bits. `safe_get_full_grad()` likewise returns an fp32 tensor reconstructed from those bf16 fragments rather than a preserved fp32 request sum.

A model-specific ZeRO-1 trace runs the Qwen3.6 DP `gas4` case at its 8-GPU placement inside a two-node, 16-H200 allocation. The traced input embedding is `[248320, 2048]`, reaches ZeRO as bf16, and is copied into a bf16 contiguous reduction/owned-partition buffer before fp32 communication. The uncontended check measures a `26.65e-03` norm disagreement against its `25.00e-03` gate, isolated to `embed_tokens.weight`.

A direct four-layer boundary driver over the frozen 8,192-token `gas4` row measures all 2,097,152 layer-0 router logits and all 65,536 selected expert indices as exact between the reference and Arctic Platform. An off-spec cast of normalized top-k scores to the model activation dtype makes all 65,536 routing weights exact (`0.000e-03` relative L2 and maximum absolute delta), while the routed-expert output retains 5,053 of 8,192 exact lane values with `2.801e-03` relative L2 and `0.122e-03` maximum absolute delta. The cast moved both eight-GPU gradient checks inside their gates but changed the existing DP end-to-end pass into a failure beginning at step 13, so it was reverted. The no-regression baseline keeps the eight-GPU DP and SP8 `gas4` gradient cases failed and the sixteen-GPU slots passed.

An embedding-backward boundary probe runs the frozen 8,192-token `gas4` row three times on eight H200s. It accumulates repeated token-index contributions in fp32 and casts only the complete leaf gradient to bf16, while the control uses the stock bf16 embedding backward; both expose the same bf16 leaf to the model and leave the forward bytes unchanged. The candidate and control gradients are bitwise identical in every replay: relative L2, maximum absolute delta, and norm delta are all `0.000e-03`. Their norm disagreements against the reference are `10.311e-03`, `24.421e-03`, and `24.564e-03`, so wider internal embedding accumulation cannot repair the mismatch once the leaf-gradient boundary remains bf16.

A layer-0 replay on eight H200s uses one saved 8,192-token input and upstream gradient for both engines (`sha256=77a8a157ff8cc578072075593f536ac5a1c2d3636644aa0d226124333393eaea`). Across five executions, the first differing layer-0 backward boundary is the routed-MoE input contribution: 5,460,384 to 5,460,628 of 16,777,216 elements are exact, relative L2 is `6.188260e-03` to `6.188564e-03`, and maximum absolute delta is `2.980232e-08`. Splitting that contribution isolates differences in both branches. The expert-path contribution has 5,595,305 to 5,595,462 exact elements, `6.703448e-03` to `6.703794e-03` relative L2, and `2.980232e-08` maximum absolute delta. The router-path measurements repeat exactly: 8,170,487 exact elements, `3.628303e-03` relative L2, and `1.490116e-08` maximum absolute delta. The expert path is the larger branch-level disagreement under the shared replay.

#### Boundary measurement setup

A tied toy isolates the proposed fp32-side-sum handoff to ZeRO; it is a storage-boundary probe, not a reproduction of Qwen3.6's untied parameter layout. The probe uses two H200 nodes with eight GPUs per node (`world_size=16`), DeepSpeed 0.19.7, NCCL 2.28.9 with CUDA 13.0, and the repository's `dss` environment. The model has one bf16 `[256, 128]` embedding matrix (32,768 elements); the same matrix is the input embedding and output projection. Each rank processes two independently seeded `[1, 96]` microbatches with cross-entropy, `gradient_accumulation_steps=2`, `train_batch_size=32`, AdamW at `1e-3`, and either ZeRO stage 1 or stage 2. Seed `20261005` fixes model initialization; rank and microbatch seeds fix token and label generation.

Three hooks expose the transition. The first records each autograd gradient. The second accumulates those bf16 tensors into a request-local fp32 sum and returns zero for the first backward. The final backward returns the complete request sum in the parameter dtype, exactly the narrow request-accumulator design. The third records what reaches ZeRO. After the gradient-accumulation boundary, the probe records `param.grad`, `param.grad_accum`, the optimizer partition tensors, the parameter's low/high-precision mapping, and `safe_get_full_grad()` before the optimizer step.

The distributed controls are computed separately from the optimizer: all ranks all-reduce (a) the local fp32 request sums and (b) the same local sums after a bf16 round trip, then divide by sixteen. This separates the final local cast from ZeRO's storage and reconstruction.

#### Boundary measurement

ZeRO stages 1 and 2 produce the same numerical result.

| Quantity | ZeRO-1 | ZeRO-2 |
| --- | ---: | ---: |
| autograd hook input dtype | bf16 | bf16 |
| request side-sum dtype | fp32 | fp32 |
| final hook output dtype | bf16 | bf16 |
| optimizer gradient-partition dtype | bf16 | bf16 |
| `param.grad_accum` | absent | absent |
| `safe_get_full_grad()` dtype | fp32 | fp32 |

The first microbatch's local fp32 side sum has norm `1.553190708`. The complete two-microbatch side sum has norm `2.385491610`; its bf16 round trip has norm `2.385478020`. The scalar norm changes little because positive and negative element errors cancel, while the vector exposes the discarded information:

| Local final cast | Value |
| --- | ---: |
| absolute norm change | `-0.013590e-03` |
| relative norm change | `0.005697e-03` |
| relative vector delta | `1.304342e-03` |

The separate distributed controls have norms `1.498084307` for the ideal fp32 average and `1.498084545` after each rank's local bf16 cast. The reconstructed ZeRO gradient has norm `1.498115778` and the following vector deltas on both stages:

| Distributed comparison | Relative vector delta |
| --- | ---: |
| locally cast bf16 average against ideal fp32 average | `0.575021e-03` |
| `safe_get_full_grad()` against ideal fp32 average | `2.943119e-03` |
| `safe_get_full_grad()` against locally cast bf16 average | `2.877908e-03` |

ZeRO-1 exposes a zero bf16 `param.grad` after the first backward; ZeRO-2 leaves it absent. Both store the owned 2,048-element gradient fragment in bf16 (`averaged_gradients` and the parameter's low-precision gradient mapping). The fp32 optimizer-master fragment is a parameter copy, not fp32 storage for this gradient. `safe_get_full_grad()` converts and gathers the low-precision fragments, which explains why its output dtype is fp32 but its values remain separated from the ideal fp32 average.

An independent cancellation-sensitive ZeRO-2 probe makes the same boundary visible without relying on model arithmetic. Eight ranks run inside a two-node, 16-H200 allocation. The model is one 65,536-element bf16 parameter, and a custom autograd function supplies four fp32 source gradients designed so their large terms cancel. DeepSpeed is configured with four gradient-accumulation steps, fp32 communication, fp32 configured gradient accumulation, ZeRO stage 2, reduce-scatter, and no communication overlap. The fp32 autograd source reaches the parameter hook as bf16; ZeRO reports `use_grad_accum_attribute=0`, reduces from bf16 `param.grad`, and allocates its partition buffer in bf16. The final reconstructed gradient has norm `0.159604475` against the exact fp32 norm `0.739019692`, with relative L2 error `0.860636592`. This adversarial probe amplifies the same storage boundary; it does not estimate the full model's ordinary error magnitude.

#### Full-model consequence

A direct product candidate attaches a request-scoped fp32 side sum to the untied `embed_tokens.weight` whenever a request has more than one model call. Each parameter hook receives the bf16 leaf gradient produced by one backward and immediately converts it to fp32. At the accumulation boundary, the candidate reduces the owned ZeRO fragment in fp32 and replaces only that parameter's averaged-gradient entry before norm measurement, master-gradient flattening, and optimizer consumption. It cannot recover precision already lost inside one bf16 model call.

A 32-H200 allocation runs the reviewed `gas4` cases with this candidate active. The DP config packs 32 sequences into four model calls; the SP8 config executes 32 model calls. Both remain localized failures:

| Configuration | Arctic Platform model calls | Target norm | Reference norm | Absolute difference | Gate | Median Arctic Platform/reference ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DP, 32 GPUs, `gas4` | 4 | `1.18904` | `1.21641` | `27.37e-03` | `25.00e-03` | `1.000289` |
| SP8, 32 GPUs, `gas4` | 32 | `1.18397` | `1.21104` | `27.07e-03` | `24.00e-03` | `1.000047` |

Every other compared parameter passes. The same candidate regresses the DP config's eight-GPU `e2e-sft-train-validate` result: final training loss differs by `105.566e-03` against the `30.000e-03` gate, while validation differs by `2.676e-03` and remains inside the gate. Step 1 differs by `0.576e-03`; the first disagreement is step 7, after six optimizer steps, and the widest recorded step differs by `183.757e-03` at step 12.

The request-side-sum candidate is not a production fix: it crosses neither gradient gate and changes the multi-step optimizer trajectory enough to turn the eight-GPU e2e cell from pass to fail. The residual gradient difference is already present in the bf16 leaf gradients that reach the hooks. Boundary bisection finds exact router logits and expert indices, then the first divergence at normalized top-k routing scores. Casting those scores to bf16 is also not a production fix: although it closes the eight-GPU gradient boundary, it regresses the previously passing DP end-to-end trajectory and has been reverted.

A global `grad_accum_dtype=fp32` changes every parameter and selects a different DeepSpeed optimizer path; it is not equivalent to one parameter-specific fp32 partition. Writing through `safe_set_full_grad()` is also insufficient while its destination is the same bf16 low-precision fragment.

## Part 1. Arctic Platform Config Onboarding

### Configs

Store configs at `arctic_platform/correctness/configs/<model>/<gpu-family>/<workload>.config`, for example `arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config`. The config ID is its path below `configs/` with slashes replaced by dashes; the ID names its reviewed spec and report rows.

Supported models and recommended starting configurations are listed at <https://docs.snowflake.com/en/LIMITEDACCESS/snowflake-cortex/cortex-training-models>. Store the reviewed native `ArcticClientConfig` JSON with top-level allocation fields and nested `training`, `sampling`, and `backend` sections.

The harness validates GPU count, GPU family, attention implementation, and sequence-parallel divisibility. It refuses an incompatible config rather than reshaping it. Arctic Platform uses the config's own `n_gpus`, and also a multiple of it where the allocation has room, as described under [The gpu dimension](#the-gpu-dimension). The reference always uses one GPU.

A config declaring more GPUs than can be placed is refused by name at load time, before any checkpoint work: the config runs as written, so there is no narrowing of its topology to make it fit. What can be placed is one node's devices when the harness starts its own gateway, and the whole allocation when a gateway already running across a hostfile is named with `DSS_GATEWAY_URL` and `DSS_GATEWAY_HOSTFILE`.

Hopper configs use `flash_attention_3`; Blackwell configs use `flash_attention_4`. Arctic Platform and the Hugging Face reference use the architecture-matched FlashAttention implementation. Data-parallel configs receive enough generated rows to keep every DP shard active. LoRA configs apply the same PEFT configuration and initial adapter weights to both engines and compare trainable adapter gradients only.

### Onboard a config

Onboarding performs the expensive, one-time work that regression must not repeat. The checkpoint defaults to the training sub-job's `model_name`; `--source-checkpoint` overrides it for a private, renamed, or local checkpoint.

Concrete example:

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config
```

Generic form:

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/<model>/<gpu-family>/<workload>.config
```

After the gradient check has produced the config's reviewed spec, onboard the optimizer-step check against the same generated cases:

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --test-id single-step-optimizer
```

Onboarding is per check, not per config: a config is onboarded again for each check it is to run, and that is what adds the check's id to its `applicable_tests`. A check that compares Arctic Platform against Arctic Platform declares `compares_to_reference=False` and takes a much thinner path, because there is no reference arm to size a model for and no engine-to-engine spread to calibrate. Its gate is a constant in its own module, so onboarding runs it once against the config's existing reviewed spec and, on a pass, adds its id to `applicable_tests`; nothing is written to `test_tolerances` or `test_settings` and no model is materialized:

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --test-id inference-checkpoint-loss
```

That path refuses a config with no reviewed spec, and refuses a config whose training content no longer hashes to the checksum its spec records rather than re-stamping it. On any failure the previous spec bytes are restored. A check declaring `requires_hosted_control_plane=True` is refused there as well, with a message naming the hosted command, because onboarding publishes only what it has run; no registered check declares it at present.

Override the source only when the configured model cannot be resolved from the Hub or local cache:

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/<model>/<gpu-family>/<workload>.config \
  --source-checkpoint <local-path-or-hub-id>
```

Onboarding process steps:

1. Reads the training sub-job. The config file is not written, now or later: a knob it leaves unset stays unset and resolves through the runtime's own default, as it does for the operator's job.
2. Resolves the source checkpoint from `model_name` or the explicit override.
3. Finds the shortest layer prefix containing every attention-layer type, with a four-layer minimum. Attention backend, LM-head and loss settings, PEFT settings, seed, and batch semantics remain those of the config. Correctness input generation applies the 64K aggregate GAS1 ceiling described in [Cases and packing](#cases-and-packing).
4. Materializes that reduced checkpoint in the model cache and runs a real single-GPU Hugging Face forward-backward to verify that the minimum representative reference fits. CUDA OOM is reported as a sizing failure; any other worker failure aborts onboarding.
5. Builds the frozen GAS1 and GAS4 cases from the config's GPU count and the 64K aggregate GAS1 ceiling. A configured sequence length or microbatch token budget above 64K is reduced for correctness testing; smaller configured limits remain in force.
6. Executes the selected check's repeatability calibration with a ten-run tqdm progress bar showing completed runs, elapsed time, and ETA. The optimizer check keeps Test 1's generated cases but materializes its own minimum representative reference so Adam states do not compete with unnecessary layers for GPU memory.
7. Writes a candidate reviewed spec, runs the complete frozen regression exactly as future operators will run it, and accepts only a semantic pass.
8. Publishes the reviewed spec only after final validation. A failed first onboarding removes the candidate spec; a failed re-onboarding restores the prior reviewed spec.

The model cache defaults to `/data-fast/base-models/synthetic`. `--cache-root`, `--output-dir`, `--test-spec`, `--tokens`, and `--reference-token-budget` override generated locations or reference limits when required.

## Part 2. Run regression testing

All checks for one config, using an existing config:

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config
```

Generic form:

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/<model>/<gpu-family>/<workload>.config
```

Every reviewed config compatible with the current GPU family:

```bash
python -m arctic_platform.correctness run --all-configs
```

One check across all configs:

```bash
python -m arctic_platform.correctness run --all-configs --test single-step-grads
```

Run only the optimizer-step check for one config:

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --test single-step-optimizer
```

The quickest generated case, using an existing config:

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --case gas1
```

Generic form:

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/<model>/<gpu-family>/<workload>.config \
  --case gas1
```

List registered checks:

```bash
python -m arctic_platform.correctness list
```

`--all-configs` searches `--config-dir` recursively and selects only configs matching the current GPU family. `--any-gpu` deliberately bypasses that protection. `--case` and `--test` are repeatable. `--fail-fast` stops after the first non-pass case. `--order batched` pays one gateway startup for all selected cases; `--order per-arm` alternates reference and Arctic Platform and restarts the gateway for each case. `--out` selects the report directory.

A missing reviewed spec is inapplicable, not a pass. Config/spec drift, runtime errors, missing parameters, incomplete comparisons, and correctness mismatches are failures.


### Reports and outcomes

Every run writes a timestamped pair under the output directory:

```text
report-YYYY-MM-DD-HH-MM.md
report-YYYY-MM-DD-HH-MM.json
```

`report.md` and `report.json` are relative symlinks to the latest timestamped pair. Earlier timestamped reports remain available.

The Markdown report includes environment identity, model hash, attention implementation, frozen gates, case shapes, active tokens, padding, expected and observed model calls, overall totals, and one row per check and case. A failed case lists every over-gate quantity with its target value, reference value, absolute difference, and ratio. The JSON report carries the same verdict data for machine consumers.

Outcomes are `pass`, `fail`, and `inapplicable`. Inapplicable is never counted as a pass. If every selected config is inapplicable, the command exits nonzero and states that nothing ran.

## Part 3. Nuances

### Reference, target, and case

A check compares two sides, and they are called `reference` and `target` everywhere: `Mismatch.reference` and `Mismatch.target` carry the two values of one compared quantity, `ctx.reference_for(case)` and `ctx.target_for(case)` fetch the two sides' outputs, and the report columns are headed the same way.

Which engine plays which role belongs to the check. In `single-step-grads` and `single-step-optimizer` the reference is the single-GPU Hugging Face run and the target is Arctic Platform. In `checkpoint-resume-loss` both sides are Arctic Platform: the reference is the uninterrupted run and the target is the run that went through a checkpoint. `dss` names an engine and is therefore not a role name.

`arm` is the third word and means something else: one generated case, `gas1` or `gas4`, produced by `arms_for` and carried on `TestResult.arm`. It never names a compared side.

### Frozen tolerances

For the gradient-norm agreement check, onboarding measures repeatability with sixteen forward-backward executions of the reduced single-GPU Hugging Face reference. Every execution uses the same checkpoint, the same frozen materialized `gas1` batch, the same seed, and the config-selected FlashAttention, LM-head, loss, PEFT, and matmul settings.

Nondeterminism is enabled during these executions, and only during them: deterministic PyTorch algorithms are disabled, deterministic cuDNN is disabled, deterministic FlashAttention is disabled, and `CUBLAS_WORKSPACE_CONFIG` is unset. Each of the sixteen runs is its own process, because a process picks its kernels and autotuning once and reuses them, which would narrow the measured spread to the part the tolerance does not need to cover. The checks themselves run the opposite way, seeded and as deterministic as the build allows.

For each parameter, onboarding computes:

```text
range(parameter) = max(gradient L2 norm across the sixteen runs)
                 - min(gradient L2 norm across the sixteen runs)
raw_variation = max(range(parameter) for every aligned parameter)
```

The absolute regression gate is:

```text
max(1e-3, ceil((2 * raw_variation) / 1e-3) * 1e-3)
```

For the optimizer-step check, onboarding reuses the exact generated `gas1` input and runs sixteen nondeterministic optimizer references, eight at a time across the node's GPUs, so sixteen samples cost about the wall clock of two. Each reference uses the config's AdamW hyperparameters except for the deliberately large **`lr=1e-2`**, matches the configured Adam master/state precision, and records the parameter update, `exp_avg`, and `exp_avg_sq` after one step. For every tensor, onboarding reconstructs the bias-corrected Adam update implied by each run's own moments and subtracts it from that run's recorded parameter update. It measures full-tensor L2 deltas across all 120 pairs for both moments and this update residual; the largest pairwise delta is the raw variation. The residual retains weight decay and optimizer arithmetic that the moments do not explain, while a near-zero gradient sign change no longer turns an accepted moment difference into a learning-rate-sized update difference.

The gradient check doubles its raw variation and the optimizer check quadruples it; each rounds upward to the next whole `1e-3` unit and applies a `1e-3` minimum. The reviewed spec stores the run count, raw variation, multiplier, computed value, rounding quantum, minimum, selected gate, and worst tensor. Regression consumes the reviewed gate and never recalibrates itself.

### Reviewed spec contract

A reviewed spec records:

- the canonical SHA-256 checksum of the normalized training sub-job;
- source and materialized model identity, reduced layer count, model hash, and measured reference peak memory, plus a check-specific reduced model when optimizer memory requires fewer layers;
- generated cases, active tokens, padding, and expected Arctic Platform model calls;
- applicable checks and their frozen gates;
- GPU-family attention implementation and parameter-name mapping.

Regression refuses a missing checksum or a checksum that differs from the current config. Any behavior-changing config edit therefore requires onboarding again. Formatting-only JSON changes do not change the canonical checksum. A changed case shape also requires re-onboarding. Missing or extra aligned parameter names fail rather than silently reducing coverage.

### The config is never modified

Onboarding and regression read a config and never write one. The file on disk stays byte-identical to the recipe it came from, including the knobs it leaves unset: an absent knob means the runtime's own default applies, and neither tool fills one in. A config that omits `matmul_precision`, `sp_size`, or a `prime_rl` block runs with whatever the training schema and `arctic_platform/common/deepspeed_worker.py` resolve for it, which is what the operator's job uses.

What the harness changes is the request payload, built per case in `arctic_platform/correctness/harness/dss_driver.py` from a deep copy of the training config:

- `mb_spec.max_tokens_per_mb` is lowered to at most 65,536. A smaller configured budget is left as written, and an absent one resolves through the runtime's own budget helper rather than a value the harness carries.
- `debug.gradient_norms_per_param` and `debug.gradient_sample_max_numel` return the per-parameter gradient norms the comparison reads.
- `debug.full_determinism` is on, together with `debug.full_determinism_must_comply: false`. A comparison that is not repeatable is not evidence, so every reduction order the build can pin is pinned: the cuBLAS workspace, NCCL, and the deterministic ATen kernels that fix the embedding gradient's scatter-add. FlashAttention 3 refuses a deterministic backward above head dimension 192, which this model family exceeds; `must_comply: false` is what lets the run proceed with the rest of the set instead of stopping, and the frozen gate covers the attention backward that stays free. `debug.fp32_precision` stays off, so the arithmetic is the operator's.
- `debug.optimizer_state_output_dir` is set for the step check, so post-step artifacts land where the check reads them.

`--attn` is the one command-line override that reaches the payload, and it is deliberate rather than silent.

Because of this, the checksum in a reviewed spec describes the recipe exactly as written. A regression that refuses its spec on a checksum mismatch is reporting an edit made by a person; the harness cannot produce one.

### Cases and packing

GAS1 contains `n_gpus` variable-length sequences and GAS4 contains `4 * n_gpus` sequences. The per-sequence padded width is at most `floor(65,536 / n_gpus)` tokens and never exceeds the config's sequence-length limit or an explicit lower onboarding limit. GAS1 therefore contains at most 65,536 padded token slots in aggregate; GAS4 contains at most 262,144 padded slots as four GAS1-sized groups. Active tokens are lower because generated sequence lengths differ and every case includes padding.

Correctness execution caps `mb_spec.max_tokens_per_mb` at 65,536 when the training config allows more; a smaller configured budget remains unchanged. The one-GPU Hugging Face reference never has to hold more than one GAS1-sized group for forward-backward. GAS4 is accumulated from four groups rather than represented as one 256K sequence.

The runner materializes each Test 1 case once and gives the exact same batch to both engines. Test 2 reuses those Test 1 batch files unchanged and adds only the optimizer step and moment/update-residual comparison. Arctic Platform strips padding and packs sequences while preserving their boundaries. The reference keeps the equivalent sequence boundaries on one GPU. Neither engine may silently interpret several packed samples as one sample.

### Attention implementation

Both engines use the accelerator-family implementation encoded by the config path: FlashAttention 3 on Hopper and FlashAttention 4 on Blackwell. This keeps packed-sample semantics and kernel-family behavior aligned. `--attn` is an explicit Arctic Platform override for a deliberate comparison; it does not silently change the reviewed default.

On a B200 with `flash-attn-4 4.0.0b32`, transformers 5.12.1 and torch 2.11.0+cu130, `Qwen3_5ForConditionalGeneration` accepts `flash_attention_4` and completes a forward and backward; `flash_attention_3` raises `ImportError` on the same host, because the FA3 package is Hopper-only and the provisioning script installs it only for H200.

### The gpu dimension

Agreement with the single-GPU reference is a claim about a configuration, not about one topology, so a config is measured at the number of GPUs it declares and, where the allocation has room, once more at a multiple of that width. The multiple is how many times the declared width fits in the allocation, never more than four. A config declaring more GPUs than the allocation holds is skipped rather than narrowed, because the config is run as written.

An allocation of one node holds eight GPUs, so the widths for the three declared sizes are:

| allocation | 4-GPU config | 8-GPU config | 16-GPU config |
| --- | --- | --- | --- |
| 1 node, 8 GPUs | 4, 8 | 8 | skipped |
| 2 nodes, 16 GPUs | 4, 16 | 8, 16 | 16 |
| 4 nodes, 32 GPUs | 4, 16 | 8, 32 | 16, 32 |

A 4-GPU config on one node therefore runs once as written and once at 8, while an 8-GPU config on that node has no second width available. On two nodes the 4-GPU config reaches the cap, so its wider run is 16 rather than 8, and the 16-GPU config becomes runnable for the first time. On four nodes the cap holds the 4-GPU config's wider run at 16 even though 32 GPUs are free.

The wider run is a placement override and not a second config. The training config's checksum is taken from the file as written, and the cases are the reviewed spec's own, carrying proportionally more sequences: a case holds one sequence per GPU at the declared width, so a placement four times as wide holds four times the rows. The padded width of each sequence, the seed, and the model are the reviewed ones, so the extra rows are the only difference. Each width is compared against its own single-GPU reference, which executes that width's rows, so a wider run costs one extra reference pass.

The extra GPUs widen data parallelism: the global batch grows by the same multiple as `n_gpus`, while `sp_size` stays as the config declares it. Two independent constraints require this. DeepSpeed asserts `train_batch_size == train_micro_batch_size_per_gpu * gradient_accumulation_steps * dp_size` at engine initialization, and Arctic Platform refuses to dispatch a batch with fewer rows than there are data-parallel shards, since a shard holding no sequence has no gradient to reduce. A config states the global batch in two places and both grow: `ds_config.train_batch_size`, which DeepSpeed reads, and the training config's own `train_batch_size`, which the multi-step checks read to size their dataset slice. A config that states no `train_batch_size` in one of those places has none written for it there.

Every measured width, including the declared one, writes that identity into both copies of `train_batch_size` on the in-memory config the engine sees. The file on disk is unchanged, and the spec checksum is taken from the file.

The widths come from `placement_widths` in `arctic_platform/correctness/harness/config.py`, which is given the pool a job can be placed on: the devices of this node when the harness starts its own gateway, or the summed slots of the allocation's hostfile when an already-running gateway spans several nodes.

### Reference and Arctic Platform execution

The reference runs on one GPU with no distributed parallelism or optimizer offload. It uses gradient checkpointing, a request-wide active-token denominator, config-selected loss and LM-head settings, and a CPU-resident float32 gradient accumulator drained after every microbatch. Chunked LM-head projection is used when the config requests it. The optimizer check applies clipping with Arctic Platform semantics, then runs one AdamW step at `lr=1e-2` using the config's master and moment precision.

Arctic Platform runs the config's declared topology. The harness enables only the telemetry needed by the selected check, sends the generated request, and collects gradients or post-step optimizer artifacts without changing training behavior. Each case starts from the same materialized checkpoint so results do not depend on case order.

The gradient-norm check requires complete parameter-name alignment and compares every aligned norm against its frozen absolute gate. The optimizer check requires complete alignment of parameter updates, first moments, and second moments. It compares the full-tensor L2 deltas of both moments and of the update residual left after subtracting each engine's moment-implied Adam update. Expert-parallel optimizer onboarding is supported only when the trainable set is limited to replicated Q/K/V/O LoRA adapters; full parameters and expert adapters are refused until rank-local expert artifacts can be reconstructed into full tensors. Scalar loss difference is reported as a diagnostic but does not determine either verdict.

For `qwen3.6-35b-a3b-h200-train-sft-lora-8gpus-sp8-64k`, sixteen reference runs put the largest pairwise moment or update-residual variation at `0.000189e-03` against the `1.000e-03` minimum gate. Validation passes all 24 quantities in both cases on eight H200s; the largest deltas are `0.225e-03` in `gas1` and `0.221e-03` in `gas4`.

### Checkpoint-and-resume check

`checkpoint-resume-loss` asks whether a training job that stops and restarts from a checkpoint continues the run it interrupted. Both sides are Arctic Platform on identical inputs, so the single-GPU reference takes no part and the check declares `compares_to_reference=False`.

Per case, twenty iterations of forward-backward and an AdamW step at `lr=1e-3`, cross entropy, on twenty distinct batches -- one per iteration, seeded by iteration, built the way every other case is built and materialized before either run starts, so both runs consume identical inputs in the same order:

- the reference run trains all twenty iterations in one job and writes no checkpoint;
- the target run trains ten, saves a `resumable` checkpoint, and runs iterations 11 to 20 in a fresh job initialized from that checkpoint, on batches 11 to 20. The resumed job continues the data order rather than replaying it, so the only difference from the reference run is that weights, optimizer moments and LR-scheduler state arrived from disk.

The verdict compares the loss at iteration 10 and at iteration 20 against the absolute gate in the config's `test_tolerances` for this check. Onboarding measures that gate with determinism off: three runs of the first ten iterations, the same batches in every run, and the gate is twice the widest per-iteration loss range across those runs and across the config's cases, raised to the next `0.001` and no smaller than `0.001`. The iteration count, checkpoint iteration, and learning rate stay constants in `arctic_platform/correctness/checks/checkpoint_resume.py`. The regression run keeps best-effort determinism. Iteration 10 is the control -- up to there the two runs are one computation, so a disagreement there leaves the iteration-20 comparison with nothing to stand on. It admits two causes and excludes a resume defect: the runs were given different work, or the jobs are not reduction-order pinned and have drifted apart under their own optimizer steps. The first iteration separates them, since identical losses there mean the inputs agree. Iteration 20 is the resume verdict.

The save goes through `POST /save-checkpoint` with `checkpoint_type: "resumable"`, writing DeepSpeed model, optimizer and LR-scheduler state under `global_step10` in the saving job's node-local checkpoint directory. On a gateway, the saved tree is moved out of the saving job's directory before that job is destroyed, because destroy removes the directory. The resumed job is created from the base model, the tree is moved into its own checkpoint directory on every node that held it, and `POST /load-checkpoint` with `{"checkpoint_id": "global_step10"}` restores weights, optimizer moments and LR-scheduler position through the zone's runtime load. A node count that differs between the hold and the placement raises rather than loading a partial checkpoint. On the hosted transport the resume rides on `POST /initialize` with `source_checkpoint_info = {"checkpoint_id": "global_step10", "source_job_id": <saving job>}`.

A zone refuses a create-time `source_checkpoint_info` that carries no stage credentials, and only the hosted control plane mints that stage before a zone reads the job config, which is why a gateway resumes through the runtime load instead. Both transports request `full_determinism`: without it two jobs that are the same computation on the same batches agree at iteration 1 and diverge to `0.923e+00` by iteration 10, which fails the control and says nothing about the resume. With it, `qwen3-8b-h200-train-sft-full-4gpus-64k` passes both cases on the hosted server, 0 of 2 recorded losses outside the fixed `1e-3`, across six hosted jobs in 37m57s. Through a gateway, the same config passes all four cases at four and at eight GPUs in 18m15s. `qwen3.6-35b-a3b-h200-train-sft-full-8gpus-sp8-64k` passes both cases at eight GPUs under its calibrated gate of 0.893, drawn from the widest determinism-off loss range of 0.446 at `gas4` iteration 5. At head dimension 256, `full_determinism_must_comply: false` leaves the FlashAttention 3 backward nondeterministic, and the calibrated gate covers that best-effort execution. `qwen3.6-35b-a3b-h200-train-rl-16gpus-64k` passes both cases at eight GPUs under its calibrated gate of 10.852. The sixteen-GPU placement was measured against the fixed `1e-3` gate: iteration 10 differs by 6.519e-03 in `gas1` and 12.040e-03 in `gas4`, and iteration 20 by 1589.000e-03 in `gas1` and 7.472e-03 in `gas4`. `qwen3.6-35b-a3b-h200-train-sft-8gpus-64k` passes both cases at eight GPUs under its calibrated gate of 1.553. `qwen3.8-27b-b200-train-sft-8gpus-2k` passes both cases at eight B200s under its calibrated gate of 23.685. `qwen3.6-35b-a3b-h200-train-sft-lora-8gpus-sp8-64k` passes both cases at eight GPUs. The current Qwen3-8B full calibration measures a maximum reference-loss range of `20.124399`, freezes a `40.249` gate, and passes both cases at four and eight GPUs. The Qwen3-8B LoRA calibration measures `0.490e-03`, freezes a `1.000e-03` gate, and passes the same placements.

The check writes nothing to `test_settings`. It reuses the reduced materialized model an already-onboarded config has. `onboard --test-id checkpoint-resume-loss` measures the gate with determinism off, writes it under `test_tolerances`, runs the check through a gateway with best-effort determinism, and, on a pass, adds the id to `applicable_tests`. A failed validation restores the previous spec bytes. The hosted command runs the checks named on its command line whether or not a spec lists them as applicable.
When `DSS_GATEWAY_HOSTFILE` names an allocation wider than the config, that command also runs the check at the widest multiple of the declared width that fits, with `train_batch_size` and the case row counts scaled the same way a local run scales them, and it does not repeat the declared width. When that allocation has no wider multiple, the command runs the declared width, which is the only placement that fits.

### Training-versus-serving validation-loss check

`inference-checkpoint-loss` asks whether a validation loss read from the engine that trained a model equals the one read from an engine serving the checkpoint that engine saved. Both sides are Arctic Platform, so the single-GPU reference takes no part and the check declares `compares_to_reference=False`. It drives its own gateway, jobs and zones, and has no case axis: each side runs one forward over one batch, and gradient accumulation is a property of a backward pass, so a second case would re-measure the first.

One configuration, ten steps of GSM8K training at the rate the config's `optimizer.lr` declares, then the same held-out batch scored twice:

- the reference is the training job that just trained. Its forward-only route runs with no gradient and the engine in `eval()`, and returns per-position log-probabilities rather than a loss (`arctic_platform/common/deepspeed_worker.py:3031`);
- the target is a sampling job started on the weights-only checkpoint that training job saved, scored through `POST /log-probs`.

Both sides produce, per row, that row's answer-token log-probabilities in order, and `cross_entropy` in `arctic_platform/correctness/checks/inference_checkpoint_loss.py` turns them into a negative mean log-probability. Nothing else reduces anything, so a disagreement between the two numbers cannot be a disagreement about how they were averaged. The verdict is a fixed `1e-3` absolute, a constant in that module and deliberately not calibrated: the two sides are one set of weights read by two engines, so there is no engine-to-engine spread for a calibration to size.

#### Data and the validation slice

`TRAIN_SPLIT` and `VALIDATION_SPLIT` in `arctic_platform/correctness/checks/inference_checkpoint_loss.py` name the GSM8K train and test parquet files on the shared filesystem. Training rows come from the first and validation rows from the second, so the held-out slice is disjoint from training by construction; the check asserts it on the questions rather than trusting the file names. A step consumes the config's `train_batch_size` rows, in file order, with no row used twice.

Loss covers answer tokens only. Prompt tokens and the padded tail carry `IGNORE_INDEX`. Prompt text is identical on both sides and far more numerous than answers, so scoring it would dilute the quantity under test with a term neither engine can get wrong.

The validation example count is measured rather than chosen. `arctic_platform/correctness/harness/gsm8k.py` tokenizes the whole held-out split with the config's own tokenizer and takes the longest leading run whose padded rectangle -- row count times the longest row in the run -- stays within the config's single-forward token budget. Extending the run can only grow that rectangle, so the largest count that fits is the last one that fits. The run is refused if it holds fewer rows than the config has data-parallel shards, since every shard needs one.

On `qwen3-8b/h200/train-sft-full-4gpus-64k`, whose budget resolves to 10,240 tokens, the 1,319 held-out examples span 72 to 523 tokens with a median of 180, and the selected slice is 33 examples at a padded width of 304: 10,032 token slots and 4,043 scored answer tokens.

#### The checkpoint round trip

The save is `POST /save-checkpoint` with `checkpoint_type: "weights-only"`, an explicit `path`, and `stage_info: null`. That writes Hugging Face shards and their sidecars under a tag directory on the node that wrote them, so the round trip needs no object storage; the endpoint requires a stage or a path (`arctic_platform/common/ray_server.py:3746`). The sampling job is then initialized with that tag directory as its `model_name` and `weight_format: "hf"`, which is the only shape it accepts -- the loader rejects a DeepSpeed directory (`arctic_platform/common/deepspeed_worker.py:275`).

A composite source -- a vision-language checkpoint such as Qwen3.5 or Qwen3.8, whose `config.json` carries a `text_config` -- trains as its text-only model, and that model saves a text-only config and no vision tower. vLLM serves the source architecture and refuses such a tree with `TypeError: Invalid type of HuggingFace config`. The dense weights-only save therefore completes it into the source's layout (`restore_source_weight_layout` in `arctic_platform/common/deepspeed_worker.py`): trained tensors keep their values under the source's names, every tensor training never held is copied unchanged from the source, and the source's config files replace the saved one. On `qwen3.8-27b/b200/train-sft-8gpus-2k` that copies 33 tensors into a 159,488,600-byte `model-source-only.safetensors`.

The two zones hold disjoint GPUs: the gateway is asked for the config's GPU count plus one, and the sampling job takes that one. One GPU holds the reduced model and runs a single prefill over a batch the token budget bounds.

#### Alignment rules, and why each is load-bearing

Four properties decide whether the two numbers are comparable at all. Each was worth one to three orders of magnitude more than the accelerator-family gate before it was applied.

- **The forward-only route does not shift labels.** `/fwd-bwd` dispatches with `attach_global_loss_counts=True` and converts HuggingFace-convention labels to logit alignment before packing, which is why `dss_driver.pack` sends them unshifted. The forward-only route dispatches with `attach_global_loss_counts=False`, and for a `model_provider: huggingface` job `_sft_labels_are_shifted` is false (`arctic_platform/common/deepspeed_worker.py:2206`), so `shift_sft_labels_for_logits` never runs and the same request is off by one. The check compensates: its validation request is packed by `dss_driver.pack_logit_aligned`. The inconsistency between the two routes is a product finding, not something this check fixes.
- **The prompt/answer boundary is a common prefix, not a prompt length.** The answer span begins at the length of the common prefix of the prompt tokenization and the joined `prompt + completion` tokenization. A prompt that does not end on a token boundary of the joined text tokenizes to a last id the joined text never contains, and taking `len(tok(prompt))` would put the two sides on different spans.
- **vLLM reports no log-probability for position 0.** Nothing precedes it, and the sampling zone's loop drops that entry (`arctic_platform/common/ray_server.py:704`), so a returned row holds one fewer value than the row has tokens. Both sides slice the same logit positions -- `p - 1` through `L - 2` for a row of length `L` whose answer begins at `p` -- and the returned length and token ids are asserted rather than inferred from the answer span starting late enough.
- **`top_k` is a correctness knob.** The zone reads the first key of each position's log-probability dictionary. Only `top_k: 0`, which asks for the sequence's own token and nothing else, scores the row that was sent; a positive value makes the first key the top-ranked token instead. The check passes `0` explicitly because the gateway's own default is `1` (`arctic_platform/client/requests.py:181`) while the zone body's is `0`.

#### The reduced model carries a tokenizer

A reduced model written by `materialize_pretrained` carries the source checkpoint's tokenizer sidecars, including when the source is a hub id. A model directory without them fails silently in two stages: it loads, and `AutoTokenizer.from_pretrained` on it returns a single-entry vocabulary that encodes every string to an empty id list rather than raising. The weights-only save copies its sidecars from the job's model directory, so a reduced model without a tokenizer produces a serving tree without one.

`materialize_pretrained` resolves the source's snapshot before globbing for sidecars, and copies them whether or not it wrote the weights, so a cache written without them gains them on the next call. Every tokenizer the harness loads has its vocabulary size asserted against the model config's embedding row count: a vocabulary wider than the table can emit an id the model cannot look up, and a vocabulary of one entry cannot represent text. Adding sidecars changes a materialized model's recorded `content_hash`, which is reported but never compared, so an existing reviewed spec stays valid until the config is onboarded again.

#### Running it

The check writes nothing to `test_settings` and nothing to `test_tolerances`: it reuses the reduced materialized model an already-onboarded config has, and its gate is fixed at `3e-3` on B200 and `1e-3` on other GPU families. Selection intersects the chosen checks with the spec's `applicable_tests`, so a config runs it only once it has been onboarded for it. Onboarding runs the check against that config's reviewed spec and adds the id on a pass:

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --test-id inference-checkpoint-loss
```

Regression then runs it for that config:

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --test inference-checkpoint-loss
```

### End-to-end SFT training and validation loss check

`e2e-sft-train-validate` asks whether a hundred Arctic Platform training steps reproduce the loss trajectory of the same hundred steps on one GPU. The reference is the single-GPU Hugging Face engine and the target is Arctic Platform at the config's own `n_gpus`, the roles `single-step-grads` and `single-step-optimizer` use, so the check declares `compares_to_reference=True`. What duration adds is the optimizer: one step compares two forwards, and a hundred steps compares two trajectories, so the state each engine carries between steps is under test alongside the arithmetic of any one step.

One configuration and no case axis. A hundred steps of GSM8K at the rate the config's own `optimizer.lr` declares, each step consuming `train_batch_size` rows in file order with no row used twice, then the held-out batch scored once on each side after the last step.

The verdict compares two quantities against a fixed absolute `3.000e-02`: the final step's training loss, and the held-out validation loss. The gate is a stated requirement rather than a measured spread, so onboarding calibrates nothing for it and no run of this check may move it. It is held as `TOLERANCE` in `arctic_platform/correctness/checks/e2e_sft_train_validate.py`, beside the other fixed gates in this package.

Every step's loss is recorded from both engines and every step's disagreement is reported, but the trajectory is a diagnostic and not a gate. It is what separates a disagreement present at step 1 from one that first appears at step 60. The first is the forward -- the loss denominator, the label alignment, or the weights the two engines loaded -- and later steps only carry it forward; the second is what the steps did, and step 1 having agreed excludes the batches and the alignment. The final loss on its own cannot tell those apart.

Measured on eight GPUs -- H200, or B200 where the config names it -- each config at its own topology and its own `optimizer.lr`, with the two gated disagreements and the widest single step of the hundred recorded:

| Config | `n_gpus` / `sp` | Final train loss | Validation | Widest step | Verdict |
| --- | --- | --- | --- | --- | --- |
| `qwen3-8b-h200-train-sft-full-4gpus-64k` | 4 / 1 | 2.226e-03 | 0.160e-03 | 3.066e-03 (step 10) | pass |
| `qwen3-8b-h200-train-sft-lora-4gpus-64k` | 4 / 1 | 0.011e-03 | 0.179e-03 | 16.027e-03 (step 9) | pass |
| `qwen3.6-35b-a3b-h200-train-sft-lora-8gpus-sp8-64k` | 8 / 8 | 2.517e-03 | 0.306e-03 | 13.002e-03 (step 7) | pass |
| `qwen3.6-35b-a3b-h200-train-sft-full-8gpus-sp8-64k` | 8 / 8 | 8.115e-03 | 1.558e-03 | 37.825e-03 (step 86) | pass |
| `qwen3.6-35b-a3b-h200-train-sft-8gpus-64k` | 8 / 1 | 68.569e-03 | 12.522e-03 | 145.766e-03 (step 52) | fail |
| `qwen3.8-27b-b200-train-sft-8gpus-2k` | 8 / 1 | 1.661e-03 | 0.013e-03 | 19.948e-03 (step 19) | pass |

Every row was measured on Arctic Platform `v0.1.5rc1`.

The trajectory is worth reading whatever the verdict says, because the gated quantity is one draw from a wider spread. Each of the five passing configs has a single step wider than its own gated disagreement, and one passes with a step wider than the gate itself. The Qwen3-8B LoRA run passes on the smallest final disagreement in the table while its widest step reads `16.027e-03`.

Every one of the six agrees at step 1 -- the Qwen3-8B full-model run to `0.001e-03` -- so in none of them are the batches, the label alignment or the loaded weights implicated, and the disagreement each accumulates is the optimizer state, the gradient reduction, or a reduction order neither engine pins. The failing row is the only one of the three Qwen3.6 configs that is purely data-parallel; its two siblings differ from it in being eight-way sequence-parallel, and both pass.

`qwen3.8-27b-h200-train-sft-8gpus-2k` is not measurable by this check on one H200: with the FP32 master copy and two Adam moments per parameter the single-GPU reference exhausts 139.59 GiB of the card's 139.80 GiB during the first optimizer step.

#### Identical inputs, and the one optimizer

Both engines read one materialized copy of the run's inputs. The per-step batches and the held-out batch are written to disk before either engine starts; the reference process loads those files and the Arctic Platform requests are packed from the same files, so the two runs consuming identical inputs is a property of the bytes rather than of two code paths agreeing. Both sides also request `debug.full_determinism`, without which two runs that are the same computation drift apart under their own optimizer steps and a hundred-step comparison measures the drift.

`reference/optimizer_step.py` applies one AdamW step and clears the optimizer state afterwards. That is what a single-step comparison wants and what a trajectory cannot use: Adam's moments and its step counter are the state that makes step *k* depend on the `k - 1` steps before it, so a loop over that function would repeat step one a hundred times. `reference/adamw_trajectory.py` holds one optimizer across the whole run and writes the FP32 master copy back into the model's parameters after every step, because the next step's forward reads the model. The parameter grouping, the optimizer-name validation and the master precision are the single step's, shared rather than restated, so a change to the grouping cannot move only one of the two.

#### The validation loss, and why it is a log-probability

The training job's forward-only route runs with no gradient and the engine in `eval()`, and returns per-position log-probabilities rather than a loss (`arctic_platform/common/deepspeed_worker.py:3031`), so there is no loss to read on the Arctic Platform side. The reference produces the same array from the same model state, and both sides hand it to the answer-span slice and the `cross_entropy` of `inference-checkpoint-loss`, so the two numbers are the same estimator and a disagreement between them cannot be a disagreement about how they were averaged. Loss covers answer tokens only, and the held-out example count is measured against the config's own single-forward token budget the way `inference-checkpoint-loss` measures it.

That route also does not shift labels. `/fwd-bwd` dispatches with `attach_global_loss_counts=True` and converts HuggingFace-convention labels to logit alignment before packing, which is why the training requests here are packed by `dss_driver.pack`; the forward-only route dispatches with `attach_global_loss_counts=False` and shifts nothing, so the held-out request is packed by `dss_driver.pack_logit_aligned` instead. An off-by-one there produces a plausible-looking wrong loss rather than an obvious failure.

A LoRA config's adapter is initialized by Arctic Platform and exported before the reference starts, as it is for the one-step comparison. Without that the two engines would train from different adapters and the trajectories would part at step 1 for a reason that is not a defect.

#### Running it

Onboarding is per test id and selection intersects the chosen checks with the reviewed spec's `applicable_tests`, so a config runs this check once its spec lists `e2e-sft-train-validate` and reports it inapplicable until then.

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --test e2e-sft-train-validate
```

### Adding a check

The onboarding already does all the work for you, but you can also do it manually

Decorate a function under `arctic_platform/correctness/checks/` and add its ID to the spec's `applicable_tests`:

```python
@correctness_test("t03_something", title="...", criterion="...")
def run(ctx) -> list[TestResult]:
    ...
```

`ctx` provides generated cases, reference outputs, Arctic Platform outputs, and the reviewed spec. A check owns any onboarding measurement needed to freeze its contract; regression only consumes that result.

### Disposable onboarding artifacts

Onboarding writes diagnostics under `/data-fast/dss-correctness/onboarding/<config-id>/`, on the node's own disk, falling back to `.agent-work/config-onboarding/<config-id>/` in the checkout when the node has no `/data-fast`. They run to tens of gigabytes per config and are entirely recreatable, so a wiped node loses nothing: regression consumes only the checked-in config and reviewed spec, and the next onboarding writes the diagnostics again. `--output-dir` selects another location.

### Reinforcement-learning configs

A config is reinforcement learning when it declares both a training sub-job and a sampling sub-job. The pair is the signal: a sampling sub-job alone serves a policy and trains nothing, and a training sub-job alone consumes a fixed dataset. The training sub-job's own contents cannot decide this, because a supervised job may still select `model_provider: prime_rl` and the PrimeRL loss options under it.

The checks that compare Arctic Platform against the reduced single-GPU Hugging Face reference do not apply to such a config: the reference executes one supervised training step and cannot generate rollouts, so there is no baseline for the quantity the job computes. Regression reports them as inapplicable rather than skipping them silently, and onboarding refuses them by name. Two checks compare the config's own zones with each other instead. They declare `compares_to_reference=False` and are onboarded and run like any other fixed-gate check.

Both checks start the config's training and sampling sub-jobs together, on disjoint GPUs, so the gateway needs the sum of the two GPU counts: sixteen for `qwen3.6-35b-a3b-h200-train-rl-16gpus-64k`, which one 8-GPU node cannot place. The two sub-jobs cannot share one set of GPUs, so the next width is thirty-two. They run against a gateway spanning a two-node allocation, named with `DSS_GATEWAY_URL` and `DSS_GATEWAY_HOSTFILE`. The sampling sub-job is the config's own, with only `model_name` replaced by the reduced model the spec names; the training sub-job is built the way every other check builds it.

#### `rl-weight-sync`

One RL fwd-bwd over one global batch of GSM8K rows and one step at the config's learning rate move the trainer's weights off the checkpoint both zones loaded; `/sync-weights` then runs with the gateway's defaults. Three records are compared, all exactly:

- the trainer's manifest (`debug.weight_sync_manifest`): shape, dtype, element count and sha256 of every tensor it sends, and the number of elements it holds across its expert-parallel shards;
- every sampler rank's record of each tensor it received, hashed the same way, with the parameters the loader wrote it into;
- every sampler rank's shadow replay: each received tensor is loaded a second time into NaN-filled copies of its destination parameters, so a NaN left behind is an element no tensor wrote, and a written element that differs from the live parameter is one the live load did not keep.

The check fails when the trainer sends fewer elements than it holds; when a rank received a different set of tensors, or one with different bytes, shape, dtype or count; when a received tensor reaches no parameter; when a destination has an unwritten or differing element, or cannot be shadow-verified; and when a sampler parameter is written by no tensor. The sampler-side records exist only when the sampling workers run with `ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY=1`. A gateway the check starts inherits it from the check's own environment; an external gateway must be started with it, and without it the check fails naming the variable.

#### `rl-router-replay`

The sampling zone generates one answer per GSM8K prompt, one prompt per row of the config's global batch, with router replay on and `dss_return_back_router_info` set, so each result carries the expert indices its routers chose at every captured position. Each prompt may generate as many tokens as the dataset's own answer to it has. The training zone then bootstraps router replay against the sampling job and runs one RL fwd-bwd over the same rows, replaying their routing, with `debug.router_replay_trace` recording on every rank the routing it received, each model call's packed tokens, positions and `routed_experts`, and every router invocation by decoder layer, on the forward pass and inside backward, where activation checkpointing recomputes the layer.

Three equalities are checked, all exact: the routing generate returned equals the routing the trainer received for that sample; the `routed_experts` a model call carries equals the sample's routing at every captured position of the tokens it packs; and every router invocation replayed, rather than computed its own routing, and used exactly its layer's slice of that tensor. Under `ac_config` mode `full` with `freq` 1, every call must show both a forward and a recompute invocation for every layer. The last token of a sequence produces no routing in the sampler, since `capture_len` is one less than the row length, so the trainer's value at that position is not compared. Tiled MLP is not covered.

#### Running them

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-rl-16gpus-64k.config \
  --test-id rl-weight-sync
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-rl-16gpus-64k.config \
  --test-id rl-router-replay
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-rl-16gpus-64k.config \
  --test rl-weight-sync --test rl-router-replay
```

Both commands need `DSS_GATEWAY_URL` and `DSS_GATEWAY_HOSTFILE` naming a gateway over at least sixteen GPUs, started with `ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY=1`. Onboarding reuses the spec the config already has, runs the check once the way regression runs it, and lists it in `applicable_tests` only on a pass.

Before any config work, `run` and `onboard` inspect every node of a multi-node `DSS_GATEWAY_HOSTFILE` over `ds_ssh` for an interrupted torch extension build: a build directory holding a lock and no shared object while no compiler runs on that node. DeepSpeed JIT-compiles `fused_adam` under that lock, every DeepSpeed worker on the node waits on it, and the workers on the other nodes wait in their first collective until the training-zone init timeout, so the run stops instead and names each such directory; removing it, or provisioning the node again, which builds `fused_adam`, clears it. Each zone loads the spec's `cache_path` from its own node's disk, so that directory must exist on every node, together with the source checkpoint's processor files: vLLM builds a multimodal model's input processor at engine start and fails without `preprocessor_config.json`. `copy_sidecars` in `arctic_platform/correctness/onboarding/synth_model.py` adds them to a reduced tree built before it copied them.

### Unresolved issues behind outcome failures

Each numbered entry is one common issue and lists every DSS outcome-table `F` marker it explains. The marker count across the entries must equal the DSS `F` tally. Commands run from the checkout root on a node of the GPU family the config names. A sixteen-GPU figure needs a two-node allocation named by `DSS_GATEWAY_HOSTFILE`.

#### 1. Qwen3.6 `gas4` input-embedding gradient disagreement

Affected markers: `qwen3.6-35b-a3b/h200/train-sft-full-8gpus-sp8-64k` at 8 GPUs and `qwen3.6-35b-a3b/h200/train-sft-8gpus-64k` at 8 GPUs. This issue accounts for 2 of the 7 `F` markers.

```bash
python -m arctic_platform.correctness run --config arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-sft-full-8gpus-sp8-64k.config --test single-step-grads
python -m arctic_platform.correctness run --config arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-sft-8gpus-64k.config --test single-step-grads
```

The SP8 config fails `gas4` on `embed_tokens.weight` alone, with every other tensor inside its gate and the next widest at `6.09e-03`. Two seeded, best-effort-deterministic sessions measured `39.17e-03` against a `25.00e-03` gate and `35.26e-03` against a `20.00e-03` gate. The disagreement is about 1.5 times its gate in both sessions, while the unpinned FlashAttention 3 backward at head dimension 256 moves it by several `1e-03` between sessions.

The DP config's verdict varies but currently contributes the second marker. Two sessions measured `embed_tokens.weight` in `gas4` at `26.63e-03` against a `23.00e-03` gate and `30.02e-03` against a `21.00e-03` gate, while an earlier session passed at `17.62e-03` against a `25.00e-03` gate. Its other 71 tensors are inside their gates. With best-effort determinism forced on every payload, eight H200s measured `28.67e-03` and sixteen measured `64.63e-03`, both against a `25.00e-03` gate; `gas1` passed at both widths.

#### 2. Qwen3.8 H200 checkpoint check blocked before measurement

Affected markers: `gas1` and `gas4` at 8 GPUs, and `gas1` and `gas4` at 16 GPUs, for `qwen3.8-27b/h200/train-sft-8gpus-2k`. This issue accounts for 4 of the 7 `F` markers.

```bash
python -m arctic_platform.correctness run --config arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config --test checkpoint-resume-loss
```

All four markers have the same failure: the first of twenty iterations stops in the first backward with `Deterministic backward not supported for hdim 256`. Every case completes 0 iterations, produces 0 loss pairs, saves 0 checkpoints, and performs 0 resume comparisons, so none of these markers measures checkpoint-resume correctness. The retained invocation for both 8-GPU cases took 8 whole minutes; the retained invocation for both 16-GPU cases took 9 whole minutes on a two-node, 16-H200 allocation. The production-image dependency is described under [Checkpoint-and-resume check](#checkpoint-and-resume-check).

#### 3. Qwen3.8 B200 `gas4` final-norm gradient disagreement

Affected marker: `qwen3.8-27b/b200/train-sft-8gpus-2k` `gas4` at 16 GPUs. This issue accounts for 1 of the 7 `F` markers.

```bash
python -m arctic_platform.correctness run --config arctic_platform/correctness/configs/qwen3.8-27b/b200/train-sft-8gpus-2k.config --test single-step-grads
```

On a two-node B200 allocation with 16 available GPUs, `gas1` passes all 56 tensors with `norm.weight` widest at `3.418e-03` against the `4.000e-03` gate. `gas4` fails only `norm.weight`: Arctic Platform reads `3.10961`, the reference reads `3.11773`, and the absolute difference is `8.116e-03` against the same gate. Its loss difference is `0.002e-03`, and the median Arctic Platform/reference gradient-norm ratio is `0.999604`, so the disagreement is localized rather than a global scale error.

### Resolved diagnostics

#### `qwen3.8-27b/b200/train-sft-8gpus-2k`: `inference-checkpoint-loss`, inside the `3e-3` gate

```bash
python -m arctic_platform.correctness onboard --config arctic_platform/correctness/configs/qwen3.8-27b/b200/train-sft-8gpus-2k.config --test-id inference-checkpoint-loss
```

With the patched FlashAttention 4 stack, the training zone reads 4.299332 and the sampling zone reads 4.297708, an absolute difference of `1.624e-03` against the B200 `3.000e-03` gate, so the check passes in 5m19.2s. The measurements below were collected against the former `1.000e-03` gate and preserve the diagnostic history.

On eight B200s the training-zone validation cross entropy is 4.299332 and the sampling zone is 4.297012, an absolute difference of `2.319e-03` against the fixed `1.000e-03` gate. An independent eight-B200 run against the rebuilt reviewed four-layer checkpoint reads 4.299332 and 4.296074, a difference of `3.257e-03` against the same gate, in 6m08s. The second node reads 4.299276 and 4.297507, a difference of `1.770e-03` against the `1.000e-03` gate, in 6m45s. On the head-node checkpoint, two in-memory training scores and two scores after fresh training-engine reloads are identical at every one of the 4,088 answer tokens, each with cross entropy 4.299332; the sampling engine reads 4.296074. Its per-token absolute differences have a maximum of `321.669e-03`, mean `24.008e-03`, and RMS `38.536e-03`. Checkpoint serialization and training forward-only scoring are therefore excluded; the divergence begins in sampling-engine prompt-logprob production. A fresh four-minute rerun reads 4.299332 in training and 4.296215 in serving, `3.117e-03` apart. The retained-code verification reads 4.299332 and 4.296231, `3.100e-03` apart against the `1.000e-03` gate, in 3m46s. Raw token auditing finds 0 missing positions and 0 ordering mismatches across 6,130 positions; the first numeric differences are row 0 token indices 69, 70 and 71, with token IDs 51864, 29772 and 220. Across the 4,088 answer tokens, same-position absolute differences have maximum `327.775e-03`, mean `23.912e-03`, and RMS `38.380e-03`. Token extraction is excluded: the divergence is already present in ArcticInference 0.1.3 with vLLM 0.26.0 model-runner logits before prompt-logprob bookkeeping. Repeating the same raw sampling request from the same checkpoint and the same 33 rows preserves exact alignment across all 6,130 positions, but the 20 recorded first-differing sampled logprobs are not bitwise repeatable; their largest run-to-run shift is `74.000e-03` at token index 85. The sampling model execution therefore contributes a run-to-run nondeterministic component before bookkeeping. In an off-spec diagnostic, reloading that saved tree through Hugging Face reads 4.299018, only `0.314e-03` from training and inside the gate; eager vLLM narrows the serving difference to `1.341e-03` but still fails. Explicit `FLASH_ATTN` reads `1.476e-03`, and row-wise serving reads `3.162e-03`, so CUDA graphs contribute to the disagreement but attention-backend selection and request batching do not explain the residual. The Blackwell branch of the provisioning script installs `quack-kernels==0.6.1`, which defines `sub_packed_f32x2`, then installs `nvidia-cutlass-dsl==4.8.0` and its libs with `--no-deps`. `quack-kernels` 0.6.1 depends on `nvidia-cutlass-dsl==4.6.0`, and neither 4.6.0 nor 4.6.1 exports `cutlass.cute.FastDivmodDivisorV2`, which `flash-attn-4` 4.0.0b33 imports. `flash-attn-4[cu13]` resolves CUTLASS 4.8.0 and, without `--no-deps`, replaces quack with 0.6.5, which does not define `sub_packed_f32x2`. An H200 run does not enter this branch.

### `qwen3.6-35b-a3b/h200/train-sft-8gpus-64k` `checkpoint-resume-loss`

```bash
python -m arctic_platform.correctness run --config arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-sft-8gpus-64k.config --test checkpoint-resume-loss
```

Both cases pass at eight and sixteen GPUs after checkpoint files are merged into the resumed job directory on every node. The complete four-case run took 13m07.5s at eight GPUs and 15m48.3s at sixteen GPUs. The checkpoint directory is node-local: the head node's saved tree holds `mp_rank_00_model_states.pt`, expert shards `experts_ep_rank_000` through `003`, and optimizer shards `bf16_zero_pp_rank_0` through `_7`, and the second node holds lock files and no tree. Hold and place run on every host in `DSS_GATEWAY_HOSTFILE`, and a file present on only one node is copied onto the others before load.

### `qwen3.8-27b/h200/train-sft-8gpus-2k` `single-step-grads`

`qwen3.8-27b/h200/train-sft-8gpus-2k` passes `single-step-grads` at eight and at sixteen GPUs against the spec gate of 1.000e-02. Sixteen H200s measure `norm.weight` at 3.418e-03 in `gas1` and 8.116e-03 in `gas4`, 56 of 56 tensors inside the gate in both cases. Eight H200s measure 2.929e-03 on `norm.weight` in `gas1` and 2.535e-03 on `lm_head.weight` in `gas4`.

```bash
python -m arctic_platform.correctness run --config arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config --test single-step-grads
```

The eight-GPU figures are reproducible rather than draw-dependent: three independent onboarding sessions measured 2.929e-03 on `norm.weight` in `gas1` and 2.535e-03 on `lm_head.weight` in `gas4` to four significant figures, the third with the seeded, best-effort-deterministic execution. The gate moves independently of them. Two sessions drew gates of 1.000e-03 and 2.000e-03 from raw same-tensor variations of 0.486e-03 and 0.576e-03, which those two tensors exceeded; the calibrated 2x upward round of that variation in 1e-3 units is 3.000e-03, and the selected gate is 1.000e-02. Loss agrees to 0.000209e-03 in `gas1` and 0.000149e-03 in `gas4`, and the median per-tensor ratio is 1.000004 and 0.999985, so the reduction agrees, and those two tensors sit inside the selected 1.000e-02 gate. The spec is sized at the minimum representative prefix, four of sixty-four layers for this architecture, peaking at 15.903 GiB.

`fused_lm_head_token_chunk_size: 8192` is set by these two configs and by no other, and the widest disagreement in each case lands on `lm_head.weight` or the final `norm.weight` immediately upstream of it. The correlation is confounded with the model, since they are also the only Qwen3.8-27B configs.

Removing the chunked head does not remove the disagreement, which excludes it as the cause. `python -m arctic_platform.correctness.diagnostics.lm_head_chunking --config arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config --test single-step-grads` drops `fused_lm_head_token_chunk_size` from the executed training sub-job after the checksum gate, which also moves the reference from the tiled log-probability path to ordinary chunked cross entropy. A plain regression run in the same environment is the control:

| width | case | chunked head (control) | chunked head removed |
| --- | --- | --- | --- |
| 8 | `gas1` | 2.929e-03 `norm.weight`, pass | 5.371e-03 `norm.weight`, fail |
| 8 | `gas4` | 2.535e-03 `lm_head.weight`, pass | 10.250e-03 `norm.weight`, fail |
| 16 | `gas1` | 3.418e-03 `norm.weight`, pass | 6.347e-03 `norm.weight`, fail |
| 16 | `gas4` | 8.116e-03 `norm.weight`, pass | 2.081e-03 `lm_head.weight`, pass |

Losses agree to within 0.597e-06 in all eight runs. In the three failing runs without the chunked head, the Arctic Platform `norm.weight` gradient norm is 3.12524 while the reference reads 3.13061, 3.13549 and 3.13159.

### TODO

Configs move to the `cortex-training` repository. Onboarding then follows a different route, and this harness stops carrying its own copy of the configs:

1. Recommended configs are added to `cortex-training` and merged there first. That step is owned by `cortex-training`, not by this code base.
2. Correctness testing then runs against an updated `cortex-training` clone, scanning it for new and changed configs and onboarding those.

Return `checkpoint-resume-loss` to the hosted transport. `requires_hosted_control_plane` stays false. A hosted `gas1` on `qwen3.6-35b-a3b/h200/train-sft-8gpus-64k` at eight GPUs completes worker init and compares losses, with no deterministic-backward refusal in the log. The two runs disagree at iteration 10, before the checkpoint is written: 20.2307 against 17.3527, a difference of 2.878e+00 against the 1.553e+00 gate. Set `requires_hosted_control_plane=True` in `checks/checkpoint_resume.py` and assert it again in `selftest/check_fixed_tolerance_onboarding.py` once a hosted run agrees at the checkpoint iteration.


### Layout

- `checks/` contains registered correctness checks.
- `configs/` contains operator-authored training inputs.
- `diagnostics/` contains standalone investigation programs and is not part of onboarding or regression.
- `harness/` contains config intake, case generation, Arctic Platform execution, parameter alignment, verdicts, frozen-spec types, and report rendering.
- `onboarding/` contains automatic reference reduction, model materialization, repeatability calibration, final validation, and reviewed-spec publication.
- `reference/` contains the single-GPU Hugging Face engine.
- `specs/` contains the onboarding-generated, checked-in regression contract for each config: its config checksum, reduced-reference identity, generated cases, attention backend, parameter mapping, and frozen gates.
