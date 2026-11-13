# Correctness Test Details

**The suite contains seven tests with different comparisons and pass criteria.** This guide explains what each test measures, what a pass establishes and how to interpret a failure. Use [Getting started](getting-started.md) for onboarding and [Running tests](running-tests.md) for execution commands and runtime prerequisites.

**A result applies only to the recorded workload and placement.** A reduced model fixture does not validate the full pretrained model, and short sequences do not validate the recipe's maximum context length. Gradient accumulation steps (GAS) describe how many microbatches contribute to an optimizer step; the first two tests compare separate GAS1 and GAS4 arms. Do not assume every later test has those two arms. Data parallelism distributes examples, sequence parallelism distributes sequence computation, tensor parallelism partitions tensors and expert parallelism distributes mixture-of-experts (MoE) experts.

**Missing prerequisites and incomplete evidence are not numerical passes.** Onboarding attempts applicable tests independently and prepares their required reference records; one test's ordinary failure does not block another test. Unsafe cleanup stops subsequent workloads. Regression reads compatible prepared records and reviewed gates rather than generating missing references or widening tolerances. A gate below is an absolute tolerance, not a percentage error.

## 1. Single-Step Gradients (`single-step-grads`)

**Compare each parameter's gradient L2 norm after one forward-backward operation against an independent single-GPU Hugging Face reference.**

### What Runs

Both engines start from the same prepared model or low-rank adaptation (LoRA) adapter weights and consume identical materialized synthetic inputs. The L2 norm is the Euclidean magnitude of a parameter's gradient tensor. The GAS1 and GAS4 arms deliberately contain padding and unequal sequence lengths. The target packs the request into microbatches according to its token budget; the Hugging Face (HF) reference performs its independently implemented token-weighted backward computation.

The first-two workflow requires BFloat16 (BF16) model weights and verified 32-bit floating-point (FP32) gradient accumulation. An omitted accumulation dtype must be authenticated against the installed target runtime; onboarding does not silently rewrite the recipe to FP32. These precision requirements do not automatically apply to later-only selections.

### What Decides The Result

Named gradient norms must align completely, be finite and nonnegative, and have nonempty active parameter coverage. Where participation records are present, both engines must agree about active and inactive parameters before comparing active norms. A missing gradient is not silently replaced with zero.

Each aligned norm must differ by no more than the config-specific prepared gate. Gradient calibration measures repeatability across independent nondeterministic reference runs; the current calibration uses four times the widest same-parameter norm range, rounded upward in `0.001` units with a `0.001` minimum. The recorded gate, not a newly measured or enlarged value, controls regression. Loss difference is reported as a diagnostic; the gradient-norm comparison determines the numerical verdict.

### Nuances And Failure Interpretation

- A broadly shared target/reference norm ratio suggests loss-denominator or accumulation scaling errors. Token counts must be weighted over the whole request; padding and unequal row lengths expose per-row, per-rank or per-microbatch normalization mistakes.
- A mismatch concentrated in particular parameters suggests a localized computation or reduction difference. The median norm ratio, worst parameter and actual model-call count help distinguish localized errors from global scaling.
- L2 norms measure magnitude, not direction. Equal norms do not prove elementwise gradient equality and can conceal sign changes or permutations.
- A GAS1 pass does not establish GAS4 correctness. Compare both arms, and keep each GPU placement's reference identity separate.

Implementation: [gradient comparison](../checks/fwd_bwd.py), [synthetic inputs and loss weighting](../harness/sft.py), [gradient calibration](../onboarding/calibration.py).

## 2. Single-Step Optimizer (`single-step-optimizer`)

**Compare Adam's first-moment, second-moment and parameter-update-residual L2 norms after one optimizer step against the independent Hugging Face reference.**

### What Runs

The target runs the forward-backward and optimizer step for each selected GAS arm and captures named optimizer artifacts. The first-step comparison uses the test's fixed learning rate of `0.01`, not the recipe's learning rate. When the first two tests are selected together, they can inspect the same execution; optimizer-only selection still performs its own required backward and step. Successful gradient onboarding is not a prerequisite for attempting optimizer onboarding.

For each active parameter, the comparison reduces three tensors to FP32 L2 norms: Adam's first moment (`exp_avg`), second moment (`exp_avg_sq`) and the parameter-update residual. Parameter names, shapes, participation and artifact step identity must match.

### Why Compare An Update Residual

The first Adam update is nearly a sign operation. A small gradient difference near zero can flip the update direction and produce a raw update difference of roughly one learning-rate unit even when both moment comparisons are acceptable.

The residual subtracts the Adam update implied by each engine's own moments from that engine's measured parameter update. Comparing the moments separately then tests update behavior not already explained by those moments. The calculation reproduces each backend's scalar rounding and operation order, including DeepSpeed's FP32 kernel arguments and bias corrections, rather than using one scalar convention for both engines.

### What Decides The Result

Every named quantity must have an absolute FP32 norm difference within the prepared reference record's reviewed `optimizer_norm_gate`. The record must identify the criterion and contain nonempty gate provenance. Zero and near-zero norms receive no exemptions. Missing artifacts, incomplete parameter coverage or an incompatible gate record cannot pass.

Fresh optimizer calibration uses eight independent nondeterministic single-GPU HF repeats per arm, alongside the deterministic baseline. The current computed gate is four times the largest pairwise scalar-norm difference, rounded upward in `0.001` units with a `0.001` minimum. The record retains the actual calibration and any explicitly reviewed acceptance gate; regression neither recalibrates nor widens that gate.

### Nuances And Failure Interpretation

- Moment mismatches point toward gradient, accumulation or state-update differences; a residual mismatch with acceptable moments points toward update arithmetic or its capture. These are diagnostic directions, not proof of a particular defect.
- Loss agreement does not replace optimizer-state evidence. The report retains loss deltas, but the three norm comparisons decide the result.
- Equal L2 norms can hide sign, permutation and directional errors. This is a scalar-norm test, not an elementwise optimizer equivalence claim.
- A changed criterion, artifact schema, fixture or GPU placement requires compatible records. Do not reuse a historical numerical pass against a different identity.

Implementation: [optimizer verdict](../checks/fwd_bwd_step.py), [artifact decoding and residual calculation](../checks/optimizer_state.py).

## 3. Checkpoint Resume Loss (`checkpoint-resume-loss`)

**Compare uninterrupted and checkpoint-resumed Arctic Platform training losses.**

### What Runs

Both trajectories use Arctic Platform; there is no Hugging Face reference. For each selected arm, one job trains for 20 iterations without interruption. A second job consumes the identical first ten batches and saves a resumable checkpoint. A fresh third job loads that checkpoint and consumes batches 11 through 20. Both trajectories use the test's fixed learning rate of `0.001`, not the recipe's learning rate.

Every iteration has a distinct seeded batch, materialized into identical request bytes for both trajectories. The resumed job must continue the data order rather than replay the first ten batches. The checkpoint must restore weights, optimizer moments and scheduler state; a weights-only inference export is insufficient.

### What Decides The Result

The losses at iterations 10 and 20 must agree within the stored config-specific absolute gate. Onboarding derives that gate from three identical ten-step target trajectories with determinism disabled: twice the widest same-iteration loss range, rounded upward in `0.001` units with a `0.001` minimum.

### Nuances And Failure Interpretation

- Iteration 10 is the pre-checkpoint control. If the trajectories already disagree there, the iteration-20 mismatch cannot be attributed specifically to restore: the jobs either received different work or accumulated execution nondeterminism before saving.
- An iteration-20 mismatch after an acceptable iteration-10 control directs investigation toward checkpoint contents, restore behavior, resumed data order or subsequent execution drift.
- Requesting determinism does not guarantee every installed attention kernel is deterministic. Read the stored calibration and determinism settings; a large calibrated gate makes the test less sensitive to a small restore defect.
- Hosted and local transports can restore differently: create-time checkpoint initialization and runtime load must each preserve resumable state. Successfully writing a directory is not proof that a fresh job restored it.
- The verdict compares two recorded losses, not every checkpoint tensor or every intermediate iteration. Loss agreement alone does not establish byte-identical restored optimizer state.

Implementation: [trajectories, control and calibration](../checks/checkpoint_resume.py), [training transport](../harness/cortex.py).

## 4. Inference Checkpoint Loss (`inference-checkpoint-loss`)

**Compare answer-token cross entropy from the training engine with the sampling engine serving that training job's saved checkpoint.**

### What Runs

The training job performs ten GSM8K supervised training steps at the configured learning rate, then scores held-out examples through its forward-only route in evaluation mode. The harness saves a weights-only checkpoint and closes the training job. A sampling job initialized from that checkpoint scores the same tokenized examples.

This test has one comparison rather than GAS1/GAS4 verdicts. It compares the trained weights across engines, not a backward accumulation calculation. The reference is the live training engine, not an independent Hugging Face run.

### What Decides The Result

Both engines must score the same answer-token count. The harness computes cross entropy as one token-weighted mean over those answer tokens, excluding prompts and padding. The absolute gate is fixed rather than calibrated: `0.001` by default and `0.003` for B200.

### Nuances And Failure Interpretation

- Label and logit alignment are part of the test. The forward-only request uses explicitly logit-aligned labels; the answer boundary comes from the common prefix of prompt-only and joined prompt/answer tokenizations. The sampler omits one final logit position, which must not shift the scored answer span.
- Sampling requests use `prompt_logprobs=0` to score the supplied token. A positive top-k request can expose a different token's probability and invalidate the comparison.
- The validation slice must fit the padded token budget and data-parallel layout. A successful run does not imply that every held-out example or the model's maximum sequence length was tested.
- Serving requires a complete compatible export: weights, architecture/config, tokenizer and required processor metadata. For a text-only training view of a multimodal source, source-layout restoration must retain trained tensors and preserve source-only tensors. Export success does not establish sampler compatibility.
- LoRA execution requires supported export and adapter loading; silently omitting an adapter is not an acceptable substitute.
- The on-prem route also reloads the checkpoint into a training engine and prints live-training, reloaded-training and sampling losses. Those extra losses help distinguish serialization/reload disagreement from sampling disagreement; they do not replace the main verdict.

Implementation: [checkpoint scoring and export validation](../checks/inference_checkpoint_loss.py), [answer-token selection and cross entropy](../harness/sft.py).

## 5. End-To-End SFT Training And Validation (`e2e-sft-train-validate`)

**Replay 100 supervised fine-tuning steps in Arctic Platform and an independent single-GPU Hugging Face reference; compare final training loss and held-out validation loss.**

### What Runs

Supervised fine-tuning (SFT) uses ordered GSM8K training examples with a separate held-out validation slice. Both engines read the same materialized batch files, start from compatible initial weights or the same deterministic LoRA adapter, and use the recipe's optimizer settings and learning rate. Each training row is consumed once in the 100-step replay.

The reference record binds the complete replay, initial model/adapter identity, optimizer, tokenizer and execution options. A first-two reduced-fixture record cannot substitute for this trajectory record. Full-model execution needs its separately admitted snapshot and enough memory for the single-GPU reference's weights and optimizer state.

### What Decides The Result

Only training loss at step 100 and held-out answer-token validation cross entropy are gated, each at a fixed absolute `0.03`. All 100 per-step training losses and their differences are retained as diagnostics. An intermediate step exceeding `0.03` does not, by itself, fail the test.

### Nuances And Failure Interpretation

- Read the entire trajectory even after a pass. The final-step difference can be smaller than earlier disagreements, so a final pass does not establish agreement throughout training.
- A step-one mismatch exists before the first optimizer update; inspect the forward computation, loaded weights, label alignment and loss normalization. Agreement at step one followed by later divergence directs investigation toward optimizer updates, gradient reduction or nondeterministic execution, but does not prove the cause.
- The held-out score covers answer tokens only, using a budget-fitting slice. The reference and target must report matching scored-token counts.
- This is SFT, not RL. A mixed training/sampling RL recipe does not become eligible by generating a supervised reference.
- Lack of memory or a missing compatible trajectory record is a prerequisite failure, not a numerical mismatch. Do not shrink a claimed full-model workload or widen the fixed gate to turn such a failure into a pass.

Implementation: [replay preparation, trajectory diagnostics and verdict](../checks/e2e_sft_train_validate.py), [HF reference identity rules](hf-reference-onboarding.md).

## 6. RL Weight Synchronization (`rl-weight-sync`)

**After one reinforcement-learning optimizer step, verify exact tensor receipt on every sampling rank, plus installed full-weight values or LoRA destination bindings.**

### What Runs

Reinforcement learning (RL) uses a native training/sampling job pair. One RL backward and optimizer step at the configured learning rate moves the trainer away from the initial checkpoint, then the normal synchronization operation transfers the updated weights.

Opt-in observers record the trainer's collective tensor inventory, each sampler rank's received tensors and actual destination-write evidence. Transfer records include names, shapes, dtypes, element counts and SHA-256 digests. For full-weight synchronization, destination verification independently replays loading into NaN-filled shadow storage and compares completed shadow values with the live installed values. LoRA has the separate installation-binding comparison described below.

### What Decides The Result

The trainer must account for its complete applicable parameter inventory. Every expected sampling rank must supply complete receive and installation evidence. Received tensor metadata and bytes must match the sent tensors exactly. For full weights, dropped tensors, missing destinations, unwritten elements, unequal values, unverifiable destinations or later mutation fail the comparison. LoRA factors must be received unchanged and bound to installed LoRA-manager destinations. There is no floating-point tolerance gate.

### Nuances And Failure Interpretation

- Sender checksums alone are insufficient: a correct transfer can still be incorrectly mapped, sliced or ignored by the receiver. Actual destination evidence is required.
- Eight-bit floating-point (FP8) destination verification independently replays the supported quantizing updater, including fused tensors assembled across multiple sends. Comparing only pre-quantization tensors does not establish the installed FP8 state.
- A frozen table deliberately excluded from synchronization must still match the trainer's checkpoint-backed copy. Sharded tables require complete per-shard identity evidence; skip-sync is not permission to omit coverage.
- LoRA synchronization transfers trainable adapter factors, not the full frozen base model. Each broadcast-group rank must prove receipt and binding to an installed LoRA-manager factor, including packed-module slots. Byte equality applies before the builder's casts and scaling; the current LoRA evidence does not independently shadow-compare the transformed installed values or validate the full frozen base model.
- This test verifies one transfer and installation, not subsequent sampler loss, generation quality or maximum-context training. Missing supported observers cannot be counted as a pass.

Implementation: [inventory and receive/destination verdict](../checks/rl_weight_sync.py), [opt-in synchronization observers](../../debug/weight_sync.py).

## 7. RL Router Replay (`rl-router-replay`)

**Verify that the expert indices captured during sampling are exactly the indices every expected training MoE router uses during forward and required backward recomputation.**

### What Runs

The sampler generates answers to GSM8K prompts with routing capture and replay enabled. Sample/replay identities bind each generated token sequence to the routing the sampler returned. The trainer performs one RL forward-backward operation over those sequences.

Opt-in request-scoped traces record routing received from the replay cache, packed model-call tokens and positions, and router invocations on every actual training rank. The expected router layers come from a model-bound inventory, not merely the routers that happened to produce traces.

### What Decides The Result

Three comparisons must be exact: sampler routing versus trainer-received routing; sample routing versus each packed model call's routing; and model-call routing versus the expert indices used by every expected router invocation. Missing sample coverage, ranks, calls or router evidence fails the test. There is no numerical tolerance.

### Nuances And Failure Interpretation

- Sequence-parallel ranks receive only their captured-token window, not the whole sample. Comparison must attribute the correct token positions and layer slices rather than concatenate unrelated windows.
- A packed call containing only dispatch padding has no attributable sample routing and is handled separately. A call containing real sample tokens without replay routing is a failure.
- The sampler captures no routing for the sequence's final token: `capture_len` is one less than the generated row length. The uncaptured position is not evidence of a routing mismatch.
- With full activation checkpointing at frequency one, every expected router must also show a backward recomputation invocation. A forward-only match cannot establish recomputation replay.
- Dense decoder layers have no MoE router and are not expected to emit router invocations. Conversely, an unobserved expected MoE router cannot disappear from coverage just because its trace is missing.
- A router that recomputes its own expert choice instead of consuming replayed indices fails, even if its independent choice happens to match. Matching training loss alone does not establish replay correctness.
- This test verifies replay for the captured workload; it does not establish that the sampler's original expert choices are optimal or that a different uncaptured sequence length works.

Implementation: [routing attribution and exact verdict](../checks/rl_router_replay.py), [trace collection and bound inventory](../harness/inference.py), [opt-in router observers](../../debug/training.py).
