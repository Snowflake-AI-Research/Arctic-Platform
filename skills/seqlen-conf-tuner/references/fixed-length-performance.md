# Fixed-Length Performance Tuning

**Apply the guide's fixed-target refinement method.** The [long-seqlen tuning guide](../../../docs/long-seqlen-tuning-guide.md#refine-a-fitting-target-configuration) owns candidate-selection heuristics, history reuse, OOM pruning, coupled topology/memory moves and timing defaults. Do not maintain a second search algorithm here. Keep model weights, precision, loss, optimizer semantics and target workload fixed; rank warm forward/backward time without optimizer `/step` time and report complete-update tokens/sec separately. Maximum-capacity timings at another length are not a fixed-target baseline.

## TODO For The Next Performance Sweep

- [ ] Check that the sweep driver ranks candidates by median warm forward/backward seconds, not forward/backward plus optimizer time; update the driver before using an older complete-update scorer.
- [ ] Reuse compatible warm measurements; for an unmeasured candidate run two total updates and score only update 2. Never include update 1; add samples only for an unresolved tie or suspect timing.
- [ ] Use the guide and existing pass/OOM history to select a tight local candidate, allowing a coupled SP/memory-control change when justified. Keep GBS=1 and validate active-rank topology before launching.
- [ ] Qualify and export only the best measured config; require a fresh two-update verification only if existing evidence is insufficient. Retain optimizer timings, complete-update throughput and memory evidence as separate reporting fields.

## Workload And Timing

**Define what the target length means before comparing candidates.** An individual sequence, total request tokens and per-data-parallel-group packed tokens are different quantities. A data-parallel (DP) group receives a share of request rows; sequence parallelism (SP) splits each group's packed call across its ranks. At fixed world size, `dp=world/sp`. Decreasing SP while filling each group to the same packed budget increases global tokens per update. Such step times are not equal-work comparisons.

**Keep GBS=1 and send the full target workload once globally.** Preserve source-row length/count, packed-call occupancy, loss boundaries and global tokens across candidates. Do not multiply target tokens by DP or halve each group's context just to populate another group. Follow the guide's engine batch constraints: a topology requiring engine GBS>1 is not eligible unless the user changes the batch requirement. For an individual-sequence target, never replace the sequence with shorter packed rows.

**Keep older DP-filled and fixed-global evidence qualified.** Historical trials that filled every DP group's budget changed global work when SP changed. Other historical trials held global tokens fixed but shortened the per-group context. Neither can be silently adopted as an identical GBS1 full-target workload. Retain their shapes and original objectives; compatible measurements can still inform which memory or communication mechanism is worth investigating.

**Check the scope of reported packing metrics.** Current DSS aggregation retains `sft_model_calls` and `sft_packed_rows` from worker 0, not every worker. Verify the DP split through the actual dispatcher and its tests, then use those metrics as a runtime worker-0 check. Do not claim that a worker-0 count observes all DP groups.

**Exclude update 1 and all optimizer time from ranking.** Follow the guide's two-update default: one cold update and one warm ranking sample, or reuse compatible saved warm samples. Execute `/step` outside the forward/backward timer to validate updates and reset gradients normally. With several valid warm samples, rank their median; report sample count and range, and do not infer variability from a single sample. HTTP-client timing includes endpoint overhead; keep the same method across candidates. Follow the [shared normalization rules](../SKILL.md#deliver) for descriptive complete-update tokens/sec.

**Include TFLOPS when a supported existing estimator is available.** Follow the [shared throughput, timing and FLOPS reporting requirements](../SKILL.md#deliver); report estimator scope and availability in the candidate table and final result. Estimated TFLOPS supplements normalized complete-step tokens/sec and latency; it does not replace the requested ranking metric.

## Execution And Delivery

**Select candidates from the guide, not an automatic grid.** Reuse the trial table and exact failure evidence before choosing the next local move. Explain the changed knobs, supporting evidence, expected benefit and memory risks before launching. A runner should execute those explicitly selected trials and return evidence for the next decision.

**Qualify the measured configuration before export.** Require finite losses and gradient norms, successful complete updates and advancing counters, exact target occupancy, zero padding, GBS1 and verified cleanup on every used node. Capture total NVML device used/total/free, allocator peaks and aggregate host/cgroup usage. Non-capacity failures are not OOM bounds. Require fresh verification only when existing compatible evidence cannot qualify the winner.

**Preserve historical outcomes and export the measured controls without debug configuration.** Follow the [shared export requirements](../SKILL.md#deliver), including a root-level hyphen-separated filename with `<count>gpus`, unchanged transport envelope, exact buffered effective-payload equality before debug removal, recorded export omissions and runtime dependencies. Include the full absolute output path in the report TLDR and final response. If an explicitly authorized candidate uses fewer active GPUs than allocated, report both counts and do not present its timing as using all allocated devices.

## Diagnostics

**Diagnostic timings are not final performance timings.** Reuse existing `DSS_MEM_LOG`, `DSS_MEM_BREAKDOWN`, `DSS_MEM_STAGED`, `training_memory_telemetry` and `step_peak_memory_log` controls for diagnosis. Disable expensive synchronized staged/breakdown/telemetry probes and allocator snapshots during clean timing comparisons. Keep the cheap per-step peak response and the same NVML sampling interval for all candidates. Record which diagnostics were enabled. Do not invent new memory variables or counters.

**Unset presence-gated debug flags instead of setting them to `"0"`.** DSS's debug memory helpers check for a nonempty environment value; `"0"` therefore enables those helpers. Remove the variables from the clean timing environment and explicitly disable structured `training_memory_telemetry`. Verify the exact imported implementation because other controls may parse boolean strings differently.

**Do not spend the safety buffer to improve timing.** Follow the guide's [GPU safety margin](../../../docs/long-seqlen-tuning-guide.md#leave-a-gpu-safety-margin): hold the current 20 GiB default fixed across baseline, candidates and verification, and reject no-buffer evidence as buffered qualification. Increasing GPU residency may remove offload copies within that constrained budget; qualify the move through complete updates and inspect warm allocator cache-flush warnings before selection. Report measured memory including the buffer, allocator cache/slack and host pressure separately.

**A completed forward/backward call is not a passing performance candidate.** The first optimizer update may allocate host state and cross Ray's node-memory threshold. Record the failed phase and any completed call duration, but leave complete-step timing unavailable when no optimizer update succeeds. An observed low GPU peak before a host OOM does not predict the GPU peak of an uncompleted backward or optimizer phase.

**Resume completed-but-unrecorded candidates without rewriting evidence.** A sweep can exit after saving `result.json` but before saving its progress entry, especially when cleanup verification fails. Validate the saved candidate envelope and effective payload, raw source SHA256, runtime identity, GPU/SP/EP degrees, target budget and request shape against the resumed candidate before adopting the result. Require an explicit OOM for capacity failures. Recover only the exact owned zone and obtain fresh process-exit and idle-device acknowledgments on every node. Save that acceptance separately with the original result hash; preserve the failed cleanup result and recovery log. Missing or mismatched artifacts are not permission to overwrite or rerun a known OOM.

**Keep backend correctness limitations visible.** Current Flash-Next QSA does not isolate packed rows. Its short-row packed stress tests can compare implementation performance with controlled shape, but cannot establish independent-sample correctness or expected short-row attention scaling. Do not change the attention algorithm, precision or pretrained weights merely to win a timing comparison.

**Check vocabulary-sharded lookup with rank-distinct IDs.** The [Flash-Next lessons](flash-next.md) record a demonstrated PLE embedding error under distinct EP inputs. Finite training metrics do not certify model correctness. A backend repair changes runtime identity and needs fresh timings rather than reuse of pre-repair results.

**Measure topology rather than inferring it from SP degree.** With world16, EP8 and eight GPUs per node, current `auto` selection uses contiguous decoupled SP groups for degrees 16, 8 and 4. SP2 instead selects node-local EP with cross-node SP pairs `[0,8]` through `[7,15]`; SP1 uses standard EP without SP collectives. Thus stepping from SP4 to SP2 does not necessarily reduce network communication. Verify the imported topology helpers and startup group logs.
