# Essential Onboarding Debug Instrumentation

These are the essentials needed for debugging most config onboardings: ZeRO initialization allocation tracing and observation of the tensors handed to the native weight sender. Both are opt-in. With the options absent, native arguments, results and tensor lists are unchanged.

## Initialization Traces

Set the DeepSpeed configuration option `arctic_platform_zero_initialization_trace` to an object with `request_id` and `output_dir`. The request ID must contain only letters, digits, underscores or hyphens, and the evidence directory must resolve under `/data-fast`. The initialization context removes this private option from the configuration sent to DeepSpeed without mutating the original recipe.

The trace records the installed ZeRO optimizer's source SHA256, rank, process and allocation events. Its request-scoped observer does not replace `torch.zeros` globally, and it restores the optimizer method after success or failure. A trace helps locate allocation stalls or failures; it is not a capacity guarantee or numerical result.

## Weight Sender Observation

Set `ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY=1` in the training worker's environment to record exact SHA256 manifests of the tensors passed to its existing `WeightSender.send`. The observer hashes bounded chunks of logical tensor bytes before and after the single native send and reports a mutation if those bytes change. It also hashes and rechecks the source files of the selected sender and export method.

The send result gains `sent_manifest`, `trainer_parameters` and `evidence_sources`. No extra send, receiver replay, sampler replacement or collective is introduced. This sender-only evidence does not prove receiver correctness. Enabled observation rejects ZeRO-3 post-gather views and expert-parallel parameter holdings because a single sender cannot prove their complete inventory. Hashing adds overhead and is intended for diagnosis, not ordinary training.

## Scope

Config-specific LoRA admission, reduced-MTP fixtures, model-capacity projections, router replay, participation capture, receiver shadow replay and custom inference startup or warmup diagnostics are excluded. The retained libraries import training dependencies only inside enabled implementations and work without the correctness harness.

## Validation

CPU tests cover disabled import and identity behavior, native call order, exact manifests, mutation detection and trace restoration. GPU-hidden full-suite results are reported separately in the PR description; these controls do not claim a numerical onboarding result or distributed receiver verification.
