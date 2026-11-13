# Correctness Code Map

**This checkout registers five correctness checks without adding production hooks or debug dependencies.** `single-step-grads`, `single-step-optimizer`, `checkpoint-resume-loss`, `inference-checkpoint-loss` and `e2e-sft-train-validate` are the registered IDs. The gradient and optimizer checks retain their reduced supervised-reference workflow. Checkpoint resume, training-to-inference checkpoint loss, and end-to-end supervised fine-tuning (SFT) retain their separate workload identities and target prerequisites. `rl-weight-sync` and `rl-router-replay` and their reinforcement-learning (RL) check modules are excluded.

Start with [the operator guide](running-tests.md). All paths below are relative to `arctic_platform/correctness/`.

| Area | Responsibility |
| --- | --- |
| `__main__.py`, `api.py` | Operator CLI, external config discovery, selection, and shared Python run interface. |
| `checks/fwd_bwd.py` | `single-step-grads` comparison. |
| `checks/fwd_bwd_step.py`, `checks/optimizer_state.py` | `single-step-optimizer` and optimizer artifact/norm validation. |
| `checks/checkpoint_resume.py` | `checkpoint-resume-loss`: uninterrupted versus resumed training, saved checkpoint identity and restored-state gates. |
| `checks/inference_checkpoint_loss.py` | `inference-checkpoint-loss`: answer-token loss from training and sampling the saved checkpoint, export schema coverage and owned teardown. |
| `checks/e2e_sft_train_validate.py` | `e2e-sft-train-validate`: the same ordered SFT inputs in HF and Arctic Platform; compact reference identity and final training/validation verdicts. |
| `harness/config.py`, `harness/sft.py` | Reviewed specs, placement arms, reproducible synthetic inputs, and configured training data. |
| `harness/runner.py`, `harness/execution.py` | Prepared-reference regression, sequencing, and CLI/report dispatch. |
| `harness/gateway.py` | Supervised training lifecycle, payloads, save/catalog identity helpers, gateway ownership, and cleanup. |
| `harness/cortex.py` | Reviewed fixture admission, training transport contracts, and explicitly requested microbatch packing. |
| `harness/inference.py`, `harness/optimizer.py` | Checkpoint export/decoding, native first-two optimizer capture, tensor-name alignment and complete participation coverage. First-two capture is not required by later-check training. |
| `harness/lora.py` | Upstream PEFT reference adapters, content verification, and correctness-owned local target loading of the same adapter. Hosted shared-adapter loading is not admitted. No deterministic recipes or SAVE0 execution. |
| `harness/storage.py`, `harness/report.py` | Compact reference records, checkpoint ownership, fast scratch containment, and reports. |
| `reference/` | Independent single-GPU reference, one-step AdamW, model features, process determinism, and Qwen4 loading. |
| `onboarding/` | Explicit sizing, calibration, fixture publication, and reference preparation. |
| `selftest/library.py`, `selftest/check_suite.py` | Synthetic CPU coverage in one library, collected by one driver with each section's original fixture boundaries. No bundled real recipe dependencies. |
| `diagnostics/` | `lm_head_chunking` and its stdlib dispatcher; altered-workload diagnostic only. |

**Configs, specs, and compact records are operator inputs, not bundled assets.** Explicit single-config runs accept `--test-spec`; multi-config runs use locally generated matching default specs. Heavyweight scratch and model assets remain under `/data-fast`.

**Checkpoint helpers support the optimizer, resume and inference checks.** Later-check selftests add focused checkpoint identity/restoration, exporter schema/teardown, and SFT trajectory/reference-record controls to the retained baseline; retired debug campaigns are not restored.

**No production change or added debug package is required by this extraction.** The correctness-owned gateway collects participation for the first two comparisons only. Later-only selection does not acquire the first-two BF16 or runtime-capture requirement. An installed target runtime, compatible model assets and check-specific execution prerequisites remain necessary.
