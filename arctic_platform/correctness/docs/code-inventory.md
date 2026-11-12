# Correctness Code Map

**The maintained library contains only the first two checks and their preparation paths.** Start with [the operator guide](running-tests.md). All paths below are relative to `arctic_platform/correctness/`.

| Area | Responsibility |
| --- | --- |
| `__main__.py`, `api.py` | Operator CLI, external config discovery, selection, and shared Python run interface. |
| `checks/fwd_bwd.py` | `single-step-grads` comparison. |
| `checks/fwd_bwd_step.py`, `checks/optimizer_state.py` | `single-step-optimizer` and optimizer artifact/norm validation. |
| `harness/config.py`, `harness/sft.py` | Reviewed specs, placement arms, reproducible synthetic inputs, and configured training data. |
| `harness/runner.py`, `harness/execution.py` | Prepared-reference regression, sequencing, and CLI/report dispatch. |
| `harness/gateway.py` | Supervised training lifecycle, payloads, save/catalog identity helpers, gateway ownership, and cleanup. |
| `harness/cortex.py` | Reviewed fixture admission, training transport contracts, and explicitly requested microbatch packing. |
| `harness/inference.py`, `harness/optimizer.py` | Test2 checkpoint export, safe decoding, native optimizer capture, tensor-name alignment, and complete participation coverage. No inference check is registered. |
| `harness/lora.py` | Upstream PEFT reference adapters, content verification, and retained native loading. Internal Ray loading is not an admitted Cortex-training target route. No deterministic recipes or SAVE0 execution. |
| `harness/storage.py`, `harness/report.py` | Compact reference records, checkpoint ownership, fast scratch containment, and reports. |
| `reference/` | Independent single-GPU reference, one-step AdamW, model features, process determinism, and Qwen4 loading. |
| `onboarding/` | Explicit sizing, calibration, fixture publication, and reference preparation. |
| `selftest/library.py`, `selftest/check_suite.py` | Synthetic CPU coverage in one library, collected by one driver with each section's original fixture boundaries. No bundled real recipe dependencies. |
| `../debug/lora.py`, `../debug/weight_sync.py`, `../debug/inference.py`, `../debug/training.py`, `../debug/deepspeed_zero_initialization.py` | Opt-in debug integrations grouped by responsibility; disabled integrations preserve ordinary production behavior. |
| `diagnostics/` | `lm_head_chunking` and its stdlib dispatcher; altered-workload diagnostic only. |

**Configs, specs, and compact records are operator inputs, not bundled assets.** Explicit single-config runs accept `--test-spec`; multi-config runs use locally generated matching default specs. Heavyweight scratch and model assets remain under `/data-fast`.

**Checkpoint save/export is retained solely as a Test2 dependency.** Resume, inference, trajectory, end-to-end SFT, and RL check modules are not part of this extraction.
