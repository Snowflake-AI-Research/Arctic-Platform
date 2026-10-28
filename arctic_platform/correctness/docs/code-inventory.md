# Correctness Code Map

**The maintained library contains only the first two checks and their preparation paths.** Start with [the operator guide](running-tests.md). All paths below are relative to `arctic_platform/correctness/`.

| Area | Responsibility |
| --- | --- |
| `__main__.py`, `api.py` | Operator CLI, external config discovery, selection, and shared Python run interface. |
| `checks/fwd_bwd.py` | `single-step-grads` comparison. |
| `checks/fwd_bwd_step.py`, `checks/optimizer_state.py` | `single-step-optimizer` and optimizer artifact/norm validation. |
| `harness/config.py`, `harness/batches.py` | Reviewed specs, placement arms, and reproducible synthetic input tensors. |
| `harness/runner.py`, `harness/execution.py` | Prepared-reference regression, sequencing, and CLI/report dispatch. |
| `harness/dss_driver.py`, `harness/transport.py` | Supervised training lifecycle, payloads, packing, and cleanup. |
| `harness/cortex.py`, `harness/hosted.py` | Reviewed fixture admission, training transport, public save/catalog identity helpers. |
| `harness/inference_export.py`, `harness/cortex_optimizer.py`, `harness/optimizer_capture.py` | Test2 resumable checkpoint export, safe decoding, and local optimizer capture. The export module's historical filename does not register an inference check. |
| `harness/shared_lora_adapter.py` | Upstream PEFT reference adapter generation and content verification; retained internal Ray worker loading is not an admitted Cortex-training target route. |
| `harness/lora.py` | Canonical tensor-name validation shared by the adapter checks. No retired deterministic recipes or SAVE0 execution. |
| `harness/participation.py` | Versioned parameter participation and complete gradient coverage. |
| `harness/reference_records.py`, `harness/storage.py`, `harness/report.py`, `harness/names.py` | Compact record identity, fast scratch containment, reports, and parameter-name alignment. |
| `reference/` | Independent single-GPU reference, one-step AdamW, model features, process determinism, and Qwen4 loading. |
| `gateway/` | `single-step-grads` for an unmodified catalog create-job body on a local cortex-training gateway: HF reference, target through the `cortex_training` client, LoRA adapter export from the resumable checkpoint, and multi-tree summaries. |
| `onboarding/` | Explicit sizing, calibration, fixture publication, and reference preparation. `fp8_fixture.py` cuts a block-FP8 checkpoint to a reduced fixture's tensors for FP8 targets. |
| `selftest/` | Synthetic CPU coverage for the retained checks and shared interfaces. No bundled real recipe dependencies. |
| `diagnostics/` | `lm_head_chunking` and its stdlib dispatcher; altered-workload diagnostic only. |

**Configs, specs, and compact records are operator inputs, not bundled assets.** Explicit single-config runs accept `--test-spec`; multi-config runs use locally generated matching default specs. Heavyweight scratch and model assets remain under `/data-fast`.

**Checkpoint save/export is retained solely as a Test2 dependency.** Resume, inference, trajectory, end-to-end SFT, and RL check modules are not part of this extraction.
