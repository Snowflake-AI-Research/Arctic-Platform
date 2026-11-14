# Maintained Correctness Code

**Operators use a few stable commands backed by the shared library.** Start with [running-tests.md](running-tests.md). Commands run on an already-prepared allocation; the library does not allocate machines or install prerequisites.

**This checkout registers seven correctness tests, including two reinforcement-learning (RL) comparisons.** `single-step-grads`, `single-step-optimizer`, `checkpoint-resume-loss`, `inference-checkpoint-loss`, `e2e-sft-train-validate`, `rl-weight-sync` and `rl-router-replay` are the registered IDs. The first five tests retain their existing workload identities and prerequisites. The two RL tests require narrowly scoped, opt-in trainer/sampler evidence: actual tensor-send and receiver-write records for weight synchronization, and received routing plus forward/backward router invocations for replay. CLI discovery and help do not import these observers; ordinary training and sampling calls leave them disabled.

## Active Layout

Paths in this table are relative to `arctic_platform/correctness/`.

| Area | Responsibility |
| --- | --- |
| `__main__.py`, `api.py` | Stable operator command and supported Python selection/run interface. |
| `harness/` | Shared execution, input preparation, transport, configuration, reporting, and record validation. |
| `harness/config.py` | Reviewed specifications, model settings, and placement arms. |
| `harness/storage.py` | Fast-storage paths, checkpoint ownership, compact reference records, and interrupted-build checks. |
| `harness/sft.py` | Synthetic and configured inputs, dataset locations, training settings, and answer-token scoring. |
| `harness/cortex.py` | Training transport contracts, local gateway client, fixture admission, and explicitly requested microbatch packing. Ordinary requests remain unsplit. |
| `harness/lora.py` | Upstream PEFT reference adapters, content verification, native loading, and legacy E2E adapter recipes. Correctness-owned local first-two loading binds the same generated adapter; hosted shared-adapter loading is not admitted. |
| `harness/gateway.py` | Gateway ownership, supervised and RL training lifecycles, payloads, and owned-worker cleanup. A standard-library-only subprocess watches the owner lifetime pipe and imports this library only after abnormal owner exit. |
| `harness/optimizer.py` | Tensor-name alignment, participation coverage, optimizer capture, and checkpoint decoding. |
| `harness/inference.py` | Checkpoint export, inference allocation validation, and router-replay evidence collection. |
| `models/` | Shared original-model metadata and exact validation; metadata alone does not establish runtime or capacity admission. |
| `checks/` | Registered numerical comparisons; adding a model does not require a new launcher. |
| `checks/fwd_bwd.py` | `single-step-grads`: named gradient norms against prepared HF measurements. |
| `checks/fwd_bwd_step.py`, `checks/optimizer_state.py` | `single-step-optimizer`: optimizer artifact coverage and scalar-norm validation. |
| `checks/checkpoint_resume.py` | `checkpoint-resume-loss`: uninterrupted versus resumed training, checkpoint identity and restored-state gates. |
| `checks/inference_checkpoint_loss.py` | `inference-checkpoint-loss`: training versus sampling answer-token loss, export schema coverage and owned teardown. |
| `checks/e2e_sft_train_validate.py` | `e2e-sft-train-validate`: ordered SFT replay, compact HF trajectory identity and final training/validation verdicts. |
| `checks/rl_weight_sync.py` | `rl-weight-sync`: collective trainer inventory, received tensors and actual destination-write or LoRA-install evidence. |
| `checks/rl_router_replay.py` | `rl-router-replay`: sampler routing, trainer-received routing and model-bound forward/backward router coverage. |
| `harness/runner.py`, `harness/execution.py` | Prepared-reference regression, selected-check sequencing and CLI/report dispatch. |
| `reference/` | Independent reference algorithms and their existing subprocess driver. |
| `onboarding/` | Explicit model/reference preparation and calibration, separate from regression. |
| `configs/`, `specs/`, `reference_data/` | Reviewed workload definitions, unchanged gates, and compact reference measurements. |
| `selftest/library.py`, `selftest/check_suite.py` | CPU checks and shared fixtures in one library, collected by one driver. Sections retain their original setup-fixture boundaries. |
| `../debug/lora.py`, `../debug/weight_sync.py`, `../debug/inference.py`, `../debug/training.py`, `../debug/deepspeed_zero_initialization.py` | Opt-in debug integrations grouped by responsibility; ordinary production behavior remains unchanged when disabled. |
| `diagnostics/` | One maintained diagnostic, `lm_head_chunking`, and the shared list/run command. |
| `docs/` | Static operator guide and maintainer instructions; no generated run updates. |

**The five non-RL tests retain their correctness-owned capture; the RL tests use opt-in production observers.** The correctness-owned gateway collects participation for the first two comparisons only. Later-only selection does not acquire the first-two BF16 or runtime-capture requirement. An installed target runtime, compatible model assets and check-specific execution prerequisites remain necessary. RL observers record existing production operations; they must not replace tensor sends, receiver writes, optimizer steps or router execution.

**Configs, specs and compact records are operator inputs, not bundled assets.** Explicit single-config runs accept `--test-spec`; multi-config runs use locally generated matching default specs. Configs and compact records remain local until explicitly selected for publication; heavyweight scratch and model assets stay under `/data-fast`.

## Retired Experiments

**Retired experiments are not part of the branch's supported workflow.** Optional local archives are not bundled, linked as prerequisites or included in operator selections, ordinary selftests or package builds.

**Standalone probes are no longer supported active commands.** This retirement includes the B200 gradient probes, Qwen3.6 width/grouping/telemetry/scaling experiments, DeepEP tracing, checkpoint-state investigation, engine ablations, and numerical-instability experiments. Their tests accompany the archived code; deleting those tests does not remove maintained-library coverage. The dispatcher now rejects retired names.

**Reusable behavior remains in the library with its tests.** `pack_microbatches` lives in `harness/cortex.py`; the runtime-client section of `selftest/library.py` exercises row splitting, shifted labels, and copied processing settings. The removed `with_accumulation` configuration override is not part of the supported interface.

**Consolidation removes replaced modules rather than leaving forwarding files.** Maintained coverage lives in `selftest/`, not in archived drivers. Historical suite counts and receipts are run results, not guarantees about the current checkout.

**Historical diagnostic campaigns are retired with their unused helpers.** The selftests no longer carry the standalone LoRA storage/scaling observers or the synthetic expert-parallel pack/prepare/send workflow. The corresponding unused debug APIs are removed. Reusable numerical comparisons, byte-bound adapter loading, compact-reference publication, ordinary disabled behavior and owned-worker cleanup retain representative coverage; the retirement does not claim equivalent coverage of every old injected failure.

**One-off fixture configurations and the old naming-source exception are retired.** Job-numbered, issue-numbered and 64-token experiment configs are not maintained operator workloads. Their frozen copies remain with the original evidence. The removed historical-gradient options cannot admit consolidated sources under obsolete `names.py` hashes; current reference loading still requires an exact identity.

**The documented LM-head command remains supported.** Both forms invoke the same maintained diagnostic:

```bash
python -m arctic_platform.correctness.diagnostics list
python -m arctic_platform.correctness.diagnostics run lm_head_chunking --help
python -m arctic_platform.correctness.diagnostics.lm_head_chunking --help
```

**Historical evidence is not rewritten.** Saved results, compact reference records, submitted scripts, and frozen source snapshots keep their existing paths and identities. Frozen copies of retired files remain history, not active dependencies. Archiving an investigation does not resolve its numerical failure or turn an old result into a new pass. New source-bound runs must bind their actual current file set.
