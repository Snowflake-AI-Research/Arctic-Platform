# Opt-In Debug Instrumentation

The `arctic_platform.debug` libraries observe training initialization, model materialization, checkpoint readers, router replay and weight transfer. Native entry points call descriptive `maybe_...` helpers. With their options absent, the helpers leave native arguments, objects, tensor iteration and cleanup behavior unchanged.

## Initialization Traces

Set the DeepSpeed configuration option `arctic_platform_zero_initialization_trace` to an object with `request_id` and `output_dir`. The request ID must contain only letters, digits, underscores or hyphens, and the evidence directory must resolve under `/data-fast`. The initialization context removes this private option from the configuration sent to DeepSpeed.

The trace binds the installed ZeRO optimizer's source SHA256, rank, process and allocation events. Its request-scoped observer does not replace `torch.zeros` globally, and it restores the optimizer method after success or failure. A trace is observation, not a capacity guarantee or numerical result.

## Dependency Boundaries

Enabled routes require their actual native implementations and source identities. Debug41 does not supply replacement training or sampling implementations:

- Initialization-capacity verification uses the sequence-topology API from #436 and the parallel-group implementation from #439. Main without that API rejects enabled verification and runs owned cleanup.
- Participation capture requires the correctness harness's LoRA and optimizer modules plus the DSS training/gateway services. An AP-only installation does not supply those services. The ownership test explicitly allows only its two function-local DSS imports in `debug/training.py`; ordinary modules and eager debug imports still cannot import DSS.
- Selected external sampler routes require the explicitly bound `arctic_inference` receiver and vLLM sources. Installing bundled `arctic_platform.inference` does not satisfy that different source contract.
- Reduced prediction-fixture authentication requires the native auxiliary/MTP capture libraries from #437/#438. Its absent-option route is a no-op. When composing those fixes, retain ordinary strict native capture; the debug hook must not replace or suppress it for an ordinary checkpoint.
- Receiver loading observation composes with fix38's ordinary loading helpers. It does not introduce those native loading changes into this branch.

The initialization-memory arithmetic lives in `arctic_platform.debug.initialization_memory`. Native model inventory is imported lazily when enabled. Unresolved memory terms remain unresolved; unit-test success does not admit a real allocation.

## Validation

CPU controls check disabled imports, native argument/result identity, actor bindings, cancellation, source hashes and trace restoration. They do not establish GPU kernel, distributed topology, external receiver or large-model capacity correctness. Enabled execution must separately admit the complete native dependency closure and exact selected source bytes.
