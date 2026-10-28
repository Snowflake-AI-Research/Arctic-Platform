# Use resolved sampler IDs after Cortex router replay bootstrap

## Problem

Router replay bootstrap resolves public sampler handles to zone IDs, but later training requests retain the public handle and cannot find the replay cache.

## Change

Record resolved IDs per training/sampling pair and translate subsequent requests without mutating caller-owned bodies. Cover synchronous and asynchronous bootstrap replies.

## Validation

- Base: `a1da584b67be01560c151b26daefd55caa82aef8` (literal copy of `arctic-platform-main`).
- An upstream-base failing control was not rerun in this review revision; the results below validate the revised branch.
- Revised-branch regression: 44 passed; configured formatting and lint hooks passed.
- Command: `python -m pytest -sv tests/client/test_cortex_sampler_identity.py tests/client/test_cortex_forward.py`.
- Correctness harness, debug instrumentation and skills are excluded; AP contract documentation is included.

## Review Follow-up

Decoded-frame contracts cover successful bootstrap followed by forward and forward-backward, for single and multi-chunk frames. The zone-visible metadata carries the resolved sampler ID without mutating caller inputs. The encoder is identical to #326, making the branch independently testable. `docs/cortex-integration.md` links the consuming zone reader at dss-platform commit `c4e35f9403972f66404e1dc2ca6bb535fea1d92e`. No live zone integration test was performed.

The external Cortex contract update is not published. Additions to Cortex-training's `docs/reference/rest-api.md` were committed locally as `66b9184` in a separate copy, but GitHub rejected the push with HTTP 403. No external PR was created. AP's own documentation does not close the outstanding canonical-contract finding, and the current writing scope prohibits further changes to external repositories.

The reviewer also named a DSS-client specification. Current AP and Cortex source searches found no import or dependency on that package; its specification has not been established as an authority for this AP transport. An earlier documentation-only DSS-client branch must not be counted as completion of this review finding.

Eight CPU checks executed the proposed documentation examples with both wire encoders, both forward operations and unchunked/multichunk requests: all passed. These checks validate frame encoding, not publication of the external contract or live SNOWAPI and zone integration.

Validation command: `python -m pytest -sv tests/client/test_cortex_sampler_identity.py tests/client/test_cortex_forward.py tests/client/test_cortex_routes.py`.
