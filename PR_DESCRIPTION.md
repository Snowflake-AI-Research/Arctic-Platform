# Preserve Cortex router replay wire metadata

## Problem

Forward requests serialize replay coordination into the model batch instead of frame metadata.

## Change

Keep replay coordination in wire metadata for both forward endpoints, including chunked requests, without mutating caller-owned requests.

## Validation

- Base: `a1da584b67be01560c151b26daefd55caa82aef8` (literal copy of `arctic-platform-main`).
- An upstream-base failing control was not rerun in this review revision; the results below validate the revised branch.
- Revised-branch regression: 41 passed; configured formatting and lint hooks passed.
- Command: `python -m pytest -sv tests/client/test_cortex_forward.py`.
- Correctness harness, debug instrumentation and skills are excluded; AP contract documentation is included.

## Review Follow-up

`docs/cortex-integration.md` documents replay metadata and links the consuming zone reader and bootstrap response at dss-platform commit `c4e35f9403972f66404e1dc2ca6bb535fea1d92e`. Decoded-frame tests cover single and chunked requests. No live zone integration test was performed.

The external Cortex contract update is not published. Additions to Cortex-training's `docs/reference/rest-api.md` were committed locally as `66b9184` in a separate copy, but GitHub rejected the push with HTTP 403. No external PR was created. AP's own documentation does not close the outstanding canonical-contract finding, and the current writing scope prohibits further changes to external repositories.

The reviewer also named a DSS-client specification. Current AP and Cortex source searches found no import or dependency on that package; its specification has not been established as an authority for this AP transport. An earlier documentation-only DSS-client branch must not be counted as completion of this review finding.

Eight CPU checks executed the proposed documentation examples with both wire encoders, both forward operations and unchunked/multichunk requests: all passed. These checks validate frame encoding, not publication of the external contract or live SNOWAPI and zone integration.

Validation command: `python -m pytest -sv tests/client/test_cortex_forward.py tests/client/test_cortex_routes.py`.
