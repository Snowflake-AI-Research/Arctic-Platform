.. _dflash2-nan-fix:

DFlash2 NaNs on GDN-backed models
=================================

ArcticInference carries a temporary compatibility fix for NaNs observed with
DFlash2 speculative decoding on ``Qwen3.8-27B`` under concurrent, long-running
decode workloads with prefix caching enabled. The failure is recorded in the
`DSS issue tracker <https://github.com/Snowflake-AI-Research/DSS-Issue-Tracker/blob/main/issues/2026-09-25-dflash-speculative-decoding-on-qwen3-8-27b-turns-running-seq-882babc67dc040c3a33d67f36d06c331/ISSUE.md>`__.

Two independent vLLM defects combine to produce the corruption:

* A discarded asynchronous speculative step can leave
  ``num_accepted_tokens == 0``. GDN state lookup subtracts one from this value,
  selecting an invalid speculative recurrent-state slot.
* Reallocated Mamba/GDN state pages are not included in vLLM's block-zeroing
  path, so a newly assigned page can retain recurrent state from its previous
  request.

The ArcticInference plugin applies both mitigations at startup for its pinned
vLLM 0.30.0 release, including when ``ARCTIC_INFERENCE_ENABLED=0`` leaves the
rest of the Arctic optimization stack disabled. Stale GDN rows have their
state slots replaced with ``NULL_BLOCK_ID`` and their accepted count clamped to
one, preventing state reads or writes for the discarded step. Newly allocated
Mamba blocks are also recorded and all of their state tensors are included in
block zeroing.

The exact 344-request workload completed without a NaN or failed request after
both changes. Mamba page zeroing alone was insufficient and produced 10 NaN
HTTP errors in 86 requests.

Upstream status
---------------

The permanent fixes are under review in vLLM:

* `vLLM PR #51508 <https://github.com/vllm-project/vllm/pull/51508>`__ fixes
  stale zero-accept GDN and KDA speculative rows.
* `vLLM PR #56524 <https://github.com/vllm-project/vllm/pull/56524>`__ adds
  Mamba state pages to allocation-time zeroing.

The ArcticInference compatibility patches should be removed after both changes
are available in the pinned vLLM release.
