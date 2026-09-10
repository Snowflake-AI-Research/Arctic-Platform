"""Does the *default* Cortex image assemble a chunked /forward correctly?

On Sep 2 the three-seed convergence run failed on the default deployed image
with ``SafetensorError: header too large``: the image could not decode a
``/operation`` forward payload that arrived as more than one chunk. The run was
completed on a pinned debug image instead. Thong's server fix merged Sep 1, so
that constraint may be gone -- this checks, rather than assumes.

Two arms over one job on the default image, same batch both times:

  control  one envelope   (16 MB ceiling, frame fits, single POST)
  test     N envelopes    (16 KB ceiling, frame is split and reassembled)

Chunking is a *transport* detail, so the two arms must return bitwise-identical
log-probs. Checking only that the chunked arm returns 200 would pass on a server
that silently assembled the frame in the wrong order; comparing against the
control is what makes this a correctness result. The control also isolates the
failure if the whole forward path is broken for reasons unrelated to chunking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from dss_client import wire  # noqa: E402
from dss_client.neutrino_client import _operation_chunk_max_bytes  # noqa: E402

from cortex_client_side_loss import (  # noqa: E402
    RESPONSE_OPTIONS,
    build_client,
    default_job_body,
    make_batch,
    training_sub_job,
)

CEILING_ENV = "DSS_FORWARD_OPERATION_MAX_JSON_BYTES"
MIN_CEILING = 16 * 1024


def forward_logprobs(client, job_id, sub_job_id, frame: bytes) -> torch.Tensor:
    body = client.forward(job_id, frame, sub_job_id=sub_job_id)
    result = client.poll_request(job_id, body["request_id"])
    if "logprobs" not in result:
        raise RuntimeError(f"/forward returned no logprobs (keys={sorted(result)})")
    return torch.as_tensor(result["logprobs"], dtype=torch.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--job-body", default=default_job_body())
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--keep-job", action="store_true")
    args = parser.parse_args(argv)

    # Any pinned tag inherited from the environment would defeat the whole point.
    os.environ.pop("NEUTRINO_ENABLE_DEBUG_OPTIONS", None)
    os.environ.pop(CEILING_ENV, None)

    cfg = json.loads(Path(args.config).read_text())
    client = build_client(cfg)

    created = False
    if args.job_id:
        job_id = args.job_id
        print(f"Reusing job {job_id}")
    else:
        body = json.loads(Path(args.job_body).read_text())
        assert "debug" not in body, "job body pins an image; this probe must use the default"
        job_id = str(client.create_job_from_body(body)["job_id"])
        created = True
        print(f"Created job {job_id} on the DEFAULT image; waiting for RUNNING ...")

    try:
        if created:
            client.wait_for_job(job_id)
            print(f"Job {job_id} is RUNNING")
        sub_job_id = training_sub_job(client, job_id)

        batch = make_batch(args.model_name, args.max_length)
        frame = wire.dumps({"kwargs": batch}, metadata=RESPONSE_OPTIONS)
        # Ask the encoder how it will actually split this rather than dividing by
        # the budget: it emits chunks well under max_bytes, so a computed estimate
        # understates the split by roughly an order of magnitude.
        n_chunks = len(
            wire.encode_byte_chunks(
                frame, kind="request", operation="forward", max_bytes=_operation_chunk_max_bytes(MIN_CEILING)
            )
        )
        print(f"batch={tuple(batch['input_ids'].shape)} frame={len(frame)}B -> {n_chunks} chunks at a {MIN_CEILING}B ceiling")
        if n_chunks < 2:
            print("FAIL: frame fits one chunk even at the floor; raise --max-length")
            return 1

        print("\n[control] single envelope")
        control = forward_logprobs(client, job_id, sub_job_id, frame)
        print(f"  logprobs={tuple(control.shape)} mean={float(control.mean()):.8f}")

        print(f"\n[test] forced into {n_chunks} chunks")
        os.environ[CEILING_ENV] = str(MIN_CEILING)
        try:
            chunked = forward_logprobs(client, job_id, sub_job_id, frame)
        finally:
            os.environ.pop(CEILING_ENV, None)
        print(f"  logprobs={tuple(chunked.shape)} mean={float(chunked.mean()):.8f}")

        if control.shape != chunked.shape:
            print(f"\nFAIL: shape {tuple(control.shape)} != {tuple(chunked.shape)}")
            return 1
        if not torch.equal(control, chunked):
            delta = (control - chunked).abs().max()
            print(f"\nFAIL: chunked forward disagrees with single-envelope, max|delta|={float(delta):.3e}")
            return 1
        print(f"\nPASS: {n_chunks}-chunk forward is bitwise identical to the single-envelope forward,")
        print("      on the default deployed image (no debug tag).")
        return 0
    finally:
        if created and not args.keep_job:
            try:
                client.cancel_job(job_id)
                print(f"Cancelled job {job_id}")
            except Exception as exc:
                print(f"Could not cancel job {job_id}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
