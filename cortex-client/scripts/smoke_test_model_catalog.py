#!/usr/bin/env python3
"""Dry-run or execute a Cortex Training catalog profile as a real job."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from validate_model_catalog import load_catalog
from validate_model_catalog import validate_catalog

CLIENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CLIENT_ROOT.parent

PROFILE_KEYS = ("inference", "sftLora", "sftFull", "rlLora", "rlFull")
TRAINING_PROFILE_KEYS = ("sftLora", "sftFull", "rlLora", "rlFull")
SAMPLING_PROFILE_KEYS = ("inference", "rlLora", "rlFull")
TOP_LEVEL_ARGS = {
    "global_batch_size",
    "dtype",
    "seed",
    "model_post_init",
    "source_checkpoint_info",
}


def _find_model(models_doc: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = models_doc.get("models", [])
    model = next((item for item in models if item.get("modelId") == model_id), None)
    if model is None:
        raise ValueError(f"unknown catalog model: {model_id}")
    return model


def _recommended_profile_id(model: dict[str, Any], profile_key: str) -> str:
    capabilities = model["capabilities"]
    if profile_key == "inference":
        recommendation = capabilities["inference"]
        if not recommendation["supported"]:
            raise ValueError(f"{model['modelId']} does not support inference")
    else:
        training = capabilities["training"]
        if not training["supported"]:
            raise ValueError(f"{model['modelId']} does not support training")
        recommendation = training["profiles"][profile_key]
    return recommendation["recommendedProfileId"]


def _build_wire_sub_job(model_id: str, sub_job: dict[str, Any]) -> dict[str, Any]:
    builder = sub_job["builder"]
    args = copy.deepcopy(sub_job["args"])
    extra_key = "extra_training" if builder == "training_job" else "extra_sampling"
    config_key = "training_config" if builder == "training_job" else "inference_config"
    job_type = (
        "training" if builder == "training_job" else args.pop("job_type", "sampling")
    )

    wire = {
        "job_type": job_type,
        "model_name": model_id,
    }
    for key in TOP_LEVEL_ARGS:
        if key in args:
            wire[key] = args.pop(key)

    extra = args.pop(extra_key, {})
    for key, value in extra.items():
        args.setdefault(key, value)
    wire[config_key] = args
    return wire


def build_profile_request(
    config_dir: Path,
    repo_root: Path,
    model_id: str,
    profile_key: str,
) -> dict[str, Any]:
    """Build a CreateJob body directly from one validated catalog recommendation."""
    if profile_key not in PROFILE_KEYS:
        raise ValueError(f"invalid profile key: {profile_key}")

    models_doc, profiles = load_catalog(config_dir)
    validate_catalog(models_doc, profiles, repo_root)
    model = _find_model(models_doc, model_id)
    profile_id = _recommended_profile_id(model, profile_key)
    profile = next(item for item in profiles if item["id"] == profile_id)

    return {
        "sub_job_configs": [
            _build_wire_sub_job(model_id, sub_job) for sub_job in profile["subJobs"]
        ]
    }


def build_forward_backward_probe_spec(request: dict[str, Any]) -> dict[str, Any]:
    """Build a small, tokenizer-independent training batch for a catalog job."""
    training = next(
        (
            sub_job["training_config"]
            for sub_job in request["sub_job_configs"]
            if sub_job["job_type"] == "training"
        ),
        None,
    )
    if training is None:
        raise ValueError("forward-backward probe requires a training sub-job")

    batch_size = training.get("train_batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("training_config.train_batch_size must be a positive integer")

    token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    return {
        "payload": {
            "input_ids": [token_ids[:] for _ in range(batch_size)],
            "position_ids": "arange",
            "labels": {
                "strategy": "next_token",
                "mask_padding": False,
            },
        }
    }


def _build_forward_backward_probe_payload(request: dict[str, Any]) -> bytes:
    sys.path.insert(0, str(CLIENT_ROOT))
    from dss_client import build_forward_backward_payload

    return build_forward_backward_payload(build_forward_backward_probe_spec(request))


def _run_generate_probe(client, job_id: str) -> dict[str, Any]:
    request_id = client.generate(
        job_id,
        prompts=["Reply with OK."],
        sampling_params={
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )
    result = client.poll_request(job_id, request_id)
    generated = result.get("results")
    if not isinstance(generated, list) or len(generated) != 1:
        raise RuntimeError(f"generate probe expected one result, got {generated!r}")
    return {
        "operation": "generate",
        "requestId": request_id,
        "resultCount": len(generated),
    }


def _run_forward_backward_probe(
    client,
    job_id: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    payload = _build_forward_backward_probe_payload(request)
    request_id = client.forward_backward(job_id, payload)
    result = client.poll_request(job_id, request_id)
    avg_loss = result.get("avg_loss")
    try:
        avg_loss_value = float(avg_loss)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"forward-backward probe returned invalid avg_loss: {avg_loss!r}"
        ) from exc
    if not math.isfinite(avg_loss_value):
        raise RuntimeError(
            f"forward-backward probe returned non-finite avg_loss: {avg_loss!r}"
        )
    return {
        "operation": "forward-backward",
        "requestId": request_id,
        "avgLoss": avg_loss_value,
    }


def run_data_plane_probes(
    client,
    job_id: str,
    profile_key: str,
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute the minimal operations required by a catalog workflow."""
    probes = []
    if profile_key in SAMPLING_PROFILE_KEYS:
        probes.append(_run_generate_probe(client, job_id))
    if profile_key in TRAINING_PROFILE_KEYS:
        probes.append(_run_forward_backward_probe(client, job_id, request))
    return probes


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _required_env(*names: str) -> str:
    value = _first_env(*names)
    if value is None:
        raise ValueError(f"set one of: {', '.join(names)}")
    return value


def _build_client(poll_timeout: float):
    sys.path.insert(0, str(CLIENT_ROOT))
    from dss_client import NeutrinoClient

    host = _required_env("NEUTRINO_HOST", "SNOWFLAKE_HOST")
    host = host.removeprefix("https://").removeprefix("http://").rstrip("/")
    return NeutrinoClient.from_pat(
        host=host,
        pat=_required_env("NEUTRINO_PAT", "SNOWFLAKE_PAT"),
        database=_required_env("NEUTRINO_DATABASE", "SNOWFLAKE_DATABASE"),
        schema=_first_env("NEUTRINO_SCHEMA", "SNOWFLAKE_SCHEMA") or "PUBLIC",
        endpoint=_first_env("NEUTRINO_ENDPOINT") or "cortex-training",
        poll_timeout=poll_timeout,
    )


def _cancel_job(client, job_id: str, active_error: BaseException | None) -> bool:
    try:
        client.cancel_job(job_id)
    except Exception as cancel_error:
        if active_error is None:
            raise
        print(
            f"WARNING: failed to cancel {job_id}: {cancel_error}",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    default_config = CLIENT_ROOT / "configs" / "cortex-training"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--profile",
        action="append",
        choices=PROFILE_KEYS,
        required=True,
        help="Catalog profile to test. Repeat to test multiple profiles serially.",
    )
    parser.add_argument("--config-dir", type=Path, default=default_config)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Submit real jobs and execute workflow data-plane probes. "
            "Without this flag, print the generated request bodies."
        ),
    )
    parser.add_argument("--poll-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    plans = []
    for profile_key in args.profile:
        request = build_profile_request(
            args.config_dir.resolve(),
            args.repo_root.resolve(),
            args.model_id,
            profile_key,
        )
        plans.append(
            {
                "profileKey": profile_key,
                "request": request,
            }
        )

    if not args.submit:
        print(
            json.dumps(
                [
                    {
                        "profileKey": plan["profileKey"],
                        "request": plan["request"],
                    }
                    for plan in plans
                ],
                indent=2,
            )
        )
        return 0

    client = _build_client(args.poll_timeout)
    results = []
    for plan in plans:
        profile_key = plan["profileKey"]
        job_id: str | None = None
        active_error: BaseException | None = None
        try:
            response = client.create_job_from_body(plan["request"])
            job_id = response["job_id"]
            print(f"Submitted {profile_key} as job {job_id}", file=sys.stderr)
            job = client.wait_for_job(job_id)
            probes = run_data_plane_probes(
                client,
                job_id,
                profile_key,
                plan["request"],
            )
            results.append(
                {
                    "profileKey": profile_key,
                    "jobId": job_id,
                    "status": str(job.get("status") or "")
                    .lower()
                    .removeprefix("job_state_"),
                    "probes": probes,
                }
            )
        except BaseException as error:
            active_error = error
            raise
        finally:
            if job_id is not None:
                if _cancel_job(client, job_id, active_error):
                    print(f"Cancelled job {job_id}", file=sys.stderr)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
