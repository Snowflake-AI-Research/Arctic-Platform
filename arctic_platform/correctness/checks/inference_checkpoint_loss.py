# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One validation cross entropy, computed by the training zone and by a sampling zone on its checkpoint.

An operator who trains in Arctic Platform and then serves the result reads a loss from two different engines. This
check asks whether those two numbers are the same number. A short GSM8K run trains the config as written;
the training job then scores a held-out batch through its forward-only route, saves a weights-only
checkpoint, and is destroyed; a sampling job started on that checkpoint scores the same rows.

The reference is the training zone, because it is the engine that produced the weights. Its forward-only
route runs with no gradient and the engine in ``eval()`` and returns per-token log-probabilities. The
target is the sampling zone's ``log-probs`` over the same answer tokens. This check reduces both arrays
with one token-weighted mean, so the two paths compare the same quantity.

There is no case axis. The two sides run one forward each over one batch, and gradient accumulation is a
property of a backward pass, so a second case would re-measure the first.

The loss covers answer tokens only: the prompt and the padded tail carry ``IGNORE_INDEX``. Prompt tokens
are the same text on both sides and are far more numerous than answers, so scoring them would dilute the
quantity under test with a term neither engine can get wrong.

Two alignment rules do all the work, and each was worth two orders of magnitude more than the gate before
it was applied:

- the labels sent to the forward-only route are shifted here, because its dispatch does not shift them for
  a ``huggingface`` model provider while ``/fwd-bwd`` does;
- the answer span is located as the common prefix of the prompt and joined tokenizations, and both sides
  slice the same logit positions out of arrays that differ in length by the one position vLLM does not
  report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import traceback
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.correctness.harness import sft as gsm8k
from arctic_platform.correctness.harness.cortex import CorrectnessClient
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.gateway import UnsafeCleanupError
from arctic_platform.correctness.harness.gateway import _inference_client_config
from arctic_platform.correctness.harness.gateway import _local_checkpoint_metadata
from arctic_platform.correctness.harness.gateway import build_payload
from arctic_platform.correctness.harness.gateway import forward_logprobs
from arctic_platform.correctness.harness.gateway import fwd_bwd_step
from arctic_platform.correctness.harness.gateway import gateway
from arctic_platform.correctness.harness.gateway import is_local_cortex_gateway
from arctic_platform.correctness.harness.gateway import log_probs
from arctic_platform.correctness.harness.gateway import minimum_context_parallel_row_tokens
from arctic_platform.correctness.harness.gateway import pack
from arctic_platform.correctness.harness.gateway import pack_logit_aligned
from arctic_platform.correctness.harness.gateway import require_local_cortex_gateway
from arctic_platform.correctness.harness.gateway import running_job
from arctic_platform.correctness.harness.gateway import sampling_payload
from arctic_platform.correctness.harness.gateway import save_and_resolve_checkpoint_id
from arctic_platform.correctness.harness.gateway import save_weights_only_checkpoint
from arctic_platform.correctness.harness.sft import SEED
from arctic_platform.correctness.harness.sft import TRAIN_SPLIT
from arctic_platform.correctness.harness.sft import VALIDATION_SPLIT
from arctic_platform.correctness.harness.sft import Batch
from arctic_platform.correctness.harness.sft import _assert_answer_span
from arctic_platform.correctness.harness.sft import configured_learning_rate
from arctic_platform.correctness.harness.sft import configured_rows_per_step
from arctic_platform.correctness.harness.sft import cross_entropy
from arctic_platform.correctness.harness.sft import reference_answer_logprobs
from arctic_platform.correctness.harness.storage import correctness_workdir
from arctic_platform.correctness.harness.storage import require_data_fast_path
from arctic_platform.debug.inference import maybe_bind_inference_startup_error
from arctic_platform.debug.inference import maybe_bind_inference_warmup_observer

from ..harness.config import correctness_microbatch_tokens
from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test

TEST_ID = "inference-checkpoint-loss"

TRAIN_STEPS = 10

_EXPORT_JSON_LIMIT_BYTES = 16 * 1024 * 1024
_TOKENIZER_JSON_LIMIT_BYTES = 32 * 1024 * 1024
_PROCESSOR_JSON_LIMIT_BYTES = 16 * 1024 * 1024
_PROCESSOR_METADATA_NAMES = ("preprocessor_config.json", "video_preprocessor_config.json")

# Fixed, and deliberately not calibrated. The two sides are one set of weights read by two engines, so
# there is no engine-to-engine spread for a calibration to size: a checkpoint that carries the model
# reproduces the loss. A measurement above the accelerator-family gate is the check's finding, not noise.
DEFAULT_TOLERANCE = 1e-3
B200_TOLERANCE = 3e-3

# The zone reads the first key of each position's log-probability dictionary, so zero -- which asks vLLM
# for the sequence's own token and nothing else -- is the only value that scores the row that was sent. A
# positive ``top_k`` makes the first key the top-ranked token instead. Passed explicitly because the
# gateway's own default is 1 (arctic_platform/client/requests.py:181) while the zone body's is 0.
LOG_PROB_TOP_K = 0

# Default sampling uses one GPU and a bounded prefill batch. Explicit sampling topology is opt-in;
# neither placement may substitute reduced weights after a memory failure.
SAMPLING_GPUS = 1
_FLASHNEXT_CONFIG_ID = "qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k"

QUANTITY = "validation cross entropy over answer tokens"


def _sampling_topology(ctx) -> tuple[int, int, int]:
    topology = getattr(ctx, "inference_sampling_topology", None)
    if topology is None:
        return SAMPLING_GPUS, SAMPLING_GPUS, 1
    if (
        not isinstance(topology, tuple)
        or len(topology) != 3
        or any(type(size) is not int or size <= 0 for size in topology)
        or topology[0] != topology[1] * topology[2]
    ):
        raise ValueError("inference sampling topology requires positive integer allocation, TP and PP with GPUs=TP*PP")
    for sub in ctx.cfg.sub_jobs:
        if sub.get("job_type") != "sampling":
            continue
        inference = sub["inference_config"]
        allocation = inference["n_gpus"]
        engine = inference.get("vllm_config", {})
        reviewed = (
            allocation,
            engine.get("tensor_parallel_size", allocation),
            engine.get("pipeline_parallel_size", 1),
        )
        if any(type(size) is not int or size <= 0 for size in reviewed) or topology != reviewed:
            raise ValueError(
                "explicit inference sampling topology differs from the configured sampling allocation/TP/PP"
            )
    return topology


def sampling_gpu_count(ctx) -> int:
    return _sampling_topology(ctx)[0]


def _sampling_payload(ctx, model_path, token_budget: int) -> dict:
    allocation, tensor, pipeline = _sampling_topology(ctx)
    payload = sampling_payload(
        model_path,
        SEED,
        dtype=str(ctx.cfg.sub_job.get("dtype") or ctx.cfg.training.get("dtype") or "bfloat16"),
        n_gpus=allocation,
        max_seq_len=token_budget + 1,
    )
    if ctx.cfg.training.get("peft_config"):
        payload["inference_config"]["peft_config"] = deepcopy(ctx.cfg.training["peft_config"])
    if getattr(ctx, "inference_sampling_topology", None) is not None:
        payload["inference_config"]["vllm_config"].update(
            tensor_parallel_size=tensor,
            pipeline_parallel_size=pipeline,
            enable_expert_parallel=True,
            extra_engine_kwargs={
                "distributed_executor_backend": "ray",
                "data_parallel_size": 1,
                "prefill_context_parallel_size": 1,
                "decode_context_parallel_size": 1,
                "enable_eplb": False,
                "enable_elastic_ep": False,
            },
        )
        if ctx.cfg.model_name == "Qwen/Qwen3.8-Flash-Next":
            payload["inference_config"]["vllm_config"]["extra_engine_kwargs"]["engram_config"] = {
                "cpu_offload": False,
                "embedding_across_dp": False,
                "dp_shared_memory": False,
                "use_thp": False,
            }
    maybe_bind_inference_warmup_observer(payload, ctx)
    maybe_bind_inference_startup_error(payload, ctx)
    return payload


def _cortex_sampling_parallelism(ctx) -> dict:
    if getattr(ctx, "inference_sampling_topology", None) is None:
        return {}
    return {"inference_tensor_parallel_size": _sampling_topology(ctx)[1]}


@correctness_test(
    TEST_ID,
    title="Validation loss agreement between the training zone and a sampling zone on its checkpoint",
    criterion=(
        "the answer-token cross entropy of a held-out GSM8K batch agrees between the training zone that "
        "trained the weights and a sampling zone serving the checkpoint it saved, within 3e-3 on "
        "B200 and 1e-3 on other GPU families"
    ),
    compares_to_reference=False,
)
def run(ctx) -> List[TestResult]:
    """One gateway, one training job, one sampling job. The check owns all three."""
    try:
        return [_run(ctx)]
    except UnsafeCleanupError:
        raise
    except Exception as exc:  # noqa: BLE001 - a failure here is this check's verdict, not a crash
        formatted = traceback.format_exc()
        print(formatted, flush=True)
        site = next(
            (line.strip() for line in reversed(formatted.splitlines()) if line.strip().startswith("File ")), ""
        )
        summary = f"{type(exc).__name__}: {exc}"
        if site:
            summary = f"{summary} at {site}"
        return [
            TestResult(
                test_id=TEST_ID,
                config_id=ctx.config_id,
                outcome=TestOutcome.FAIL,
                summary=summary,
                error_type=type(exc).__name__,
            )
        ]


def target_answer_logprobs(results: Sequence[dict], examples: Sequence[gsm8k.Example]) -> List[List[float]]:
    """The sampling zone's log-probabilities over the same logit positions.

    vLLM reports nothing for position 0, because no token precedes it, and the zone's loop drops that
    entry (arctic_platform/common/ray_server.py:704). A returned row therefore holds ``L - 1`` entries, and
    entry ``k`` scores the token at ``k + 1`` -- the same indexing the training side's array has, one
    entry shorter. Both sides consequently slice ``p - 1`` through ``L - 2``, and the length is asserted
    rather than inferred from the answer span happening to start late enough.

    The token ids the zone returns are compared against the row's own ids. A row scored against a
    different token stream -- a different tokenizer in the served tree, or a ``top_k`` that reports ranked
    tokens instead of the sequence's -- then fails here instead of quietly moving the measurement.
    """
    if len(results) != len(examples):
        raise RuntimeError(f"sampling returned {len(results)} rows for {len(examples)} validation examples")
    rows: List[List[float]] = []
    for index, (result, example) in enumerate(zip(results, examples)):
        _assert_answer_span(example, index)
        returned_ids = [int(token) for token in result["token_ids"]]
        expected_ids = list(example.input_ids[1:])
        if returned_ids != expected_ids:
            raise RuntimeError(
                f"row {index}: the sampling zone scored {len(returned_ids)} tokens that are not this "
                f"row's tokens 1..{example.length - 1}; the served tree and the batch disagree on the "
                "token stream"
            )
        values = _as_float_list(result["logprobs"])
        if len(values) != example.length - 1:
            raise RuntimeError(
                f"row {index}: the sampling zone returned {len(values)} log-probabilities for a row of "
                f"{example.length} tokens; expected {example.length - 1}, one per position after the first"
            )
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"row {index}: sampling log-probabilities must be finite")
        span = values[example.n_prompt - 1 : example.length - 1]
        if len(span) != example.answer_tokens:
            raise RuntimeError(f"row {index}: sampling response does not cover the complete answer span")
        rows.append(span)
    return rows


def cortex_prompt_logprobs(results: Sequence[dict], examples: Sequence[gsm8k.Example]) -> List[dict]:
    """Normalize generate's per-position dictionaries without selecting a ranked alternative token."""
    if not isinstance(results, (list, tuple)) or len(results) != len(examples):
        raise RuntimeError("generate must return one row per validation example")
    normalized: List[dict] = []
    for index, (result, example) in enumerate(zip(results, examples)):
        _assert_answer_span(example, index)
        if not isinstance(result, dict):
            raise RuntimeError(f"row {index}: generate result must be a dictionary")
        positions = result.get("prompt_logprobs")
        if not isinstance(positions, (list, tuple)) or len(positions) != example.length or positions[0] is not None:
            raise RuntimeError(
                f"row {index}: prompt_logprobs must contain {example.length} positions with initial None"
            )
        if "prompt_len" in result and result["prompt_len"] != example.length:
            raise RuntimeError(f"row {index}: generate prompt length disagrees with the validation token stream")
        values = []
        for position, token_id in enumerate(example.input_ids[1:], start=1):
            entry = positions[position]
            if not isinstance(entry, dict) or not entry:
                raise RuntimeError(f"row {index} position {position}: missing prompt log-probabilities")
            keys = [key for key in (token_id, str(token_id)) if key in entry]
            if len(keys) != 1:
                raise RuntimeError(
                    f"row {index} position {position}: expected token {token_id} is missing or ambiguous"
                )
            value = entry[keys[0]]
            if isinstance(value, dict):
                value = value.get("logprob")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"row {index} position {position}: expected token logprob must be finite")
            values.append(float(value))
        normalized.append({"token_ids": list(example.input_ids[1:]), "logprobs": values, "seq_len": len(values)})
    return normalized


def cortex_sampling_config(
    ctx, backend: CortexConfig, checkpoint_id: str, source_job_id: str, token_budget: int
) -> CortexTrainingConfig:
    """Initialize standalone managed sampling from the saved weights, not a node-local export."""
    model_name = _full_model_name(ctx)
    if not checkpoint_id.startswith("cp_") or not source_job_id:
        raise ValueError("managed sampling requires a durable checkpoint_id and source_job_id")
    payload = _sampling_payload(ctx, model_name, token_budget)
    payload["job_type"] = "sampling"
    config = _inference_client_config(payload, backend=backend)
    return CortexTrainingConfig.model_validate(
        {
            **config.model_dump(),
            "hardware": ctx.cfg.gpu_type.upper() if ctx.cfg.gpu_type is not None else None,
            "source_checkpoint_info": {"checkpoint_id": checkpoint_id, "source_job_id": source_job_id},
            **_cortex_sampling_parallelism(ctx),
        }
    )


def _full_model_name(ctx) -> str:
    """Use the configured model, never the reduced HF reference or its publication binding."""
    model_name = ctx.cfg.model_name
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("inference comparison requires a full configured model")
    if isinstance(getattr(ctx, "client_backend", None), CortexConfig) and model_name.startswith("/"):
        raise ValueError("Cortex inference requires a full configured model available in its catalog")
    return model_name


def _as_float_list(values) -> List[float]:
    """The wire format hands back either a sequence or an array; both become one flat list of floats."""
    if hasattr(values, "reshape"):
        values = values.reshape(-1).tolist()
    return [float(value) for value in values]


def training_batches(examples: Sequence[gsm8k.Example], rows_per_step: int, pad_id: int) -> List[Batch]:
    """One batch per step, consuming the training slice in order with no row used twice."""
    if len(examples) != TRAIN_STEPS * rows_per_step:
        raise ValueError(
            f"{TRAIN_STEPS} steps of {rows_per_step} rows need {TRAIN_STEPS * rows_per_step} examples, "
            f"got {len(examples)}"
        )
    return [
        gsm8k.build_batch(
            f"train-step-{step:02d}", examples[(step - 1) * rows_per_step : step * rows_per_step], pad_id
        )
        for step in range(1, TRAIN_STEPS + 1)
    ]


def _minimum_context_parallel_row_tokens(ctx, model_path: str) -> int:
    return minimum_context_parallel_row_tokens(ctx.cfg, model_path)


def _extend_short_causal_rows(batch: Batch, minimum_tokens: int, pad_id: int) -> Batch:
    """Keep every possible packed row group wide enough without adding loss-bearing tokens."""
    if minimum_tokens == 0:
        return batch
    import torch

    from arctic_platform.correctness.harness.gateway import _attention_mask_from_position_ids
    from arctic_platform.correctness.harness.sft import IGNORE_INDEX

    lengths = _attention_mask_from_position_ids(batch.position_ids).sum(dim=1)
    if bool((lengths >= minimum_tokens).all()):
        return batch
    width = max(batch.seq_len, minimum_tokens)
    ids = batch.input_ids.new_full((batch.rows, width), pad_id)
    positions = batch.position_ids.new_zeros((batch.rows, width))
    labels = batch.labels.new_full((batch.rows, width), IGNORE_INDEX)
    ids[:, : batch.seq_len] = batch.input_ids
    positions[:, : batch.seq_len] = batch.position_ids
    labels[:, : batch.seq_len] = batch.labels
    for row, length in enumerate(lengths.tolist()):
        if length < minimum_tokens:
            # Monotone positions keep the loss-free causal tail through attention-mask packing.
            ids[row, length:minimum_tokens] = pad_id
            positions[row, length:minimum_tokens] = torch.arange(
                length, minimum_tokens, dtype=positions.dtype, device=positions.device
            )
            labels[row, length:minimum_tokens] = IGNORE_INDEX
    return Batch(name=batch.name, input_ids=ids, position_ids=positions, labels=labels)


def materialize_checkpoint_tokenizer(tokenizer, checkpoint_path: str | Path) -> None:
    """Store the validated model tokenizer beside an exported weights-only checkpoint."""
    tokenizer.save_pretrained(checkpoint_path)


def _processor_metadata_identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_processor_metadata_directory(path: Path, fd: int, label: str):
    try:
        held = os.fstat(fd)
        slot = os.stat(path, follow_symlinks=False)
    except OSError:
        raise RuntimeError(f"processor metadata {label} changed") from None
    if not stat.S_ISDIR(held.st_mode) or _processor_metadata_identity(slot) != _processor_metadata_identity(held):
        raise RuntimeError(f"processor metadata {label} changed")
    return held


def _read_processor_metadata(fd: int, name: str) -> bytes:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"processor metadata snapshot requires a regular file: {name}")
    if metadata.st_size > _PROCESSOR_JSON_LIMIT_BYTES:
        raise RuntimeError(f"processor metadata exceeds its bound: {name}")
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), "rb") as stream:
        raw = stream.read(_PROCESSOR_JSON_LIMIT_BYTES + 1)
    if len(raw) > _PROCESSOR_JSON_LIMIT_BYTES:
        raise RuntimeError(f"processor metadata exceeds its bound: {name}")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate processor metadata JSON key: {key}")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError(f"processor metadata JSON contains a nonfinite value: {value}")

    value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise RuntimeError(f"processor metadata JSON must be an object: {name}")
    return raw


def _open_processor_metadata_file(parent_fd: int, name: str, *, required: bool) -> int | None:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        if not required:
            return None
        raise RuntimeError(f"processor metadata snapshot requires a regular file: {name}") from None
    except OSError:
        raise RuntimeError(f"processor metadata snapshot requires a regular file: {name}") from None
    try:
        metadata = os.fstat(descriptor)
        slot = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or _processor_metadata_identity(slot) != _processor_metadata_identity(
            metadata
        ):
            raise RuntimeError(f"processor metadata snapshot requires a regular file: {name}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _copy_processor_metadata_file(
    source_fd: int, destination_parent_fd: int, name: str
) -> tuple[int, os.stat_result, bytes, bool]:
    raw = _read_processor_metadata(source_fd, name)
    destination_created = False
    try:
        destination_fd = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_parent_fd,
        )
    except FileExistsError:
        try:
            destination_fd = os.open(
                name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=destination_parent_fd,
            )
        except OSError:
            raise RuntimeError(
                f"processor metadata destination already exists but is not a stable regular file: {name}"
            ) from None
    else:
        destination_created = True
    try:
        if destination_created:
            offset = 0
            while offset < len(raw):
                written = os.write(destination_fd, raw[offset:])
                if written <= 0:
                    raise RuntimeError(f"processor metadata destination write stopped: {name}")
                offset += written
            os.fsync(destination_fd)
        destination = os.fstat(destination_fd)
        slot = os.stat(name, dir_fd=destination_parent_fd, follow_symlinks=False)
        copied = _read_processor_metadata(destination_fd, name)
        if (
            not stat.S_ISREG(destination.st_mode)
            or (destination_created and stat.S_IMODE(destination.st_mode) != 0o600)
            or _processor_metadata_identity(slot) != _processor_metadata_identity(destination)
            or copied != raw
            or hashlib.sha256(copied).digest() != hashlib.sha256(raw).digest()
        ):
            raise RuntimeError(f"processor metadata destination already exists with different bytes: {name}")
        return destination_fd, destination, raw, destination_created
    except BaseException:
        try:
            if destination_created:
                destination = os.fstat(destination_fd)
                slot = os.stat(name, dir_fd=destination_parent_fd, follow_symlinks=False)
                if _processor_metadata_identity(slot) == _processor_metadata_identity(destination):
                    os.unlink(name, dir_fd=destination_parent_fd)
        except OSError:
            pass
        finally:
            os.close(destination_fd)
        raise


def materialize_checkpoint_processor_metadata(ctx, checkpoint_path: str | Path) -> None:
    """Preserve exact offline Qwen3.5 processor JSON before publishing a local export."""
    source_path = getattr(ctx, "inference_initial_model_dir", None)
    if source_path is None:
        return

    from arctic_platform.correctness.harness.inference import validate_initial_training_model_dir

    source_path = validate_initial_training_model_dir(source_path)
    checkpoint_path = Path(checkpoint_path)
    source_parent_fd = os.open(source_path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        destination_parent_fd = os.open(checkpoint_path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except BaseException:
        os.close(source_parent_fd)
        raise
    destinations = []
    try:
        source_parent = os.fstat(source_parent_fd)
        if _processor_metadata_identity(os.stat(source_path, follow_symlinks=False)) != _processor_metadata_identity(
            source_parent
        ):
            raise RuntimeError("processor metadata source snapshot changed")
        destination_parent = _require_processor_metadata_directory(
            checkpoint_path, destination_parent_fd, "destination"
        )
        if (source_parent.st_dev, source_parent.st_ino) == (
            destination_parent.st_dev,
            destination_parent.st_ino,
        ):
            raise RuntimeError("processor metadata source and destination must differ")

        with ExitStack() as held:
            sources = []
            config_fd = _open_processor_metadata_file(source_parent_fd, "config.json", required=True)
            held.callback(os.close, config_fd)
            config_raw = _read_processor_metadata(config_fd, "config.json")
            config = json.loads(config_raw)
            config_metadata = os.fstat(config_fd)
            sources.append(("config.json", config_fd, config_metadata, config_raw))
            if config.get("model_type") != "qwen3_5_moe":
                if (
                    _processor_metadata_identity(os.stat(source_path, follow_symlinks=False))
                    != _processor_metadata_identity(source_parent)
                    or _processor_metadata_identity(os.fstat(config_fd))
                    != _processor_metadata_identity(config_metadata)
                    or _processor_metadata_identity(
                        os.stat("config.json", dir_fd=source_parent_fd, follow_symlinks=False)
                    )
                    != _processor_metadata_identity(config_metadata)
                    or _read_processor_metadata(config_fd, "config.json") != config_raw
                ):
                    raise RuntimeError("processor metadata source changed: config.json")
                return

            index_fd = _open_processor_metadata_file(source_parent_fd, "model.safetensors.index.json", required=True)
            held.callback(os.close, index_fd)
            index_raw = _read_processor_metadata(index_fd, "model.safetensors.index.json")
            sources.append(("model.safetensors.index.json", index_fd, os.fstat(index_fd), index_raw))
            processor_sources = []
            for position, name in enumerate(_PROCESSOR_METADATA_NAMES):
                descriptor = _open_processor_metadata_file(source_parent_fd, name, required=position == 0)
                if descriptor is not None:
                    held.callback(os.close, descriptor)
                    raw = _read_processor_metadata(descriptor, name)
                    sources.append((name, descriptor, os.fstat(descriptor), raw))
                    processor_sources.append((name, descriptor))

            for name, source_fd in processor_sources:
                destination_fd, destination_metadata, raw, destination_created = _copy_processor_metadata_file(
                    source_fd, destination_parent_fd, name
                )
                destinations.append((name, destination_fd, destination_metadata, raw, destination_created))

            if _processor_metadata_identity(
                os.stat(source_path, follow_symlinks=False)
            ) != _processor_metadata_identity(source_parent):
                raise RuntimeError("processor metadata source snapshot changed")
            for name, source_fd, source_metadata, raw in sources:
                if (
                    _processor_metadata_identity(os.fstat(source_fd)) != _processor_metadata_identity(source_metadata)
                    or _processor_metadata_identity(os.stat(name, dir_fd=source_parent_fd, follow_symlinks=False))
                    != _processor_metadata_identity(source_metadata)
                    or _read_processor_metadata(source_fd, name) != raw
                ):
                    raise RuntimeError(f"processor metadata source changed: {name}")
            for name, destination_fd, destination_metadata, raw, _ in destinations:
                try:
                    destination = os.fstat(destination_fd)
                    slot = os.stat(name, dir_fd=destination_parent_fd, follow_symlinks=False)
                    copied = _read_processor_metadata(destination_fd, name)
                except (OSError, RuntimeError, ValueError):
                    raise RuntimeError(f"processor metadata destination changed: {name}") from None
                if (
                    _processor_metadata_identity(destination) != _processor_metadata_identity(destination_metadata)
                    or _processor_metadata_identity(slot) != _processor_metadata_identity(destination_metadata)
                    or copied != raw
                    or hashlib.sha256(copied).digest() != hashlib.sha256(raw).digest()
                ):
                    raise RuntimeError(f"processor metadata destination changed: {name}")
            _require_processor_metadata_directory(checkpoint_path, destination_parent_fd, "destination")
    except BaseException:
        for name, destination_fd, _, _, destination_created in reversed(destinations):
            if not destination_created:
                continue
            try:
                destination = os.fstat(destination_fd)
                slot = os.stat(name, dir_fd=destination_parent_fd, follow_symlinks=False)
                if _processor_metadata_identity(slot) == _processor_metadata_identity(destination):
                    os.unlink(name, dir_fd=destination_parent_fd)
            except OSError:
                pass
        raise
    finally:
        for _, destination_fd, _, _, _ in destinations:
            try:
                os.close(destination_fd)
            except OSError:
                pass
        try:
            os.close(destination_parent_fd)
        finally:
            os.close(source_parent_fd)


def _export_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"local HF export requires a regular file: {path.name}")
    limit = _TOKENIZER_JSON_LIMIT_BYTES if path.name == "tokenizer.json" else _EXPORT_JSON_LIMIT_BYTES
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError(f"local HF export metadata exceeds its bound: {path.name}")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise RuntimeError(f"local HF export metadata must be an object: {path.name}")
    return value


def validate_local_inference_export(ctx, weights_dir: Path) -> Path:
    """Validate a complete HF export without loading or copying its tensor payloads."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.utils.hub import cached_file

    from arctic_platform.correctness.harness.inference import FAMILIES
    from arctic_platform.correctness.harness.inference import read_export_json
    from arctic_platform.correctness.harness.inference import read_inventory
    from arctic_platform.correctness.harness.inference import validate_glm53_full_config
    from arctic_platform.correctness.harness.inference import validate_glm53_full_schema
    from arctic_platform.correctness.harness.inference import validate_initial_training_model_dir
    from arctic_platform.correctness.harness.inference import validate_qwen4_local_export
    from arctic_platform.correctness.harness.inference import validate_text_projection

    canonical = require_data_fast_path(weights_dir, label="local inference HF export")
    if weights_dir != canonical or weights_dir.is_symlink() or not weights_dir.is_dir():
        raise RuntimeError("local inference requires a regular exported directory")
    weights_dir = canonical
    source_model = _full_model_name(ctx)
    initial_model_dir = getattr(ctx, "inference_initial_model_dir", None)
    if initial_model_dir is not None:
        source_model = str(validate_initial_training_model_dir(initial_model_dir))
    source = AutoConfig.from_pretrained(source_model, local_files_only=True)
    exported = _export_json(weights_dir / "config.json")
    if source.model_type in FAMILIES and exported.get("model_type") != source.model_type:
        source_config = require_data_fast_path(
            Path(cached_file(source_model, "config.json", local_files_only=True)),
            label="local full-model config",
        )
        source_index = require_data_fast_path(
            Path(cached_file(source_model, "model.safetensors.index.json", local_files_only=True)),
            label="local full-model index",
        )
        if source_config.parent != source_index.parent:
            raise RuntimeError("local full-model config and tensor index must belong to one snapshot")
        original_config = read_export_json(source_config)
        if original_config.get("model_type") != source.model_type:
            raise RuntimeError("local full-model configuration identity changed")
        original_schema = read_inventory(source_index.parent, read_export_json(source_index))
        exported_schema = read_inventory(weights_dir, read_export_json(weights_dir / "model.safetensors.index.json"))
        receipt = validate_text_projection(
            original_config, read_export_json(weights_dir / "config.json"), original_schema, exported_schema
        )
        if receipt is None:
            raise RuntimeError("local text-export configuration identity changed")
        return weights_dir
    if source.model_type == "glm5_next":
        source_config = require_data_fast_path(
            Path(cached_file(source_model, "config.json", local_files_only=True)),
            label="local full-model config",
        )
        source_index = require_data_fast_path(
            Path(cached_file(source_model, "model.safetensors.index.json", local_files_only=True)),
            label="local full-model index",
        )
        if source_config.parent != source_index.parent:
            raise RuntimeError("local full-model config and tensor index must belong to one snapshot")
        original_config = read_export_json(source_config)
        validate_glm53_full_config(original_config, read_export_json(weights_dir / "config.json"))
        quantization = original_config.get("quantization_config")
        original_schema = read_inventory(
            source_index.parent,
            read_export_json(source_index),
            original_quantization_config=quantization,
        )
        exported_schema = read_inventory(
            weights_dir,
            read_export_json(weights_dir / "model.safetensors.index.json"),
            original_quantization_config=quantization,
        )
        validate_glm53_full_schema(original_config, original_schema, exported_schema)
        return weights_dir
    for field in (
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "tie_word_embeddings",
    ):
        if exported.get(field) != getattr(source, field, None):
            raise RuntimeError(f"local HF export differs from the full configured model: {field}")
    if exported.get("auto_map"):
        raise RuntimeError("local HF export must not require remote code")
    source_index = cached_file(source_model, "model.safetensors.index.json", local_files_only=True)
    if source.model_type == "qwen4_exp":
        source_index = require_data_fast_path(Path(source_index), label="local full-model index")
        source_config = require_data_fast_path(
            Path(cached_file(source_model, "config.json", local_files_only=True)),
            label="local full-model config",
        )
        if source_config.parent != source_index.parent:
            raise RuntimeError("local full-model config and tensor index must belong to one snapshot")
        validate_qwen4_local_export(source_config, source_index, weights_dir)
        return weights_dir
    expected = _export_json(require_data_fast_path(Path(source_index), label="local full-model index"))
    index = _export_json(weights_dir / "model.safetensors.index.json")
    mapping = index.get("weight_map")
    expected_mapping = expected.get("weight_map")
    if not isinstance(mapping, dict) or not mapping or not isinstance(expected_mapping, dict) or not expected_mapping:
        raise RuntimeError(
            "local HF export does not contain the full configured tensor inventory: invalid or empty weight_map"
        )
    missing = sorted(set(expected_mapping) - set(mapping))
    unexpected = sorted(set(mapping) - set(expected_mapping))
    if missing or unexpected:
        raise RuntimeError(
            "local HF export does not contain the full configured tensor inventory: "
            f"missing={len(missing)} {missing[:8]!r}; unexpected={len(unexpected)} {unexpected[:8]!r}"
        )
    if any(
        not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors")
        for name in mapping.values()
    ):
        raise RuntimeError("local HF export index contains an invalid shard path")
    shards = set(mapping.values())
    if {path.name for path in weights_dir.glob("*.safetensors")} != shards:
        raise RuntimeError("local HF export weights inventory differs from its shard index")
    total_bytes = 0
    for name in sorted(shards):
        path = weights_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("local HF export requires regular contained weight shards")
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            if set(shard.keys()) != {tensor for tensor, file in mapping.items() if file == name}:
                raise RuntimeError("local HF export tensor names disagree with its shard index")
            for tensor in shard.keys():
                view = shard.get_slice(tensor)
                if view.get_dtype() != "BF16":
                    raise RuntimeError("local full-model HF export requires BF16 weights")
                total_bytes += math.prod(view.get_shape()) * 2
    original_metadata = expected.get("metadata")
    exported_metadata = index.get("metadata")
    expected_bytes = original_metadata.get("total_size") if isinstance(original_metadata, dict) else None
    exported_bytes = exported_metadata.get("total_size") if isinstance(exported_metadata, dict) else None
    # Original HF indexes can encode an exact integral byte count as a JSON float.
    if (
        total_bytes <= 0
        or type(expected_bytes) not in (int, float)
        or type(exported_bytes) not in (int, float)
        or expected_bytes != total_bytes
        or exported_bytes != total_bytes
    ):
        raise RuntimeError("local HF export tensor byte inventory differs from the full configured model")
    return weights_dir


def save_local_inference_checkpoint(ctx, backend: CortexConfig, client, tokenizer) -> Path:
    from arctic_platform.correctness.harness.inference import distribute_local_inference_export

    require_local_cortex_gateway(backend)
    root = require_data_fast_path(correctness_workdir("cortex-inference-"), label="local inference save")
    result = client.save_checkpoint(path=str(root), checkpoint_type="weights-only", export_hf=False)
    tag = _local_checkpoint_metadata(result, root, "save")
    if tag != f"global_step{TRAIN_STEPS}":
        raise RuntimeError("local inference save did not capture the completed training step")
    weights_dir = root / tag
    if weights_dir.is_symlink():
        raise RuntimeError("local inference export must not be a symlink")
    weights_dir = validate_local_inference_export(ctx, weights_dir)
    materialize_checkpoint_tokenizer(tokenizer, weights_dir)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        _export_json(weights_dir / name)
    restored = gsm8k.load_tokenizer(str(weights_dir))
    if restored.get_vocab() != tokenizer.get_vocab() or restored.special_tokens_map != tokenizer.special_tokens_map:
        raise RuntimeError("local inference export tokenizer differs from the configured tokenizer")
    materialize_checkpoint_processor_metadata(ctx, weights_dir)
    ctx._inference_export_locality = distribute_local_inference_export(
        weights_dir, getattr(ctx, "inference_export_allocation", None)
    )
    print("LOCAL_INFERENCE_CHECKPOINT", json.dumps(result, sort_keys=True), "HF_EXPORT", weights_dir, flush=True)
    return weights_dir


def local_cortex_sampling_config(ctx, backend: CortexConfig, weights_dir: Path, token_budget: int):
    from arctic_platform.correctness.harness.inference import require_local_inference_export_locality

    require_local_cortex_gateway(backend)
    weights_dir = validate_local_inference_export(ctx, weights_dir)
    require_local_inference_export_locality(ctx, weights_dir)
    payload = _sampling_payload(ctx, str(weights_dir), token_budget)
    payload["job_type"] = "sampling"
    config = _inference_client_config(payload, backend=backend)
    return CortexTrainingConfig.model_validate(
        {
            **config.model_dump(),
            "hardware": ctx.cfg.gpu_type.upper() if ctx.cfg.gpu_type is not None else None,
            **_cortex_sampling_parallelism(ctx),
        }
    )


def _run(ctx) -> TestResult:
    sampling_gpus = sampling_gpu_count(ctx)
    backend = getattr(ctx, "client_backend", None)
    backend = backend if backend is not None else OnPremConfig()
    managed = isinstance(backend, CortexConfig)
    local = managed and is_local_cortex_gateway(backend)
    if local:
        from arctic_platform.correctness.harness.inference import validate_local_inference_allocation

        require_local_cortex_gateway(backend)
        validate_local_inference_allocation(getattr(ctx, "inference_export_allocation", None))
    initial_model_dir = getattr(ctx, "inference_initial_model_dir", None)
    if initial_model_dir is not None:
        from arctic_platform.correctness.harness.inference import validate_initial_training_model_dir

        if not local:
            raise ValueError("initial training model directory requires local Cortex inference")
        initial_model_dir = validate_initial_training_model_dir(initial_model_dir)
    model_path = _full_model_name(ctx)
    input_model_path = model_path
    if initial_model_dir is not None:
        input_model_path = str(initial_model_dir)
    elif local and getattr(ctx, "local_gateway_full_model", False):
        from arctic_platform.correctness.harness.inference import validate_initial_training_model_dir

        input_model_path = str(validate_initial_training_model_dir(getattr(ctx, "local_full_model_path", None)))
    gateway_slots = (
        max(ctx.cfg.n_gpus, sampling_gpus)
        if getattr(ctx, "inference_sampling_topology", None) is not None
        else ctx.cfg.n_gpus + sampling_gpus
    )
    tokenizer = gsm8k.load_tokenizer(input_model_path)
    pad_id = gsm8k.padding_token_id(tokenizer)
    learning_rate = configured_learning_rate(ctx.cfg)
    rows_per_step = configured_rows_per_step(ctx.cfg)
    token_budget = correctness_microbatch_tokens(ctx.cfg.max_tokens_per_mb)

    training_examples = gsm8k.tokenize_examples(
        tokenizer, gsm8k.read_rows(TRAIN_SPLIT, count=TRAIN_STEPS * rows_per_step)
    )
    held_out = gsm8k.tokenize_examples(tokenizer, gsm8k.read_rows(VALIDATION_SPLIT))
    distribution = gsm8k.length_distribution(held_out)
    validation_examples, padded_width = gsm8k.select_by_token_budget(held_out, token_budget, ctx.cfg.dp_size)
    gsm8k.assert_disjoint(training_examples, validation_examples)

    validation_batch = gsm8k.build_batch("validation", validation_examples, pad_id)
    train_batches = training_batches(training_examples, rows_per_step, pad_id)
    minimum_row_tokens = _minimum_context_parallel_row_tokens(ctx, input_model_path)
    validation_batch = _extend_short_causal_rows(validation_batch, minimum_row_tokens, pad_id)
    train_batches = [_extend_short_causal_rows(batch, minimum_row_tokens, pad_id) for batch in train_batches]
    if minimum_row_tokens and validation_batch.total_tokens > token_budget:
        raise ValueError(
            "context-parallel validation padding exceeds the unchanged token budget; rows cannot be dropped"
        )

    workdir = Path(ctx.workdir) / TEST_ID
    # ``serve_gateway`` writes its hostfile into the directory it is handed and does not create it.
    gateway_dir = workdir / "gateway"
    gateway_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = workdir / "weights-only"
    if not managed:
        shutil.rmtree(checkpoint_root, ignore_errors=True)

    payload = build_payload(
        ctx.cfg.training,
        model_path,
        SEED,
        attn_implementation=ctx.attn_implementation,
        gradient_norms_per_param=False,
        weight_format=ctx.cfg.sub_job.get("weight_format"),
    )
    if managed and ctx.cfg.gpu_type is not None:
        payload["hardware"] = ctx.cfg.gpu_type.upper()
    if initial_model_dir is not None:
        payload["local_model_dir"] = str(initial_model_dir)
    provider = str(ctx.cfg.training.get("model_provider", "huggingface"))
    # The forward-only route is the one that does not shift, so the validation request carries logit
    # aligned labels while the training requests carry the convention the provider's dispatch expects.
    validation_body = pack_logit_aligned(validation_batch)
    if managed:
        # Cortex's forward operation always returns logprobs; the onprem request option is not accepted.
        validation_body["processing"].pop("return_logprobs")
    training_bodies = [pack(batch, model_provider=provider) for batch in train_batches]

    with gateway(gateway_dir, gateway_slots, backend=backend) as url:
        with running_job(url, payload) as training_job:
            losses = [
                fwd_bwd_step(url, training_job, body, learning_rate=learning_rate).avg_loss for body in training_bodies
            ]
            reference_rows = reference_answer_logprobs(
                forward_logprobs(url, training_job, validation_body), validation_examples
            )
            if local:
                weights_dir = save_local_inference_checkpoint(ctx, backend, training_job.client, tokenizer)
            elif managed:
                source_job_id = training_job.client.transport.job_id
                if not isinstance(source_job_id, str) or not source_job_id:
                    raise RuntimeError("managed training has no parent job identity for its checkpoint")
                checkpoint_id = save_and_resolve_checkpoint_id(training_job.client, source_job_id, "weights-only")
            else:
                weights_dir = save_weights_only_checkpoint(url, training_job, checkpoint_root)
                materialize_checkpoint_tokenizer(tokenizer, weights_dir)

        if managed:
            config = (
                local_cortex_sampling_config(ctx, backend, weights_dir, token_budget)
                if local
                else cortex_sampling_config(ctx, backend, checkpoint_id, source_job_id, token_budget)
            )
            with CorrectnessClient(config) as sampling_client:
                results = sampling_client.generate(
                    [example.prompt + example.completion for example in validation_examples],
                    sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": LOG_PROB_TOP_K},
                )
                target_rows = target_answer_logprobs(
                    cortex_prompt_logprobs(results, validation_examples), validation_examples
                )
        else:
            target_rows, reloaded_rows = _onprem_sampling_and_reload(
                ctx, url, weights_dir, token_budget, validation_examples, validation_body
            )

    reference_ce, reference_count = cross_entropy(reference_rows)
    target_ce, target_count = cross_entropy(target_rows)
    if not managed:
        reloaded_ce, reloaded_count = cross_entropy(reloaded_rows)
        print(
            "INFERENCE_DIAGNOSTIC "
            f"live_training={reference_ce:.9f} reloaded_training={reloaded_ce:.9f} "
            f"sampling={target_ce:.9f} live_reload_abs={abs(reference_ce - reloaded_ce):.9f} "
            f"reload_sampling_abs={abs(reloaded_ce - target_ce):.9f}",
            flush=True,
        )
        if reloaded_count != reference_count:
            raise RuntimeError(
                f"the reloaded training engine scored {reloaded_count} answer tokens, expected {reference_count}"
            )
    if reference_count != target_count:
        raise RuntimeError(
            f"the two sides scored different numbers of answer tokens: {reference_count} in the training "
            f"zone against {target_count} in the sampling zone"
        )

    measured: Dict[str, float] = {
        "validation_examples": float(len(validation_examples)),
        "validation_padded_width": float(padded_width),
        "validation_token_slots": float(len(validation_examples) * padded_width),
        "single_forward_token_budget": float(token_budget),
        "training_minimum_context_parallel_row_tokens": float(minimum_row_tokens),
        "training_validation_padded_width": float(validation_batch.seq_len if minimum_row_tokens else padded_width),
        "scored_answer_tokens": float(reference_count),
        "held_out_examples": float(distribution["examples"]),
        "held_out_minimum_tokens": float(distribution["minimum_tokens"]),
        "held_out_median_tokens": float(distribution["median_tokens"]),
        "held_out_maximum_tokens": float(distribution["maximum_tokens"]),
        "train_steps": float(TRAIN_STEPS),
        "train_rows_per_step": float(rows_per_step),
        "learning_rate": learning_rate,
        "train_loss_first_step": float(losses[0]),
        "train_loss_final_step": float(losses[-1]),
    }
    return verdict(ctx, reference_ce, target_ce, measured)


def _onprem_sampling_and_reload(ctx, session, weights_dir, token_budget, examples, validation_body):
    sampling = _sampling_payload(ctx, weights_dir, token_budget)
    with running_job(session, sampling) as sampling_job:
        target_rows = target_answer_logprobs(
            log_probs(
                session,
                sampling_job,
                [example.prompt for example in examples],
                [example.completion for example in examples],
                top_k=LOG_PROB_TOP_K,
            ),
            examples,
        )
    reload_payload = build_payload(
        ctx.cfg.training,
        weights_dir,
        SEED,
        attn_implementation=ctx.attn_implementation,
        gradient_norms_per_param=False,
        weight_format=ctx.cfg.sub_job.get("weight_format"),
    )
    with running_job(session, reload_payload) as reloaded_training_job:
        reloaded_rows = reference_answer_logprobs(
            forward_logprobs(session, reloaded_training_job, validation_body), examples
        )
    return target_rows, reloaded_rows


def tolerance_for(ctx) -> float:
    """Return the fixed gate for the accelerator family named by the config path."""
    return B200_TOLERANCE if ctx.cfg.gpu_type == "b200" else DEFAULT_TOLERANCE


def verdict(ctx, reference_ce: float, target_ce: float, measured: Dict[str, float] | None = None) -> TestResult:
    """The single compared quantity, against the fixed accelerator-family gate."""
    compared = Mismatch(name=QUANTITY, target=target_ce, reference=reference_ce)
    tolerance = tolerance_for(ctx)
    failed = compared.abs_diff > tolerance
    metrics: Dict[str, float] = {
        "reference_cross_entropy": reference_ce,
        "target_cross_entropy": target_ce,
        "max_abs_diff": compared.abs_diff,
        "stated_criterion_abs": tolerance,
    }
    metrics.update(measured or {})
    return TestResult(
        test_id=TEST_ID,
        config_id=ctx.config_id,
        outcome=TestOutcome.FAIL if failed else TestOutcome.PASS,
        summary=(
            f"{QUANTITY}: training zone {reference_ce:.6f}, sampling zone {target_ce:.6f}, "
            f"absolute difference {compared.abs_diff * 1e3:.3f}e-03 against a "
            f"{tolerance * 1e3:.3f}e-03 gate"
        ),
        mismatches=[compared] if failed else [],
        worst_name=QUANTITY,
        metrics=metrics,
    )
