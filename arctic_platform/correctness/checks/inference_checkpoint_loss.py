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

import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig

from ..harness import gsm8k
from ..harness.arms import correctness_microbatch_tokens
from ..harness.batches import Batch
from ..harness.cortex import CorrectnessClient
from ..harness.cortex import CortexTrainingConfig
from ..harness.dss_driver import _inference_client_config
from ..harness.dss_driver import build_payload
from ..harness.dss_driver import forward_logprobs
from ..harness.dss_driver import fwd_bwd_step
from ..harness.dss_driver import gateway
from ..harness.dss_driver import log_probs
from ..harness.dss_driver import pack
from ..harness.dss_driver import pack_logit_aligned
from ..harness.dss_driver import running_job
from ..harness.dss_driver import sampling_payload
from ..harness.dss_driver import save_weights_only_checkpoint
from ..harness.hosted import _local_checkpoint_metadata
from ..harness.hosted import is_local_cortex_gateway
from ..harness.hosted import require_local_cortex_gateway
from ..harness.hosted import save_and_resolve_checkpoint_id
from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test
from ..harness.seeds import SEED
from ..harness.workdir import correctness_workdir
from ..harness.workdir import require_data_fast_path

TEST_ID = "inference-checkpoint-loss"

TRAIN_SPLIT = Path("/code/shared/gsm8k/train.parquet")
VALIDATION_SPLIT = Path("/code/shared/gsm8k/test.parquet")

TRAIN_STEPS = 10

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

# The sampling zone uses a separate GPU and a bounded prefill batch. The full configured model must fit
# this serving placement; a memory failure must not silently substitute reduced weights.
SAMPLING_GPUS = 1

QUANTITY = "validation cross entropy over answer tokens"


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
            )
        ]


def configured_learning_rate(cfg) -> float:
    """The rate the config's own optimizer block declares.

    Not the check's to choose. Ten steps at another rate reach a different set of weights and therefore a
    different validation loss than the operator's job would reach, and the point of the comparison is that
    both engines read the weights this config produces.
    """
    optimizer = cfg.training.get("optimizer")
    rate = optimizer.get("lr") if isinstance(optimizer, dict) else None
    if rate is None:
        raise ValueError(
            f"{cfg.config_id}: the training config declares no optimizer learning rate. This check trains "
            "the config as written, so there is nothing to run rather than a rate to pick."
        )
    return float(rate)


def configured_rows_per_step(cfg) -> int:
    """The global batch size the config trains at, which is how many dataset rows a step consumes."""
    rows = cfg.training.get("train_batch_size")
    if rows is None:
        raise ValueError(f"{cfg.config_id}: the training config declares no train_batch_size")
    return int(rows)


def cross_entropy(per_row_answer_logprobs: Sequence[Sequence[float]]) -> Tuple[float, int]:
    """Negative mean log-probability over every answer token in the batch.

    The one reduction both sides use. Each side is responsible only for producing, per row, that row's
    answer-token log-probabilities in order; the sign, the denominator and the pooling across rows are
    decided here, once.
    """
    total = 0.0
    count = 0
    for row in per_row_answer_logprobs:
        for value in row:
            total += float(value)
            count += 1
    if count == 0:
        raise ValueError("no answer tokens were scored; a cross entropy over nothing is not a measurement")
    return -total / count, count


def reference_answer_logprobs(logprobs, examples: Sequence[gsm8k.Example]) -> List[List[float]]:
    """The training zone's log-probabilities over each row's answer span.

    The route returns one entry per position, scoring the label at that position. The labels it was sent
    are logit aligned, so the entry at ``s`` is the log-probability of the token at ``s + 1``. A row of
    length ``L`` whose answer begins at ``p`` therefore has its answer scored by entries ``p - 1`` through
    ``L - 2``.
    """
    if logprobs.ndim != 2 or logprobs.shape[0] != len(examples):
        raise RuntimeError("forward-only logprobs must have one two-dimensional row per validation example")
    rows: List[List[float]] = []
    for index, example in enumerate(examples):
        _assert_answer_span(example, index)
        span = [float(value) for value in logprobs[index, example.n_prompt - 1 : example.length - 1].tolist()]
        if len(span) != example.answer_tokens:
            raise RuntimeError(
                f"row {index}: the forward-only response covers {len(span)} of the row's "
                f"{example.answer_tokens} answer tokens"
            )
        if not all(math.isfinite(value) for value in span):
            raise RuntimeError(f"row {index}: forward-only answer log-probabilities must be finite")
        rows.append(span)
    return rows


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


def _assert_answer_span(example: gsm8k.Example, index: int) -> None:
    if not 1 <= example.n_prompt < example.length:
        raise RuntimeError(f"row {index}: answer span must follow a prompt token and contain at least one token")


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
    payload = sampling_payload(
        model_name,
        SEED,
        dtype=str(ctx.cfg.sub_job.get("dtype") or ctx.cfg.training.get("dtype") or "bfloat16"),
        n_gpus=SAMPLING_GPUS,
        max_seq_len=token_budget + 1,
    )
    payload["job_type"] = "sampling"
    config = _inference_client_config(payload, backend=backend)
    return CortexTrainingConfig.model_validate(
        {
            **config.model_dump(),
            "hardware": ctx.cfg.gpu_type.upper() if ctx.cfg.gpu_type is not None else None,
            "source_checkpoint_info": {"checkpoint_id": checkpoint_id, "source_job_id": source_job_id},
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


def materialize_checkpoint_tokenizer(tokenizer, checkpoint_path: str | Path) -> None:
    """Store the validated model tokenizer beside an exported weights-only checkpoint."""
    tokenizer.save_pretrained(checkpoint_path)


def _export_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"local HF export requires a regular file: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
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

    canonical = require_data_fast_path(weights_dir, label="local inference HF export")
    if weights_dir != canonical or weights_dir.is_symlink() or not weights_dir.is_dir():
        raise RuntimeError("local inference requires a regular exported directory")
    weights_dir = canonical
    source = AutoConfig.from_pretrained(_full_model_name(ctx), local_files_only=True)
    exported = _export_json(weights_dir / "config.json")
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
    source_index = cached_file(_full_model_name(ctx), "model.safetensors.index.json", local_files_only=True)
    expected = _export_json(require_data_fast_path(Path(source_index), label="local full-model index"))
    index = _export_json(weights_dir / "model.safetensors.index.json")
    mapping = index.get("weight_map")
    expected_mapping = expected.get("weight_map")
    if (
        not isinstance(mapping, dict)
        or not mapping
        or not isinstance(expected_mapping, dict)
        or set(mapping) != set(expected_mapping)
    ):
        raise RuntimeError("local HF export does not contain the full configured tensor inventory")
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
    expected_bytes = (expected.get("metadata") or {}).get("total_size")
    if (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or (total_bytes != expected_bytes or (index.get("metadata") or {}).get("total_size") != total_bytes)
    ):
        raise RuntimeError("local HF export tensor byte inventory differs from the full configured model")
    return weights_dir


def save_local_inference_checkpoint(ctx, backend: CortexConfig, client, tokenizer) -> Path:
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
    print("LOCAL_INFERENCE_CHECKPOINT", json.dumps(result, sort_keys=True), "HF_EXPORT", weights_dir, flush=True)
    return weights_dir


def local_cortex_sampling_config(ctx, backend: CortexConfig, weights_dir: Path, token_budget: int):
    require_local_cortex_gateway(backend)
    weights_dir = validate_local_inference_export(ctx, weights_dir)
    payload = sampling_payload(
        str(weights_dir),
        SEED,
        dtype=str(ctx.cfg.sub_job.get("dtype") or ctx.cfg.training.get("dtype") or "bfloat16"),
        n_gpus=SAMPLING_GPUS,
        max_seq_len=token_budget + 1,
    )
    payload["job_type"] = "sampling"
    config = _inference_client_config(payload, backend=backend)
    return CortexTrainingConfig.model_validate(
        {
            **config.model_dump(),
            "hardware": ctx.cfg.gpu_type.upper() if ctx.cfg.gpu_type is not None else None,
        }
    )


def _run(ctx) -> TestResult:
    backend = getattr(ctx, "client_backend", None)
    backend = backend if backend is not None else OnPremConfig()
    managed = isinstance(backend, CortexConfig)
    local = managed and is_local_cortex_gateway(backend)
    if local:
        require_local_cortex_gateway(backend)
    model_path = _full_model_name(ctx)
    tokenizer = gsm8k.load_tokenizer(model_path)
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
    )
    if managed and ctx.cfg.gpu_type is not None:
        payload["hardware"] = ctx.cfg.gpu_type.upper()
    provider = str(ctx.cfg.training.get("model_provider", "huggingface"))
    # The forward-only route is the one that does not shift, so the validation request carries logit
    # aligned labels while the training requests carry the convention the provider's dispatch expects.
    validation_body = pack_logit_aligned(validation_batch)
    if managed:
        # Cortex's forward operation always returns logprobs; the onprem request option is not accepted.
        validation_body["processing"].pop("return_logprobs")
    training_bodies = [pack(batch, model_provider=provider) for batch in train_batches]

    with gateway(gateway_dir, ctx.cfg.n_gpus + SAMPLING_GPUS, backend=backend) as url:
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
    sampling = sampling_payload(
        weights_dir,
        SEED,
        dtype=str(ctx.cfg.sub_job.get("dtype") or ctx.cfg.training.get("dtype") or "bfloat16"),
        n_gpus=SAMPLING_GPUS,
        max_seq_len=token_budget + 1,
    )
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
