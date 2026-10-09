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

import os
import shutil
import traceback
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import reserve_free_port

from ..harness import gsm8k
from ..harness.arms import correctness_microbatch_tokens
from ..harness.batches import Batch
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
from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test
from ..harness.seeds import SEED

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

# The sampling zone runs beside the training zone's GPUs rather than on them, so the gateway is asked for
# the config's GPU count plus these. One GPU is enough: the zone holds the reduced model and runs a single
# prefill over a batch that the config's own single-forward token budget bounds.
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
    rows: List[List[float]] = []
    for index, example in enumerate(examples):
        span = [float(value) for value in logprobs[index, example.n_prompt - 1 : example.length - 1].tolist()]
        if len(span) != example.answer_tokens:
            raise RuntimeError(
                f"row {index}: the forward-only response covers {len(span)} of the row's "
                f"{example.answer_tokens} answer tokens"
            )
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
    rows: List[List[float]] = []
    for index, (result, example) in enumerate(zip(results, examples)):
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
        rows.append(values[example.n_prompt - 1 : example.length - 1])
    return rows


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


def _run(ctx) -> TestResult:
    model_path = ctx.spec.model.cache_path
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
    shutil.rmtree(checkpoint_root, ignore_errors=True)

    payload = build_payload(
        ctx.cfg.training,
        model_path,
        SEED,
        attn_implementation=ctx.attn_implementation,
        gradient_norms_per_param=False,
    )
    provider = str(ctx.cfg.training.get("model_provider", "huggingface"))
    # The forward-only route is the one that does not shift, so the validation request carries logit
    # aligned labels while the training requests carry the convention the provider's dispatch expects.
    validation_body = pack_logit_aligned(validation_batch)
    training_bodies = [pack(batch, model_provider=provider) for batch in train_batches]

    with gateway(gateway_dir, ctx.cfg.n_gpus + SAMPLING_GPUS) as url:
        with running_job(url, payload) as training_job:
            losses = [
                fwd_bwd_step(url, training_job, body, learning_rate=learning_rate).avg_loss for body in training_bodies
            ]
            reference_rows = reference_answer_logprobs(
                forward_logprobs(url, training_job, validation_body), validation_examples
            )
            weights_dir = save_weights_only_checkpoint(url, training_job, checkpoint_root)
            materialize_checkpoint_tokenizer(tokenizer, weights_dir)

        sampling = sampling_payload(
            weights_dir,
            SEED,
            dtype=str(ctx.cfg.sub_job.get("dtype") or ctx.cfg.training.get("dtype") or "bfloat16"),
            n_gpus=SAMPLING_GPUS,
            # One prefill of the widest batch the budget allows, plus the single token ``log_probs`` asks
            # the engine to generate beside it (arctic_platform/common/ray_server.py:695).
            max_seq_len=token_budget + 1,
        )
        with running_job(url, sampling) as sampling_job:
            target_rows = target_answer_logprobs(
                log_probs(
                    url,
                    sampling_job,
                    [example.prompt for example in validation_examples],
                    [example.completion for example in validation_examples],
                    top_k=LOG_PROB_TOP_K,
                ),
                validation_examples,
            )

        reload_payload = build_payload(
            ctx.cfg.training,
            weights_dir,
            SEED,
            attn_implementation=ctx.attn_implementation,
            gradient_norms_per_param=False,
        )
        reload_master_port = reserve_free_port(get_unique_port_number() + 1, span=7)
        previous_master_port = os.environ.get("MASTER_PORT")
        os.environ["MASTER_PORT"] = str(reload_master_port)
        try:
            with running_job(url, reload_payload) as reloaded_training_job:
                reloaded_rows = reference_answer_logprobs(
                    forward_logprobs(url, reloaded_training_job, validation_body), validation_examples
                )
        finally:
            if previous_master_port is None:
                os.environ.pop("MASTER_PORT", None)
            else:
                os.environ["MASTER_PORT"] = previous_master_port

    reference_ce, reference_count = cross_entropy(reference_rows)
    reloaded_ce, reloaded_count = cross_entropy(reloaded_rows)
    target_ce, target_count = cross_entropy(target_rows)
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
