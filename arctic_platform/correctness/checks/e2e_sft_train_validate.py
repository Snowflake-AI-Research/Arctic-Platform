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

"""A hundred SFT steps in Arctic Platform reproduce the training and validation losses of a single-GPU HF run.

The reference is the single-GPU Hugging Face engine and the target is Arctic Platform at the config's own topology,
the same roles the gradient and optimizer checks use. What is new is duration: one step compares two
forwards, and a hundred steps compares two *trajectories*, so the optimizer state each engine carries is
under test as well as the arithmetic of any one step.

Both engines read one materialized copy of the run's inputs. The per-step batches and the held-out batch
are written to disk before either engine starts, the reference process loads those files and the Arctic Platform
requests are packed from the same files, so "the two runs consumed identical inputs" is a property of the
bytes rather than of two code paths agreeing.

The verdict compares two numbers: the final step's training loss and the held-out validation loss. The
whole per-step trajectory is recorded from both engines and reported beside them as a diagnostic. It is
not gated, and it is not decoration either: a disagreement that is present at step 1 and one that first
appears at step 60 have disjoint causes -- the first is the forward, the second is what the steps did --
and the trajectory is the only thing that separates them once the final loss has failed.

The held-out loss is derived from per-position log-probabilities on both sides. The Arctic Platform training job's
forward-only route returns those and never a loss (``arctic_platform/common/deepspeed_worker.py:3031``), so the
reference produces the same array and one reduction turns either engine's output into the compared number.

That route also does not shift labels. ``/fwd-bwd`` dispatches with ``attach_global_loss_counts=True`` and
converts HuggingFace-convention labels to logit alignment before packing, which is why the training
requests here are packed by ``pack``; the forward-only route dispatches with
``attach_global_loss_counts=False`` and shifts nothing, so the held-out request is packed by
``pack_logit_aligned`` instead. An off-by-one there produces a plausible-looking wrong loss rather than an
obvious failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import torch

from arctic_platform.model.implementations.debug.determinism import CUBLAS_WORKSPACE

from ..harness import gsm8k
from ..harness.arms import correctness_microbatch_tokens
from ..harness.batches import Batch
from ..harness.batches import load
from ..harness.batches import save
from ..harness.dss_driver import build_payload
from ..harness.dss_driver import forward_logprobs
from ..harness.dss_driver import fwd_bwd_step
from ..harness.dss_driver import gateway
from ..harness.dss_driver import pack
from ..harness.dss_driver import pack_logit_aligned
from ..harness.dss_driver import running_job
from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test
from ..harness.seeds import SEED
from ..harness.spec import tolerance_text
from ..reference.model_features import uses_mixer_packing
from .inference_checkpoint_loss import TRAIN_SPLIT
from .inference_checkpoint_loss import VALIDATION_SPLIT
from .inference_checkpoint_loss import configured_learning_rate
from .inference_checkpoint_loss import configured_rows_per_step
from .inference_checkpoint_loss import cross_entropy
from .inference_checkpoint_loss import reference_answer_logprobs

TEST_ID = "e2e-sft-train-validate"

# The recipe length the criterion names: "~100 step SFT recipe with HF".
TRAIN_STEPS = 100

# The gate both compared losses must meet: an absolute difference of at most 3e-2.
#
# Fixed rather than calibrated, and held here the way the other fixed gates in this package are
# (``checkpoint_resume.TOLERANCE``, ``inference_checkpoint_loss.TOLERANCE``): the number is a stated
# criterion, not a measured spread, so onboarding calibrates nothing for it and no measurement this
# check takes may be used to move it.
TOLERANCE = 3e-2

TRAIN_LOSS_QUANTITY = "training loss at step {step}"
VALIDATION_QUANTITY = "validation cross entropy over answer tokens"

# The held-out loss is read once, after the last training step, and compared. Reading it periodically
# would insert forward-only passes between the training steps of both engines, which is a different run
# than the one being measured.
GATE_VALIDATION_LOSS = True


def gated_training_steps(recorded: int) -> Tuple[int, ...]:
    """Which of the recorded per-step training losses decide the verdict: the last one.

    Every step is recorded and every step's disagreement is reported. This selects the subset the outcome
    turns on, so widening it to ``tuple(range(1, recorded + 1))`` changes what fails without changing
    what is measured.
    """
    if recorded < 1:
        raise ValueError(f"a trajectory has at least one step, got {recorded}")
    return (recorded,)


@dataclass(frozen=True)
class Replay:
    """The one materialized copy of the run's inputs, which both engines read from disk.

    Files rather than in-memory batches, for the reason ``harness.batches`` exists: the reference runs in
    its own process and may hold different library versions, so loading identical bytes removes the
    question of whether two generators agreed.
    """

    train_batch_paths: List[Path]
    validation_batch_path: Path

    @property
    def steps(self) -> int:
        return len(self.train_batch_paths)


def replay_batches(examples: Sequence[gsm8k.Example], rows_per_step: int, steps: int, pad_id: int) -> List[Batch]:
    """One batch per step, consuming the training slice in order with no row used twice.

    A run that replayed a batch instead of advancing would reach step 100 on the wrong data, and both
    engines would do it identically, so the agreement would say nothing.
    """
    if len(examples) != steps * rows_per_step:
        raise ValueError(
            f"{steps} steps of {rows_per_step} rows need {steps * rows_per_step} examples, got {len(examples)}"
        )
    return [
        gsm8k.build_batch(
            f"train-step-{step:03d}", examples[(step - 1) * rows_per_step : step * rows_per_step], pad_id
        )
        for step in range(1, steps + 1)
    ]


def materialize_replay(train_batches: Sequence[Batch], validation_batch: Batch, directory: Path) -> Replay:
    """Write every batch once, before either engine starts, and name the files in step order."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for step, batch in enumerate(train_batches, start=1):
        path = directory / f"train-step-{step:03d}.pt"
        save(batch, path)
        paths.append(path)
    validation_path = directory / "validation.pt"
    save(validation_batch, validation_path)
    return Replay(train_batch_paths=paths, validation_batch_path=validation_path)


def request_bodies(replay: Replay, *, model_provider: str) -> Tuple[List[bytes], bytes]:
    """The Arctic Platform request bytes for the replay, read back from the files the reference reads.

    The training bodies carry the label convention the provider's dispatch expects, because ``/fwd-bwd``
    shifts them itself. The held-out body carries logit-aligned labels, because the forward-only route
    does not shift and the same bytes would otherwise score one position off.
    """
    training = [pack(load(path), model_provider=model_provider) for path in replay.train_batch_paths]
    return training, pack_logit_aligned(load(replay.validation_batch_path))


def write_replay_manifest(replay: Replay, path: Path) -> Path:
    """Name the replay's files, in step order, for the reference process to load.

    The reference reads this list and the Arctic Platform requests are packed from the same paths, so the two engines
    consuming identical inputs is a property of one list of filenames.
    """
    path.write_text(
        json.dumps(
            {
                "train_batches": [str(batch_path) for batch_path in replay.train_batch_paths],
                "validation_batch": str(replay.validation_batch_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def reference_optimizer_backend(cfg) -> str:
    """The backend the product resolves for this optimizer block.

    Qwen training defaults AdamW to DeepSpeed FusedAdam. An explicit torch name or ``fused: false`` keeps the
    independent PyTorch implementation; the trajectory must otherwise use the product backend because its scalar
    rounding compounds across every step.
    """
    optimizer = cfg.training["optimizer"]
    name = str(optimizer.get("name", "AdamW")).strip().lower().replace("-", "_")
    if name == "torch_adamw" or optimizer.get("fused") is False:
        return "torch"
    return "fused_adam"


def reference_command(
    ctx, replay: Replay, manifest: Path, out_path: Path, peft_adapter_path: Path | None
) -> List[str]:
    """The subprocess invocation for the single-GPU trajectory, built from the config as written.

    Every value here is read from the config or from the reviewed spec. The reference is a subprocess for
    the reason the one-step reference is: its CUDA context and allocator caches are reclaimed before the
    gateway claims the node's GPUs.
    """
    training = ctx.cfg.effective_training
    command = [
        sys.executable,
        "-m",
        "arctic_platform.correctness.reference.sft_trajectory",
        "--model",
        str(ctx.spec.model.cache_path),
        "--replay",
        str(manifest),
        "--out",
        str(out_path),
        "--seed",
        str(SEED),
        "--attn",
        str(ctx.attn_implementation),
        "--token-budget",
        str(correctness_microbatch_tokens(ctx.cfg.max_tokens_per_mb)),
        "--matmul-precision",
        str(training.get("matmul_precision", "highest")),
        "--dtype",
        str(ctx.cfg.sub_job.get("dtype") or training.get("dtype") or "bfloat16"),
        "--learning-rate",
        str(configured_learning_rate(ctx.cfg)),
        "--optimizer-config",
        json.dumps(ctx.cfg.training["optimizer"], sort_keys=True),
        "--optimizer-dtype",
        ctx.cfg.optimizer_dtype,
        "--optimizer-backend",
        reference_optimizer_backend(ctx.cfg),
        "--deterministic",
    ]
    if uses_mixer_packing(ctx.spec.model.cache_path):
        command.append("--mixer-packing")
    if training.get("fp32_lm_head", False):
        command.append("--fp32-lm-head")
    fused = ctx.cfg.fused_cross_entropy
    if fused:
        command += ["--fused-cross-entropy", "liger" if fused is True else str(fused)]
    if ctx.cfg.lm_head_token_chunk_size is not None:
        command += [
            "--lm-head-token-chunk",
            str(ctx.cfg.lm_head_token_chunk_size),
            "--lm-head-vocab-chunk",
            str(training.get("fused_lm_head_vocab_chunk_size", 8192)),
        ]
    clipping = ctx.cfg.training.get("gradient_clipping")
    if clipping is not None:
        command += ["--gradient-clipping", str(clipping)]
    if peft_adapter_path is not None:
        command += ["--peft-adapter", str(peft_adapter_path)]
    elif training.get("peft_config"):
        command += ["--peft-config", json.dumps(training["peft_config"], sort_keys=True)]
    return command


def run_reference_trajectory(ctx, replay: Replay, workdir: Path, peft_adapter_path: Path | None = None) -> dict:
    """The single-GPU trajectory's losses and held-out log-probabilities."""
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = write_replay_manifest(replay, workdir / "replay.json")
    out_path = workdir / "trajectory.json"
    command = reference_command(ctx, replay, manifest, out_path, peft_adapter_path)
    # Expandable segments for the reason the one-step reference uses them: the margin left for activations
    # beside an fp32 master copy and two Adam moments is thin, and the default allocator spends it on
    # fragmentation. The workspace string has to be in the environment before CUDA initializes.
    env = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE,
    }
    # Streamed rather than captured, because this subprocess runs a hundred steps: buffered output tells
    # the reader nothing until it exits, and a run that is killed part way through leaves no evidence at
    # all. The lines are echoed as they arrive and kept so a failure still carries its own tail.
    transcript: List[str] = []
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
    ) as process:
        for line in process.stdout:
            transcript.append(line)
            print(f"  [reference] {line.rstrip()}", flush=True)
        returncode = process.wait()
    if returncode != 0:
        tail = "".join(transcript[-60:])
        raise RuntimeError(f"reference trajectory failed ({returncode}):\n{tail}")
    payload = json.loads(out_path.read_text())
    if len(payload["train_losses"]) != replay.steps:
        raise RuntimeError(
            f"the reference reported {len(payload['train_losses'])} step losses for a {replay.steps}-step replay"
        )
    if payload["optimizer_steps"] != replay.steps:
        raise RuntimeError(
            f"the reference took {payload['optimizer_steps']} optimizer steps over a "
            f"{replay.steps}-step replay; one optimizer must carry the whole trajectory"
        )
    return payload


def run_dss_trajectory(ctx, replay: Replay, workdir: Path) -> Tuple[List[float], "torch.Tensor"]:
    """The same replay through Arctic Platform at the config's own topology, then the held-out forward.

    One job for the whole trajectory: a job restarted mid-run would reload the checkpoint, and the point
    of a hundred steps is that step 100 stands on the ninety-nine before it.
    """
    provider = str(ctx.cfg.training.get("model_provider", "huggingface"))
    training_bodies, validation_body = request_bodies(replay, model_provider=provider)
    learning_rate = configured_learning_rate(ctx.cfg)
    payload = build_payload(
        ctx.cfg.training, ctx.spec.model.cache_path, SEED, attn_implementation=ctx.attn_implementation
    )
    # ``serve_gateway`` writes its hostfile into the directory it is handed and does not create it.
    gateway_dir = workdir / "gateway"
    gateway_dir.mkdir(parents=True, exist_ok=True)
    with gateway(gateway_dir, ctx.cfg.n_gpus) as url:
        with running_job(url, payload) as job_id:
            losses = [
                fwd_bwd_step(url, job_id, body, learning_rate=learning_rate).avg_loss for body in training_bodies
            ]
            return losses, forward_logprobs(url, job_id, validation_body)


def trajectory_comparisons(reference: Sequence[float], target: Sequence[float]) -> List[Tuple[int, Mismatch]]:
    """Every recorded step as a compared quantity, in step order, paired with its step number.

    All of them, whether or not they gate. The onset of a disagreement is read off this list, and a list
    that held only the gated step could not distinguish a defect in the first forward from one the
    optimizer steps accumulated.
    """
    if len(reference) != len(target):
        raise ValueError(
            f"the two trajectories recorded {len(reference)} and {len(target)} step losses; a "
            "step-by-step comparison needs one loss per step from each engine"
        )
    return [
        (
            step,
            Mismatch(
                name=TRAIN_LOSS_QUANTITY.format(step=step),
                target=float(target[step - 1]),
                reference=float(reference[step - 1]),
            ),
        )
        for step in range(1, len(reference) + 1)
    ]


def gated_comparisons(trajectory: Sequence[Tuple[int, Mismatch]], validation: Mismatch) -> List[Mismatch]:
    """The quantities the outcome turns on: the selected training step(s), then the validation loss."""
    selected = set(gated_training_steps(len(trajectory)))
    compared = [mismatch for step, mismatch in trajectory if step in selected]
    if GATE_VALIDATION_LOSS:
        compared.append(validation)
    return compared


def divergence_reason(failing_steps: Sequence[int], recorded: int) -> str:
    """Where in the trajectory the disagreement starts, and what that placement rules out.

    Read from the whole trajectory rather than from the gated step, because the gated step is the last
    one and by itself says only that the two runs ended apart. ``failing_steps`` is every step whose two
    losses differ by more than the gate, gated or not.
    """
    if not failing_steps:
        return ""
    onset = min(failing_steps)
    if onset == 1:
        return (
            "The two engines already disagree at step 1, before either has taken an optimizer step, so "
            "this is a disagreement in the first forward and not one that accumulated. Both engines read "
            "the same materialized batch files, so the inputs are excluded and what remains is the "
            "forward itself: the loss denominator, the label alignment, or the weights the two engines "
            "loaded. Later steps carry this one forward and say nothing further, which is why the final "
            "step's disagreement is not evidence of anything the optimizer did."
        )
    elapsed = onset - 1
    return (
        f"The two engines agree at step 1 and first disagree at step {onset} of {recorded}, so the "
        f"disagreement accumulated over {elapsed} optimizer "
        f"{'step' if elapsed == 1 else 'steps'} rather than being present in the "
        "first forward. Step 1 already agreed, which excludes the batches and the label alignment and "
        "leaves what the steps did: the optimizer state, the gradient reduction, or a reduction order "
        "neither engine has pinned."
    )


def verdict(
    ctx,
    reference_losses: Sequence[float],
    target_losses: Sequence[float],
    reference_validation: float,
    target_validation: float,
    measured: Dict[str, float] | None = None,
) -> TestResult:
    """The gated quantities against the fixed gate, with the whole trajectory reported beside them."""
    trajectory = trajectory_comparisons(reference_losses, target_losses)
    validation = Mismatch(name=VALIDATION_QUANTITY, target=target_validation, reference=reference_validation)
    compared = gated_comparisons(trajectory, validation)

    failures = sorted(
        (mismatch for mismatch in compared if mismatch.abs_diff > TOLERANCE),
        key=lambda mismatch: mismatch.abs_diff,
        reverse=True,
    )
    worst = max(compared, key=lambda mismatch: mismatch.abs_diff)
    # Diagnostic, over every step rather than the gated one. None of it decides the outcome. The first
    # step and the widest step are reported whatever the verdict, because together they separate a
    # disagreement the two runs started with from one they accumulated: a first step inside the noise with a
    # wide step later means the runs began as the same computation and parted during it.
    drifting = [step for step, mismatch in trajectory if mismatch.abs_diff > TOLERANCE]
    worst_step, worst_step_mismatch = max(trajectory, key=lambda pair: pair[1].abs_diff)
    first_step_mismatch = trajectory[0][1]
    final_training = trajectory[-1][1]

    metrics: Dict[str, float] = {
        "max_abs_diff": worst.abs_diff,
        "stated_criterion_abs": TOLERANCE,
        "gated_quantities": float(len(compared)),
        "final_train_loss_abs_diff": final_training.abs_diff,
        "validation_abs_diff": validation.abs_diff,
        "train_steps_recorded": float(len(trajectory)),
        "trajectory_steps_over_gate": float(len(drifting)),
        "first_disagreeing_train_step": float(min(drifting)) if drifting else 0.0,
        "worst_train_loss_abs_diff": worst_step_mismatch.abs_diff,
        "worst_train_loss_step": float(worst_step),
        "first_train_loss_abs_diff": first_step_mismatch.abs_diff,
        "reference_train_loss_first_step": float(reference_losses[0]),
        "target_train_loss_first_step": float(target_losses[0]),
        "reference_train_loss_final_step": float(reference_losses[-1]),
        "target_train_loss_final_step": float(target_losses[-1]),
        "reference_validation_cross_entropy": reference_validation,
        "target_validation_cross_entropy": target_validation,
    }
    # The per-step disagreement for every recorded step, which is what locates the onset.
    for step, mismatch in trajectory:
        metrics[f"train_loss_abs_diff_step_{step:03d}"] = mismatch.abs_diff
    metrics.update(measured or {})

    return TestResult(
        test_id=TEST_ID,
        config_id=ctx.config_id,
        outcome=TestOutcome.PASS if not failures else TestOutcome.FAIL,
        summary=(
            f"{len(failures)}/{len(compared)} gated losses failed to match "
            f"{tolerance_text(TOLERANCE)}: final training loss "
            f"{final_training.abs_diff * 1e3:.3f}e-03, validation "
            f"{validation.abs_diff * 1e3:.3f}e-03; step 1 "
            f"{first_step_mismatch.abs_diff * 1e3:.3f}e-03 and widest of the {len(trajectory)} "
            f"recorded steps {worst_step_mismatch.abs_diff * 1e3:.3f}e-03 at step {worst_step}, "
            "neither gated"
        ),
        reason=divergence_reason(drifting, len(trajectory)),
        mismatches=failures,
        worst_name=worst.name,
        metrics=metrics,
    )


@correctness_test(
    TEST_ID,
    title="Training and validation loss agreement over a 100-step SFT run replayed in Arctic Platform",
    criterion=(
        "the final training loss and the held-out validation loss of a 100-step SFT run agree between "
        "Arctic Platform and the single-GPU Hugging Face reference on the same replayed batches; the whole per-step "
        "trajectory is recorded and reported as a diagnostic"
    ),
)
def run(ctx) -> List[TestResult]:
    """One reference process and one Arctic Platform job, both over one materialized replay.

    No case axis. The cases vary gradient-accumulation depth in one step, and this check varies step
    count instead; a second case would run the same trajectory at another accumulation depth, which is
    what the one-step checks already measure.
    """
    try:
        return [_run(ctx)]
    except Exception as exc:  # noqa: BLE001 - a failure here is this check's verdict, not a crash
        return [
            TestResult(
                test_id=TEST_ID,
                config_id=ctx.config_id,
                outcome=TestOutcome.FAIL,
                summary=f"{type(exc).__name__}: {exc}",
            )
        ]


def _run(ctx) -> TestResult:
    import torch

    model_path = ctx.spec.model.cache_path
    tokenizer = gsm8k.load_tokenizer(model_path)
    pad_id = gsm8k.padding_token_id(tokenizer)
    rows_per_step = configured_rows_per_step(ctx.cfg)
    learning_rate = configured_learning_rate(ctx.cfg)
    token_budget = correctness_microbatch_tokens(ctx.cfg.max_tokens_per_mb)

    training_examples = gsm8k.tokenize_examples(
        tokenizer, gsm8k.read_rows(TRAIN_SPLIT, count=TRAIN_STEPS * rows_per_step)
    )
    held_out = gsm8k.tokenize_examples(tokenizer, gsm8k.read_rows(VALIDATION_SPLIT))
    distribution = gsm8k.length_distribution(held_out)
    validation_examples, padded_width = gsm8k.select_by_token_budget(held_out, token_budget, ctx.cfg.dp_size)
    gsm8k.assert_disjoint(training_examples, validation_examples)

    workdir = Path(ctx.workdir) / TEST_ID
    replay = materialize_replay(
        replay_batches(training_examples, rows_per_step, TRAIN_STEPS, pad_id),
        gsm8k.build_batch("validation", validation_examples, pad_id),
        workdir / "replay",
    )

    # A LoRA config's adapter is initialized by Arctic Platform and exported so the reference starts from identical
    # bytes, which is what the one-step reference comparison does. Without it the two engines would train
    # from different adapters and the trajectories would part at step 1 for a reason that is not a defect.
    peft_adapter_path = None
    if ctx.cfg.training.get("peft_config"):
        from ..harness.runner import materialize_dss_peft_adapter

        peft_adapter_path = materialize_dss_peft_adapter(
            ctx.cfg, model_path, workdir / "reference-adapter-init", attn_implementation=ctx.attn_implementation
        )

    reference = run_reference_trajectory(ctx, replay, workdir / "reference", peft_adapter_path)
    target_losses, target_logprobs = run_dss_trajectory(ctx, replay, workdir / "dss")

    reference_ce, reference_count = cross_entropy(
        reference_answer_logprobs(
            torch.as_tensor(reference["validation_logprobs"], dtype=torch.float32), validation_examples
        )
    )
    target_ce, target_count = cross_entropy(reference_answer_logprobs(target_logprobs, validation_examples))
    if reference_count != target_count:
        raise RuntimeError(
            f"the two engines scored different numbers of answer tokens: {reference_count} in the "
            f"single-GPU reference against {target_count} in Arctic Platform"
        )

    reference_losses = [float(value) for value in reference["train_losses"]]
    (workdir / "trajectories.json").write_text(
        json.dumps(
            {
                "reference_train_losses": reference_losses,
                "target_train_losses": [float(value) for value in target_losses],
                "reference_train_gradient_norms": reference["train_gradient_norms"],
                "reference_validation_cross_entropy": reference_ce,
                "target_validation_cross_entropy": target_ce,
            },
            indent=2,
        )
        + "\n"
    )

    measured: Dict[str, float] = {
        "train_rows_per_step": float(rows_per_step),
        "learning_rate": learning_rate,
        "n_gpus_dss": float(ctx.cfg.n_gpus),
        "validation_examples": float(len(validation_examples)),
        "validation_padded_width": float(padded_width),
        "validation_token_slots": float(len(validation_examples) * padded_width),
        "single_forward_token_budget": float(token_budget),
        "scored_answer_tokens": float(reference_count),
        "held_out_examples": float(distribution["examples"]),
        "held_out_minimum_tokens": float(distribution["minimum_tokens"]),
        "held_out_median_tokens": float(distribution["median_tokens"]),
        "held_out_maximum_tokens": float(distribution["maximum_tokens"]),
        "reference_peak_gib": float(reference["peak_gib"]),
        "reference_trainable_parameters": float(reference["trainable_parameters"]),
    }
    return verdict(ctx, reference_losses, target_losses, reference_ce, target_ce, measured)
