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

"""A multi-step single-GPU SFT run and one held-out scoring: the golden truth for a Arctic Platform batch replay.

``hf_single_gpu.run`` executes one request. It builds a model, forward-backwards it once, optionally
applies one independent AdamW step, and frees the model, so a trajectory cannot be assembled by calling it
repeatedly: every call would start from the checkpoint again and Adam's moments would restart with it.
This module builds the engine once, drives ``forward_backward`` over a materialized list of per-step
batches, and holds one ``AdamWTrajectory`` across all of them, so step *k* reads the state the ``k - 1``
steps before it wrote.

The held-out score is per-position log-probabilities rather than a loss, because that is what the Arctic Platform
training job's forward-only route returns (``arctic_platform/common/deepspeed_worker.py:3031``). Producing the same
array here lets one reduction turn either engine's output into the compared number, so a disagreement
between the two cannot be a disagreement about how they were averaged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional

import torch

from .adamw_trajectory import AdamWTrajectory
from .hf_single_gpu import IGNORE_INDEX
from .hf_single_gpu import ReferenceEngine
from .hf_single_gpu import build_engine
from .hf_single_gpu import forward_backward
from .hf_single_gpu import mixer_boundaries
from .hf_single_gpu import reference_row_groups


@dataclass
class TrajectoryResult:
    """Every step's loss, the optimizer state that produced them, and the held-out scoring."""

    train_losses: List[float]
    train_gradient_norms: List[float]
    active_tokens: List[int]
    # One entry per logit position of every held-out row, in the shape the forward-only route returns.
    validation_logprobs: List[List[float]] = field(default_factory=list)
    optimizer_steps: int = 0
    trainable_parameters: int = 0
    peak_gib: float = 0.0


def positional_logprobs(
    engine: ReferenceEngine,
    batch,
    *,
    token_budget: int = 65536,
    ce_chunk: int = 2048,
    fp32_lm_head: bool = False,
    mixer_packing: bool = False,
    lm_head_token_chunk_size: int | None = None,
    lm_head_vocab_chunk_size: int = 8192,
) -> "torch.Tensor":
    """The log-probability of every logit-aligned label, shaped as the Arctic Platform forward-only route shapes it.

    Entry ``[b, s]`` is the log-probability the model assigns to row ``b``'s token at ``s + 1``. Positions
    whose label is ``IGNORE_INDEX`` -- the prompt and the padded tail -- hold zero and are never read; the
    caller slices each row's answer span out of this array, on both engines, with one function.

    No gradient and the engine in ``eval()``, which is how that route runs. The projection follows the
    config the same way the forward-backward path does: the engine's own chunked head when the config
    selects one, and otherwise a token-chunked ``F.linear`` whose operand width follows ``fp32_lm_head``.
    """
    import torch
    import torch.nn.functional as F

    from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs

    engine.model.eval()
    device = engine.device
    input_ids = batch.input_ids.to(device)
    labels = batch.shifted_labels().to(device)
    rows, width = int(input_ids.shape[0]), int(input_ids.shape[1])
    out = torch.zeros((rows, width), dtype=torch.float32)

    with torch.no_grad():
        groups = reference_row_groups([width] * rows, width, token_budget, mixer_packing=mixer_packing)
        for group in groups:
            index = torch.tensor(group, device=device)
            ids_mb = input_ids.index_select(0, index)
            labels_mb = labels.index_select(0, index)

            mixer_kwargs = {}
            if mixer_packing:
                rows_here, row_len = ids_mb.shape
                mixer_kwargs = mixer_boundaries(row_len, rows_here, device)
            hidden = engine.decoder(input_ids=ids_mb, use_cache=False, **mixer_kwargs).last_hidden_state

            if lm_head_token_chunk_size is not None:
                values = chunked_lm_head_logprobs(
                    hidden,
                    engine.head.weight,
                    labels_mb,
                    bias=getattr(engine.head, "bias", None),
                    token_chunk_size=lm_head_token_chunk_size,
                    vocab_chunk_size=lm_head_vocab_chunk_size,
                    fp32_lm_head=fp32_lm_head,
                ).float()
            else:
                flat_hidden = hidden.reshape(-1, engine.hidden_size)
                flat_labels = labels_mb.reshape(-1)
                flat = torch.zeros(flat_labels.shape, dtype=torch.float32, device=device)
                for start in range(0, flat_labels.numel(), ce_chunk):
                    chunk_labels = flat_labels[start : start + ce_chunk]
                    scored = chunk_labels != IGNORE_INDEX
                    if not bool(scored.any()):
                        continue
                    chunk_hidden = flat_hidden[start : start + ce_chunk]
                    # ``fp32_lm_head`` widens the projection's operands and not only its output, which is
                    # the same conversion the loss path applies.
                    if fp32_lm_head:
                        logits = F.linear(chunk_hidden.float(), engine.head.weight.float())
                    else:
                        logits = F.linear(chunk_hidden, engine.head.weight).float()
                    # ``IGNORE_INDEX`` is negative and cannot index the vocabulary, so it is clamped for
                    # the gather and masked out of the result.
                    gathered = (
                        torch.log_softmax(logits, dim=-1).gather(1, chunk_labels.clamp(min=0).unsqueeze(1)).squeeze(1)
                    )
                    flat[start : start + ce_chunk] = torch.where(scored, gathered, torch.zeros_like(gathered))
                values = flat.reshape(len(group), width)
            out[group] = values.cpu()
    return out


def run(
    model_path: str,
    train_batches: Iterable,
    validation_batch,
    *,
    optimizer_config: dict,
    learning_rate: float,
    gradient_clipping: float | None,
    optimizer_dtype: str = "float32",
    token_budget: int = 65536,
    ce_chunk: int = 2048,
    dtype: Optional[str] = "bfloat16",
    attn_implementation: str = "sdpa",
    fp32_lm_head: bool = False,
    fused_cross_entropy: bool | str = False,
    mixer_packing: bool = False,
    matmul_precision: str = "highest",
    peft_config: Optional[dict] = None,
    peft_adapter_path: str | None = None,
    seed: int = 0,
    lm_head_token_chunk_size: int | None = None,
    lm_head_vocab_chunk_size: int = 8192,
    optimizer_backend: str = "torch",
    device: str = "cuda",
) -> TrajectoryResult:
    """Train every batch in order through one engine and one optimizer, then score the held-out batch."""
    import torch

    engine = build_engine(
        model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        peft_config=peft_config,
        peft_adapter_path=peft_adapter_path,
        seed=seed,
        matmul_precision=matmul_precision,
        device=device,
    )
    optimizer = AdamWTrajectory(
        engine.model,
        optimizer_config,
        learning_rate=learning_rate,
        gradient_clipping=gradient_clipping,
        optimizer_dtype=optimizer_dtype,
        optimizer_backend=optimizer_backend,
    )
    torch.cuda.reset_peak_memory_stats(device)

    losses: List[float] = []
    norms: List[float] = []
    active: List[int] = []
    for batch in train_batches:
        forward = forward_backward(
            engine,
            batch,
            token_budget=token_budget,
            ce_chunk=ce_chunk,
            fp32_lm_head=fp32_lm_head,
            fused_cross_entropy=fused_cross_entropy,
            mixer_packing=mixer_packing,
            lm_head_token_chunk_size=lm_head_token_chunk_size,
            lm_head_vocab_chunk_size=lm_head_vocab_chunk_size,
        )
        losses.append(forward.loss)
        active.append(forward.active_tokens)
        norms.append(optimizer.step(forward.gradients))
        print(
            f"step {len(losses)} loss {forward.loss:.6f} grad_norm {norms[-1]:.6f} "
            f"active_tokens {forward.active_tokens}",
            flush=True,
        )
        # The float32 accumulator is as large as the trainable set, so it is released before the next
        # step allocates its own rather than being held for the length of the run.
        del forward

    validation = positional_logprobs(
        engine,
        validation_batch,
        token_budget=token_budget,
        ce_chunk=ce_chunk,
        fp32_lm_head=fp32_lm_head,
        mixer_packing=mixer_packing,
        lm_head_token_chunk_size=lm_head_token_chunk_size,
        lm_head_vocab_chunk_size=lm_head_vocab_chunk_size,
    )
    result = TrajectoryResult(
        train_losses=losses,
        train_gradient_norms=norms,
        active_tokens=active,
        validation_logprobs=validation.tolist(),
        optimizer_steps=optimizer.steps,
        trainable_parameters=len(optimizer.trainable_names),
        peak_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
    )
    del engine, optimizer
    torch.cuda.empty_cache()
    return result


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--replay",
        required=True,
        help=(
            "JSON naming the materialized per-step batch files, in order, and the "
            "held-out batch; both engines read these same files"
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--optimizer-config", required=True, help="JSON AdamW config; the trajectory holds one optimizer built from it"
    )
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--gradient-clipping", type=float)
    parser.add_argument("--optimizer-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--optimizer-backend", choices=("torch", "fused_adam"), default="torch")
    parser.add_argument("--token-budget", type=int, default=65536)
    parser.add_argument("--ce-chunk", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument(
        "--matmul-precision",
        default="highest",
        choices=("highest", "high", "medium"),
        help="float32 matmul precision; the engine's matmul_precision setting",
    )
    parser.add_argument(
        "--fp32-lm-head", dest="fp32_lm_head", action="store_true", help="apply the engine's fp32_lm_head behavior"
    )
    parser.add_argument(
        "--fused-cross-entropy",
        choices=("liger",),
        help="use the config's fused output projection and cross-entropy kernel",
    )
    parser.add_argument(
        "--mixer-packing",
        dest="mixer_packing",
        action="store_true",
        help="hand the gated delta net the sequence boundaries the engine hands it",
    )
    parser.add_argument("--lm-head-token-chunk", type=int)
    parser.add_argument("--lm-head-vocab-chunk", type=int, default=8192)
    parser.add_argument("--peft-config", help="JSON PEFT config to apply before the run")
    parser.add_argument("--peft-adapter", help="Arctic Platform-initialized PEFT adapter directory")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "pin reduction order so the trajectory repeats; the caller also sets "
            "CUBLAS_WORKSPACE_CONFIG, which has to be in the environment before CUDA "
            "initializes"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="arctic_platform.correctness.reference.sft_trajectory")
    _add_arguments(parser)
    args = parser.parse_args()

    if args.deterministic:
        from .process_determinism import pin_reduction_order

        pin_reduction_order(args.model, args.attn)

    from ..harness.batches import load

    replay = json.loads(Path(args.replay).read_text())
    result = run(
        args.model,
        # Loaded one at a time: the run holds the engine, an fp32 master copy and two Adam moments, and
        # every step's batch resident at once would add a copy of the whole replay to that.
        (load(Path(path)) for path in replay["train_batches"]),
        load(Path(replay["validation_batch"])),
        optimizer_config=json.loads(args.optimizer_config),
        learning_rate=args.learning_rate,
        gradient_clipping=args.gradient_clipping,
        optimizer_dtype=args.optimizer_dtype,
        optimizer_backend=args.optimizer_backend,
        token_budget=args.token_budget,
        ce_chunk=args.ce_chunk,
        dtype=args.dtype,
        attn_implementation=args.attn,
        fp32_lm_head=args.fp32_lm_head,
        fused_cross_entropy=args.fused_cross_entropy or False,
        mixer_packing=args.mixer_packing,
        matmul_precision=args.matmul_precision,
        peft_config=json.loads(args.peft_config) if args.peft_config else None,
        peft_adapter_path=args.peft_adapter,
        seed=args.seed,
        lm_head_token_chunk_size=args.lm_head_token_chunk,
        lm_head_vocab_chunk_size=args.lm_head_vocab_chunk,
    )
    Path(args.out).write_text(
        json.dumps(
            {
                "train_losses": result.train_losses,
                "train_gradient_norms": result.train_gradient_norms,
                "active_tokens": result.active_tokens,
                "validation_logprobs": result.validation_logprobs,
                "optimizer_steps": result.optimizer_steps,
                "trainable_parameters": result.trainable_parameters,
                "peak_gib": result.peak_gib,
            }
        )
    )


if __name__ == "__main__":
    main()
