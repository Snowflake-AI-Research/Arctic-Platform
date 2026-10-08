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

"""Router replay: the sampling zone's routing is the routing every training-zone MoE router uses.

The sampling zone generates one answer per GSM8K prompt with router replay on and returns, beside each answer,
the expert indices its routers chose at every captured position (``dss_return_back_router_info``). The training
zone then runs one RL fwd-bwd over those same rows, replaying the sampler's routing, with
``debug.router_replay_trace`` recording on every rank:

- the routing each sample's replay id delivered from the sampler's cache;
- for every model call, the packed tokens, their positions, and the ``routed_experts`` tensor handed to the
  model;
- every router invocation, by decoder layer, on the forward pass and inside backward, where activation
  checkpointing recomputes the layer: whether it replayed supplied routing, and the indices it used.

Three equalities are checked, all exact: the routing generate returned equals the routing the trainer
received for that sample; the ``routed_experts`` a model call carries equals the sample's routing at every
captured position of the tokens it packs; and every router invocation replayed and used exactly its layer's
slice of that tensor. Under ``ac_config`` mode ``full`` with ``freq`` 1 every layer is recomputed, so each
call must show a recompute invocation for every layer as well as a forward one. A router that computed its
own routing, a sample whose routing reached no model call, and a router invocation outside any traced call
are failures.

The last token of a sequence produces no routing in the sampler: ``capture_len`` is one less than the row
length, and the trainer's value at that position is not compared.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from ..harness import gsm8k
from ..harness.dss_driver import build_payload
from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test
from ..harness.rl_driver import generate
from ..harness.rl_driver import rl_batch
from ..harness.rl_driver import rl_fwd_bwd
from ..harness.rl_driver import running_rl_client
from ..harness.seeds import SEED
from .inference_checkpoint_loss import TRAIN_SPLIT
from .inference_checkpoint_loss import configured_rows_per_step

TEST_ID = "rl-router-replay"

_MARKER_IDENTITY_FIELDS = (
    "sample_id",
    "replay_id",
    "trajectory_id",
    "request_id",
    "replica",
    "prompt_len",
    "generation_len",
    "capture_len",
)


@correctness_test(
    TEST_ID,
    title="Router replay: the training zone routes exactly as the sampling zone did",
    criterion=(
        "for every generated sample, the expert indices the sampling zone returns, the indices the training "
        "zone receives for it, and the indices every MoE router uses on the forward pass and on the "
        "activation-checkpoint recompute are identical, element for element"
    ),
    compares_to_reference=False,
)
def run(ctx) -> List[TestResult]:
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


def expects_full_recompute(training_config: dict) -> bool:
    ac = training_config.get("ac_config") or {}
    return ac.get("mode") == "full" and int(ac.get("freq", 0)) == 1


def _run(ctx) -> TestResult:
    cfg = ctx.cfg
    model_path = ctx.spec.model.cache_path
    tokenizer = gsm8k.load_tokenizer(model_path)
    pad_id = gsm8k.padding_token_id(tokenizer)
    examples = gsm8k.tokenize_examples(tokenizer, gsm8k.read_rows(TRAIN_SPLIT, count=configured_rows_per_step(cfg)))
    prompts = [list(e.input_ids[: e.n_prompt]) for e in examples]
    # Each prompt may generate as many tokens as the dataset's own answer to it has; nothing else bounds the
    # rollout, and the sampler's own defaults decide everything else about sampling.
    params = [
        {
            "max_tokens": int(e.answer_tokens),
            "dss_sample_id": f"row-{index}",
            "dss_router_replay_id": f"rr1:{uuid.uuid4()}",
            "dss_return_back_router_info": True,
        }
        for index, e in enumerate(examples)
    ]

    training = build_payload(cfg.training, model_path, SEED, attn_implementation=ctx.attn_implementation)
    training["training_config"]["debug"]["router_replay_trace"] = True
    gateway_dir = Path(ctx.workdir) / TEST_ID / "gateway"
    gateway_dir.mkdir(parents=True, exist_ok=True)
    with running_rl_client(cfg, training, model_path, gateway_dir) as client:
        results = generate(client, prompts, params)
        samples = [sample_from_result(prompt, result) for prompt, result in zip(prompts, results)]
        batch = rl_batch([s["tokens"] for s in samples], [s["prompt_len"] for s in samples], pad_id)
        batch["rr_sample_ids"] = [s["marker"]["replay_id"] for s in samples]
        batch["rr_replay_meta"] = [{k: s["marker"].get(k) for k in _MARKER_IDENTITY_FIELDS} for s in samples]
        batch["rr_discard"] = True
        response = rl_fwd_bwd(client, batch, replay_sampling=True)

    traces = response.get("router_replay_trace")
    if not traces:
        raise RuntimeError(
            "fwd-bwd returned no router_replay_trace although the payload set debug.router_replay_trace"
        )
    failed, metrics = judge(samples, traces, full_recompute=expects_full_recompute(cfg.training))
    metrics["train_loss"] = float(response.get("avg_loss", float("nan")))
    return verdict(ctx, failed, metrics)


def sample_from_result(prompt: Sequence[int], result: dict) -> dict:
    marker = result.get("router_replay")
    if not isinstance(marker, dict) or "routed_experts" not in marker:
        raise RuntimeError(
            f"generate returned no router_replay routing for prompt of {len(prompt)} tokens: keys {sorted(result)}"
        )
    tokens = list(prompt) + [int(t) for t in result["token_ids"]]
    if int(marker["prompt_len"]) != len(prompt):
        raise RuntimeError(f"sampler prompt_len {marker['prompt_len']} for a prompt of {len(prompt)} tokens")
    return {"tokens": tokens, "prompt_len": len(prompt), "marker": marker, "routing": marker["routed_experts"]}


def _segments(positions: Sequence[int]) -> List[Tuple[int, int]]:
    """Spans of a packed call, split where the position restarts at zero."""
    starts = [i for i, p in enumerate(positions) if p == 0] or [0]
    if starts[0] != 0:
        starts.insert(0, 0)
    return [(start, end) for start, end in zip(starts, starts[1:] + [len(positions)])]


def _attribute(
    tokens: Sequence[int], positions: Sequence[int], start: int, end: int, samples: Sequence[dict]
) -> Optional[int]:
    """The one sample whose row holds this span's tokens at this span's positions."""
    first = positions[start]
    span = list(tokens[start:end])
    matches = [
        i
        for i, s in enumerate(samples)
        if s["tokens"][first : first + len(span)] == span and first + len(span) <= len(s["tokens"])
    ]
    return matches[0] if len(matches) == 1 else None


def judge(
    samples: Sequence[dict], traces: Sequence[Optional[dict]], *, full_recompute: bool
) -> Tuple[List[Mismatch], Dict[str, float]]:
    failed: List[Mismatch] = []
    by_replay_id = {s["marker"]["replay_id"]: i for i, s in enumerate(samples)}
    layers = len(samples[0]["routing"][0]) if samples and samples[0]["routing"] else 0
    received_by: Dict[int, int] = {}
    covered_positions = [0] * len(samples)
    counts = {
        "calls": 0,
        "padding_calls": 0,
        "forward_invocations": 0,
        "recompute_invocations": 0,
        "compared_positions": 0,
        "unattributed_tokens": 0,
    }

    for trace in traces:
        if trace is None:
            continue
        rank = trace.get("rank")
        for replay_id, routing in (trace.get("received") or {}).items():
            index = by_replay_id.get(replay_id)
            if index is None:
                failed.append(Mismatch(f"rank {rank}: received routing for unknown id {replay_id}", 1.0, 0.0))
                continue
            received_by[index] = received_by.get(index, 0) + 1
            if routing != samples[index]["routing"]:
                failed.append(
                    Mismatch(f"rank {rank}: sample {index} routing received differs from generate", 1.0, 0.0)
                )
        for entry in trace.get("unattributed_router_calls") or []:
            failed.append(
                Mismatch(f"rank {rank}: router layer {entry.get('layer')} ran outside a traced call", 1.0, 0.0)
            )
        for call_index, call in enumerate(trace.get("calls") or []):
            counts["calls"] += 1
            if call.get("padding"):
                counts["padding_calls"] += 1
                continue
            label = f"rank {rank} call {call_index}"
            routed = call.get("routed_experts")
            if routed is None:
                failed.append(Mismatch(f"{label}: no routed_experts handed to the model", 1.0, 0.0))
                continue
            tokens, positions = call["input_ids"], call["position_ids"]
            for start, end in _segments(positions):
                index = _attribute(tokens, positions, start, end, samples)
                if index is None:
                    counts["unattributed_tokens"] += end - start
                    continue
                capture = samples[index]["routing"]
                differing = 0
                for t in range(start, end):
                    position = positions[t]
                    if position >= len(capture):
                        continue
                    counts["compared_positions"] += 1
                    covered_positions[index] += 1
                    if routed[t] != capture[position]:
                        differing += 1
                if differing:
                    failed.append(
                        Mismatch(
                            f"{label}: sample {index} positions whose routed_experts differ from the sampler's",
                            float(differing),
                            0.0,
                        )
                    )
            seen = {"forward": set(), "recompute": set()}
            for invocation in call.get("router") or []:
                layer, phase = invocation.get("layer"), invocation.get("phase")
                counts[f"{phase}_invocations"] += 1
                if layer is None:
                    failed.append(Mismatch(f"{label}: router with no decoder layer index", 1.0, 0.0))
                    continue
                seen[phase].add(layer)
                if not invocation.get("replayed"):
                    failed.append(Mismatch(f"{label}: layer {layer} {phase} computed its own routing", 1.0, 0.0))
                    continue
                expected = [row[layer] for row in routed]
                used = invocation.get("indices")
                if used != expected:
                    rows = sum(1 for a, b in zip(used or [], expected) if a != b)
                    rows += abs(len(used or []) - len(expected))
                    failed.append(
                        Mismatch(
                            f"{label}: layer {layer} {phase} tokens whose indices differ from the replayed slice",
                            float(rows),
                            0.0,
                        )
                    )
            for phase, required in (("forward", True), ("recompute", full_recompute)):
                if required:
                    for layer in sorted(set(range(layers)) - seen[phase]):
                        failed.append(Mismatch(f"{label}: layer {layer} has no {phase} router invocation", 1.0, 0.0))

    for index, sample in enumerate(samples):
        if index not in received_by:
            failed.append(Mismatch(f"sample {index}: no training rank received its routing", 1.0, 0.0))
        expected_positions = int(sample["marker"]["capture_len"])
        if covered_positions[index] < expected_positions:
            failed.append(
                Mismatch(
                    f"sample {index}: captured positions compared in a model call",
                    float(covered_positions[index]),
                    float(expected_positions),
                )
            )

    metrics: Dict[str, float] = {
        "samples": float(len(samples)),
        "layers": float(layers),
        "generated_tokens": float(sum(int(s["marker"]["generation_len"]) for s in samples)),
        "captured_positions": float(sum(int(s["marker"]["capture_len"]) for s in samples)),
        "full_recompute_expected": 1.0 if full_recompute else 0.0,
    }
    metrics.update({key: float(value) for key, value in counts.items()})
    return failed, metrics


def verdict(ctx, failed: List[Mismatch], metrics: Dict[str, float]) -> TestResult:
    summary = (
        f"{int(metrics['samples'])} samples, {int(metrics['captured_positions'])} captured positions over "
        f"{int(metrics['layers'])} layers; {int(metrics['compared_positions'])} positions compared, "
        f"{int(metrics['forward_invocations'])} forward and {int(metrics['recompute_invocations'])} recompute "
        f"router invocations; {len(failed)} disagreements"
    )
    if failed:
        summary += f" (first: {failed[0].name})"
    return TestResult(
        test_id=TEST_ID,
        config_id=ctx.config_id,
        outcome=TestOutcome.FAIL if failed else TestOutcome.PASS,
        summary=summary,
        mismatches=failed,
        worst_name=failed[0].name if failed else None,
        metrics=metrics,
    )
