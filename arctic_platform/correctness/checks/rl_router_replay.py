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
``debug.training`` recording on every rank:

- the routing each sample's replay id delivered from the sampler's cache;
- for every model call, the packed tokens, their positions, and the ``routed_experts`` tensor handed to the
  model;
- every router invocation, by decoder layer, on the forward pass and inside backward, where activation
  checkpointing recomputes the layer: whether it replayed supplied routing, and the indices it used.

Three equalities are checked, all exact: the routing generate returned equals the routing the trainer
received for that sample; the ``routed_experts`` a model call carries equals the sample's routing at every
captured position of the tokens it packs; and every router invocation replayed and used exactly its layer's
slice of that tensor. Under sequence parallelism a rank receives only the ``[start, end)`` window of the capture
spanned by the positions its own calls carry for that sample, clipped to ``capture_len``, so that window is what
its received routing is compared with. The capture is as wide as the decoder; a dense layer has no router, so the
layers that must show invocations come from the model's bound router inventory, not observed invocations. Under
``ac_config`` mode ``full`` with ``freq`` 1 every layer is recomputed, so each call must show a recompute invocation for every such
layer as well as a forward one. A router that computed its own routing, a sample whose routing reached no model
call, no router invocation in any call, and a router invocation outside any traced call are failures.

Under sequence parallelism dispatch right-pads a packed call so it divides across the ranks, and a rank whose window
holds only that padding holds no token of any sample. Its call carries no ``routed_experts`` and its routers route by
the gate, so the call is not compared and its tokens count as unattributed; a call holding a sample's tokens without
``routed_experts`` is a failure.

The last token of a sequence produces no routing in the sampler: ``capture_len`` is one less than the row
length, and the trainer's value at that position is not compared.
"""

from __future__ import annotations

import traceback
import uuid
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from arctic_platform.correctness.harness import sft as gsm8k
from arctic_platform.correctness.harness.gateway import UnsafeCleanupError
from arctic_platform.correctness.harness.gateway import build_payload
from arctic_platform.correctness.harness.gateway import generate
from arctic_platform.correctness.harness.gateway import minimum_context_parallel_row_tokens
from arctic_platform.correctness.harness.gateway import rl_batch
from arctic_platform.correctness.harness.gateway import rl_fwd_bwd
from arctic_platform.correctness.harness.gateway import running_rl_client
from arctic_platform.correctness.harness.inference import router_replay_bound_inventory
from arctic_platform.correctness.harness.inference import router_replay_expected_layers
from arctic_platform.correctness.harness.inference import router_replay_routing_geometry
from arctic_platform.correctness.harness.sft import SEED
from arctic_platform.correctness.harness.sft import TRAIN_SPLIT
from arctic_platform.correctness.harness.sft import configured_rows_per_step

from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test

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
    workload_modes=("rl",),
)
def run(ctx) -> List[TestResult]:
    try:
        return [_run(ctx)]
    except UnsafeCleanupError:
        raise
    except Exception as exc:  # noqa: BLE001 - a failure here is this check's verdict, not a crash
        traceback.print_exc()
        return [
            TestResult(
                test_id=TEST_ID,
                config_id=ctx.config_id,
                outcome=TestOutcome.FAIL,
                summary=f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            )
        ]


def expects_full_recompute(training_config: dict) -> bool:
    ac = training_config.get("ac_config") or {}
    return ac.get("mode") == "full" and int(ac.get("freq", 0)) == 1


def _run(ctx) -> TestResult:
    from arctic_platform.correctness.harness.inference import collect_traces
    from arctic_platform.debug.training import TRAINING_OPTION

    cfg = ctx.cfg
    model_path = cfg.model_name
    tokenizer = gsm8k.load_tokenizer(model_path)
    pad_id = gsm8k.padding_token_id(tokenizer)
    examples = replay_examples(
        tokenizer, configured_rows_per_step(cfg), minimum_context_parallel_row_tokens(cfg, model_path)
    )
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
    training["training_config"].setdefault("router_replay", {})["enabled"] = True
    option = {
        "request_id": uuid.uuid4().hex,
        "output_dir": str(Path(ctx.workdir) / TEST_ID / "traces"),
    }
    training["training_config"].setdefault("ds_config", {})[TRAINING_OPTION] = option
    gateway_dir = Path(ctx.workdir) / TEST_ID / "gateway"
    gateway_dir.mkdir(parents=True, exist_ok=True)
    with running_rl_client(
        cfg,
        training,
        model_path,
        gateway_dir,
        backend=getattr(ctx, "client_backend", None),
        router_replay_verification=True,
    ) as client:
        results = generate(client, prompts, params)
        samples = [sample_from_result(prompt, result) for prompt, result in zip(prompts, results)]
        batch = rl_batch([s["tokens"] for s in samples], [s["prompt_len"] for s in samples], pad_id)
        mask = batch["kwargs"]["attention_mask"]
        batch["kwargs"]["position_ids"] = (mask.cumsum(dim=-1) - 1).clamp_min(0) * mask
        batch["kwargs"]["rr_sample_ids"] = [s["marker"]["replay_id"] for s in samples]
        batch["kwargs"]["rr_replay_meta"] = [{k: s["marker"].get(k) for k in _MARKER_IDENTITY_FIELDS} for s in samples]
        batch["kwargs"]["rr_discard"] = True
        response = rl_fwd_bwd(client, batch, replay_sampling=True)
        traces = collect_traces(option, response, ranks=cfg.native.training_gpus)
    failed, metrics = judge(samples, traces, full_recompute=expects_full_recompute(cfg.training))
    metrics["train_loss"] = float(response.get("avg_loss", float("nan")))
    return verdict(ctx, failed, metrics)


def replay_examples(tokenizer, rows: int, minimum_prompt_tokens: int) -> List[gsm8k.Example]:
    """The train split's first ``rows`` examples, in file order, whose prompt holds ``minimum_prompt_tokens`` or more.

    Router replay admits a row only when its attended tokens are exactly the prompt plus the generation the sampler
    captured, so a row too short for context parallelism cannot be widened with loss-free tail tokens the way
    ``rl_batch(minimum_row_tokens=...)`` widens the other RL checks' rows. A packed window holds whole rows, so a
    prompt that alone spans the minimum keeps every rank's shard of every window wide enough, however few tokens the
    sampler generates. A minimum of zero selects the leading rows.
    """
    if type(minimum_prompt_tokens) is not int or minimum_prompt_tokens < 0:
        raise ValueError("minimum_prompt_tokens must be a nonnegative integer")
    if minimum_prompt_tokens == 0:
        return gsm8k.tokenize_examples(tokenizer, gsm8k.read_rows(TRAIN_SPLIT, count=rows))
    selected: List[gsm8k.Example] = []
    for row in gsm8k.read_rows(TRAIN_SPLIT):
        (example,) = gsm8k.tokenize_examples(tokenizer, [row])
        if example.n_prompt >= minimum_prompt_tokens:
            selected.append(example)
            if len(selected) == rows:
                return selected
    raise ValueError(
        f"{TRAIN_SPLIT} holds {len(selected)} prompt(s) of at least {minimum_prompt_tokens} tokens; the config trains "
        f"{rows} row(s) per step"
    )


def sample_from_result(prompt: Sequence[int], result: dict) -> dict:
    marker = result.get("router_replay")
    if not isinstance(marker, dict) or "routed_experts" not in marker:
        raise RuntimeError(
            f"generate returned no router_replay routing for prompt of {len(prompt)} tokens: keys {sorted(result)}"
        )
    tokens = list(prompt) + [int(t) for t in result["token_ids"]]
    if int(marker["prompt_len"]) != len(prompt):
        raise RuntimeError(f"sampler prompt_len {marker['prompt_len']} for a prompt of {len(prompt)} tokens")
    return {
        "tokens": tokens,
        "prompt_len": len(prompt),
        "marker": marker,
        "routing": marker["routed_experts"],
    }


def _segments(positions: Sequence[int]) -> List[Tuple[int, int]]:
    """Spans of a packed call, split where the position restarts at zero."""
    starts = [i for i, p in enumerate(positions) if p == 0] or [0]
    if starts[0] != 0:
        starts.insert(0, 0)
    return [(start, end) for start, end in zip(starts, starts[1:] + [len(positions)])]


def _attribute(
    tokens: Sequence[int],
    positions: Sequence[int],
    start: int,
    end: int,
    samples: Sequence[dict],
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


def _holds_sample(tokens: Sequence[int], positions: Sequence[int], samples: Sequence[dict]) -> bool:
    """Whether a span of a packed call belongs to a sample; a window of sequence-parallel padding alone holds none."""
    return any(_attribute(tokens, positions, start, end, samples) is not None for start, end in _segments(positions))


def _sample_geometry(sample: dict, inventory: dict) -> None:
    marker = sample["marker"]
    tokens, routing = sample["tokens"], sample["routing"]
    prompt, generated, captured = (marker.get(key) for key in ("prompt_len", "generation_len", "capture_len"))
    if (
        not isinstance(tokens, list)
        or any(type(token) is not int or token < 0 for token in tokens)
        or any(type(value) is not int or value <= 0 for value in (prompt, captured))
        or type(generated) is not int
        or generated < 0
        or type(sample.get("prompt_len")) is not int
        or sample["prompt_len"] != prompt
        or prompt + generated != len(tokens)
        or captured != len(tokens) - 1
    ):
        raise ValueError("sampler capture_len must equal prompt_len + generation_len - 1 with exact token metadata")
    replay_id = marker.get("replay_id")
    if not isinstance(replay_id, str) or not replay_id:
        raise ValueError("sampler replay_id must be nonempty")
    router_replay_routing_geometry(routing, inventory, tokens=captured)
    shape = marker.get("routed_experts_shape")
    if (
        not isinstance(shape, list)
        or any(type(size) is not int for size in shape)
        or shape != [captured, inventory["num_hidden_layers"], inventory["num_experts_per_tok"]]
    ):
        raise ValueError("sampler routed_experts_shape differs from model_inventory or capture_len")
    dtype = marker.get("routed_experts_dtype")
    if dtype not in ("uint8", "uint16"):
        raise ValueError("sampler routed_experts_dtype must be native uint8 or uint16")
    limit = 256 if dtype == "uint8" else 65536
    if any(index >= limit for row in routing for indices in row for index in indices):
        raise ValueError("sampler expert index does not fit its declared native dtype")
    router_replay_routing_geometry(marker.get("routed_experts"), inventory, tokens=captured)
    if marker["routed_experts"] != routing:
        raise ValueError("sampler marker routing differs from the compared routing")


def judge(
    samples: Sequence[dict], traces: Sequence[Optional[dict]], *, full_recompute: bool
) -> Tuple[List[Mismatch], Dict[str, float]]:
    failed: List[Mismatch] = []
    if not samples:
        failed.append(Mismatch("no generated samples to compare", 1.0, 0.0))
    layers = 0
    received_by: Dict[int, int] = {}
    covered_positions = [set() for _ in samples]
    router_layers = None
    model_inventory = None
    for trace_index, trace in enumerate(traces):
        try:
            inventory = router_replay_expected_layers(trace.get("expected_layers") if trace is not None else None)
            binding = router_replay_bound_inventory(
                trace.get("model_inventory") if trace is not None else None, expected_layers=inventory
            )
        except ValueError as exc:
            failed.append(Mismatch(f"trace {trace_index}: invalid bound router inventory: {exc}", 1.0, 0.0))
            continue
        if router_layers is None:
            router_layers = inventory
            model_inventory = binding
        elif inventory != router_layers:
            failed.append(Mismatch(f"rank {trace.get('rank')}: expected_layers differs across ranks", 1.0, 0.0))
        elif binding != model_inventory:
            failed.append(Mismatch(f"rank {trace.get('rank')}: model_inventory differs across ranks", 1.0, 0.0))
    if samples and router_layers is None:
        failed.append(Mismatch("no bound model router inventory in training traces", 1.0, 0.0))
    router_layers = router_layers or []
    counts = {
        "calls": 0,
        "padding_calls": 0,
        "forward_invocations": 0,
        "recompute_invocations": 0,
        "compared_positions": 0,
        "unattributed_tokens": 0,
    }
    if model_inventory is not None:
        layers = model_inventory["num_hidden_layers"]
        for index, sample in enumerate(samples):
            try:
                _sample_geometry(sample, model_inventory)
            except (KeyError, TypeError, ValueError) as exc:
                failed.append(Mismatch(f"sample {index}: invalid captured geometry: {exc}", 1.0, 0.0))
    replay_ids = [sample.get("marker", {}).get("replay_id") for sample in samples]
    if len({str(replay_id) for replay_id in replay_ids}) != len(samples):
        failed.append(Mismatch("sampler replay identities are duplicated", 1.0, 0.0))
    metrics: Dict[str, float] = {
        "samples": float(len(samples)),
        "layers": float(layers),
        "generated_tokens": 0.0,
        "captured_positions": 0.0,
        "full_recompute_expected": 1.0 if full_recompute else 0.0,
        **{key: float(value) for key, value in counts.items()},
    }
    if failed:
        return failed, metrics
    by_replay_id = {s["marker"]["replay_id"]: i for i, s in enumerate(samples)}

    for trace in traces:
        if trace is None:
            continue
        rank = trace.get("rank")
        windows: Dict[int, Tuple[int, int]] = {}
        for entry in trace.get("unattributed_router_calls") or []:
            failed.append(
                Mismatch(
                    f"rank {rank}: router layer {entry.get('layer')} ran outside a traced call",
                    1.0,
                    0.0,
                )
            )
        for call_index, call in enumerate(trace.get("calls") or []):
            counts["calls"] += 1
            if call.get("padding"):
                counts["padding_calls"] += 1
                continue
            label = f"rank {rank} call {call_index}"
            routed = call.get("routed_experts")
            tokens, positions = call["input_ids"], call["position_ids"]
            try:
                if (
                    not isinstance(tokens, list)
                    or not isinstance(positions, list)
                    or len(tokens) != len(positions)
                    or any(type(token) is not int or token < 0 for token in tokens)
                    or any(type(position) is not int or position < 0 for position in positions)
                ):
                    raise ValueError("model token and position vectors have invalid geometry")
                if routed is not None:
                    router_replay_routing_geometry(routed, model_inventory, tokens=len(tokens))
            except ValueError as exc:
                failed.append(Mismatch(f"{label}: invalid captured geometry: {exc}", 1.0, 0.0))
                continue
            if routed is None:
                if _holds_sample(tokens, positions, samples):
                    failed.append(Mismatch(f"{label}: no routed_experts handed to the model", 1.0, 0.0))
                else:
                    counts["unattributed_tokens"] += len(positions)
                continue
            for start, end in _segments(positions):
                index = _attribute(tokens, positions, start, end, samples)
                if index is None:
                    counts["unattributed_tokens"] += end - start
                    continue
                if positions[start:end] != list(range(positions[start], positions[start] + end - start)):
                    failed.append(Mismatch(f"{label}: sample {index} has noncontiguous captured positions", 1.0, 0.0))
                    continue
                low, high = windows.get(index, (positions[start], positions[end - 1] + 1))
                windows[index] = (min(low, positions[start]), max(high, positions[end - 1] + 1))
                capture = samples[index]["routing"]
                differing = 0
                for t in range(start, end):
                    position = positions[t]
                    if position < 0:
                        failed.append(Mismatch(f"{label}: sample {index} has a negative captured position", 1.0, 0.0))
                        continue
                    if position >= len(capture):
                        continue
                    counts["compared_positions"] += 1
                    covered_positions[index].add(position)
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
                if phase not in seen:
                    failed.append(Mismatch(f"{label}: router invocation has invalid phase {phase!r}", 1.0, 0.0))
                    continue
                counts[f"{phase}_invocations"] += 1
                if layer is None:
                    failed.append(Mismatch(f"{label}: router with no decoder layer index", 1.0, 0.0))
                    continue
                if type(layer) is not int or layer not in router_layers:
                    failed.append(
                        Mismatch(f"{label}: router layer {layer!r} is outside the bound inventory", 1.0, 0.0)
                    )
                    continue
                seen[phase].add(layer)
                if not invocation.get("replayed"):
                    failed.append(
                        Mismatch(
                            f"{label}: layer {layer} {phase} computed its own routing",
                            1.0,
                            0.0,
                        )
                    )
                    continue
                expected = [row[layer] for row in routed]
                used = invocation.get("indices")
                if (
                    not isinstance(used, list)
                    or len(used) != len(expected)
                    or any(
                        not isinstance(row, list)
                        or len(row) != model_inventory["num_experts_per_tok"]
                        or any(
                            type(index) is not int or not 0 <= index < model_inventory["num_experts"] for index in row
                        )
                        for row in used
                    )
                ):
                    failed.append(Mismatch(f"{label}: layer {layer} {phase} has invalid expert geometry", 1.0, 0.0))
                    continue
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
                    for layer in sorted(set(router_layers) - seen[phase]):
                        failed.append(
                            Mismatch(
                                f"{label}: layer {layer} has no {phase} router invocation",
                                1.0,
                                0.0,
                            )
                        )
        received = trace.get("received") or {}
        for index, (low, high) in windows.items():
            if (
                low < min(high, len(samples[index]["routing"]))
                and samples[index]["marker"]["replay_id"] not in received
            ):
                failed.append(
                    Mismatch(f"rank {rank}: sample {index} has no RX record for its captured window", 1.0, 0.0)
                )
        for replay_id, routing in received.items():
            index = by_replay_id.get(replay_id)
            if index is None:
                failed.append(
                    Mismatch(
                        f"rank {rank}: received routing for unknown id {replay_id}",
                        1.0,
                        0.0,
                    )
                )
                continue
            received_by[index] = received_by.get(index, 0) + 1
            capture = samples[index]["routing"]
            low, high = windows.get(index, (0, 0))
            try:
                router_replay_routing_geometry(routing, model_inventory, tokens=max(0, min(high, len(capture)) - low))
            except ValueError as exc:
                failed.append(Mismatch(f"rank {rank}: sample {index} has invalid RX geometry: {exc}", 1.0, 0.0))
                continue
            if routing != capture[low : min(high, len(capture))]:
                failed.append(
                    Mismatch(
                        f"rank {rank}: sample {index} routing received differs from generate",
                        1.0,
                        0.0,
                    )
                )

    for index, sample in enumerate(samples):
        if index not in received_by:
            failed.append(Mismatch(f"sample {index}: no training rank received its routing", 1.0, 0.0))
        expected_positions = int(sample["marker"]["capture_len"])
        if covered_positions[index] != set(range(expected_positions)):
            failed.append(
                Mismatch(
                    f"sample {index}: captured positions compared in a model call lack exact [0, capture_len)"
                    " coverage",
                    float(len(covered_positions[index])),
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
