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

"""Plugin conformance: the checks a model plugin must pass before the training service admits it.

Every CPU tier here runs locally with ``python -m arctic_platform.model_plugin.conformance --plugin pkg:PLUGIN
--spec conformance.toml``. The service reruns them at registration on the plugin's fixture, then runs the GPU
tiers (sampler log-probability parity and a real-weights step), which need the training runtime and vLLM.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import tomllib
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from torch import nn

from arctic_platform.common.packing import IGNORE_INDEX
from arctic_platform.model_plugin.experts import ROUTER_REPLAY_FRESH
from arctic_platform.model_plugin.host import bind_layers
from arctic_platform.model_plugin.host import forward
from arctic_platform.model_plugin.host import next_token_labels
from arctic_platform.model_plugin.host import pack_sequences
from arctic_platform.model_plugin.host import prepare_model
from arctic_platform.model_plugin.host import token_logprobs
from arctic_platform.model_plugin.parallel import shard_sequence
from arctic_platform.model_plugin.plugin import ModelPlugin
from arctic_platform.model_plugin.plugin import load_plugin
from arctic_platform.model_plugin.recompute import apply_activation_checkpointing
from arctic_platform.model_plugin.weights import INDEX_FILE
from arctic_platform.model_plugin.weights import STAMP_PLUGIN
from arctic_platform.model_plugin.weights import checkpoint_files
from arctic_platform.model_plugin.weights import checkpoint_stamp
from arctic_platform.model_plugin.weights import iter_hf_weights
from arctic_platform.model_plugin.weights import layer_index
from arctic_platform.model_plugin.weights import load_hf_checkpoint
from arctic_platform.model_plugin.weights import read_stamp
from arctic_platform.model_plugin.weights import save_hf_checkpoint
from arctic_platform.model_plugin.weights import weight_sync_skipped


class Tolerances(BaseModel):
    """Absolute tolerances; every comparison uses ``rtol = 0``. Defaults suit an fp32 CPU fixture."""

    model_config = ConfigDict(extra="forbid")

    reference: float = Field(1e-5, ge=0)
    packing: float = Field(1e-5, ge=0)
    parallel: float = Field(1e-5, ge=0)
    rl_step: float = Field(1e-5, ge=0)


class ConformanceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture: dict[str, Any]
    seed: int = 0
    sequence_lengths: list[int] = Field(default_factory=lambda: [5, 6, 5], min_length=2)
    sp_sizes: list[int] = Field(default_factory=lambda: [2])
    ep_sizes: list[int] = Field(default_factory=lambda: [2])
    tolerances: Tolerances = Field(default_factory=Tolerances)

    @model_validator(mode="after")
    def _check(self) -> ConformanceSpec:
        total = sum(self.sequence_lengths)
        row_starts = {sum(self.sequence_lengths[:i]) for i in range(len(self.sequence_lengths))}
        for size in self.sp_sizes:
            if total % size != 0:
                raise ValueError(f"total sequence length {total} must be divisible by every sp size, not {size}")
            # A row that starts on a shard boundary never needs its predecessor shard's tokens, which would let a
            # mixer that ignores state from the previous rank pass.
            shard_starts = {rank * total // size for rank in range(1, size)}
            if shard_starts <= row_starts:
                raise ValueError(f"with sp size {size}, at least one row must cross a shard boundary")
        return self


def load_conformance_spec(path: str | Path) -> ConformanceSpec:
    with open(path, "rb") as f:
        return ConformanceSpec.model_validate(tomllib.load(f))


@dataclass
class TierResult:
    tier: str
    passed: bool
    detail: str = ""


def build_fixture(plugin: ModelPlugin, spec: ConformanceSpec) -> tuple[Any, nn.Module]:
    torch.manual_seed(spec.seed)
    config = plugin.config_class(**spec.fixture)
    return config, plugin.build_model(config).float()


def fixture_sequences(config: Any, spec: ConformanceSpec, offset: int = 0) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(spec.seed + 1000 + offset)
    return [torch.randint(0, config.vocab_size, (length,), generator=generator) for length in spec.sequence_lengths]


def _logits(output) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, what: str) -> None:
    diff = (actual.double() - expected.double()).abs().max().item() if actual.numel() > 0 else 0.0
    if actual.shape != expected.shape or not diff <= atol:
        raise AssertionError(f"{what}: shapes {tuple(actual.shape)} vs {tuple(expected.shape)}, max |diff| {diff}")


def _token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"
    )


def _grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}


def check_static(plugin: ModelPlugin, spec: ConformanceSpec) -> None:
    plugin.manifest.check_contract()
    config, model = build_fixture(plugin, spec)
    info = plugin.parallelism(config)
    bind_layers(model, plugin, info)
    if info.moe_layers and plugin.expert_layer is None:
        raise AssertionError("the config has expert layers but the plugin has no expert_layer")


def check_reference_parity(plugin: ModelPlugin, spec: ConformanceSpec) -> None:
    if plugin.reference_forward is None:
        raise AssertionError("the plugin supplies no reference_forward")
    config, model = build_fixture(plugin, spec)
    with torch.no_grad():
        for index, sequence in enumerate(fixture_sequences(config, spec)):
            input_ids, position_ids = pack_sequences([sequence])
            actual = _logits(forward(model, input_ids, position_ids))
            expected = plugin.reference_forward(model, sequence)
            _assert_close(actual[0], expected, spec.tolerances.reference, f"sequence {index} vs reference")


def check_packing(plugin: ModelPlugin, spec: ConformanceSpec) -> None:
    """Packed rows give the logits, loss and gradients of the same rows run one at a time."""
    config, model = build_fixture(plugin, spec)
    sequences = fixture_sequences(config, spec)
    separate = copy.deepcopy(model)

    input_ids, position_ids = pack_sequences(sequences)
    packed_logits = _logits(forward(model, input_ids, position_ids))
    _token_loss(packed_logits, next_token_labels(input_ids, position_ids)).backward()

    start = 0
    for index, sequence in enumerate(sequences):
        row_ids, row_positions = pack_sequences([sequence])
        row_logits = _logits(forward(separate, row_ids, row_positions))
        _token_loss(row_logits, next_token_labels(row_ids, row_positions)).backward()
        _assert_close(
            packed_logits[:, start : start + len(sequence)].detach(),
            row_logits.detach(),
            spec.tolerances.packing,
            f"row {index} packed vs alone",
        )
        start += len(sequence)
    expected = _grads(separate)
    for name, grad in _grads(model).items():
        _assert_close(grad, expected[name], spec.tolerances.packing, f"grad {name} packed vs rows")


def check_router_replay(plugin: ModelPlugin, spec: ConformanceSpec) -> None:
    """Replaying captured routing reproduces the forward, fresh ids fall back to the gate, and replay is honoured."""
    config, model = build_fixture(plugin, spec)
    info = plugin.parallelism(config)
    input_ids, position_ids = pack_sequences(fixture_sequences(config, spec))
    with torch.no_grad():
        logits, routing = forward(model, input_ids, position_ids, return_routing=True)
        if set(routing) != set(info.moe_layers):
            raise AssertionError(f"routing covers layers {sorted(routing)}, config declares {list(info.moe_layers)}")
        replayed = _logits(forward(model, input_ids, position_ids, routed_experts=routing))
        _assert_close(replayed, logits, 0.0, "replayed captured routing")
        fresh = {layer: torch.full_like(route, ROUTER_REPLAY_FRESH) for layer, route in routing.items()}
        _assert_close(_logits(forward(model, input_ids, position_ids, routed_experts=fresh)), logits, 0.0, "fresh")
        shifted = {layer: (route + 1) % info.num_experts for layer, route in routing.items()}
        moved = _logits(forward(model, input_ids, position_ids, routed_experts=shifted))
        if (moved - logits).abs().max().item() == 0.0:
            raise AssertionError("forcing different experts did not change the output: routed_experts is ignored")


def check_checkpoint_and_sync(plugin: ModelPlugin, spec: ConformanceSpec) -> None:
    """A saved Hugging Face checkpoint reloads to the same model, and the sampler stream is the checkpoint minus
    trainer-only tensors."""
    config, model = build_fixture(plugin, spec)
    input_ids, position_ids = pack_sequences(fixture_sequences(config, spec))
    stamp = {key: value for key, value in checkpoint_stamp(plugin).items() if key.startswith("arctic_plugin")}
    with torch.no_grad():
        expected = _logits(forward(model, input_ids, position_ids))
        for max_shard_bytes in (None, 4096):
            with tempfile.TemporaryDirectory() as path:
                save_hf_checkpoint(model, plugin, config, path, max_shard_bytes=max_shard_bytes)
                files = set(checkpoint_files(path).values())
                if max_shard_bytes is not None and (len(files) < 2 or not (Path(path) / INDEX_FILE).is_file()):
                    raise AssertionError("a sharded save must write several files and an index")
                if read_stamp(path) != stamp:
                    raise AssertionError(f"checkpoint stamp {read_stamp(path)} does not name this plugin {stamp}")
                _, reloaded = load_hf_checkpoint(plugin, path)
                what = "reloaded checkpoint" + ("" if max_shard_bytes is None else " (sharded)")
                _assert_close(_logits(forward(reloaded, input_ids, position_ids)), expected, 0.0, what)
    checkpoint = dict(iter_hf_weights(model, plugin, for_sampler=False))
    stream = dict(iter_hf_weights(model, plugin, for_sampler=True))
    skipped = weight_sync_skipped(model)
    if not set(stream) <= set(checkpoint):
        raise AssertionError(
            f"sampler stream has names not in the checkpoint: {sorted(set(stream) - set(checkpoint))}"
        )
    if len(skipped) == 0 and set(stream) != set(checkpoint):
        raise AssertionError(f"sampler stream drops {sorted(set(checkpoint) - set(stream))} with nothing marked skip")
    for name, tensor in stream.items():
        _assert_close(tensor, checkpoint[name], 0.0, f"stream tensor {name}")


def check_activation_checkpointing(plugin: ModelPlugin, spec: ConformanceSpec) -> None:
    """Recomputing checkpointed decoder layers changes no output, gradient or buffer: side effects run once."""
    config, model = build_fixture(plugin, spec)
    checkpointed = copy.deepcopy(model)
    apply_activation_checkpointing(checkpointed)
    input_ids, position_ids = pack_sequences(fixture_sequences(config, spec))
    labels = next_token_labels(input_ids, position_ids)
    logits = {}
    for name, candidate in (("plain", model), ("checkpointed", checkpointed)):
        logits[name] = _logits(forward(candidate, input_ids, position_ids))
        _token_loss(logits[name], labels).backward()
    _assert_close(logits["checkpointed"].detach(), logits["plain"].detach(), 0.0, "checkpointed logits")
    expected = _grads(model)
    for name, grad in _grads(checkpointed).items():
        _assert_close(grad, expected[name], 0.0, f"recomputed grad {name}")
    buffers = dict(model.named_buffers())
    for name, buffer in checkpointed.named_buffers():
        _assert_close(buffer, buffers[name], 0.0, f"buffer {name} after recompute")


def check_weights_directory(plugin: ModelPlugin, path: str | Path) -> None:
    """A real weights directory is servable: config, every tensor's name and shape, and the tokenizer.

    Shapes are checked on the meta device, so this runs on a laptop even for a large model.
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer

    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    if raw.get("model_type") not in plugin.manifest.hf_model_types:
        raise AssertionError(
            f"config.json model_type {raw.get('model_type')!r} is not in {plugin.manifest.hf_model_types}"
        )
    sampler = plugin.manifest.sampler
    architecture = (raw.get("architectures") or [None])[0]
    if sampler is not None and architecture != sampler.vllm_architecture:
        raise AssertionError(f"config.json architectures[0] {architecture!r} is not {sampler.vllm_architecture!r}")
    stamp = read_stamp(path)
    if STAMP_PLUGIN in stamp and stamp[STAMP_PLUGIN] != plugin.manifest.name:
        raise AssertionError(f"checkpoint was written by plugin {stamp[STAMP_PLUGIN]}")

    config = plugin.config_class.from_pretrained(path)
    plugin.parallelism(config)
    with torch.device("meta"):
        expected = {name: tuple(tensor.shape) for name, tensor in plugin.build_model(config).state_dict().items()}
    files = checkpoint_files(path)
    for file in set(files.values()):
        if not file.is_file():
            raise AssertionError(f"the checkpoint index names {file.name}, which is missing")
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for file in set(files.values()):
        with safe_open(str(file), framework="pt") as f:
            for name in f.keys():
                shape = f.get_slice(name).get_shape()
                by_layer.setdefault(layer_index(name), {})[name] = torch.empty(shape, device="meta")
    found: dict[str, tuple[int, ...]] = {}
    problems: list[str] = []
    for index, layer in sorted(by_layer.items()):
        if index >= 0 and plugin.convert_layer_to_prime is not None:
            try:
                layer = plugin.convert_layer_to_prime(layer, index, config)
            except ValueError as error:  # the converter's own shape and index checks
                problems.append(f"layer {index} does not convert: {error}")
                continue
        found.update({name: tuple(tensor.shape) for name, tensor in layer.items()})
    if len(problems) > 0:
        raise AssertionError(f"{len(problems)} tensor problems, first: {problems[:5]}")
    problems += [f"missing {name}" for name in sorted(set(expected) - set(found))]
    problems += [f"unexpected {name}" for name in sorted(set(found) - set(expected))]
    problems += [
        f"{name}: checkpoint {found[name]} vs model {expected[name]}"
        for name in sorted(set(found) & set(expected))
        if found[name] != expected[name]
    ]
    if len(problems) > 0:
        raise AssertionError(f"{len(problems)} tensor problems, first: {problems[:5]}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    except Exception as error:
        raise AssertionError(f"the tokenizer does not load without remote code: {error}") from error
    if tokenizer.eos_token_id is None:
        raise AssertionError("the tokenizer defines no EOS token")
    if len(tokenizer) > config.vocab_size:
        raise AssertionError(f"the tokenizer has {len(tokenizer)} tokens but the model only {config.vocab_size}")


def check_rl_step(plugin: ModelPlugin, spec: ConformanceSpec) -> None:
    """A GRPO step on packed rows matches the same step on padded rows: loss and every gradient."""
    from arctic_platform.rl.processors.grpo import grpo_loss

    config, model = build_fixture(plugin, spec)
    padded_model = copy.deepcopy(model)
    sequences = fixture_sequences(config, spec)
    generator = torch.Generator().manual_seed(spec.seed + 7)

    input_ids, position_ids = pack_sequences(sequences)
    labels = next_token_labels(input_ids, position_ids)
    packed = token_logprobs(_logits(forward(model, input_ids, position_ids)), labels)
    advantages = torch.randn(packed.shape, generator=generator)
    old = packed.detach() + 0.01 * torch.randn(packed.shape, generator=generator)
    mask = labels != IGNORE_INDEX
    packed_loss, _ = grpo_loss(
        dict(logprobs=packed), dict(old_log_probs_shifted=old, advantages=advantages, loss_mask=mask), {}, {}, "cpu"
    )
    packed_loss.backward()

    width = max(len(sequence) for sequence in sequences)
    rows = len(sequences)
    padded_old = torch.zeros(rows, width)
    padded_advantages = torch.zeros(rows, width)
    padded_mask = torch.zeros(rows, width, dtype=torch.bool)
    rows_logprobs = []
    start = 0
    for row, sequence in enumerate(sequences):
        row_ids, row_positions = pack_sequences([sequence])
        row_labels = next_token_labels(row_ids, row_positions)
        rows_logprobs.append(token_logprobs(_logits(forward(padded_model, row_ids, row_positions)), row_labels)[0])
        stop = start + len(sequence)
        padded_old[row, : len(sequence)] = old[0, start:stop]
        padded_advantages[row, : len(sequence)] = advantages[0, start:stop]
        padded_mask[row, : len(sequence)] = mask[0, start:stop]
        start = stop
    padded = torch.stack([torch.nn.functional.pad(lp, (0, width - len(lp))) for lp in rows_logprobs])
    padded_loss, _ = grpo_loss(
        dict(logprobs=padded),
        dict(old_log_probs_shifted=padded_old, advantages=padded_advantages, loss_mask=padded_mask),
        {},
        {},
        "cpu",
    )
    padded_loss.backward()
    _assert_close(packed_loss.detach(), padded_loss.detach(), spec.tolerances.rl_step, "GRPO loss packed vs padded")
    expected = _grads(padded_model)
    for name, grad in _grads(model).items():
        _assert_close(grad, expected[name], spec.tolerances.rl_step, f"GRPO grad {name}")


def _parallel_worker(rank: int, size: int, init_file: str, target: str, spec_json: str, mode: str, path: list[str]):
    sys.path[:] = path
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", world_size=size, rank=rank, timeout=timedelta(seconds=120)
    )
    try:
        plugin = load_plugin(target)
        spec = ConformanceSpec.model_validate_json(spec_json)
        _parallel_checks(plugin, spec, mode, rank, size)
    finally:
        dist.destroy_process_group()


def _parallel_checks(plugin: ModelPlugin, spec: ConformanceSpec, mode: str, rank: int, size: int) -> None:
    group = dist.group.WORLD
    atol = spec.tolerances.parallel
    config, baseline = build_fixture(plugin, spec)
    model = copy.deepcopy(baseline)
    sp_group = group if mode in ("sp", "sp+ep") else None
    ep_group = group if mode in ("ep", "sp+ep") else None
    prepare_model(model, plugin, config, sp_group=sp_group, ep_group=ep_group)
    if mode == "sp+ep":
        # The production layout: recompute then reruns the sequence and expert collectives inside backward.
        apply_activation_checkpointing(model)
    expert_names = {
        f"{path}.{name}"
        for path, module in model.named_modules()
        if plugin.expert_layer is not None and plugin.expert_layer.is_instance(module)
        for name in plugin.expert_layer.expert_parameters(module)
    }

    # Sequence-parallel ranks hold shards of the same rows; pure expert parallelism gives each rank its own rows.
    sequences = fixture_sequences(config, spec, offset=rank if mode == "ep" else 0)
    input_ids, position_ids = pack_sequences(sequences)
    labels = next_token_labels(input_ids, position_ids)

    baseline_logits = _logits(forward(baseline, input_ids, position_ids))
    _token_loss(baseline_logits, labels).backward()
    logits = _logits(forward(model, input_ids, position_ids, sp_group=sp_group))
    _token_loss(logits, shard_sequence(labels, sp_group)).backward()
    _assert_close(logits.detach(), shard_sequence(baseline_logits.detach(), sp_group), atol, f"{mode} logits")

    sharded = expert_names if ep_group is not None else set()
    expected = _grads(baseline)
    for name, grad in _grads(model).items():
        reference = expected[name]
        if mode == "ep":
            # Each rank's baseline saw only its own rows; summing over ranks gives the step's gradient.
            dist.all_reduce(reference, group=group)
        if name in sharded:
            # Expert gradients already hold every rank's tokens, routed here by combine_tokens' backward.
            per_rank = reference.shape[0] // size
            reference = reference[rank * per_rank : (rank + 1) * per_rank]
        else:
            # Replicated weights: each rank holds a partial gradient that data/sequence parallelism sums.
            dist.all_reduce(grad, group=group)
        _assert_close(grad, reference, atol, f"{mode} grad {name}")

    if ep_group is not None:
        exported = dict(iter_hf_weights(model, plugin, for_sampler=True, ep_group=ep_group))
        unsharded = dict(iter_hf_weights(baseline, plugin, for_sampler=True))
        if set(exported) != set(unsharded):
            raise AssertionError(f"{mode} export names differ: {sorted(set(exported) ^ set(unsharded))[:5]}")
        for name, tensor in unsharded.items():
            _assert_close(exported[name], tensor, 0.0, f"{mode} exported {name}")


def check_parallel(plugin_target: str, spec: ConformanceSpec, mode: str, size: int) -> None:
    """Run ``mode`` ('sp', 'ep' or 'sp+ep') on ``size`` gloo ranks and compare with the unsharded model."""
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _parallel_worker,
            args=(
                size,
                str(Path(directory) / "rendezvous"),
                plugin_target,
                spec.model_dump_json(),
                mode,
                list(sys.path),
            ),
            nprocs=size,
        )


def run_conformance(plugin_target: str, spec: ConformanceSpec, weights: str | Path | None = None) -> list[TierResult]:
    """Every applicable CPU tier; with ``weights``, also check that real weights directory."""
    plugin = load_plugin(plugin_target)
    capabilities = plugin.manifest.capabilities
    config, _ = build_fixture(plugin, spec)
    has_experts = len(plugin.parallelism(config).moe_layers) > 0
    tiers: list[tuple[str, Callable[[], None]]] = [
        ("static", lambda: check_static(plugin, spec)),
        ("reference_parity", lambda: check_reference_parity(plugin, spec)),
        ("packing", lambda: check_packing(plugin, spec)),
        ("activation_checkpointing", lambda: check_activation_checkpointing(plugin, spec)),
        ("checkpoint_and_sync", lambda: check_checkpoint_and_sync(plugin, spec)),
        ("rl_step", lambda: check_rl_step(plugin, spec)),
    ]
    if weights is not None:
        tiers.append(("weights_directory", partial(check_weights_directory, plugin, weights)))
    if capabilities.router_replay and has_experts:
        tiers.append(("router_replay", lambda: check_router_replay(plugin, spec)))
    if capabilities.sequence_parallel:
        tiers += [(f"sp{size}", partial(check_parallel, plugin_target, spec, "sp", size)) for size in spec.sp_sizes]
    if capabilities.expert_parallel and has_experts:
        tiers += [(f"ep{size}", partial(check_parallel, plugin_target, spec, "ep", size)) for size in spec.ep_sizes]
    if capabilities.sequence_parallel and capabilities.expert_parallel and has_experts:
        both = sorted(set(spec.sp_sizes) & set(spec.ep_sizes))
        tiers += [(f"sp{size}+ep{size}", partial(check_parallel, plugin_target, spec, "sp+ep", size)) for size in both]

    results = []
    for name, check in tiers:
        try:
            check()
            results.append(TierResult(name, True))
        except Exception as error:  # a failing tier is reported, then the next tier runs
            detail = "".join(traceback.format_exception_only(type(error), error)).strip()
            results.append(TierResult(name, False, detail))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plugin", required=True, help="'module:attribute' of the ModelPlugin")
    parser.add_argument("--spec", required=True, help="path to the plugin's conformance.toml")
    parser.add_argument("--weights", help="a real Hugging Face weights directory to check (shapes only, no GPU)")
    args = parser.parse_args(argv)
    results = run_conformance(args.plugin, load_conformance_spec(args.spec), weights=args.weights)
    width = max(len(result.tier) for result in results)
    for result in results:
        print(f"{result.tier:<{width}}  {'PASS' if result.passed else 'FAIL'}  {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
