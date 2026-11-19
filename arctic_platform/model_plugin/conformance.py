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
--spec conformance.toml``; ``--device cuda`` runs the same tiers on GPUs, with the parallel tiers over NCCL. The
``rl_step`` tier uses the platform's GRPO loss, so it needs ``arctic-platform[rl]`` and is skipped without it. The
service reruns every tier at registration on the plugin's fixture, then runs the GPU tiers (sampler
log-probability parity and a real-weights step), which need the training runtime and vLLM.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
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
from typing import Literal

import torch
import torch.distributed as dist
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from torch import nn

from arctic_platform.common.packing import IGNORE_INDEX
from arctic_platform.model_plugin.build import build
from arctic_platform.model_plugin.experts import ROUTER_REPLAY_FRESH
from arctic_platform.model_plugin.host import bind_layers
from arctic_platform.model_plugin.host import compute_logits
from arctic_platform.model_plugin.host import forward
from arctic_platform.model_plugin.host import next_token_labels
from arctic_platform.model_plugin.host import pack_sequences
from arctic_platform.model_plugin.host import prepare_model
from arctic_platform.model_plugin.host import token_logprobs
from arctic_platform.model_plugin.launch import spawn_ranks
from arctic_platform.model_plugin.marks import cast_model
from arctic_platform.model_plugin.marks import tensor_marks
from arctic_platform.model_plugin.marks import weight_sync_skipped
from arctic_platform.model_plugin.parallel import shard_sequence
from arctic_platform.model_plugin.plugin import ModelPlugin
from arctic_platform.model_plugin.plugin import load_plugin
from arctic_platform.model_plugin.recompute import apply_activation_checkpointing
from arctic_platform.model_plugin.weights import INDEX_FILE
from arctic_platform.model_plugin.weights import STAMP_PLUGIN
from arctic_platform.model_plugin.weights import _check_layer_pattern
from arctic_platform.model_plugin.weights import _safetensors_dtype
from arctic_platform.model_plugin.weights import checkpoint_files
from arctic_platform.model_plugin.weights import checkpoint_stamp
from arctic_platform.model_plugin.weights import iter_hf_weights
from arctic_platform.model_plugin.weights import load_hf_checkpoint
from arctic_platform.model_plugin.weights import read_stamp
from arctic_platform.model_plugin.weights import require_declared_dtype
from arctic_platform.model_plugin.weights import save_hf_checkpoint


class Tolerances(BaseModel):
    """Absolute tolerances; every comparison uses ``rtol = 0``. Defaults suit an fp32 CPU fixture."""

    model_config = ConfigDict(extra="forbid")

    reference: float = Field(1e-5, ge=0)
    packing: float = Field(1e-5, ge=0)
    chunking: float = Field(
        1e-6,
        ge=0,
        description=(
            "Chunking reorders the sum behind the output projection's gradient, and short chunks can take a "
            "different GEMM kernel, so fp32 results agree to rounding, not bitwise."
        ),
    )
    parallel: float = Field(1e-5, ge=0)
    rl_step: float = Field(1e-5, ge=0)


class ConformanceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture: dict[str, Any]
    options: dict[str, Any] = Field(default_factory=dict, description="The plugin's job options for the fixture.")
    seed: int = 0
    logprob_chunk: int = Field(4, ge=1, description="Tokens per log-probability chunk; small, so rows span chunks.")
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


Device = str | torch.device


class TierSkipped(Exception):
    """A tier that cannot run in this environment; the service, which has the full runtime, always runs it."""


@dataclass
class TierResult:
    tier: str
    status: Literal["PASS", "FAIL", "SKIP"]
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def build_fixture(plugin: ModelPlugin, spec: ConformanceSpec, device: Device = "cpu") -> tuple[Any, nn.Module]:
    """The fixture config and model, prepared as the platform runs it (routing installed, no parallelism)."""
    torch.manual_seed(spec.seed)
    config = plugin.config_class(**spec.fixture)
    model = cast_model(build(plugin, config, spec.options), torch.float32).to(device)
    prepare_model(model, plugin, config)
    return config, model


def fixture_sequences(
    config: Any, spec: ConformanceSpec, offset: int = 0, device: Device = "cpu"
) -> list[torch.Tensor]:
    """The fixture's rows, drawn on CPU so they are the same on every device."""
    generator = torch.Generator().manual_seed(spec.seed + 1000 + offset)
    return [
        torch.randint(0, config.vocab_size, (length,), generator=generator).to(device)
        for length in spec.sequence_lengths
    ]


def _hidden(output) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _logits(model: nn.Module, output) -> torch.Tensor:
    """Full logits of a forward's hidden states: the fixture is tiny, so they fit."""
    return compute_logits(model, _hidden(output))


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, what: str) -> None:
    diff = (actual.double() - expected.double()).abs().max().item() if actual.numel() > 0 else 0.0
    if actual.shape != expected.shape or not diff <= atol:
        raise AssertionError(f"{what}: shapes {tuple(actual.shape)} vs {tuple(expected.shape)}, max |diff| {diff}")


def _token_loss(model: nn.Module, output, labels: torch.Tensor, spec: ConformanceSpec) -> torch.Tensor:
    """Summed negative log-likelihood through the platform's chunked log-probabilities."""
    return -token_logprobs(model, _hidden(output), labels, chunk_size=spec.logprob_chunk).sum()


def _grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}


PUBLIC_API = "arctic_platform.model_plugin"


def _plugin_sources(plugin: ModelPlugin) -> list[Path]:
    """The source files of the package (or single module) that defines the plugin's model."""
    top = plugin.build_model.__module__.split(".")[0]
    found = importlib.util.find_spec(top)
    if found is None:
        return []
    if found.submodule_search_locations:
        return sorted(path for root in found.submodule_search_locations for path in Path(root).rglob("*.py"))
    return [Path(found.origin)] if found.origin and found.origin.endswith(".py") else []


def private_imports(plugin: ModelPlugin) -> list[str]:
    """Every import of the platform other than its public API, ``arctic_platform.model_plugin`` itself, as
    ``file:line: module``. Only the package root is supported across releases; its submodules may change."""
    found = []
    for path in _plugin_sources(plugin):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module.split(".")[0] == "arctic_platform" and module != PUBLIC_API:
                    found.append(f"{path}:{node.lineno}: {module}")
    return found


def check_static(plugin: ModelPlugin, spec: ConformanceSpec) -> None:
    plugin.manifest.check_contract()
    private = private_imports(plugin)
    if len(private) > 0:
        raise AssertionError(
            f"import the platform only from {PUBLIC_API}; these imports use other modules: " + "; ".join(private)
        )
    config, model = build_fixture(plugin, spec)
    info = plugin.parallelism(config)
    bind_layers(model, plugin, info)
    if info.moe_layers and plugin.expert_layer is None:
        raise AssertionError("the config has expert layers but the plugin has no expert_layer")


def check_reference_parity(plugin: ModelPlugin, spec: ConformanceSpec, device: Device = "cpu") -> None:
    if plugin.reference_forward is None:
        raise AssertionError("the plugin supplies no reference_forward")
    config, model = build_fixture(plugin, spec, device)
    with torch.no_grad():
        for index, sequence in enumerate(fixture_sequences(config, spec, device=device)):
            input_ids, position_ids = pack_sequences([sequence])
            actual = _logits(model, forward(model, input_ids, position_ids))
            expected = plugin.reference_forward(model, sequence)
            _assert_close(actual[0], expected, spec.tolerances.reference, f"sequence {index} vs reference")


def check_packing(plugin: ModelPlugin, spec: ConformanceSpec, device: Device = "cpu") -> None:
    """Packed rows give the logits, loss and gradients of the same rows run one at a time."""
    config, model = build_fixture(plugin, spec, device)
    sequences = fixture_sequences(config, spec, device=device)
    separate = copy.deepcopy(model)

    input_ids, position_ids = pack_sequences(sequences)
    packed = forward(model, input_ids, position_ids)
    packed_logits = _logits(model, packed)
    _token_loss(model, packed, next_token_labels(input_ids, position_ids), spec).backward()

    start = 0
    for index, sequence in enumerate(sequences):
        row_ids, row_positions = pack_sequences([sequence])
        row = forward(separate, row_ids, row_positions)
        row_logits = _logits(separate, row)
        _token_loss(separate, row, next_token_labels(row_ids, row_positions), spec).backward()
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


def check_logprob_chunking(plugin: ModelPlugin, spec: ConformanceSpec, device: Device = "cpu") -> None:
    """Log-probabilities computed ``logprob_chunk`` tokens at a time, recomputed in backward, equal one pass over
    the whole row: values and every gradient, through the model's own ``compute_logits``."""
    config, chunked = build_fixture(plugin, spec, device)
    whole = copy.deepcopy(chunked)
    input_ids, position_ids = pack_sequences(fixture_sequences(config, spec, device=device))
    labels = next_token_labels(input_ids, position_ids)
    if spec.logprob_chunk >= input_ids.shape[1]:
        raise AssertionError(
            f"logprob_chunk {spec.logprob_chunk} must be shorter than the {input_ids.shape[1]}-token fixture"
        )
    values = {}
    for name, model, chunk in (("chunked", chunked, spec.logprob_chunk), ("whole", whole, input_ids.shape[1])):
        values[name] = token_logprobs(model, forward(model, input_ids, position_ids), labels, chunk_size=chunk)
        values[name].sum().backward()
    _assert_close(values["chunked"].detach(), values["whole"].detach(), spec.tolerances.chunking, "chunked log-probs")
    expected = _grads(whole)
    for name, grad in _grads(chunked).items():
        _assert_close(grad, expected[name], spec.tolerances.chunking, f"chunked grad {name}")


def check_meta_build(plugin: ModelPlugin, spec: ConformanceSpec, device: Device = "cpu") -> None:
    """Built on the meta device, allocated with ``to_empty``, re-initialized with ``ap_init_buffers`` and loaded from
    a checkpoint, the model equals one built normally: every buffer, checkpointed or not, and the logits."""
    config, model = build_fixture(plugin, spec, device)
    input_ids, position_ids = pack_sequences(fixture_sequences(config, spec, device=device))
    with tempfile.TemporaryDirectory() as path:
        save_hf_checkpoint(model, plugin, config, path)
        _, loaded = load_hf_checkpoint(plugin, path, options=spec.options, device=device)
    buffers = dict(loaded.named_buffers())
    for name, buffer in model.named_buffers():
        try:
            _assert_close(buffers[name], buffer, 0.0, f"buffer {name}")
        except AssertionError as error:
            raise AssertionError(f"{error} after a meta-device build: does its module implement ap_init_buffers?")
    with torch.no_grad():
        expected = _logits(model, forward(model, input_ids, position_ids))
        _assert_close(_logits(loaded, forward(loaded, input_ids, position_ids)), expected, 0.0, "meta-built logits")


def check_dtype_marks(plugin: ModelPlugin, spec: ConformanceSpec, device: Device = "cpu") -> None:
    """Under a bf16 build, tensors marked ``dtype="fp32"`` stay fp32 and the rest are bf16; a checkpoint, written
    from the same stream the sampler receives, carries those mixed dtypes and reloads every tensor exactly."""
    config, model = build_fixture(plugin, spec, device)
    cast_model(model, torch.bfloat16)
    marks = tensor_marks(model)
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    for name, tensor in tensors.items():
        if tensor.is_floating_point():
            expected = torch.float32 if marks[name].dtype == "fp32" else torch.bfloat16
            if tensor.dtype != expected:
                raise AssertionError(f"{name} is {tensor.dtype} under a bf16 build, expected {expected}")
    with tempfile.TemporaryDirectory() as path:
        save_hf_checkpoint(model, plugin, config, path)
        _, loaded = load_hf_checkpoint(plugin, path, options=spec.options, device=device)
    reloaded = loaded.state_dict()
    for name, tensor in model.state_dict().items():
        if reloaded[name].dtype != tensor.dtype:
            raise AssertionError(f"{name} reloads as {reloaded[name].dtype}, saved as {tensor.dtype}")
        _assert_close(reloaded[name], tensor, 0.0, f"reloaded {name}")


def check_router_replay(plugin: ModelPlugin, spec: ConformanceSpec, device: Device = "cpu") -> None:
    """Replaying captured routing reproduces the forward, fresh ids fall back to the gate, and replay is honoured."""
    config, model = build_fixture(plugin, spec, device)
    info = plugin.parallelism(config)
    input_ids, position_ids = pack_sequences(fixture_sequences(config, spec, device=device))
    with torch.no_grad():
        hidden, routing = forward(model, input_ids, position_ids, return_routing=True)
        logits = _logits(model, hidden)
        if set(routing) != set(info.moe_layers):
            raise AssertionError(f"routing covers layers {sorted(routing)}, config declares {list(info.moe_layers)}")
        replayed = _logits(model, forward(model, input_ids, position_ids, routed_experts=routing))
        _assert_close(replayed, logits, 0.0, "replayed captured routing")
        fresh = {layer: torch.full_like(route, ROUTER_REPLAY_FRESH) for layer, route in routing.items()}
        _assert_close(
            _logits(model, forward(model, input_ids, position_ids, routed_experts=fresh)), logits, 0.0, "fresh"
        )
        shifted = {layer: (route + 1) % info.num_experts for layer, route in routing.items()}
        moved = _logits(model, forward(model, input_ids, position_ids, routed_experts=shifted))
        if (moved - logits).abs().max().item() == 0.0:
            raise AssertionError("forcing different experts did not change the output: routed_experts is ignored")


def check_checkpoint_and_sync(plugin: ModelPlugin, spec: ConformanceSpec, device: Device = "cpu") -> None:
    """A saved Hugging Face checkpoint reloads to the same model, and the sampler stream is the checkpoint minus
    trainer-only tensors."""
    config, model = build_fixture(plugin, spec, device)
    input_ids, position_ids = pack_sequences(fixture_sequences(config, spec, device=device))
    stamp = {key: value for key, value in checkpoint_stamp(plugin).items() if key.startswith("arctic_plugin")}
    with torch.no_grad():
        expected = _logits(model, forward(model, input_ids, position_ids))
        for max_shard_bytes in (None, 4096):
            with tempfile.TemporaryDirectory() as path:
                save_hf_checkpoint(model, plugin, config, path, max_shard_bytes=max_shard_bytes)
                files = set(checkpoint_files(path).values())
                if max_shard_bytes is not None and (len(files) < 2 or not (Path(path) / INDEX_FILE).is_file()):
                    raise AssertionError("a sharded save must write several files and an index")
                if read_stamp(path) != stamp:
                    raise AssertionError(f"checkpoint stamp {read_stamp(path)} does not name this plugin {stamp}")
                _, reloaded = load_hf_checkpoint(plugin, path, options=spec.options, device=device)
                what = "reloaded checkpoint" + ("" if max_shard_bytes is None else " (sharded)")
                _assert_close(_logits(reloaded, forward(reloaded, input_ids, position_ids)), expected, 0.0, what)
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


def check_activation_checkpointing(plugin: ModelPlugin, spec: ConformanceSpec, device: Device = "cpu") -> None:
    """Recomputing checkpointed decoder layers changes no output, gradient or buffer: side effects run once."""
    config, model = build_fixture(plugin, spec, device)
    checkpointed = copy.deepcopy(model)
    apply_activation_checkpointing(checkpointed)
    input_ids, position_ids = pack_sequences(fixture_sequences(config, spec, device=device))
    labels = next_token_labels(input_ids, position_ids)
    logits = {}
    for name, candidate in (("plain", model), ("checkpointed", checkpointed)):
        output = forward(candidate, input_ids, position_ids)
        logits[name] = _logits(candidate, output)
        _token_loss(candidate, output, labels, spec).backward()
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
    expected = {name: tuple(tensor.shape) for name, tensor in build(plugin, config, meta=True).state_dict().items()}
    files = checkpoint_files(path)
    for file in set(files.values()):
        if not file.is_file():
            raise AssertionError(f"the checkpoint index names {file.name}, which is missing")
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for file in set(files.values()):
        with safe_open(str(file), framework="pt") as f:
            for name in f.keys():
                if plugin.is_passthrough(name, config):
                    continue
                header = f.get_slice(name)
                layer = plugin.manifest.checkpoint.layer_of(name)
                tensor = torch.empty(header.get_shape(), dtype=_safetensors_dtype(header.get_dtype()), device="meta")
                by_layer.setdefault(layer, {})[name] = tensor
    require_declared_dtype(plugin, path, {name for name in files if plugin.is_passthrough(name, config)})
    _check_layer_pattern(by_layer.get(-1, {}))
    found: dict[str, tuple[int, ...]] = {}
    problems: list[str] = []
    for index, layer in sorted(by_layer.items()):
        if plugin.convert_layer_to_prime is not None:
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


def check_rl_step(plugin: ModelPlugin, spec: ConformanceSpec, device: Device = "cpu") -> None:
    """A GRPO step on packed rows matches the same step on padded rows: loss and every gradient."""
    try:
        from arctic_platform.rl.processors.grpo import grpo_loss
    except ImportError as error:
        raise TierSkipped(f"needs the platform's GRPO loss: {error}") from error

    config, model = build_fixture(plugin, spec, device)
    padded_model = copy.deepcopy(model)
    sequences = fixture_sequences(config, spec, device=device)
    generator = torch.Generator().manual_seed(spec.seed + 7)

    input_ids, position_ids = pack_sequences(sequences)
    labels = next_token_labels(input_ids, position_ids)
    packed = token_logprobs(model, forward(model, input_ids, position_ids), labels, chunk_size=spec.logprob_chunk)
    advantages = torch.randn(packed.shape, generator=generator).to(device)
    old = packed.detach() + 0.01 * torch.randn(packed.shape, generator=generator).to(device)
    mask = labels != IGNORE_INDEX
    packed_loss, _ = grpo_loss(
        dict(logprobs=packed),
        dict(old_log_probs_shifted=old, advantages=advantages, loss_mask=mask),
        {},
        {},
        str(device),
    )
    packed_loss.backward()

    width = max(len(sequence) for sequence in sequences)
    rows = len(sequences)
    padded_old = torch.zeros(rows, width, device=device)
    padded_advantages = torch.zeros(rows, width, device=device)
    padded_mask = torch.zeros(rows, width, dtype=torch.bool, device=device)
    rows_logprobs = []
    start = 0
    for row, sequence in enumerate(sequences):
        row_ids, row_positions = pack_sequences([sequence])
        row_labels = next_token_labels(row_ids, row_positions)
        hidden = forward(padded_model, row_ids, row_positions)
        rows_logprobs.append(token_logprobs(padded_model, hidden, row_labels, chunk_size=spec.logprob_chunk)[0])
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
        str(device),
    )
    padded_loss.backward()
    _assert_close(packed_loss.detach(), padded_loss.detach(), spec.tolerances.rl_step, "GRPO loss packed vs padded")
    expected = _grads(padded_model)
    for name, grad in _grads(model).items():
        _assert_close(grad, expected[name], spec.tolerances.rl_step, f"GRPO grad {name}")


def _parallel_worker(
    rank: int, size: int, init_file: str, target: str, spec_json: str, mode: str, path: list[str], device_type: str
):
    sys.path[:] = path
    torch.set_num_threads(1)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device(device_type)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device.type == "cuda" else "gloo",
        init_method=f"file://{init_file}",
        world_size=size,
        rank=rank,
        timeout=timedelta(seconds=120),
        device_id=device if device.type == "cuda" else None,
    )
    try:
        plugin = load_plugin(target)
        spec = ConformanceSpec.model_validate_json(spec_json)
        _parallel_checks(plugin, spec, mode, rank, size, device)
    finally:
        dist.destroy_process_group()


def _parallel_checks(
    plugin: ModelPlugin, spec: ConformanceSpec, mode: str, rank: int, size: int, device: Device = "cpu"
) -> None:
    group = dist.group.WORLD
    atol = spec.tolerances.parallel
    config, baseline = build_fixture(plugin, spec, device)
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
    sequences = fixture_sequences(config, spec, offset=rank if mode == "ep" else 0, device=device)
    input_ids, position_ids = pack_sequences(sequences)
    labels = next_token_labels(input_ids, position_ids)

    unsharded = forward(baseline, input_ids, position_ids)
    baseline_logits = _logits(baseline, unsharded)
    _token_loss(baseline, unsharded, labels, spec).backward()
    output = forward(model, input_ids, position_ids, sp_group=sp_group)
    logits = _logits(model, output)
    _token_loss(model, output, shard_sequence(labels, sp_group), spec).backward()
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


def check_parallel(plugin_target: str, spec: ConformanceSpec, mode: str, size: int, device: Device = "cpu") -> None:
    """Run ``mode`` ('sp', 'ep' or 'sp+ep') on ``size`` ranks and compare with the unsharded model.

    On CPU the ranks are gloo processes; with ``device="cuda"`` each rank drives its own GPU over NCCL.
    """
    kind = torch.device(device).type
    if kind == "cuda" and torch.cuda.device_count() < size:
        raise TierSkipped(f"needs {size} GPUs, found {torch.cuda.device_count()}")
    with tempfile.TemporaryDirectory() as directory:
        spawn_ranks(
            _parallel_worker,
            args=(
                size,
                str(Path(directory) / "rendezvous"),
                plugin_target,
                spec.model_dump_json(),
                mode,
                list(sys.path),
                kind,
            ),
            nprocs=size,
        )


def run_conformance(
    plugin_target: str, spec: ConformanceSpec, weights: str | Path | None = None, device: Device = "cpu"
) -> list[TierResult]:
    """Every applicable tier on ``device``; with ``weights``, also check that real weights directory."""
    plugin = load_plugin(plugin_target)
    capabilities = plugin.manifest.capabilities
    config, _ = build_fixture(plugin, spec)
    has_experts = len(plugin.parallelism(config).moe_layers) > 0
    tiers: list[tuple[str, Callable[[], None]]] = [
        ("static", lambda: check_static(plugin, spec)),
        ("reference_parity", lambda: check_reference_parity(plugin, spec, device)),
        ("packing", lambda: check_packing(plugin, spec, device)),
        ("logprob_chunking", lambda: check_logprob_chunking(plugin, spec, device)),
        ("activation_checkpointing", lambda: check_activation_checkpointing(plugin, spec, device)),
        ("meta_build", lambda: check_meta_build(plugin, spec, device)),
        ("dtype_marks", lambda: check_dtype_marks(plugin, spec, device)),
        ("checkpoint_and_sync", lambda: check_checkpoint_and_sync(plugin, spec, device)),
        ("rl_step", lambda: check_rl_step(plugin, spec, device)),
    ]
    if weights is not None:
        tiers.append(("weights_directory", partial(check_weights_directory, plugin, weights)))
    if capabilities.router_replay and has_experts:
        tiers.append(("router_replay", lambda: check_router_replay(plugin, spec, device)))
    if capabilities.sequence_parallel:
        tiers += [
            (f"sp{size}", partial(check_parallel, plugin_target, spec, "sp", size, device)) for size in spec.sp_sizes
        ]
    if capabilities.expert_parallel and has_experts:
        tiers += [
            (f"ep{size}", partial(check_parallel, plugin_target, spec, "ep", size, device)) for size in spec.ep_sizes
        ]
    if capabilities.sequence_parallel and capabilities.expert_parallel and has_experts:
        both = sorted(set(spec.sp_sizes) & set(spec.ep_sizes))
        tiers += [
            (f"sp{size}+ep{size}", partial(check_parallel, plugin_target, spec, "sp+ep", size, device))
            for size in both
        ]

    results = []
    for name, check in tiers:
        try:
            check()
            results.append(TierResult(name, "PASS"))
        except TierSkipped as skipped:
            results.append(TierResult(name, "SKIP", str(skipped)))
        except Exception as error:  # a failing tier is reported, then the next tier runs
            detail = "".join(traceback.format_exception_only(type(error), error)).strip()
            results.append(TierResult(name, "FAIL", detail))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plugin", required=True, help="'module:attribute' of the ModelPlugin")
    parser.add_argument("--spec", required=True, help="path to the plugin's conformance.toml")
    parser.add_argument("--weights", help="a real Hugging Face weights directory to check (shapes only, no GPU)")
    parser.add_argument("--device", default="cpu", help="'cpu' (gloo ranks) or 'cuda' (one GPU per rank, NCCL)")
    args = parser.parse_args(argv)
    results = run_conformance(args.plugin, load_conformance_spec(args.spec), weights=args.weights, device=args.device)
    width = max(len(result.tier) for result in results)
    for result in results:
        print(f"{result.tier:<{width}}  {result.status}  {result.detail}")
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
