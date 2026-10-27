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

"""Build and materialize the reduced model both engines load.

Every dimension of the source checkpoint is preserved and only the layer count is reduced, which keeps the
architecture-specific kernels in the Arctic Platform path under test. A generic small transformer would exercise none
of them.

The kept layers carry the checkpoint's own weights. A freshly initialized model on uniform random tokens
sits at maximum entropy, where each parameter's gradient is a small residual of many opposing per-token
contributions and bf16 rounding is largest relative to the result; comparing two engines there measures the
fixture. On a 2,048-token case the random-init model put Arctic Platform 4.98e-04 above the reference with 312 of 374
tensors on the same side, and the checkpoint's weights bring that to -2.01e-04 with a 161/374 split.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Tuple

from ..harness.batches import SEED
from ..harness.config import ModelSpec
from ..harness.config import hash_directory
from ..harness.storage import require_data_fast_path

# A single H200 must hold the reference's weights, its fp32 gradient accumulator, the live gradients of
# one backward pass, one microbatch of activations, and the chunked cross-entropy workspace. The margin
# below that keeps the largest case from sitting at the edge of the card.
SINGLE_GPU_BUDGET_GIB = 100.0

# bfloat16 weights (2) + the fp32 gradient accumulator (4) + the bfloat16 gradients a backward pass leaves
# on the parameters before they are drained into that accumulator (2). All three are resident at once, so
# sizing on weights and accumulator alone understates the requirement by a quarter and selects a layer
# count that does not fit.
BYTES_PER_PARAM = 8


def describe_source_model(source: str) -> tuple[ModelSpec, int, dict]:
    """Read original config and tensor headers for reference-free checks, without constructing a model.

    Header element counts include buffers and do not establish architecture parameter counts. Neither a
    reference sizing measurement nor an initialization seed exists for this original pretrained model.
    """
    from arctic_platform.correctness.harness.inference_export import original_schema_bytes
    from arctic_platform.correctness.harness.inference_export import read_export_json
    from arctic_platform.correctness.harness.inference_export import read_inventory
    from arctic_platform.correctness.harness.inference_export import schema_sha

    root = Path(source)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(source, local_files_only=True))
    canonical = require_data_fast_path(root, label="reference-free original model")
    if root.is_symlink() or root.absolute() != canonical or not root.is_dir():
        raise ValueError("reference-free metadata requires a canonical regular original model directory")
    root = canonical

    def fingerprint(path: Path) -> tuple[int, int, int, int, int]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    index_path = root / "model.safetensors.index.json"

    def metadata_paths() -> list[Path]:
        result = [root, root / "config.json", *sorted(root.glob("*.safetensors"))]
        if index_path.exists() or index_path.is_symlink():
            result.append(index_path)
        return result

    paths = metadata_paths()
    if any(path.is_symlink() for path in paths):
        raise ValueError("reference-free metadata requires regular original config and weight files")
    before = {path: fingerprint(path) for path in paths}
    config = read_export_json(root / "config.json")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("original text configuration object required")
    for name in ("num_hidden_layers", "vocab_size"):
        if type(text.get(name)) is not int or text[name] <= 0:
            raise ValueError(f"original positive integer {name} required")
    quantization_config = config.get("quantization_config", text.get("quantization_config"))
    vision = config.get("vision_config")
    vision_depth = None
    if vision is not None:
        if not isinstance(vision, dict):
            raise ValueError("original vision configuration object required")
        vision_depth = vision.get("depth", vision.get("num_hidden_layers"))
        if type(vision_depth) is not int or vision_depth <= 0:
            raise ValueError("original positive integer vision depth required")

    if index_path in before:
        index = read_export_json(index_path)
        index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    else:
        from safetensors import safe_open

        weights = root / "model.safetensors"
        if weights not in before:
            raise ValueError("original safetensors index or single model.safetensors required")
        with safe_open(str(weights), framework="pt", device="cpu") as stream:
            schema = {
                name: [stream.get_slice(name).get_dtype(), stream.get_slice(name).get_shape()]
                for name in stream.keys()
            }
        index = {
            "metadata": {"total_size": original_schema_bytes(schema, quantization_config)},
            "weight_map": {name: weights.name for name in schema},
        }
        index_sha = None
    schema = read_inventory(root, index, original_quantization_config=quantization_config)
    evidence = {
        "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
        "index_sha256": index_sha,
        "tensor_schema_sha256": schema_sha(schema),
        "tensor_count": len(schema),
        "stored_tensor_elements": sum(math.prod(shape) for _, shape in schema.values()),
        "tensor_payload_bytes": original_schema_bytes(schema, quantization_config),
        "tensor_payloads_read": False,
    }
    if quantization_config is not None:
        evidence["original_quantization_config"] = quantization_config
    model = ModelSpec(
        source_checkpoint=source,
        num_hidden_layers=text["num_hidden_layers"],
        vision_depth=vision_depth,
        seed=None,
        param_count=None,
        cache_path=str(root),
        content_hash=hash_directory(root),
        sized_against_gib=None,
    )
    after_paths = metadata_paths()
    if (
        paths != after_paths
        or any(path.is_symlink() for path in after_paths)
        or before != {path: fingerprint(path) for path in after_paths}
    ):
        raise ValueError("original source metadata or weights changed during preparation")
    return model, text["vocab_size"], evidence


def detect_block_period(layer_types: List[str]) -> int:
    """Smallest period whose repetitions have the layer list as a prefix.

    Derived from the list rather than read from an interval field, because not every architecture has such
    a field and the ones that do are not always consistent with the list.
    """
    n = len(layer_types)
    for p in range(1, n + 1):
        if all(layer_types[i] == layer_types[i % p] for i in range(n)):
            return p
    return n


def reduce_config(source: str, num_layers: int, vision_depth: int = 2):
    """Rewrite the model config to ``num_layers``, truncating every entry correlated with the layer count.

    Correlated entries are found by shape rather than by name: any list whose length equals the original
    layer count is truncated alongside it. That catches per-layer type lists without naming them, which is
    what lets this work on an architecture the harness has not seen.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(source, trust_remote_code=True)
    text = getattr(cfg, "text_config", cfg)
    original = int(text.num_hidden_layers)

    period = detect_block_period(list(getattr(text, "layer_types", []))) if hasattr(text, "layer_types") else 1
    if period > 1 and num_layers % period != 0:
        raise ValueError(
            f"num_layers={num_layers} must be a multiple of the block period {period}; a partial block "
            "would drop a layer type from the comparison entirely"
        )

    text.num_hidden_layers = num_layers
    for attr in dir(text):
        if attr.startswith("_"):
            continue
        value = getattr(text, attr, None)
        if isinstance(value, list) and len(value) == original:
            setattr(text, attr, list(value[:num_layers]))

    if hasattr(cfg, "vision_config") and vision_depth is not None:
        cfg.vision_config.depth = vision_depth
    return cfg, original, period


def _meta_model(cfg):
    import torch
    import transformers

    cls = getattr(transformers, cfg.architectures[0])
    with torch.device("meta"):
        return cls(cfg)


def count_parameters(cfg) -> int:
    """Exact full parameter count with no tensor payload allocation or weight loading."""
    return sum(p.numel() for p in _meta_model(cfg).parameters())


@dataclass(frozen=True)
class ReferenceParameterResidency:
    total_parameter_elements: int
    cpu_frozen_parameter_elements: int
    accelerator_frozen_parameter_elements: int
    accelerator_trainable_parameter_elements: int
    cpu_frozen_weight_bytes: int
    accelerator_weight_bytes: int
    accelerator_live_gradient_bytes: int
    cpu_gradient_accumulator_bytes: int
    accelerator_buffer_bytes: int
    conservative_accelerator_parameter_charge_bytes: int
    sizing_scope: str = "bf16-forward-backward-parameters-only-not-peak-runtime-or-optimizer-admission"


def reference_parameter_residency(cfg, bytes_per_param: int = BYTES_PER_PARAM) -> ReferenceParameterResidency:
    """Count unique architecture parameters by the maintained BF16 reference's placement policy.

    CPU bytes are obligations, not available-memory claims. The Qwen4 charge retains the conservative
    trainable rate (eight bytes by default) and charges accelerator-frozen weights at two bytes each.
    Buffers are reported separately; activations, workspace and optimizer state are outside this estimate.
    Other families retain the original all-parameter charge, regardless of their trainability.
    """
    model = _meta_model(cfg)
    qwen4 = getattr(cfg, "model_type", None) in {"qwen4_exp", "qwen4_exp_text"}
    cpu_ids = frozenset()
    frozen_ids = frozenset()
    if qwen4:
        from ..reference.qwen4_reference import qwen4_parameter_policy

        policy = qwen4_parameter_policy(model)
        cpu_ids = policy.cpu_parameter_ids
        frozen_ids = policy.frozen_parameter_ids
    cpu_frozen = accelerator_frozen = accelerator_trainable = 0
    for parameter in model.parameters():
        if id(parameter) in cpu_ids:
            cpu_frozen += parameter.numel()
        elif id(parameter) in frozen_ids or not parameter.requires_grad:
            accelerator_frozen += parameter.numel()
        else:
            accelerator_trainable += parameter.numel()
    total = cpu_frozen + accelerator_frozen + accelerator_trainable
    charge = accelerator_trainable * bytes_per_param + accelerator_frozen * 2 if qwen4 else total * bytes_per_param
    return ReferenceParameterResidency(
        total_parameter_elements=total,
        cpu_frozen_parameter_elements=cpu_frozen,
        accelerator_frozen_parameter_elements=accelerator_frozen,
        accelerator_trainable_parameter_elements=accelerator_trainable,
        cpu_frozen_weight_bytes=cpu_frozen * 2,
        accelerator_weight_bytes=(accelerator_frozen + accelerator_trainable) * 2,
        accelerator_live_gradient_bytes=accelerator_trainable * 2,
        cpu_gradient_accumulator_bytes=accelerator_trainable * 4,
        accelerator_buffer_bytes=sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()),
        conservative_accelerator_parameter_charge_bytes=charge,
    )


def size_for_single_gpu(
    source: str, budget_gib: float = SINGLE_GPU_BUDGET_GIB, bytes_per_param: int = BYTES_PER_PARAM
) -> Tuple[int, int]:
    """Largest whole-block layer count whose reference parameter residency fits the budget.

    Activations and the cross-entropy workspace are not counted here; they are bounded separately by the
    reference's own microbatch token budget, and the margin in ``budget_gib`` covers them. CPU table and
    accumulator capacity, buffer residency and optimizer peak memory are not admitted by this selection.
    """
    from transformers import AutoConfig

    probe = AutoConfig.from_pretrained(source, trust_remote_code=True)
    text = getattr(probe, "text_config", probe)
    full = int(text.num_hidden_layers)
    period = detect_block_period(list(getattr(text, "layer_types", []))) or 1

    for n in range(full - (full % period), 0, -period):
        cfg, _, _ = reduce_config(source, n)
        if getattr(cfg, "model_type", None) in {"qwen4_exp", "qwen4_exp_text"}:
            residency = reference_parameter_residency(cfg, bytes_per_param=bytes_per_param)
            params = residency.total_parameter_elements
            charge = residency.conservative_accelerator_parameter_charge_bytes
        else:
            params = count_parameters(cfg)
            charge = params * bytes_per_param
        if charge / 1024**3 <= budget_gib:
            return n, params
    raise RuntimeError(f"no layer count fits {budget_gib} GiB; the embedding pair alone may exceed it")


# Tokenizer, merge table, vocabulary, chat template and image/video processor files: everything a serving engine
# needs beside the weights, and nothing a reduced ``save_pretrained`` writes for itself. vLLM builds a multimodal
# checkpoint's input processor at engine start and refuses to initialize without ``preprocessor_config.json``.
SIDECAR_PATTERNS = ("*token*", "*.txt", "*processor*", "chat_template*")


def sidecar_files(source: str) -> List[Path]:
    """The source checkpoint's tokenizer sidecars, as paths on this filesystem.

    ``source`` is a Hugging Face hub id as often as it is a directory, and a glob over a hub id matches
    nothing. Resolving the snapshot first makes both forms local. The failure this prevents is silent in
    two stages: a reduced model written without tokenizer sidecars still loads, and
    ``AutoTokenizer.from_pretrained`` on it returns a single-entry vocabulary that encodes every string to
    an empty id list rather than raising. The weights-only checkpoint a serving engine is pointed at then
    inherits the gap, because that save copies its sidecars from the job's model directory.
    """
    root = source
    if not Path(source).is_dir():
        from huggingface_hub import snapshot_download

        root = snapshot_download(source, allow_patterns=list(SIDECAR_PATTERNS))
    found: List[Path] = []
    for pattern in SIDECAR_PATTERNS:
        found.extend(Path(path) for path in glob.glob(f"{root}/{pattern}"))
    return sorted(set(found))


def copy_sidecars(source: str, dest_path: Path) -> List[Path]:
    """Place the source's tokenizer sidecars beside a reduced model's weights.

    Run whether or not the weights were just written, so a cache produced before this copy existed gains
    them on the next call rather than staying unusable. Copying is idempotent: a hub snapshot's files are
    read-only, so an earlier copy is removed rather than opened for writing, and the replacement is created
    with the ambient mode instead of inheriting the snapshot's.
    """
    dest_path = require_data_fast_path(dest_path, label="reduced model sidecar destination")
    copied: List[Path] = []
    for sidecar in sidecar_files(source):
        target = dest_path / sidecar.name
        target.unlink(missing_ok=True)
        shutil.copyfile(sidecar, target)
        copied.append(target)
    if not any(path.name == "tokenizer_config.json" for path in copied):
        raise RuntimeError(
            f"no tokenizer sidecars were found for {source!r}; a reduced model without them loads but "
            "tokenizes every string to nothing, and so does the weights-only checkpoint saved from it"
        )
    return copied


def _save_reduced_pretrained(model, dest_path: Path) -> None:
    generation = getattr(model, "generation_config", None)
    preserve_legacy_defaults = False
    if generation is not None:
        try:
            generation.validate(strict=True)
        except ValueError:
            # Loading permits inherited, inactive generation flags; keep hard-invalid values rejected.
            generation.validate()
            preserve_legacy_defaults = True
            model.generation_config = type(generation)()
    try:
        model.save_pretrained(dest_path, safe_serialization=True, max_shard_size="5GB")
    finally:
        if preserve_legacy_defaults:
            model.generation_config = generation
    if preserve_legacy_defaults:
        generation.to_json_file(dest_path / "generation_config.json", use_diff=True, keys_to_pop=["compile_config"])


def _load_reduced_pretrained(model_class, source: str, cfg, num_layers: int, original_layers: int, **options):
    import torch
    from transformers import PreTrainedModel

    if num_layers >= original_layers:
        return model_class.from_pretrained(source, config=cfg, **options)

    def adjust_loading_info(model, loading_info):
        model_class._adjust_missing_and_unexpected_keys(model, loading_info)
        text_config = getattr(model.config, "text_config", model.config)
        omitted_prefixes = []
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.ModuleList) or len(module) != num_layers:
                continue
            parent = model.get_submodule(name.rpartition(".")[0])
            if isinstance(parent, PreTrainedModel) and parent.config is text_config:
                omitted_prefixes.extend(f"{name}.{index}." for index in range(num_layers, original_layers))
        loading_info.unexpected_keys = {
            key for key in loading_info.unexpected_keys if not key.startswith(tuple(omitted_prefixes))
        }

    # Scope reporting rules to this load, preserving architecture-specific conversion names and model identity.
    loading_class = type(
        model_class.__name__,
        (model_class,),
        {
            "__module__": model_class.__module__,
            "_adjust_missing_and_unexpected_keys": adjust_loading_info,
        },
    )
    model = loading_class.from_pretrained(source, config=cfg, **options)
    model.__class__ = model_class
    return model


def materialize_pretrained(
    source: str, dest: str, num_layers: int, vision_depth: int = 2, force: bool = False
) -> ModelSpec:
    """Write the reduced config filled with the source checkpoint's own weights for the layers it keeps.

    The layers above ``num_layers`` are unexpected keys and are dropped. The result is not a trained model
    -- a prefix of a stack feeding the final norm and the lm_head predicts badly -- but every tensor holds
    a value the checkpoint was trained to, so the gradients are not the near-cancelling residual that a
    fresh initializer on random tokens produces.

    The written tree carries the source's tokenizer sidecars, so it can be tokenized against and a
    weights-only checkpoint saved from a job that loaded it is self-contained for a serving engine.
    """
    dest_path = require_data_fast_path(dest, label="reduced pretrained model destination")

    import torch
    import transformers

    cfg, original_layers, _ = reduce_config(source, num_layers, vision_depth)
    load_options = {}
    quantization = getattr(cfg, "quantization_config", None)
    if quantization is not None:
        quantization = quantization.to_dict() if hasattr(quantization, "to_dict") else dict(quantization)
        if quantization.get("quant_method") != "fp8":
            raise ValueError(
                "reduced pretrained fixtures support only unquantized or explicitly dequantized FP8 sources"
            )
        quantization["dequantize"] = True
        load_options["quantization_config"] = transformers.FineGrainedFP8Config.from_dict(quantization)

    if force or not (dest_path / "config.json").exists():
        dest_path.mkdir(parents=True, exist_ok=True)
        cls = getattr(transformers, cfg.architectures[0])
        model = _load_reduced_pretrained(
            cls,
            source,
            cfg,
            num_layers,
            original_layers,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            **load_options,
        )
        if quantization is not None:
            if (
                getattr(model, "hf_quantizer", None) is not None
                or getattr(model.config, "quantization_config", None) is not None
                or getattr(model, "is_quantized", False)
            ):
                raise ValueError("the FP8 source loader did not complete explicit dequantization")
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                if name.endswith(("weight_scale_inv", "activation_scale")) or (
                    tensor.is_floating_point() and tensor.dtype not in (torch.bfloat16, torch.float32)
                ):
                    raise ValueError(f"the dequantized fixture still contains a quantized tensor: {name}")
            from transformers.conversion_mapping import get_model_conversion_mapping

            # Undo architecture transforms when saving, but never the already-completed FP8 dequantization.
            model._weight_conversions = get_model_conversion_mapping(model, add_legacy=False)
        _save_reduced_pretrained(model, dest_path)
        del model
    elif json.loads((dest_path / "config.json").read_text()).get("quantization_config") is not None:
        raise ValueError("the cached reduced fixture is quantized; rebuild it with force=True")
    copy_sidecars(source, dest_path)

    params = count_parameters(cfg)
    return ModelSpec(
        source_checkpoint=source,
        num_hidden_layers=num_layers,
        vision_depth=vision_depth,
        seed=0,
        param_count=params,
        cache_path=str(dest_path),
        content_hash=hash_directory(dest_path),
        sized_against_gib=SINGLE_GPU_BUDGET_GIB,
    )


def materialize(
    source: str, dest: str, num_layers: int, vision_depth: int = 2, seed: int = SEED, force: bool = False
) -> ModelSpec:
    """Generate the model once and write it to ``dest``; both engines then load that directory by path.

    The weights are regenerable from the seed, so ``dest`` is a cache rather than an input. A caller that
    finds a matching content hash reuses it; anything else rebuilds.
    """
    dest_path = require_data_fast_path(dest, label="reduced model destination")

    import torch
    import transformers

    cfg, _, _ = reduce_config(source, num_layers, vision_depth)
    # These weights are initialized in BF16, not loaded from the source's quantized tensors.
    for config in (cfg, getattr(cfg, "text_config", cfg)):
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")

    if force or not (dest_path / "config.json").exists():
        dest_path.mkdir(parents=True, exist_ok=True)
        cls = getattr(transformers, cfg.architectures[0])
        torch.manual_seed(seed)
        model = cls(cfg).to(torch.bfloat16)
        model.save_pretrained(dest_path, safe_serialization=True, max_shard_size="5GB")
        for extra in glob.glob(f"{source}/*token*") + glob.glob(f"{source}/*.txt"):
            shutil.copy2(extra, dest_path)
        del model
    else:
        cached = json.loads((dest_path / "config.json").read_text())
        if any(config.get("quantization_config") is not None for config in (cached, cached.get("text_config", {}))):
            raise ValueError("the cached reduced fixture is quantized; rebuild it with force=True")

    params = count_parameters(cfg)
    return ModelSpec(
        source_checkpoint=source,
        num_hidden_layers=num_layers,
        vision_depth=vision_depth,
        seed=seed,
        param_count=params,
        cache_path=str(dest_path),
        content_hash=hash_directory(dest_path),
        sized_against_gib=SINGLE_GPU_BUDGET_GIB,
    )
