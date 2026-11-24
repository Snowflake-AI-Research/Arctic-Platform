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

import copy
import glob
import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Tuple

from arctic_platform.correctness.harness.sft import SEED
from arctic_platform.correctness.harness.storage import require_data_fast_path

from ..harness.config import ModelSpec
from ..harness.config import hash_directory

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
    from arctic_platform.correctness.harness.inference import original_schema_bytes
    from arctic_platform.correctness.harness.inference import read_export_json
    from arctic_platform.correctness.harness.inference import read_inventory
    from arctic_platform.correctness.harness.inference import schema_sha

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


def sidecar_files(source: str, patterns: Tuple[str, ...] = SIDECAR_PATTERNS) -> List[Path]:
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

        root = snapshot_download(source, allow_patterns=list(patterns))
    found: List[Path] = []
    for pattern in patterns:
        found.extend(Path(path) for path in glob.glob(f"{root}/{pattern}"))
    return sorted(set(found))


def copy_sidecars(source: str, dest_path: Path, patterns: Tuple[str, ...] = SIDECAR_PATTERNS) -> List[Path]:
    """Place the source's tokenizer sidecars beside a reduced model's weights.

    Run whether or not the weights were just written, so a cache produced before this copy existed gains
    them on the next call rather than staying unusable. Copying is idempotent: a hub snapshot's files are
    read-only, so an earlier copy is removed rather than opened for writing, and the replacement is created
    with the ambient mode instead of inheriting the snapshot's.
    """
    dest_path = require_data_fast_path(dest_path, label="reduced model sidecar destination")
    copied: List[Path] = []
    for sidecar in sidecar_files(source, patterns):
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
    index_path = dest_path / "model.safetensors.index.json"
    if not index_path.exists():
        # HF omits the index for a single shard; fixture authorization still needs its exact tensor inventory.
        with (dest_path / "model.safetensors").open("rb") as stream:
            header = json.loads(stream.read(int.from_bytes(stream.read(8), "little")))
        tensors = {name: entry for name, entry in header.items() if name != "__metadata__"}
        index = dict(
            metadata=dict(
                total_size=sum(entry["data_offsets"][1] - entry["data_offsets"][0] for entry in tensors.values())
            ),
            weight_map=dict.fromkeys(tensors, "model.safetensors"),
        )
        index_path.write_text(json.dumps(index, indent=2) + "\n")
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


# The tiny model's sidecars: the reduced model's set, the slow tokenizer's vocabulary, and the original's generation
# defaults, which a freshly built model's save would replace with values derived from its configuration.
TINY_SIDECAR_PATTERNS = (*SIDECAR_PATTERNS, "vocab*", "generation_config.json")

# Decoder layers, vision blocks and routed experts, under the prefixes both checkpoint tensors and quantization module
# lists use (``model.language_model.layers.3``, ``model.layers.3``, ``model.visual.blocks.0``, ``visual.blocks.0``).
_MAIN_LAYER = re.compile(r"^((?:model\.)?(?:language_model\.)?layers\.)(\d+)(\..+)?$")
_VISION_BLOCK = re.compile(r"^((?:.+\.)?visual\.blocks\.)(\d+)(\..+)?$")
_EXPERT = re.compile(r"\.experts\.(\d+)\.")
_MTP_LAYER = re.compile(r"^mtp\.layers\.(\d+)\.")

# The RMSNorm weight that leaves its normalized input unscaled: GLM-5.3 multiplies by the weight, Qwen4-Exp by one
# plus the weight.
_NORM_IDENTITY = {"glm5_next_text": 1.0, "qwen4_exp_text": 0.0}

# Shapes of the prediction-layer tensors Hugging Face does not build, by name suffix: GLM-5.3's projection of the
# concatenated embedding and hidden state with its three RMSNorms, and Qwen3.8's MTP embedding and hidden projections
# with their RMSNorms, the hidden one spanning every hyper-connection stream. Qwen3.8's MTP hyper-connection mixer
# takes the shapes of the decoder's.
_PREDICTION_SHAPES = {
    "eh_proj.weight": lambda text: (text["hidden_size"], 2 * text["hidden_size"]),
    "enorm.weight": lambda text: (text["hidden_size"],),
    "hnorm.weight": lambda text: (text["hidden_size"],),
    "shared_head.norm.weight": lambda text: (text["hidden_size"],),
    "fc_embedding.weight": lambda text: (text["hidden_size"], text["hidden_size"]),
    "fc_hidden.weight": lambda text: (text["hidden_size"], text["hidden_size"]),
    "pre_fc_norm_embedding.weight": lambda text: (text["hidden_size"],),
    "pre_fc_norm_hidden.weight": lambda text: (text["hc_count"] * text["hidden_size"],),
}


def _tiny_name(
    name: str,
    layers: int,
    original_layers: int,
    prediction_layers: int,
    vision_depth: int | None,
    experts: int | None = None,
) -> str | None:
    """``name`` in the tiny model's numbering, or ``None`` when the tiny model drops it.

    Decoder layers below ``layers`` keep their index and the ``prediction_layers`` stored after the original's last
    decoder layer move to follow the tiny model's last. Vision blocks at or past ``vision_depth`` and routed experts
    at or past ``experts`` are dropped.
    """
    if match := _MAIN_LAYER.match(name):
        index = int(match[2])
        if original_layers <= index < original_layers + prediction_layers:
            index += layers - original_layers
        elif index >= layers:
            return None
        name = f"{match[1]}{index}{match[3] or ''}"
    elif vision_depth is not None and (match := _VISION_BLOCK.match(name)) and int(match[2]) >= vision_depth:
        return None
    if experts is not None and (match := _EXPERT.search(name)) and int(match[1]) >= experts:
        return None
    return name


def tiny_config_dict(
    original: dict,
    num_layers: int,
    text_overrides: dict,
    vision_overrides: dict | None = None,
    vision_depth: int | None = None,
) -> dict:
    """The original ``config.json`` with the named fields replaced and every per-layer entry cut to ``num_layers``.

    Every other field keeps the original's value, and an override of a field the original does not carry is refused,
    so the tiny model has exactly the original's fields. A list as long as the original layer count is truncated, and
    the kept prefix must still contain every distinct string the list holds, so no layer or MLP kind disappears. A
    layer-index list in a nested section, such as ``linear_attn_config.kda_layers``, keeps the indices below
    ``num_layers``. A block-FP8 ``modules_to_not_convert`` list is renumbered the way the checkpoint is.
    """
    config = copy.deepcopy(original)
    text = config.get("text_config", config)
    original_layers = text["num_hidden_layers"]
    if type(num_layers) is not int or not 0 < num_layers <= original_layers:
        raise ValueError(f"num_layers must be in [1, {original_layers}], got {num_layers!r}")
    for key, value in list(text.items()):
        if isinstance(value, list) and len(value) == original_layers:
            kept = value[:num_layers]
            if all(isinstance(entry, str) for entry in value) and set(kept) != set(value):
                raise ValueError(f"{num_layers} layers drop {sorted(set(value) - set(kept))} from {key}")
            text[key] = kept
        elif isinstance(value, dict):
            for sub, entries in value.items():
                if (
                    sub.endswith("_layers")
                    and isinstance(entries, list)
                    and all(type(index) is int and 0 <= index < original_layers for index in entries)
                ):
                    value[sub] = [index for index in entries if index < num_layers]
    text["num_hidden_layers"] = num_layers
    vision = config.get("vision_config")
    for section, overrides in ((text, text_overrides), (vision, vision_overrides or {})):
        for key, value in overrides.items():
            if section is None or key not in section:
                raise ValueError(f"the original configuration has no {key!r} to override")
            section[key] = value
    if vision_depth is not None:
        if vision is None:
            raise ValueError("vision_depth was given for a configuration without a vision tower")
        vision["depth" if "depth" in vision else "num_hidden_layers"] = vision_depth
    prediction_layers = text.get("num_nextn_predict_layers", 0)
    for holder in [config] if text is config else [config, text]:
        quantization = holder.get("quantization_config")
        if isinstance(quantization, dict) and isinstance(quantization.get("modules_to_not_convert"), list):
            quantization["modules_to_not_convert"] = [
                renamed
                for module in quantization["modules_to_not_convert"]
                if (renamed := _tiny_name(module, num_layers, original_layers, prediction_layers, vision_depth))
                is not None
            ]
    return config


def config_diff(original: dict, tiny: dict, prefix: str = "") -> List[dict]:
    """Every field whose value differs between two ``config.json`` dictionaries, by dotted key.

    A list longer than 16 entries is summarized by its length and its distinct entries rather than printed whole.
    """

    def summary(value):
        if isinstance(value, list) and len(value) > 16:
            distinct = sorted({json.dumps(entry, sort_keys=True) for entry in value})
            return {"length": len(value), "distinct": distinct if len(distinct) <= 16 else len(distinct)}
        return value

    diff: List[dict] = []
    for key in sorted(original.keys() | tiny.keys()):
        old, new = original.get(key), tiny.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            diff.extend(config_diff(old, new, f"{prefix}{key}."))
        elif old != new:
            diff.append({"key": f"{prefix}{key}", "original": summary(old), "tiny": summary(new)})
    return diff


def checkpoint_schema(root: Path) -> dict:
    """Every tensor's safetensors dtype and shape, read from the shard headers without reading any payload."""
    schema = {}
    for shard in sorted(Path(root).glob("*.safetensors")):
        with shard.open("rb") as stream:
            header = json.loads(stream.read(int.from_bytes(stream.read(8), "little")))
        for name, entry in header.items():
            if name != "__metadata__":
                schema[name] = (entry["dtype"], tuple(entry["shape"]))
    return schema


def _is_prediction(name: str, num_layers: int) -> bool:
    return name.startswith("mtp.") or bool((match := _MAIN_LAYER.match(name)) and int(match[2]) >= num_layers)


def _prediction_layer_count(names, num_layers: int) -> int:
    main = {int(match[2]) for name in names if (match := _MAIN_LAYER.match(name)) and int(match[2]) >= num_layers}
    mtp = {num_layers + int(match[1]) for name in names if (match := _MTP_LAYER.match(name))}
    if main and mtp:
        raise ValueError("the original stores prediction layers both after its decoder layers and under mtp.layers")
    found = main or mtp
    if found != set(range(num_layers, num_layers + len(found))):
        raise ValueError(f"the original's prediction layers are not contiguous: {sorted(found)}")
    return len(found)


def _with_prediction_layers(config: dict, count: int) -> Tuple[dict, dict]:
    """``config`` with ``count`` prediction layers appended as decoder layers, and their built-to-stored prefixes.

    Hugging Face builds neither family's prediction layers, so they are built as the decoder layers they are:
    GLM-5.3 stores a sparse-MLA MoE layer after its last decoder layer, and Qwen3.8 stores its MTP layers' decoder
    blocks under ``mtp.layers``.
    """
    config = copy.deepcopy(config)
    text = config.get("text_config", config)
    layers = text["num_hidden_layers"]
    renames = {}
    if text.get("model_type") == "glm5_next_text":
        sparse = text["layer_types"].index("deepseek_sparse_attention")
        text["layer_types"] += ["deepseek_sparse_attention"] * count
        text["mlp_layer_types"] += ["sparse"] * count
        text["indexer_types"] += [text["indexer_types"][sparse]] * count
        text["linear_attn_config"]["full_attn_layers"] += list(range(layers, layers + count))
    elif text.get("model_type") == "qwen4_exp_text":
        mtp_types = list(text["mtp"]["layer_types"])
        if len(mtp_types) != count:
            raise ValueError(f"the original stores {count} MTP layers and declares {len(mtp_types)} MTP layer types")
        text["layer_types"] += mtp_types
        renames = {f"model.language_model.layers.{layers + j}.": f"mtp.layers.{j}." for j in range(count)}
    else:
        raise ValueError(f"no prediction-layer layout is known for {text.get('model_type')!r}")
    text["num_hidden_layers"] = layers + count
    return config, renames


def _built_layouts(config: dict, seed: int, scratch: Path) -> List[dict]:
    """The tensors a model built from ``config`` with ``seed`` saves, in both layouts Hugging Face writes.

    The first has the architecture's load-time conversions reverted, which stores one tensor per expert as GLM-5.3
    checkpoints do; the second keeps the layout the model holds in memory, with fused experts as Qwen3.8 checkpoints
    store them.
    """
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.conversion_mapping import get_model_conversion_mapping

    (scratch / "config").mkdir(parents=True)
    (scratch / "config" / "config.json").write_text(json.dumps(config))
    cfg = AutoConfig.from_pretrained(scratch / "config")
    torch.manual_seed(seed)
    model = getattr(transformers, cfg.architectures[0])(cfg)
    layouts = []
    for index, conversions in enumerate((get_model_conversion_mapping(model, add_legacy=False), [])):
        # ``save_pretrained`` reverts exactly the conversions attached to the model; none keeps the in-memory names.
        model._weight_conversions = conversions
        _save_reduced_pretrained(model, scratch / f"saved-{index}")
        tensors = {}
        for shard in sorted((scratch / f"saved-{index}").glob("*.safetensors")):
            tensors.update(load_file(shard))
        layouts.append(tensors)
    del model
    return layouts


def _split_ngram_tables(produced: dict, target: dict) -> None:
    """Store each Qwen3.8 n-gram embedding table as the ``shard_<i>`` row blocks the original checkpoint holds."""
    for name in [name for name in produced if name.endswith(".ngram_embedding.weight") and name not in target]:
        prefix = name.removesuffix(".weight")
        shard = re.compile(rf"{re.escape(prefix)}\.shard_\d+\.weight")
        parts = sum(1 for other in target if shard.fullmatch(other))
        if parts:
            for index, block in enumerate(produced.pop(name).tensor_split(parts, dim=0)):
                produced[f"{prefix}.shard_{index}.weight"] = block.contiguous()


def _prediction_tensor(name: str, text: dict, produced: dict, generator):
    import torch

    shape = None
    for suffix, rule in _PREDICTION_SHAPES.items():
        if name == suffix or name.endswith(f".{suffix}"):
            shape = rule(text)
    if shape is None and name.startswith("mtp.") and (main := f"model.language_model.{name[4:]}") in produced:
        shape = tuple(produced[main].shape)
    if shape is None:
        raise ValueError(f"no shape is known for the prediction-layer tensor {name}")
    if len(shape) == 1:
        return torch.full(shape, _NORM_IDENTITY[text["model_type"]])
    return torch.empty(shape).normal_(0.0, text["initializer_range"], generator=generator)


def _checkpoint_tensors(produced: dict, target: dict, block: int | None) -> dict:
    """``produced`` in the original's dtypes, with block-FP8 weights and their inverse scales where it is quantized."""
    import torch

    from arctic_platform.model.implementations.fp8 import quantize_to_fp8_blockwise

    dtypes = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "I64": torch.int64,
        "I32": torch.int32,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
    }
    scales = {f"{name}_scale_inv" for name, (dtype, _) in target.items() if dtype == "F8_E4M3"}
    missing = sorted(target.keys() - produced.keys() - scales)
    unexpected = sorted(produced.keys() - target.keys())
    unscaled = sorted(scales - target.keys())
    if missing or unexpected or unscaled:
        raise ValueError(
            f"the tiny tensors do not match the original's layout: {len(missing)} missing {missing[:8]}, "
            f"{len(unexpected)} unexpected {unexpected[:8]}, {len(unscaled)} FP8 weights without a stored scale "
            f"{unscaled[:8]}"
        )
    tensors = {}
    for name, tensor in produced.items():
        dtype, shape = target[name]
        if tensor.dim() != len(shape):
            raise ValueError(f"{name} has rank {tensor.dim()} and the original's tensor rank {len(shape)}")
        if dtype == "F8_E4M3":
            tensors[name], tensors[f"{name}_scale_inv"] = quantize_to_fp8_blockwise(tensor.to(torch.bfloat16), block)
        else:
            tensors[name] = tensor.to(dtypes[dtype]).contiguous()
    wrong = sorted(name for name, tensor in tensors.items() if tensor.dtype != dtypes[target[name][0]])
    if wrong:
        raise ValueError(f"{len(wrong)} tiny tensors are not in the original's dtype: {wrong[:8]}")
    return tensors


def _write_checkpoint(tensors: dict, dest: Path, max_shard_bytes: int = 2 * 1024**3) -> None:
    """Write ``tensors`` as sorted safetensors shards and a ``model.safetensors.index.json``, the hub layout."""
    from safetensors.torch import save_file

    shards: List[dict] = [{}]
    size = 0
    for name in sorted(tensors):
        nbytes = tensors[name].numel() * tensors[name].element_size()
        if shards[-1] and size + nbytes > max_shard_bytes:
            shards.append({})
            size = 0
        shards[-1][name] = tensors[name]
        size += nbytes
    weight_map = {}
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, dest / filename, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(shard, filename))
    total = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (dest / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_tiny(
    source: str,
    dest: str,
    *,
    num_layers: int,
    text_overrides: dict,
    vision_overrides: dict | None = None,
    vision_depth: int | None = 1,
    seed: int = SEED,
    force: bool = False,
) -> dict:
    """Write a random-weight model with the original's architecture and checkpoint layout at reduced size to ``dest``.

    ``materialize`` keeps every dimension and writes an unquantized BF16 model without prediction layers. This keeps
    the original's configuration fields with the named dimensions replaced (``tiny_config_dict``) and writes the
    checkpoint the original's loaders read: the same tensor names, with prediction layers renumbered after the last
    kept decoder layer and experts and vision blocks past the reduced counts dropped; every tensor in the original's
    dtype; block-FP8 weights with their ``weight_scale_inv`` wherever the original is quantized; and the original's
    tokenizer, processor and generation sidecars. Weights are random from ``seed``. An FP8 original and its BF16
    sibling reduced with the same fields and seed get the same weights, the FP8 copy block-quantized from the BF16
    values.

    Returns the manifest, also written beside ``dest`` as ``<name>.manifest.json``: the configuration diff against the
    original, the model's parameter count, the stored tensor elements, each file's SHA-256, and the content hash a
    ``ModelSpec`` binds.
    """
    import torch
    from transformers import AutoConfig

    dest_path = require_data_fast_path(dest, label="tiny model destination")
    if dest_path.exists() and not force:
        raise FileExistsError(f"{dest_path} exists; rebuild it with force=True")
    root = Path(source)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(source, local_files_only=True))
    original = json.loads((root / "config.json").read_text())
    tiny = tiny_config_dict(original, num_layers, text_overrides, vision_overrides, vision_depth)
    original_text = original.get("text_config", original)
    text = tiny.get("text_config", tiny)
    experts = text.get("n_routed_experts", text.get("num_experts"))
    target = {}
    for name, entry in checkpoint_schema(root).items():
        renamed = _tiny_name(
            name,
            num_layers,
            original_text["num_hidden_layers"],
            original_text.get("num_nextn_predict_layers", 0),
            vision_depth,
            experts,
        )
        if renamed is not None:
            target[renamed] = entry
    block = None
    if any(dtype == "F8_E4M3" for dtype, _ in target.values()):
        quantization = tiny.get("quantization_config", text.get("quantization_config")) or {}
        block_size = quantization.get("weight_block_size")
        if quantization.get("quant_method") != "fp8" or not block_size or len(set(block_size)) != 1:
            raise ValueError("an FP8 original needs a square block-FP8 quantization_config")
        block = block_size[0]

    build = copy.deepcopy(tiny)
    for holder in [build] if "text_config" not in build else [build, build["text_config"]]:
        holder.pop("quantization_config", None)
    count = _prediction_layer_count(target, num_layers)
    renames = {}
    if count:
        build, renames = _with_prediction_layers(build, count)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest_path.parent, prefix=f".{dest_path.name}.") as scratch:
        scratch = Path(scratch)
        layouts = []
        for layout in _built_layouts(build, seed, scratch / "build"):
            produced = {}
            for name, tensor in layout.items():
                for built, stored in renames.items():
                    if name.startswith(built):
                        name = stored + name[len(built) :]
                        break
                produced[name] = tensor
            _split_ngram_tables(produced, target)
            layouts.append(produced)
        # Compared only after the renames and the n-gram split, which both layouts need, so that hundreds of n-gram
        # shard names cannot outweigh the expert names that tell the layouts apart.
        produced = max(layouts, key=lambda layout: len(target.keys() & layout.keys()))
        del layouts
        for name in [name for name in produced if _is_prediction(name, num_layers) and name not in target]:
            # A prediction layer is a decoder layer without the parts that only the decoder stack carries, such as
            # GLM-5.3's per-layer hyper-connection mixers.
            del produced[name]
        generator = torch.Generator().manual_seed(seed)
        for name in sorted(target):
            if _is_prediction(name, num_layers) and name not in produced and not name.endswith("_scale_inv"):
                produced[name] = _prediction_tensor(name, text, produced, generator)
        tensors = _checkpoint_tensors(produced, target, block)
        del produced
        staged = scratch / "model"
        staged.mkdir()
        _write_checkpoint(tensors, staged)
        (staged / "config.json").write_text(json.dumps(tiny, indent=2) + "\n")
        copy_sidecars(str(root), staged, TINY_SIDECAR_PATTERNS)
        if dest_path.exists():
            shutil.rmtree(dest_path)
        staged.rename(dest_path)

    manifest = {
        "path": str(dest_path),
        "source": source,
        "source_path": str(root),
        "seed": seed,
        "num_hidden_layers": num_layers,
        "prediction_layers": count,
        "vision_depth": vision_depth,
        "config_diff": config_diff(original, tiny),
        "parameter_count": count_parameters(AutoConfig.from_pretrained(dest_path)),
        "stored_tensor_count": len(tensors),
        "stored_tensor_elements": sum(tensor.numel() for tensor in tensors.values()),
        "stored_tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
        "content_hash": hash_directory(dest_path),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
            for path in sorted(dest_path.iterdir())
            if path.is_file()
        },
    }
    (dest_path.parent / f"{dest_path.name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
