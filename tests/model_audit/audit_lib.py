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
"""Model-agnostic pieces of the model audit: profiles, tiny checkpoints, inputs, metrics.

Not collected by pytest. Imported by the tests and by the probe subprocesses, so it must not import CUDA-only or
vLLM-only modules at import time.

A *profile* (``profiles/<name>.yaml``) is the only per-model input. It names the source checkpoint, the
configuration we will launch first, how to shrink the config into a tiny random-init checkpoint that keeps every
layer kind, the token positions where code paths switch, and the known issues that turn a check into a strict
xfail. Everything else is derived from the checkpoint's own ``config.json``.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

import yaml

AUDIT_DIR = Path(__file__).resolve().parent
PROFILES_DIR = AUDIT_DIR / "profiles"

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)
_EXPERT_COUNT_KEYS = ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts")
_EXPERT_TOPK_KEYS = ("num_experts_per_tok", "moe_topk", "top_k")
_SUBCONFIG_DEPTH_KEYS = ("depth", "num_hidden_layers", "num_layers", "encoder_layers")
# Config keys whose value is a token position where an attention or rotary code path changes.
_THRESHOLD_KEYS = ("sliding_window", "attention_chunk_size", "original_max_position_embeddings")

CHECK_IDS: dict[str, str] = {}


def register_check(check_id: str, claim: str) -> str:
    """Record a check id and the claim it tries to disprove; returns the id for use in a marker."""
    assert check_id not in CHECK_IDS, f"duplicate check id {check_id!r}"
    CHECK_IDS[check_id] = claim
    return check_id


def audit_root() -> Path:
    root = Path(os.environ.get("AP_MODEL_AUDIT_DIR") or Path(tempfile.gettempdir()) / "ap-model-audit")
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    path: Path
    raw: dict

    @property
    def checkpoint(self) -> str:
        return os.path.expandvars(self.raw["checkpoint"])

    @property
    def launch(self) -> dict:
        return self.raw.get("launch", {})

    @property
    def tiny(self) -> dict:
        return self.raw.get("tiny", {})

    @property
    def expect(self) -> dict:
        return self.raw.get("expect", {})

    @property
    def tolerances(self) -> dict:
        defaults = {
            "noise_factor": 4.0,
            "noise_slack": 2e-3,
            "adamw_mismatch_fraction": 1e-3,
            "max_lost_update_fraction": 0.5,
        }
        return {**defaults, **self.raw.get("tolerances", {})}

    @property
    def dtype(self) -> str:
        return self.launch.get("dtype", "bfloat16")

    def known_issue(self, check_id: str) -> dict | None:
        return (self.raw.get("known_issues") or {}).get(check_id)

    def not_applicable(self, check_id: str) -> str | None:
        return (self.raw.get("not_applicable") or {}).get(check_id)

    def trainer_spec(self, *, model_path: str, tiny: bool = True, parallelism: dict | None = None) -> dict:
        """``ModelSpec`` keyword arguments for the trainer, as the platform sends them for the launch config."""
        spec = copy.deepcopy(self.launch.get("trainer", {}))
        if tiny:
            spec = _deep_merge(spec, self.tiny.get("trainer", {}))
        if parallelism is not None:
            spec["parallelism"] = {**spec.get("parallelism", {}), **parallelism}
        spec["model_path_or_name"] = model_path
        spec.setdefault("dtype", self.dtype)
        return spec

    def engine(self) -> dict:
        """``TrainingEngineSpec`` fields other than dtype, world size, SP degree and optimizer."""
        return copy.deepcopy(self.launch.get("engine", {}))

    def tiny_parallelism(self) -> dict:
        return self.trainer_spec(model_path="", tiny=True).get("parallelism", {})

    @property
    def baseline_gpus(self) -> int:
        """GPUs of the baseline layout: the smallest one the production loader builds (EP > 1 for most MoE models)."""
        return layout_world(self.tiny_parallelism())

    def sampler_args(self, *, tiny: bool = True) -> dict:
        args = copy.deepcopy(self.launch.get("sampler", {}))
        if tiny:
            args = _deep_merge(args, self.tiny.get("sampler", {}))
        return args

    @property
    def sync_formats(self) -> list[str]:
        """Weight-export formats production ships to the sampler's receiver."""
        return list(self.raw.get("probe", {}).get("sync_formats", ["vllm"]))

    @property
    def sampler_gpus(self) -> int:
        return int(self.sampler_args().get("tensor_parallel_size", 1))

    def optimizer(self, lr: float | None = None) -> dict:
        """The launch optimizer; ``lr`` overrides the learning rate (the direction check uses a visible one)."""
        settings = {
            "lr": 1e-6,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            **self.launch.get("optimizer", {}),
        }
        if lr is not None:
            settings["lr"] = lr
        return settings

    @property
    def audit_lr(self) -> float:
        return float(self.raw.get("probe", {}).get("audit_lr", 1e-3))


def load_profile(name_or_path: str) -> Profile:
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = PROFILES_DIR / f"{name_or_path}.yaml"
    raw = yaml.safe_load(path.read_text())
    assert raw.get("name") == path.stem, f"{path}: name must equal the file stem"
    return Profile(name=raw["name"], path=path, raw=raw)


def all_profiles() -> list[Profile]:
    return [load_profile(str(p)) for p in sorted(PROFILES_DIR.glob("*.yaml"))]


def layout_world(layout: dict) -> int:
    """World size of a parallel layout; ``world_size`` adds data-parallel replicas beyond max(EP, SP)."""
    return int(
        layout.get("world_size", max(int(layout.get("expert_parallel", 1)), int(layout.get("sequence_parallel", 1))))
    )


def layout_id(layout: dict) -> str:
    base = f"ep{layout.get('expert_parallel', 1)}-sp{layout.get('sequence_parallel', 1)}"
    return base if "world_size" not in layout else f"{base}-w{layout['world_size']}"


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(base.get(key), value) if key in base else copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


# ---------------------------------------------------------------------------------------------------------------------
# Source config and checkpoint index (never loads weights)
# ---------------------------------------------------------------------------------------------------------------------


def _resolve_file(checkpoint: str, filename: str) -> str | None:
    local = Path(checkpoint, filename)
    if local.is_file():
        return str(local)
    if Path(checkpoint).is_dir():
        return None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(checkpoint, filename)
    except EntryNotFoundError:
        return None


def source_config(profile: Profile) -> dict:
    path = _resolve_file(profile.checkpoint, "config.json")
    assert path is not None, f"{profile.checkpoint} has no config.json"
    return json.loads(Path(path).read_text())


def source_weight_names(profile: Profile) -> set[str]:
    """Tensor names of the real checkpoint, from its safetensors index or header; no tensor data is read."""
    index = _resolve_file(profile.checkpoint, "model.safetensors.index.json")
    if index is not None:
        return set(json.loads(Path(index).read_text())["weight_map"])
    single = _resolve_file(profile.checkpoint, "model.safetensors")
    assert single is not None, f"{profile.checkpoint} has neither model.safetensors.index.json nor model.safetensors"
    from safetensors import safe_open

    with safe_open(single, framework="pt") as handle:
        return set(handle.keys())


def text_config_of(config: dict) -> dict:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


# ---------------------------------------------------------------------------------------------------------------------
# Shrinking a config into a tiny one that keeps every layer kind
# ---------------------------------------------------------------------------------------------------------------------


def _per_layer_lists(text: dict, num_layers: int) -> dict[str, list]:
    return {
        key: value
        for key, value in text.items()
        if isinstance(value, list) and len(value) == num_layers and key not in ("eos_token_id",)
    }


def _layer_index_lists(text: dict, num_layers: int) -> list[tuple[tuple[str, ...], list[int]]]:
    """Lists of layer indices inside nested dicts (e.g. ``linear_attn_config.kda_layers``)."""
    found = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, path + (key,))
        elif (
            isinstance(node, list)
            and node
            and "layer" in path[-1]
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < num_layers for v in node)
            and len(node) != num_layers
        ):
            found.append((path, node))

    for key, value in text.items():
        if isinstance(value, dict):
            visit(value, (key,))
        elif isinstance(value, list) and key.endswith("layers") and len(value) != num_layers:
            visit(value, (key,))
    return found


def layer_signatures(text: dict) -> list[tuple]:
    """One hashable signature per layer: every per-layer property the config declares."""
    num_layers = int(text["num_hidden_layers"])
    lists = _per_layer_lists(text, num_layers)
    index_lists = _layer_index_lists(text, num_layers)
    first_dense = text.get("first_k_dense_replace")
    moe_freq = text.get("moe_layer_freq") if isinstance(text.get("moe_layer_freq"), int) else None
    signatures = []
    for i in range(num_layers):
        sig = [(key, json.dumps(values[i], sort_keys=True)) for key, values in sorted(lists.items())]
        sig += [(".".join(path), i in set(indices)) for path, indices in index_lists]
        if isinstance(first_dense, int):
            sig.append(("first_k_dense_replace", i < first_dense))
        if moe_freq:
            sig.append(("moe_layer_freq", i % moe_freq == 0))
        signatures.append(tuple(sig))
    return signatures


def select_layers(text: dict, explicit: list[int] | None = None, min_layers: int = 2) -> list[int]:
    """First layer of every distinct kind, in order; padded to ``min_layers`` with repeats of the commonest kind."""
    if explicit:
        return sorted(explicit)
    signatures = layer_signatures(text)
    keep: list[int] = []
    seen: set[tuple] = set()
    for i, sig in enumerate(signatures):
        if sig not in seen:
            seen.add(sig)
            keep.append(i)
    for i in range(len(signatures)):
        if len(keep) >= min_layers:
            break
        if i not in keep:
            keep.append(i)
    return sorted(keep)


def shrink_config(full: dict, tiny: dict) -> tuple[dict, list[int]]:
    """Tiny config with one layer of every kind, fewer experts and the profile's explicit overrides."""
    config = copy.deepcopy(full)
    text = text_config_of(config)
    num_layers = int(text["num_hidden_layers"])
    keep = select_layers(text, tiny.get("layers"), tiny.get("min_layers", 2))
    remap = {old: new for new, old in enumerate(keep)}

    for key, values in _per_layer_lists(text, num_layers).items():
        text[key] = [values[i] for i in keep]
    for path, indices in _layer_index_lists(text, num_layers):
        node = text
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = [remap[i] for i in indices if i in remap]
    if isinstance(text.get("first_k_dense_replace"), int):
        text["first_k_dense_replace"] = sum(1 for i in keep if i < full_text(full)["first_k_dense_replace"])
    text["num_hidden_layers"] = len(keep)

    topk = next((int(text[k]) for k in _EXPERT_TOPK_KEYS if isinstance(text.get(k), int)), None)
    for key in _EXPERT_COUNT_KEYS:
        if isinstance(text.get(key), int) and text[key] > 1:
            text[key] = int(tiny.get("experts", max(8, 2 * (topk or 1))))

    for name, sub in config.items():
        if name != "text_config" and isinstance(sub, dict) and name.endswith("_config"):
            for key in _SUBCONFIG_DEPTH_KEYS:
                if isinstance(sub.get(key), int):
                    sub[key] = int(tiny.get("subconfig_layers", 1))

    for key, value in (tiny.get("text_overrides") or {}).items():
        text[key] = _deep_merge(text.get(key), value) if isinstance(value, dict) else value
    for key, value in (tiny.get("config_overrides") or {}).items():
        config[key] = _deep_merge(config.get(key), value) if isinstance(value, dict) else value
    return config, keep


def full_text(full: dict) -> dict:
    return text_config_of(full)


def thresholds(profile: Profile, tiny_config: dict) -> list[dict]:
    """Token positions where the tiny model switches code paths: declared by the profile plus generic config keys."""
    text = text_config_of(tiny_config)
    found = [dict(t) for t in profile.tiny.get("thresholds", [])]
    rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
    for key in _THRESHOLD_KEYS:
        value = text.get(key, rope.get(key) if isinstance(rope, dict) else None)
        if isinstance(value, int) and value > 0:
            found.append({"name": key, "position": value, "reason": f"config.{key}"})
    unique = {t["position"]: t for t in found}
    return [unique[p] for p in sorted(unique)]


SHORT_ROW_LENGTHS = tuple(range(1, 9))


def row_lengths(profile: Profile, tiny_config: dict) -> list[int]:
    """Packed-row lengths: short rows plus rows that straddle every threshold, capped at ``tiny.max_row_len``."""
    if profile.tiny.get("row_lengths"):
        return list(profile.tiny["row_lengths"])
    cap = int(profile.tiny.get("max_row_len", 2048))
    lengths = [7, 61, 130]
    for t in thresholds(profile, tiny_config):
        if t["position"] + 64 <= cap:
            lengths.append(t["position"] + 64)
    return lengths


# ---------------------------------------------------------------------------------------------------------------------
# Tiny checkpoints
# ---------------------------------------------------------------------------------------------------------------------


def _config_hash(config: dict, seed: int, dtype: str) -> str:
    import torch
    import transformers

    payload = json.dumps([config, seed, dtype, transformers.__version__, torch.__version__], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def tiny_config(profile: Profile) -> dict:
    return shrink_config(source_config(profile), profile.tiny)[0]


def build_tiny_checkpoint(profile: Profile, seed: int = 0) -> Path:
    """Random-init tiny checkpoint saved the way ``save_pretrained`` writes it, plus the source tokenizer files.

    Cached under ``audit_root()`` by config hash, so every probe of a session reads the same weights.
    """
    import torch
    import transformers

    config_dict = tiny_config(profile)
    out = audit_root() / "checkpoints" / f"{profile.name}-{_config_hash(config_dict, seed, profile.dtype)}-s{seed}"
    if (out / ".complete").exists():
        _copy_tokenizer_files(profile, out)
        return out
    staging = Path(tempfile.mkdtemp(prefix=f"{out.name}.", dir=out.parent if out.parent.exists() else None))
    try:
        (staging / "config.json").write_text(json.dumps(config_dict, indent=2))
        config = transformers.AutoConfig.from_pretrained(str(staging))
        arch = (config_dict.get("architectures") or [None])[0]
        model_cls = getattr(transformers, arch, None) if arch else None
        if model_cls is None:
            model_cls = transformers.AutoModelForCausalLM._model_mapping[type(config)]
        torch.manual_seed(seed)
        with torch.device("cpu"):
            model = model_cls._from_config(config, dtype=torch.float32)
        _perturb_zero_init(model, seed)
        model.to(getattr(torch, profile.dtype)).save_pretrained(str(staging), safe_serialization=True)
        _copy_tokenizer_files(profile, staging)
        (staging / ".complete").write_text("")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out


def _copy_tokenizer_files(profile: Profile, directory: Path) -> None:
    """Tokenizer and processor files from the source checkpoint; existing ones are kept."""
    for name in _TOKENIZER_FILES:
        if (directory / name).exists():
            continue
        src = _resolve_file(profile.checkpoint, name)
        if src is not None:
            shutil.copy(src, directory / name)


def _perturb_zero_init(model, seed: int) -> None:
    """Give zero- or one-initialized parameters (biases, norms, gates) random values so their bugs are visible."""
    import torch

    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.numel() == 0 or not param.is_floating_point():
                continue
            if bool((param == param.flatten()[0]).all()):
                noise = torch.randn(param.shape, generator=gen, dtype=param.dtype) * 0.02
                param.add_(noise)
        for name, buf in model.named_buffers():
            if buf.is_floating_point() and buf.numel() > 1 and "bias" in name and bool((buf == 0).all()):
                buf.add_(torch.randn(buf.shape, generator=gen, dtype=buf.dtype) * 0.02)


def read_checkpoint_tensors(path: Path) -> dict:
    from safetensors.torch import load_file

    tensors = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        tensors.update(load_file(str(file)))
    return tensors


# ---------------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------------


def make_rows(vocab_size: int, lengths: list[int], seed: int = 0) -> list[list[int]]:
    import torch

    gen = torch.Generator().manual_seed(seed)
    low, high = min(100, vocab_size // 4), min(vocab_size, 32000)
    return [torch.randint(low, high, (n,), generator=gen).tolist() for n in lengths]


def pack_rows(rows: list[list[int]], pad_to_multiple: int = 1, pad_token_id: int = 0):
    """One packed row: ``input_ids``/``position_ids`` ``[1, T]`` and next-token ``labels`` (-100 at row ends).

    Padding matches ``pad_packed_microbatch``: position 0 per pad token, so each reads as its own sequence.
    """
    import torch

    ids, pos, labels = [], [], []
    for row in rows:
        ids += row
        pos += list(range(len(row)))
        labels += row[1:] + [-100]
    pad = (-len(ids)) % pad_to_multiple
    if pad:
        ids += [pad_token_id] * pad
        pos += [0] * pad
        labels += [-100] * pad
    as_tensor = lambda x: torch.tensor([x], dtype=torch.long)  # noqa: E731
    return as_tensor(ids), as_tensor(pos), as_tensor(labels)


def probe_inputs(profile: Profile, ckpt: str) -> tuple[dict, list[list[int]], int]:
    """The tiny config, the audit rows and the packing multiple; identical in every probe."""
    config = json.loads(Path(ckpt, "config.json").read_text())
    rows = make_rows(int(text_config_of(config)["vocab_size"]), row_lengths(profile, config), seed=0)
    return config, rows, int(profile.raw.get("probe", {}).get("pad_to_multiple", 8))


def row_slices(rows: list[list[int]]) -> list[slice]:
    out, start = [], 0
    for row in rows:
        out.append(slice(start, start + len(row)))
        start += len(row)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------


def k3(actual, expected) -> float:
    """k3 KL estimate between two per-token log-prob tensors (``actual`` playing the trainer)."""
    d = (actual.double() - expected.double()).flatten()
    d = d[~d.isnan()]
    return float((d.exp() - 1 - d).mean()) if d.numel() else 0.0


def logprob_errors(actual, expected, mask=None) -> dict:
    d = (actual.double() - expected.double()).abs()
    if mask is not None:
        d = d[mask]
    d = d[~d.isnan()]
    if d.numel() == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "n": 0}
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "n": int(d.numel())}


def bucketed_errors(actual, expected, positions, labels, cuts: list[int]) -> dict[str, dict]:
    """Errors per position bucket split at ``cuts``, so an error that starts past a threshold is not diluted."""
    valid = labels != -100
    edges = [0] + sorted(set(cuts)) + [math.inf]
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = valid & (positions >= lo) & (positions < hi)
        if bool(mask.any()):
            out[f"[{lo},{hi})"] = {**logprob_errors(actual, expected, mask), "k3": k3(actual[mask], expected[mask])}
    return out


def relative_l2(actual, expected) -> float:
    a, e = actual.double().flatten(), expected.double().flatten()
    denom = float(e.norm())
    return float((a - e).norm()) / denom if denom > 0 else float(a.norm())


def within_noise(error: float, floor: float, profile: Profile) -> bool:
    """``error`` is at most ``noise_factor`` times the reference's own bf16-vs-fp32 error, plus a small slack."""
    tol = profile.tolerances
    return error <= tol["noise_factor"] * floor + tol["noise_slack"]


def matches(pattern_entries: list[dict], name: str) -> dict | None:
    for entry in pattern_entries or []:
        if re.search(entry["pattern"], name):
            return entry
    return None


def assert_gradients_match(profile: Profile, actual: dict, reference: dict, record) -> None:
    """Every reference gradient has a production gradient of the same shape within the per-tensor noise floor."""
    import torch

    truth, floor = reference["fp32_grads"], reference["lowp_grads"]
    mass = reference.get("fp32_grad_mass", {})
    allowed = profile.expect.get("no_grad_params", [])
    missing = sorted(n for n in truth if n not in actual and matches(allowed, n) is None)
    # Low-precision noise per unit of summed |row gradient|. A tensor whose rows cancel (e.g. a 3-element gate
    # scale) can land on a lucky floor; the median rate across tensors bounds how lucky.
    rate = {
        n: float((floor[n] - truth[n]).norm() / mass[n].norm())
        for n in truth
        if n in mass and n in floor and bool(mass[n].abs().sum() > 0)
    }
    typical_rate = statistics.median(rate.values()) if rate else 0.0
    rows, bad = [], []
    for name, expected in truth.items():
        if name not in actual or matches(allowed, name):
            continue
        got = actual[name]
        if got.shape != expected.shape:
            bad.append((name, f"shape {tuple(got.shape)} != {tuple(expected.shape)}"))
        elif not bool(expected.abs().sum() > 0):
            if bool(torch.nan_to_num(got).abs().sum() > 0):
                bad.append((name, "gradient where the reference has none"))
        elif bool(torch.isnan(got).all()):
            bad.append((name, "no gradient"))
        else:
            err, noise = relative_l2(got, expected), relative_l2(floor[name], expected)
            if name in rate:
                noise = max(noise, typical_rate * float(mass[name].norm() / expected.norm()))
            rows.append((err, noise, name))
            if not within_noise(err, noise, profile):
                bad.append((name, f"rel_l2={err:.4g} floor={noise:.4g}"))
    rows.sort(reverse=True)
    record(worst=[(n, round(e, 5), round(f, 5)) for e, f, n in rows[:15]], missing=missing[:30], compared=len(rows))
    assert not missing, f"reference gradients with no production gradient: {missing[:20]}"
    assert not bad, f"gradients beyond the noise floor: {bad[:20]}"


def assert_update_is_adamw(profile: Profile, step: dict, record) -> None:
    """The stored update of every tensor is one AdamW step on the engine's reduced gradient, element for element."""
    allowed = profile.expect.get("no_grad_params", [])
    agreement = {n: f for n, f in step["adamw_agreement"].items() if matches(allowed, n) is None}
    limit = 1.0 - profile.tolerances["adamw_mismatch_fraction"]
    worst = sorted((f, n) for n, f in agreement.items())
    record(lr=step["lr"], compared=len(agreement), worst=[(n, round(f, 6)) for f, n in worst[:15]])
    assert agreement, "the step reported no tensors to compare"
    bad = [(n, round(f, 5)) for f, n in worst if f < limit]
    assert (
        not bad
    ), f"tensors whose update is not AdamW of the reduced gradient (fraction of elements agreeing): {bad[:20]}"
