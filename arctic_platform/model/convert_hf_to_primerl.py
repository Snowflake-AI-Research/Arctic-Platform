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
"""Offline HF conversion into a prepared ``<checkpoint>/prime`` cache."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.conversion_cache import conversion_cache_ready
from arctic_platform.model.implementations.moe.conversion_cache import ensure_node_local_conversion_cache
from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path
from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_root
from arctic_platform.model.implementations.moe.conversion_cache import sibling_conversion_cache_path
from arctic_platform.model.implementations.moe.weights import load_state_dict
from arctic_platform.model.implementations.moe.weights import load_state_dict_keys
from arctic_platform.model.implementations.moe.weights import save_state_dict

# model_type from config.json → (import_path, class_name).
_AP_CONVERTERS: dict[str, tuple[str, str]] = {
    "glm5_next": (
        "arctic_platform.model.implementations.glm53.modeling_glm5_next",
        "Glm5NextForConditionalGenerationPrimeRL",
    ),
    "qwen4_exp": (
        "arctic_platform.model.implementations.qwen38.modeling_qwen4_exp",
        "Qwen4ExpForConditionalGenerationPrimeRL",
    ),
    "glm_moe_dsa": (
        "arctic_platform.model.implementations.glm52.models.glm_moe_dsa.modeling_glm_moe_dsa",
        "GlmMoeDsaForCausalLM",
    ),
    "qwen3_5_moe": (
        "arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "Qwen3_5MoeForCausalLM",
    ),
    "qwen3_5_moe_text": (
        "arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "Qwen3_5MoeForCausalLM",
    ),
    "qwen3_moe": (
        "arctic_platform.model.implementations.qwen3_moe.modeling_qwen3_moe",
        "Qwen3MoeForCausalLM",
    ),
    "glm4_moe": (
        "arctic_platform.model.implementations.glm4_moe.modeling_glm4_moe",
        "Glm4MoeForCausalLM",
    ),
    "minimax_m2": (
        "arctic_platform.model.implementations.minimax_m2.modeling_minimax_m2",
        "MiniMaxM2ForCausalLM",
    ),
    "nemotron_h": (
        "arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h",
        "NemotronHForCausalLM",
    ),
    "afmoe": (
        "arctic_platform.model.implementations.afmoe.modeling_afmoe",
        "AfmoeForCausalLM",
    ),
}

# Llama is already PrimeRL-compatible; conversion is a no-op.
_NOOP_TYPES = frozenset({"llama"})


@dataclass(frozen=True)
class ConverterSpec:
    model_type: str
    cls: type[PreTrainedModelPrimeRL]
    source: str


def supported_model_types() -> list[str]:
    return sorted(set(_AP_CONVERTERS) | _NOOP_TYPES)


def _load_class(module_name: str, class_name: str) -> type[PreTrainedModelPrimeRL]:
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def resolve_converter(model_type: str) -> ConverterSpec:
    if model_type in _NOOP_TYPES:
        raise ValueError(f"{model_type} is already PrimeRL-compatible; no conversion needed")

    errors: list[str] = []
    if model_type in _AP_CONVERTERS:
        module_name, class_name = _AP_CONVERTERS[model_type]
        try:
            return ConverterSpec(model_type, _load_class(module_name, class_name), "arctic-platform")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"arctic-platform:{module_name}.{class_name}: {exc}")

    known = ", ".join(supported_model_types())
    detail = f" ({'; '.join(errors)})" if errors else ""
    raise ValueError(f"unsupported model_type={model_type!r}. known: {known}{detail}")


def read_model_type(checkpoint: Path) -> str:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if not model_type:
        raise ValueError(f"{config_path} has no model_type")
    return str(model_type)


def inspect_checkpoint(checkpoint: Path) -> dict[str, object]:
    model_type = read_model_type(checkpoint)
    dest = sibling_conversion_cache_path(checkpoint, "prime")
    keys = load_state_dict_keys(checkpoint)
    if not keys:
        raise FileNotFoundError(f"no *.safetensors under {checkpoint}")

    info: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "model_type": model_type,
        "n_keys": len(keys),
        "dest": str(dest),
        "dest_ready": conversion_cache_ready(dest),
        "needs_conversion": False,
        "already_prime": False,
        "converter": None,
    }
    if model_type in _NOOP_TYPES:
        info["already_prime"] = True
        return info

    spec = resolve_converter(model_type)
    info["converter"] = f"{spec.source}:{spec.cls.__name__}"
    key_set = dict.fromkeys(keys)
    is_hf = bool(spec.cls.is_hf_state_dict(key_set))
    is_prime = bool(spec.cls.is_prime_state_dict(key_set))
    info["already_prime"] = is_prime and not is_hf
    info["needs_conversion"] = is_hf
    return info


def _hardlink_or_copy_tree(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(f"{dest.name}.tmp-pid{os.getpid()}")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    def copy_file(source: str, target: str) -> str:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        return target

    committed = False
    try:
        shutil.copytree(src, tmp_path, copy_function=copy_file)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp_path, dest)
        committed = True
    finally:
        if not committed and tmp_path.exists():
            shutil.rmtree(tmp_path)


def promote_cache(src: Path, dest: Path, *, force: bool = False) -> Path:
    """Hardlink/copy a ready PrimeRL cache onto ``dest``."""
    src = Path(src)
    dest = Path(dest)
    if conversion_cache_ready(dest) and not force:
        return dest
    if not conversion_cache_ready(src):
        raise ValueError(f"source cache is incomplete: {src}")
    print(f"promoting {src} → {dest}")
    _hardlink_or_copy_tree(src, dest)
    if not conversion_cache_ready(dest):
        raise RuntimeError(f"promote finished but cache is incomplete: {dest}")
    return dest


def promote_hashed_cache(
    checkpoint: Path,
    dest: Path,
    *,
    cache_root: str | Path | None = None,
    force: bool = False,
) -> Path | None:
    """Copy/hardlink a ready hashed scratch cache onto ``<checkpoint>/prime``."""
    if conversion_cache_ready(dest) and not force:
        return dest
    root = resolve_conversion_cache_root(cache_root)
    hashed = hashed_conversion_cache_path(root, checkpoint, "prime")
    if not conversion_cache_ready(hashed):
        return None
    return promote_cache(hashed, dest, force=True)


def convert_checkpoint(
    checkpoint: Path,
    *,
    dest: Path | None = None,
    force: bool = False,
    promote: bool = False,
    promote_from: Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    checkpoint = checkpoint.resolve()
    dest = dest.resolve() if dest is not None else sibling_conversion_cache_path(checkpoint, "prime")
    info = inspect_checkpoint(checkpoint)
    if info["already_prime"] and not info["needs_conversion"]:
        print(f"skip {checkpoint}: already PrimeRL format")
        return dest
    if not info["needs_conversion"]:
        raise ValueError(
            f"{checkpoint} is not an HF PrimeRL-convertible checkpoint "
            f"(model_type={info['model_type']}, converter={info['converter']})"
        )
    if conversion_cache_ready(dest) and not force:
        print(f"skip {checkpoint}: cache already ready at {dest} (pass --force to rebuild)")
        return dest
    if promote_from is not None:
        promote_cache(promote_from, dest, force=force)
        print(f"ready {dest} (promoted)")
        return dest
    if promote:
        promoted = promote_hashed_cache(checkpoint, dest, cache_root=cache_root, force=force)
        if promoted is not None:
            print(f"ready {dest} (promoted)")
            return dest

    spec = resolve_converter(str(info["model_type"]))
    print(f"converting {checkpoint} ({info['model_type']}, {info['n_keys']} keys) via {info['converter']} → {dest}")
    if force and dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        ensure_node_local_conversion_cache(
            checkpoint,
            dest,
            spec.cls.convert_to_prime,
            "HF",
            "PrimeRL",
            load_state_dict,
            save_state_dict,
            rank=0,
            local_rank=0,
        )
        if not conversion_cache_ready(dest):
            raise RuntimeError(f"conversion finished but cache is incomplete: {dest}")
    except Exception:
        try:
            _cleanup_incomplete_conversion_dest(dest)
        except OSError:
            pass
        raise
    print(f"ready {dest}")
    return dest


def _cleanup_incomplete_conversion_dest(dest: Path) -> None:
    """Remove a failed conversion dest and leftover ``prime.tmp-*`` siblings."""
    if dest.exists() and not conversion_cache_ready(dest):
        shutil.rmtree(dest)
    parent = dest.parent
    if not parent.is_dir():
        return
    prefix = f"{dest.name}.tmp-"
    for child in parent.iterdir():
        if child.name.startswith(prefix):
            shutil.rmtree(child)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a PrimeRL weight cache next to one or more HF checkpoints.")
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        help="Local HF checkpoint directories (must contain config.json + *.safetensors)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override dest (default: <checkpoint>/prime). Only valid with a single checkpoint.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if dest is already complete")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="If a hashed scratch cache is already ready, hardlink/copy it to dest instead of reconverting",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help=(
            "Hashed-cache root used by --promote (default: $DSS_WEIGHT_CONVERSION_CACHE_DIR or"
            " /data-fast/prime-rl-weight-cache)"
        ),
    )
    parser.add_argument(
        "--promote-from",
        type=Path,
        default=None,
        help="Explicit ready PrimeRL cache to hardlink/copy to dest (skips reconversion)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect only; do not convert")
    parser.add_argument("--list-models", action="store_true", help="Print supported model_type values")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_models:
        for name in supported_model_types():
            print(name)
        return 0

    if not args.checkpoints:
        parser.error("at least one checkpoint is required (or pass --list-models)")
    if args.output_dir is not None and len(args.checkpoints) != 1:
        parser.error("--output-dir requires exactly one checkpoint")

    failed = 0
    for checkpoint in args.checkpoints:
        try:
            if args.dry_run:
                print(json.dumps(inspect_checkpoint(checkpoint), indent=2, sort_keys=True))
                continue
            convert_checkpoint(
                checkpoint,
                dest=args.output_dir,
                force=args.force,
                promote=args.promote,
                promote_from=args.promote_from,
                cache_root=args.cache_root,
            )
        except Exception as exc:  # noqa: BLE001 — CLI reports and continues
            failed += 1
            print(f"ERROR {checkpoint}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
