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

"""Cut a block-FP8 checkpoint down to a reduced fixture's tensor set, keeping FP8 weights and their scales.

``materialize_pretrained`` dequantizes, which suits the HF reference; a target whose catalog config trains on the
FP8 checkpoint (for example LoRA over frozen FP8 weights) needs the same reduced model still in FP8.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable

SHARD_BYTES = 5 * 1024**3
DEFAULT_CARRY = ("quantization_config",)


def scale_keys(key: str) -> tuple[str, ...]:
    """Scale tensor names a quantized ``*.weight`` may carry: HF's ``weight_scale_inv`` or DeepSeek's ``scale``."""
    if not key.endswith(".weight"):
        return ()
    return (f"{key}_scale_inv", key.removesuffix("weight") + "scale")


def fp8_keys(source_map: dict, wanted: list[str]) -> list[str]:
    keys = []
    for key in wanted:
        if key not in source_map:
            raise KeyError(f"{key} is missing from the FP8 source")
        keys.append(key)
        keys.extend(scale for scale in scale_keys(key) if scale in source_map and scale not in wanted)
    return keys


def build(
    source: str | Path,
    like: str | Path,
    dest: str | Path,
    *,
    carry: tuple[str, ...] = DEFAULT_CARRY,
    shard_bytes: int = SHARD_BYTES,
) -> dict:
    """Write ``dest`` with ``like``'s tensor names, config and sidecars, plus the source's ``carry`` config keys."""
    source, like, dest = Path(source), Path(like), Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    source_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = sorted(json.loads((like / "model.safetensors.index.json").read_text())["weight_map"])
    keys = fp8_keys(source_map, wanted)
    shards = write_shards(source, source_map, keys, dest, shard_bytes=shard_bytes)

    config = json.loads((like / "config.json").read_text())
    source_config = json.loads((source / "config.json").read_text())
    for key in carry:
        config[key] = source_config[key]
    (dest / "config.json").write_text(json.dumps(config, indent=2))
    copy_sidecars(like, dest)
    return {"tensors": len(keys), "scales": len(keys) - len(wanted), "shards": shards}


def copy_sidecars(source: Path, dest: Path) -> None:
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith((".safetensors", ".index.json")) and path.name != "config.json":
            shutil.copyfile(path, dest / path.name)


def write_shards(
    source: Path,
    source_map: dict,
    keys: list[str],
    dest: Path,
    *,
    shard_bytes: int = SHARD_BYTES,
    renamed: dict[str, str] | None = None,
    transform: Callable | None = None,
) -> int:
    """Copy ``keys`` from ``source`` into ``dest`` as an indexed safetensors checkpoint; returns the shard count.

    ``renamed`` maps a destination key to the source key it is read from, and ``transform(key, tensor)``
    rewrites a tensor on the way.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    renamed = renamed or {}
    shards, shard, size = [], {}, 0
    handles = {}
    for key in keys:
        source_key = renamed.get(key, key)
        file = source_map[source_key]
        if file not in handles:
            handles[file] = safe_open(source / file, framework="pt")
        tensor = handles[file].get_tensor(source_key)
        if transform is not None:
            tensor = transform(key, tensor).contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if shard and size + nbytes > shard_bytes:
            shards.append(shard)
            shard, size = {}, 0
        shard[key] = tensor
        size += nbytes
    if shard:
        shards.append(shard)

    weight_map = {}
    for i, tensors in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(tensors, dest / name, metadata={"format": "pt"})
        weight_map.update({k: name for k in tensors})
    index = {"metadata": {}, "weight_map": weight_map}
    (dest / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return len(shards)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arctic_platform.correctness.onboarding.fp8_fixture")
    parser.add_argument("--source", required=True, help="full block-FP8 checkpoint")
    parser.add_argument("--like", required=True, help="reduced fixture whose tensor set and config to mirror")
    parser.add_argument("--dest", required=True)
    parser.add_argument(
        "--carry", nargs="+", default=list(DEFAULT_CARRY), help="source config keys to copy, e.g. expert_dtype"
    )
    args = parser.parse_args(argv)
    print(json.dumps(build(args.source, args.like, args.dest, carry=tuple(args.carry))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
