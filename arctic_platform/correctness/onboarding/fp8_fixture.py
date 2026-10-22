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

SHARD_BYTES = 5 * 1024**3
SCALE_SUFFIX = "_scale_inv"


def fp8_keys(source_map: dict, wanted: list[str]) -> list[str]:
    keys = []
    for key in wanted:
        if key not in source_map:
            raise KeyError(f"{key} is missing from the FP8 source")
        keys.append(key)
        if key.endswith(".weight") and f"{key}{SCALE_SUFFIX}" in source_map:
            keys.append(f"{key}{SCALE_SUFFIX}")
    return keys


def build(source: str | Path, like: str | Path, dest: str | Path, *, shard_bytes: int = SHARD_BYTES) -> dict:
    """Write ``dest`` with ``like``'s tensor names, config and sidecars, plus the source's quantization config."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, like, dest = Path(source), Path(like), Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    source_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = sorted(json.loads((like / "model.safetensors.index.json").read_text())["weight_map"])
    keys = fp8_keys(source_map, wanted)

    shards, shard, size = [], {}, 0
    handles = {}
    for key in keys:
        file = source_map[key]
        if file not in handles:
            handles[file] = safe_open(source / file, framework="pt")
        tensor = handles[file].get_tensor(key)
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

    config = json.loads((like / "config.json").read_text())
    config["quantization_config"] = json.loads((source / "config.json").read_text())["quantization_config"]
    (dest / "config.json").write_text(json.dumps(config, indent=2))
    for path in like.iterdir():
        if path.is_file() and not path.name.endswith((".safetensors", ".index.json")) and path.name != "config.json":
            shutil.copyfile(path, dest / path.name)
    return {"tensors": len(keys), "scales": sum(k.endswith(SCALE_SUFFIX) for k in keys), "shards": len(shards)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arctic_platform.correctness.onboarding.fp8_fixture")
    parser.add_argument("--source", required=True, help="full block-FP8 checkpoint")
    parser.add_argument("--like", required=True, help="reduced fixture whose tensor set and config to mirror")
    parser.add_argument("--dest", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.source, args.like, args.dest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
