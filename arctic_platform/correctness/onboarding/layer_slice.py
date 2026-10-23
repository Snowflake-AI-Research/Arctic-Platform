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

"""Reduce a checkpoint to its first layers by copying tensors, keeping the on-disk names and dtypes.

``materialize_pretrained`` round-trips through a transformers model, and its save writes transformers' module
names. Checkpoints whose loaders read the original names (DeepSeek-V4's ``layers.N.attn.wq_a``) need the
reduced fixture in that format instead.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..harness.storage import require_data_fast_path
from .fp8_fixture import SHARD_BYTES
from .fp8_fixture import copy_sidecars
from .fp8_fixture import write_shards

LAYER = re.compile(r"^(?:model\.)?layers\.(\d+)\.")
DROP_PREFIXES = ("mtp.", "model.mtp.")


def kept_keys(source_map: dict, num_layers: int) -> list[str]:
    keys = []
    for key in sorted(source_map):
        match = LAYER.match(key)
        if key.startswith(DROP_PREFIXES) or (match and int(match.group(1)) >= num_layers):
            continue
        keys.append(key)
    return keys


def reduce_config(config: dict, num_layers: int) -> dict:
    """Set the layer count and truncate every per-layer list, including ones padded past it (``compress_ratios``)."""
    original = int(config["num_hidden_layers"])
    reduced = dict(config, num_hidden_layers=num_layers)
    for key, value in config.items():
        if isinstance(value, list) and len(value) >= original:
            reduced[key] = value[:num_layers]
    if "num_nextn_predict_layers" in reduced:
        reduced["num_nextn_predict_layers"] = 0
    return reduced


def slice_layers(source: str | Path, dest: str | Path, num_layers: int, *, shard_bytes: int = SHARD_BYTES) -> dict:
    source = Path(source)
    dest = require_data_fast_path(dest, label="layer-sliced fixture destination")
    dest.mkdir(parents=True, exist_ok=False)
    source_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    keys = kept_keys(source_map, num_layers)
    shards = write_shards(source, source_map, keys, dest, shard_bytes=shard_bytes)
    config = reduce_config(json.loads((source / "config.json").read_text()), num_layers)
    (dest / "config.json").write_text(json.dumps(config, indent=2))
    copy_sidecars(source, dest)
    return {"tensors": len(keys), "shards": shards}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arctic_platform.correctness.onboarding.layer_slice")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--layers", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(slice_layers(args.source, args.dest, args.layers)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
