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

"""Reduce a checkpoint to some of its layers by copying tensors, keeping the on-disk names and dtypes.

``materialize_pretrained`` round-trips through a transformers model, and its save writes transformers' module
names. Checkpoints whose loaders read the original names (DeepSeek-V4's ``layers.N.attn.wq_a``) need the
reduced fixture in that format instead.

The kept layers are either the first N or an explicit list, renumbered from 0, so a fixture can carry one
layer of every kind a model mixes. Routed experts can be capped as well; the router rows go with them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from ..harness.storage import require_data_fast_path
from .fp8_fixture import SHARD_BYTES
from .fp8_fixture import copy_sidecars
from .fp8_fixture import write_shards

LAYER = re.compile(r"^(?P<prefix>(?:model\.)?layers\.)(?P<index>\d+)\.")
EXPERT = re.compile(r"\.experts\.(\d+)\.")
ROUTER_ROWS = re.compile(r"\.gate\.(?:weight|bias|bias_vl|e_score_correction_bias)$")
EXPERT_COUNT_KEYS = ("n_routed_experts", "num_experts", "num_local_experts")
DROP_PREFIXES = ("mtp.", "model.mtp.")


def selected_layers(layers: int | list[int]) -> list[int]:
    return list(range(layers)) if isinstance(layers, int) else [int(i) for i in layers]


def kept_keys(
    source_map: dict, layers: int | list[int], *, experts: int | None = None, drop: tuple[str, ...] = ()
) -> dict[str, str]:
    """Destination key to source key for the selected layers, renumbered, minus dropped prefixes and experts."""
    position = {old: new for new, old in enumerate(selected_layers(layers))}
    kept = {}
    for key in sorted(source_map):
        if key.startswith(DROP_PREFIXES + tuple(drop)):
            continue
        match = LAYER.match(key)
        if match is None:
            kept[key] = key
            continue
        old = int(match.group("index"))
        expert = EXPERT.search(key)
        if old not in position or (experts is not None and expert and int(expert.group(1)) >= experts):
            continue
        kept[f"{match.group('prefix')}{position[old]}.{key[match.end():]}"] = key
    return kept


def reduce_config(
    config: dict, layers: int | list[int], *, experts: int | None = None, overrides: dict | None = None
) -> dict:
    """Keep the selected layers' entries of every per-layer list, including ones padded past the layer count.

    ``compress_ratios`` is one such list. A nested ``text_config`` is reduced in place of the top level, and
    ``overrides`` land at whichever level holds the layer count.
    """
    if "text_config" in config and "num_hidden_layers" not in config:
        text = reduce_config(config["text_config"], layers, experts=experts, overrides=overrides)
        return dict(config, text_config=text)
    selected = selected_layers(layers)
    original = int(config["num_hidden_layers"])
    reduced = dict(config, num_hidden_layers=len(selected))
    for key, value in config.items():
        if isinstance(value, list) and len(value) >= original:
            reduced[key] = [value[i] for i in selected]
    if "num_nextn_predict_layers" in reduced:
        reduced["num_nextn_predict_layers"] = 0
    if experts is not None:
        keys = [key for key in EXPERT_COUNT_KEYS if key in reduced]
        if not keys:
            raise ValueError(f"config names its routed experts under none of {EXPERT_COUNT_KEYS}")
        reduced.update(dict.fromkeys(keys, experts))
    reduced.update(overrides or {})
    return reduced


def _routed_experts(config: dict) -> int | None:
    text = config.get("text_config", config)
    return next((int(text[key]) for key in EXPERT_COUNT_KEYS if key in text), None)


def slice_layers(
    source: str | Path,
    dest: str | Path,
    layers: int | list[int],
    *,
    experts: int | None = None,
    drop: tuple[str, ...] = (),
    config: dict | None = None,
    shard_bytes: int = SHARD_BYTES,
) -> dict:
    """``config`` overrides keys the selection leaves stale, such as layer ids that point at other layers."""
    source = Path(source)
    dest = require_data_fast_path(dest, label="layer-sliced fixture destination")
    dest.mkdir(parents=True, exist_ok=False)
    source_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    source_config = json.loads((source / "config.json").read_text())
    kept = kept_keys(source_map, layers, experts=experts, drop=drop)
    total = _routed_experts(source_config)

    def cap_router(key, tensor):
        if experts is not None and LAYER.match(key) and ROUTER_ROWS.search(key) and tensor.shape[0] == total:
            return tensor[:experts]
        return tensor

    shards = write_shards(
        source, source_map, list(kept), dest, shard_bytes=shard_bytes, renamed=kept, transform=cap_router
    )
    reduced = reduce_config(source_config, layers, experts=experts, overrides=config)
    (dest / "config.json").write_text(json.dumps(reduced, indent=2))
    copy_sidecars(source, dest)
    # Model code shipped beside the weights (DeepSeek's ``inference/``) is what an independent reference runs.
    for path in source.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            shutil.copytree(path, dest / path.name)
    return {"tensors": len(kept), "shards": shards}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arctic_platform.correctness.onboarding.layer_slice")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--layers", type=int, help="keep the first N layers")
    group.add_argument("--layer-ids", type=int, nargs="+", help="keep these layers, renumbered from 0")
    parser.add_argument("--experts", type=int, default=None, help="keep the first N routed experts")
    parser.add_argument("--drop", nargs="*", default=[], help="also drop tensors under these prefixes")
    parser.add_argument("--config", type=json.loads, default=None, help="JSON object of config overrides")
    args = parser.parse_args(argv)
    layers = args.layers if args.layers is not None else args.layer_ids
    result = slice_layers(
        args.source, args.dest, layers, experts=args.experts, drop=tuple(args.drop), config=args.config
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
