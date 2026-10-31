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
"""CPU contracts: the launch configuration resolves and validates, and the tiny checkpoint is representative."""

from __future__ import annotations

import re

import audit_lib
import pytest

from arctic_platform.model import ModelSpec

LOADER = audit_lib.register_check(
    "contract.loader_resolution", "the launch ModelSpec builds the model family's own implementation"
)
LAUNCH_SPEC = audit_lib.register_check(
    "contract.launch_spec_valid", "the launch ModelSpec (dtype, parallelism, length, loader options) validates"
)
LAUNCH_LENGTH = audit_lib.register_check(
    "contract.launch_length_supported", "the model config represents positions up to the launch sequence length"
)
LAYER_KINDS = audit_lib.register_check(
    "contract.tiny_covers_layer_kinds", "the tiny checkpoint keeps every distinct layer kind of the real model"
)
TINY_LAYOUT = audit_lib.register_check(
    "contract.tiny_layout_matches_source",
    "the tiny checkpoint has the real checkpoint's tensor-name layout, so loading it exercises the real conversion",
)


def _launch_spec(profile, tiny_ckpt) -> ModelSpec:
    return ModelSpec(**profile.trainer_spec(model_path=str(tiny_ckpt), tiny=False))


@pytest.mark.audit_check(LOADER)
def test_loader_resolution(profile, tiny_ckpt, record):
    spec = _launch_spec(profile, tiny_ckpt)
    explicit = profile.launch.get("trainer", {}).get("loader")
    resolved = ModelSpec(**{**profile.trainer_spec(model_path=str(tiny_ckpt), tiny=False), "loader": None}).loader
    record(explicit=explicit, auto_resolved=resolved, expected=profile.expect.get("loader"))
    assert spec.loader == profile.expect["loader"]
    # A launcher that does not pin the loader gets whatever resolution picks.
    assert (
        resolved == profile.expect["loader"]
    ), f"without an explicit loader the launch spec resolves to {resolved!r}, not {profile.expect['loader']!r}"


@pytest.mark.audit_check(LAUNCH_SPEC)
def test_launch_spec_valid(profile, tiny_ckpt, record):
    spec = _launch_spec(profile, tiny_ckpt)
    record(loader=spec.loader, parallelism=spec.parallelism.model_dump(), dtype=spec.dtype)
    assert spec.dtype == profile.dtype


@pytest.mark.audit_check(LAUNCH_LENGTH)
def test_launch_length_supported(profile, record):
    text = audit_lib.text_config_of(audit_lib.source_config(profile))
    max_len = int(profile.launch["max_seq_len"])
    rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
    max_pos = text.get("max_position_embeddings")
    record(max_position_embeddings=max_pos, launch_max_seq_len=max_len, rope=rope)
    assert max_pos is None or int(max_pos) >= max_len, f"max_position_embeddings={max_pos} < launch {max_len}"
    rope_type = (rope or {}).get("rope_type", (rope or {}).get("type"))
    if rope_type == "yarn":
        assert rope.get("original_max_position_embeddings"), "YaRN without original_max_position_embeddings"


@pytest.mark.audit_check(LAYER_KINDS)
def test_tiny_covers_layer_kinds(profile, record):
    full = audit_lib.source_config(profile)
    tiny, keep = audit_lib.shrink_config(full, profile.tiny)
    full_kinds = set(audit_lib.layer_signatures(audit_lib.text_config_of(full)))
    kept_kinds = {audit_lib.layer_signatures(audit_lib.text_config_of(full))[i] for i in keep}
    record(source_layers=audit_lib.text_config_of(full)["num_hidden_layers"], kept=keep, kinds=len(full_kinds))
    assert kept_kinds == full_kinds


def _layout(names, keep: list[int] | None = None) -> set[str]:
    """Name patterns with layer and expert indices abstracted; restricted to kept layers when given."""
    out = set()
    for name in names:
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match and keep is not None and int(match.group(1)) not in keep:
            continue
        out.add(re.sub(r"\.\d+(?=\.|$)", ".N", name))
    return out


@pytest.mark.audit_check(TINY_LAYOUT)
def test_tiny_layout_matches_source(profile, tiny_ckpt, record):
    full = audit_lib.source_config(profile)
    _, keep = audit_lib.shrink_config(full, profile.tiny)
    source = _layout(audit_lib.source_weight_names(profile), keep)
    tiny = _layout(audit_lib.read_checkpoint_tensors(tiny_ckpt).keys())
    ignore = [re.compile(p) for p in profile.expect.get("layout_ignore", [])]
    if full.get("tie_word_embeddings", audit_lib.text_config_of(full).get("tie_word_embeddings")):
        ignore.append(re.compile(r"(^|\.)lm_head\.weight$"))
    source = {n for n in source if not any(p.search(n) for p in ignore)}
    tiny = {n for n in tiny if not any(p.search(n) for p in ignore)}
    only_source, only_tiny = sorted(source - tiny), sorted(tiny - source)
    record(only_in_source=only_source[:40], only_in_tiny=only_tiny[:40])
    assert not only_source and not only_tiny, (
        f"{len(only_source)} name pattern(s) only in the real checkpoint (e.g. {only_source[:5]}), "
        f"{len(only_tiny)} only in the tiny one (e.g. {only_tiny[:5]})"
    )
