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
"""CPU contracts: the launch configuration fits the model's config, and the reference reads the config as shipped."""

from __future__ import annotations

import audit_lib
import pytest

LAUNCH_LENGTH = audit_lib.register_check(
    "contract.launch_length_supported", "the model config represents positions up to the launch sequence length"
)
REFERENCE_CONFIG = audit_lib.register_check(
    "contract.reference_config_matches_shipped",
    "the reference's mapping of config.json onto the checkpoint's inference code reproduces its shipped config",
)


@pytest.mark.audit_check(LAUNCH_LENGTH)
def test_launch_length_supported(profile, record):
    try:
        launch = profile.launch_training()
    except (LookupError, FileNotFoundError) as err:
        pytest.skip(f"no launch config: {err}")
    text = audit_lib.text_config_of(audit_lib.source_config(profile))
    max_len = int(launch["training_config"]["max_seq_len"])
    rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
    max_pos = text.get("max_position_embeddings")
    record(max_position_embeddings=max_pos, launch_max_seq_len=max_len, rope=rope)
    assert max_pos is None or int(max_pos) >= max_len, f"max_position_embeddings={max_pos} < launch {max_len}"
    rope_type = (rope or {}).get("rope_type", (rope or {}).get("type"))
    if rope_type == "yarn":
        assert rope.get("original_max_position_embeddings"), "YaRN without original_max_position_embeddings"


@pytest.mark.audit_check(REFERENCE_CONFIG)
def test_reference_config_matches_shipped(profile, record):
    if profile.reference != "deepseek_inference":
        pytest.skip("the reference is the model's transformers class")
    import deepseek_inference

    mismatches = deepseek_inference.shipped_args_mismatches(profile.checkpoint)
    record(mismatches=mismatches)
    assert not mismatches, mismatches
