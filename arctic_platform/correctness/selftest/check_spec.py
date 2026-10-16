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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral checks for frozen correctness specifications."""

from pathlib import Path

from arctic_platform.correctness.harness.spec import hash_directory


def test_model_hash_ignores_non_model_processor_sidecars(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("model")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    expected = hash_directory(tmp_path)

    (tmp_path / "preprocessor_config.json").write_text("image")
    (tmp_path / "video_preprocessor_config.json").write_text("video")

    assert hash_directory(tmp_path) == expected


def test_model_hash_keeps_tokenizer_metadata_in_identity(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("model")
    expected = hash_directory(tmp_path)

    (tmp_path / "tokenizer_config.json").write_text("tokenizer")

    assert hash_directory(tmp_path) != expected
