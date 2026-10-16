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

"""Optional Neutrino uploader integration; the harness only consumes the resulting fixture binding."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..harness.cortex_fixture import MANIFEST
from ..harness.cortex_fixture import describe_fixture
from ..harness.cortex_fixture import prepare_fixture
from ..harness.cortex_fixture import validate_publication
from ..harness.spec import TestSpec


def publish_fixture(
    spec: TestSpec,
    staging_root: Path,
    uploader: Path,
    uploader_python: str,
    receipt: Path,
    *,
    stage: str,
) -> dict:
    """Let the uploader authenticate, then bind its successful receipt to the reviewed weights."""
    if not stage.startswith("@") or stage.endswith("/"):
        raise ValueError("stage must be an explicit Snowflake stage without a trailing slash")
    receipt = Path(receipt)
    source = Path(spec.model.cache_path).resolve()
    if receipt.resolve() == source or source in receipt.resolve().parents:
        raise ValueError("publication receipt must be outside the reviewed reference checkpoint")
    if receipt.exists():
        raise ValueError(f"refusing to overwrite fixture publication receipt: {receipt}")
    staging, binding = prepare_fixture(spec, staging_root)
    excluded_formats = {".bin", ".pt", ".pth", ".ckpt", ".gguf", ".msgpack", ".h5", ".onnx", ".tflite"}
    for item in binding["files"]:
        if Path(item["path"]).suffix in excluded_formats:
            raise ValueError(f"reference contains a weight format excluded by the uploader: {item['path']}")
    subprocess.run(
        [
            uploader_python,
            str(uploader),
            "--local-path",
            str(staging),
            "--model-name",
            binding["model_name"],
            "--stage",
            stage,
        ],
        check=True,
    )
    manifest = json.loads((staging / MANIFEST).read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("model_name") != binding["model_name"]
        or manifest.get("stage_path") != f"{stage}/{binding['model_name']}"
    ):
        raise ValueError("upload manifest does not match the reference fixture binding")
    binding["publication"] = validate_publication(manifest, binding)
    # Check that the reference remained unchanged while the external publisher ran.
    if describe_fixture(spec, source) != {key: value for key, value in binding.items() if key != "publication"}:
        raise ValueError("reference files changed during publication")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x") as stream:
        stream.write(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    return binding
