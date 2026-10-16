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

"""Bind a published Cortex model to the independent HF fixture, without choosing an uploader."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from arctic_platform.client.config import CortexConfig

from .spec import TestSpec
from .spec import hash_directory
from .workdir import require_data_fast_path

MANIFEST = "_neutrino_manifest.json"
TOKENIZER_JSON_SIDECARS = frozenset(
    {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_fixture(spec: TestSpec, root: Path) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"reviewed reference checkpoint is absent: {root}")
    # Legacy reduced specs predate tokenizer sidecars; publication still binds every file below.
    if (
        hash_directory(root) != spec.model.content_hash
        and hash_directory(root, exclude_names=TOKENIZER_JSON_SIDECARS) != spec.model.content_hash
    ):
        raise ValueError("reference files do not match the reviewed spec's file-name/size hash")
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    if text.get("num_hidden_layers") != spec.model.num_hidden_layers:
        raise ValueError("reference layer count does not match the reviewed spec")
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.name == MANIFEST or any(part in {".cache", ".git"} for part in rel.parts):
            continue
        if not path.is_file():
            continue
        files.append({"path": rel.as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    names = {item["path"] for item in files}
    if not any(name.endswith(".safetensors") for name in names) or "tokenizer_config.json" not in names:
        raise ValueError("reference checkpoint requires safetensors weights and tokenizer sidecars")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    basename = re.sub(r"[^A-Za-z0-9_-]", "-", Path(spec.model.cache_path).name)
    return {
        "schema_version": 1,
        "config_id": spec.config_id,
        "reference_path": spec.model.cache_path,
        "reviewed_hash": spec.model.content_hash,
        "fixture_sha256": digest,
        "model_name": f"ap-correctness/{basename}-{digest}",
        "files": files,
    }


def prepare_fixture(spec: TestSpec, staging_root: Path) -> tuple[Path, dict]:
    staging_root = require_data_fast_path(staging_root, label="Cortex fixture staging")
    source = Path(spec.model.cache_path).resolve()
    if staging_root == source or source in staging_root.parents:
        raise ValueError("staging must be outside the reviewed reference checkpoint")
    binding = describe_fixture(spec, source)
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="ap-cortex-fixture-", dir=staging_root))
    for item in binding["files"]:
        relative = Path(item["path"])
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    if describe_fixture(spec, staging) != binding:
        raise ValueError("reference files changed while preparing publication")
    return staging, binding


def validate_publication(publication: dict, binding: dict) -> dict:
    """Check a publisher's file receipt without prescribing its stage or authentication."""
    stage_path = publication.get("stage_path")
    if (
        not isinstance(stage_path, str)
        or not stage_path.startswith("@")
        or not stage_path.endswith(f"/{binding['model_name']}")
        or publication.get("files") != binding["files"]
    ):
        raise ValueError("publication does not match the reference fixture")
    return {"stage_path": stage_path, "files": publication["files"]}


def load_fixture(receipt: Path, spec: TestSpec) -> dict:
    binding = json.loads(Path(receipt).read_text())
    publication = binding.get("publication")
    if not isinstance(publication, dict):
        raise ValueError("fixture binding has no successful publication receipt")
    actual = describe_fixture(spec, Path(spec.model.cache_path))
    if any(binding.get(key) != value for key, value in actual.items()):
        raise ValueError("reference files or model identity no longer match the published fixture")
    validate_publication(publication, actual)
    return binding


def load_local_fixture(backend: CortexConfig, spec: TestSpec) -> dict:
    """Validate local weights and gateway health without attesting the server backend."""
    import requests

    try:
        return _load_local_fixture(backend, spec)
    except (requests.RequestException, OSError) as exc:
        raise ValueError(f"local Cortex fixture verification failed: {type(exc).__name__}: {exc}") from exc


def _load_local_fixture(backend: CortexConfig, spec: TestSpec) -> dict:
    import requests

    parsed = urlsplit(backend.base_url or "")
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or backend.host
        or backend.pat is not None
    ):
        raise ValueError("local Cortex fixture requires an unauthenticated loopback HTTP base URL")
    base_url = backend.base_url.rstrip("/")
    model_path = require_data_fast_path(spec.model.cache_path, label="local Cortex reference model")
    with requests.Session() as session:
        session.trust_env = False
        health = session.get(f"{base_url}/health", timeout=30, allow_redirects=False)
        if health.status_code != 200:
            raise ValueError("local gateway is not healthy")
        status = health.json()
        if not isinstance(status, dict) or status.get("status") != "OK":
            raise ValueError("local gateway is not healthy")
        binding = describe_fixture(spec, model_path)
    return {**binding, "model_name": str(model_path), "distribution": "local-gateway"}
