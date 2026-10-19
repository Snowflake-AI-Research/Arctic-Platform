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

"""Header-only controls for the explicitly pinned full dense27B source."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors
from transformers.utils import hub

from arctic_platform.client import CortexConfig
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result
from arctic_platform.correctness.harness.spec import TestSpec as Spec

BASE = Path(__file__).resolve().parents[1]
CONFIG = BASE / "configs/qwen3.8-27b/b200/train-sft-8gpus-2k.config"
SPEC = BASE / "specs/qwen3.8-27b-b200-train-sft-8gpus-2k.json"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CONFIG_SHA = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
INDEX_SHA = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
SCHEMA_SHA = "9c0eb904ed294a62882adb287b81e36fdb48e5fb8e006ee25b084341ac78fd55"
TENSOR_BYTES = 55562855904


def schema_bytes(schema):
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


@pytest.fixture
def full_model(tmp_path, monkeypatch):
    cfg, spec = load_config(CONFIG), Spec.read(SPEC)
    cache = tmp_path / "hub"
    root = cache / "models--Qwen--Qwen3.8-27B" / "snapshots" / REVISION
    root.mkdir(parents=True)
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {"num_hidden_layers": 64, "hidden_size": 5120, "vocab_size": 248320},
        "vision_config": {"depth": 27},
    }
    schema = {
        **{f"model.language_model.layers.{i}.input_layernorm.weight": ["BF16", [5120]] for i in range(64)},
        **{f"model.visual.blocks.{i}.norm.weight": ["BF16", [4]] for i in range(27)},
        "lm_head.weight": ["BF16", [248320, 5120]],
        "model.language_model.embed_tokens.weight": ["BF16", [248320, 5120]],
        **{f"mtp.synthetic.{i}.weight": ["BF16", [4]] for i in range(1106)},
    }
    used = sum(2 * _elements(shape) for _, shape in schema.values())
    schema["mtp.synthetic.0.weight"][1] = [(TENSOR_BYTES - used) // 2 + 4]
    mapping = {name: f"model-{i % 18 + 1:05d}-of-00018.safetensors" for i, name in enumerate(schema)}
    index = {"metadata": {"total_size": TENSOR_BYTES}, "weight_map": mapping}
    config_bytes = json.dumps(config).encode()
    index_bytes = json.dumps(index).encode()
    (root / "config.json").write_bytes(config_bytes)
    (root / "model.safetensors.index.json").write_bytes(index_bytes)
    for filename in {"tokenizer.json", "tokenizer_config.json", *mapping.values()}:
        (root / filename).write_text("{}")

    # Synthetic headers exercise the admission logic without creating model-sized weights.
    pins = {config_bytes: CONFIG_SHA, index_bytes: INDEX_SHA, schema_bytes(schema): SCHEMA_SHA}
    real_sha256 = hashlib.sha256

    def digest(data):
        if data in pins:
            return SimpleNamespace(hexdigest=lambda: pins[data])
        return real_sha256(data)

    monkeypatch.setattr(runner, "hashlib", SimpleNamespace(sha256=digest))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DSS_ALLOW_REMOTE_MODELS", "1")
    calls = []

    def cached_file(model, filename, **kwargs):
        assert model == cfg.model_name and filename == "config.json"
        assert kwargs == {"local_files_only": True}
        return str(root / filename)

    def safe_open(path, **kwargs):
        assert kwargs == {"framework": "pt", "device": "cpu"}
        shard = Path(path).name
        calls.append(shard)

        class Headers:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def keys(self):
                return [name for name in schema if mapping[name] == shard]

            def get_slice(self, name):
                dtype, shape = schema[name]
                return SimpleNamespace(get_shape=lambda: shape, get_dtype=lambda: dtype)

            def get_tensor(self, name):
                pytest.fail("full-model admission must never load tensor payloads")

        return Headers()

    monkeypatch.setattr(hub, "cached_file", cached_file)
    monkeypatch.setattr(safetensors, "safe_open", safe_open)
    return SimpleNamespace(root=root, cfg=cfg, spec=spec, schema=schema, mapping=mapping, calls=calls)


def _elements(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def validate(case):
    return runner.validate_local_full_model(CortexConfig(base_url="http://127.0.0.1:8003"), case.cfg, case.spec)


@pytest.mark.parametrize("source", ["Qwen/Qwen3.8-27B", "/data-fast/base-models/Qwen/Qwen3.8-27B"])
def test_full_27b_original_source_is_admitted_header_only(full_model, source):
    original = CONFIG.read_bytes(), SPEC.read_bytes()
    full_model.spec.model.source_checkpoint = source
    assert validate(full_model) == full_model.root
    assert len(full_model.calls) == 18
    assert (CONFIG.read_bytes(), SPEC.read_bytes()) == original


@pytest.mark.parametrize(
    "failure",
    [
        "source",
        "config",
        "index",
        "missing-shard",
        "missing-tokenizer",
        "missing-text-layer",
        "missing-vision-layer",
        "vocabulary-shape",
        "same-size-shape",
        "same-size-vision-shape",
        "same-size-mtp-shape",
        "dtype",
        "same-size-dtype",
    ],
)
def test_full_27b_invalid_source_is_rejected(full_model, failure):
    root, schema = full_model.root, full_model.schema
    if failure == "source":
        full_model.spec.model.source_checkpoint = "/data-fast/base-models/synthetic/Qwen3.8-27B-4L"
    elif failure in {"config", "index"}:
        filename = "config.json" if failure == "config" else "model.safetensors.index.json"
        with (root / filename).open("ab") as stream:
            stream.write(b"\n")
    elif failure == "missing-shard":
        (root / next(iter(full_model.mapping.values()))).unlink()
    elif failure == "missing-tokenizer":
        (root / "tokenizer.json").unlink()
    elif failure in {"missing-text-layer", "missing-vision-layer"}:
        name = (
            "model.language_model.layers.63.input_layernorm.weight"
            if failure == "missing-text-layer"
            else "model.visual.blocks.26.norm.weight"
        )
        del schema[name]
    elif failure == "vocabulary-shape":
        schema["lm_head.weight"][1] = [248319, 5120]
    elif failure == "same-size-shape":
        schema["model.language_model.layers.0.input_layernorm.weight"][1] = [2560, 2]
    elif failure == "same-size-vision-shape":
        schema["model.visual.blocks.0.norm.weight"][1] = [2, 2]
    elif failure == "same-size-mtp-shape":
        schema["mtp.synthetic.1.weight"][1] = [2, 2]
    elif failure == "dtype":
        schema["model.language_model.layers.0.input_layernorm.weight"][0] = "F16"
    else:
        schema["model.language_model.layers.0.input_layernorm.weight"] = ["F32", [2560]]
    with pytest.raises((ValueError, OSError)):
        validate(full_model)


@pytest.mark.parametrize("failure", ["revision", "repository", "cache-escape"])
def test_full_27b_snapshot_root_is_pinned(full_model, monkeypatch, failure):
    root = full_model.root
    wrong = {
        "revision": root.with_name("0" * 40),
        "repository": root.parent.parent.with_name("models--other"),
        "cache-escape": Path("/tmp/unapproved-full-model"),
    }[failure]
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(wrong / "config.json"))
    with pytest.raises(ValueError):
        validate(full_model)


def test_full_27b_rejects_other_config_before_cache_lookup(full_model, monkeypatch):
    full_model.cfg.config_id = "unreviewed-full-model-config"
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: pytest.fail("unreviewed source lookup"))
    with pytest.raises(ValueError, match="supported pinned original"):
        validate(full_model)


def test_full_27b_shard_symlink_cannot_escape_cache(full_model, tmp_path):
    shard = full_model.root / next(iter(full_model.mapping.values()))
    shard.unlink()
    outside = tmp_path / "outside.safetensors"
    outside.write_text("{}")
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="outside its cache"):
        validate(full_model)


def test_full_27b_inference_dispatch_never_runs_hf_reference(full_model, tmp_path, monkeypatch):
    observed = []

    def measure(ctx):
        observed.append(ctx)
        assert ctx.local_gateway_full_model is True
        assert ctx.local_full_model_path == str(full_model.root)
        assert ctx.target_model_name is None
        assert ctx.cfg.model_name == "Qwen/Qwen3.8-27B"
        assert ctx.vocab_size == 248320
        yield Result(test_id="inference-checkpoint-loss", config_id=ctx.config_id, outcome=Outcome.PASS)

    monkeypatch.setattr(runner.registered_tests()["inference-checkpoint-loss"], "fn", measure)
    monkeypatch.setattr(runner, "run_reference", lambda *args, **kwargs: pytest.fail("no HF reference for inference"))
    results = runner.execute(
        full_model.cfg,
        full_model.spec,
        [],
        vocab_size=248320,
        select=["inference-checkpoint-loss"],
        workdir=tmp_path,
        client_backend=CortexConfig(base_url="http://127.0.0.1:8003"),
        local_gateway_full_model=True,
    )
    assert len(observed) == 1
    assert [result.outcome for result in results] == [Outcome.PASS]
