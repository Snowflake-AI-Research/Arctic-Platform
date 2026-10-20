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

"""Header-only admission controls for the pinned original Flash-Next checkpoint."""

import hashlib
import json
import math
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
from arctic_platform.correctness.models import original

BASE = Path(__file__).resolve().parents[1]
CONFIG = BASE / "configs/qwen3.8-flash-next/h200/train-sft-full-16gpus-sp16-256k.config"
SPEC = BASE / "specs/qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k.json"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
CONFIG_SHA = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
INDEX_SHA = "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de"
SCHEMA_SHA = "1d55a6d3f64b1b2b004302864c0e74e11871ddf9f3481b10fc21fc8b0ec53d26"
PAYLOAD_BYTES = 359999963128
PLE = "model.language_model.layers.1.ple.ple_embedding."
BUFFERS = {"layer_multipliers": [3], "ngram_heads_offsets": [16], "ngram_heads_vocab_sizes": [16]}


def schema_bytes(schema):
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


@pytest.fixture
def full_model(tmp_path, monkeypatch):
    cfg, spec = load_config(CONFIG), Spec.read(SPEC)
    cache = tmp_path / "hub"
    root = cache / "models--Qwen--Qwen3.8-Flash-Next" / "snapshots" / REVISION
    root.mkdir(parents=True)
    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {"num_hidden_layers": 48, "hidden_size": 2560, "vocab_size": 248320},
        "vision_config": {"depth": 27},
    }
    schema = {
        **{f"model.language_model.layers.{i}.norm.weight": ["BF16", [2560]] for i in range(48)},
        **{f"model.visual.blocks.{i}.norm.weight": ["BF16", [4]] for i in range(27)},
        "lm_head.weight": ["BF16", [248320, 2560]],
        "model.language_model.embed_tokens.weight": ["BF16", [248320, 2560]],
        **{PLE + name: ["I64", shape.copy()] for name, shape in BUFFERS.items()},
        **{f"mtp.synthetic.{i}.weight": ["BF16", [4]] for i in range(1578)},
    }
    used = sum(math.prod(shape) * (8 if dtype == "I64" else 2) for dtype, shape in schema.values())
    schema["mtp.synthetic.0.weight"][1] = [(PAYLOAD_BYTES - used) // 2 + 4]
    mapping = {name: f"model-{i % 131 + 1:05d}-of-00131.safetensors" for i, name in enumerate(schema)}
    index = {"metadata": {"total_size": PAYLOAD_BYTES}, "weight_map": mapping}
    raw_config, raw_index = json.dumps(config).encode(), json.dumps(index).encode()
    (root / "config.json").write_bytes(raw_config)
    (root / "model.safetensors.index.json").write_bytes(raw_index)
    for name in {"tokenizer.json", "tokenizer_config.json", *mapping.values()}:
        (root / name).write_text("{}")

    # Fixed synthetic headers exercise the real gate without model-sized tensor payloads.
    pins = {raw_config: CONFIG_SHA, raw_index: INDEX_SHA, schema_bytes(schema): SCHEMA_SHA}
    real_sha = hashlib.sha256

    def digest(raw):
        return SimpleNamespace(hexdigest=lambda: pins[raw]) if raw in pins else real_sha(raw)

    monkeypatch.setattr(original, "hashlib", SimpleNamespace(sha256=digest))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DSS_ALLOW_REMOTE_MODELS", "1")
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(root / "config.json"))
    calls = []

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
                return [name for name in schema if mapping.get(name) == shard]

            def get_slice(self, name):
                dtype, shape = schema[name]
                return SimpleNamespace(get_dtype=lambda: dtype, get_shape=lambda: shape)

            def get_tensor(self, name):
                pytest.fail("source admission must not materialize tensor payloads")

        return Headers()

    monkeypatch.setattr(safetensors, "safe_open", safe_open)
    return SimpleNamespace(root=root, cfg=cfg, spec=spec, schema=schema, mapping=mapping, calls=calls)


def validate(case):
    return runner.validate_local_full_model(CortexConfig(base_url="http://127.0.0.1:8003"), case.cfg, case.spec)


@pytest.mark.parametrize("source", ["model-id", "pinned-snapshot"])
def test_original_flashnext_is_admitted_header_only(full_model, source):
    original = CONFIG.read_bytes(), SPEC.read_bytes()
    if source == "model-id":
        full_model.spec.model.source_checkpoint = full_model.cfg.model_name
    assert validate(full_model) == full_model.root
    assert len(full_model.calls) == 131
    assert (CONFIG.read_bytes(), SPEC.read_bytes()) == original


@pytest.mark.parametrize("buffer", sorted(BUFFERS))
@pytest.mark.parametrize("failure", ["missing", "dtype", "shape"])
def test_every_original_integer_buffer_is_pinned(full_model, buffer, failure):
    name = PLE + buffer
    if failure == "missing":
        full_model.schema.pop(name)
    elif failure == "dtype":
        full_model.schema[name][0] = "BF16"
    else:
        full_model.schema[name][1] = [1, *full_model.schema[name][1]]
    with pytest.raises(ValueError):
        validate(full_model)


@pytest.mark.parametrize(
    "failure",
    [
        "source",
        "config-id",
        "model-id",
        "config",
        "index",
        "missing-shard",
        "tokenizer",
        "text-layer",
        "vision-layer",
        "vocab",
        "shape",
        "dtype",
        "same-byte-dtype",
    ],
)
def test_original_flashnext_rejects_drift(full_model, failure):
    case = full_model
    if failure == "source":
        case.spec.model.source_checkpoint = "/data-fast/base-models/synthetic/FlashNext-4L"
    elif failure == "config-id":
        case.cfg.config_id = "unreviewed-flashnext"
    elif failure == "model-id":
        case.cfg.sub_job["model_name"] = "Qwen/other"
    elif failure in {"config", "index"}:
        name = "config.json" if failure == "config" else "model.safetensors.index.json"
        with (case.root / name).open("ab") as stream:
            stream.write(b"\n")
    elif failure in {"missing-shard", "tokenizer"}:
        name = next(iter(case.mapping.values())) if failure == "missing-shard" else "tokenizer.json"
        (case.root / name).rename(case.root / (name + ".retained"))
    elif failure in {"text-layer", "vision-layer"}:
        name = (
            "model.language_model.layers.47.norm.weight"
            if failure == "text-layer"
            else "model.visual.blocks.26.norm.weight"
        )
        case.schema.pop(name)
    elif failure == "vocab":
        case.schema["lm_head.weight"][1] = [248319, 2560]
    elif failure == "shape":
        case.schema["model.language_model.layers.0.norm.weight"][1] = [1280, 2]
    elif failure == "dtype":
        case.schema["model.language_model.layers.0.norm.weight"][0] = "F32"
    else:
        case.schema["model.language_model.layers.0.norm.weight"] = ["I64", [640]]
    with pytest.raises((ValueError, OSError)):
        validate(case)


@pytest.mark.parametrize("failure", ["revision", "repository", "offline", "remote"])
def test_original_flashnext_rejects_wrong_cache_or_online_mode(full_model, monkeypatch, failure):
    if failure in {"revision", "repository"}:
        root = full_model.root
        wrong = root.with_name("0" * 40) if failure == "revision" else root.parent.parent.with_name("other")
        monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(wrong / "config.json"))
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE" if failure == "offline" else "DSS_ALLOW_REMOTE_MODELS", "0")
    with pytest.raises(ValueError):
        validate(full_model)


def test_original_flashnext_inference_dispatch_never_runs_hf(full_model, tmp_path, monkeypatch):
    observed = []
    full_model.spec.applicable_tests.append("inference-checkpoint-loss")

    def measure(ctx):
        assert ctx.local_gateway_full_model is True
        assert ctx.local_full_model_path == str(full_model.root)
        observed.append(ctx)
        yield Result(test_id="inference-checkpoint-loss", config_id=ctx.config_id, outcome=Outcome.PASS)

    monkeypatch.setattr(runner.registered_tests()["inference-checkpoint-loss"], "fn", measure)
    monkeypatch.setattr(runner, "run_reference", lambda *args, **kwargs: pytest.fail("no inference HF reference"))
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


def test_original_flashnext_source_admission_does_not_change_applicability(full_model, tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.registered_tests()["inference-checkpoint-loss"],
        "fn",
        lambda *args: pytest.fail("source admission cannot make a check applicable"),
    )
    monkeypatch.setattr(runner, "run_reference", lambda *args, **kwargs: pytest.fail("no inference HF reference"))
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
    assert [result.outcome for result in results] == [Outcome.INAPPLICABLE]
