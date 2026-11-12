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
"""Prediction authentication requires the separately supplied native capture contract."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from arctic_platform.debug.training import maybe_capture_reduced_fixture_mtp
from arctic_platform.testing_utils import torch_assert_equal

mtp_weights = pytest.importorskip(
    "arctic_platform.model.implementations.qwen35.mtp_weights",
    reason="Enabled prediction authentication requires the separate #437/#438 native capture implementation",
)


@pytest.mark.parametrize("model_type", ["qwen3_5_moe_text", "qwen4_exp_text"])
@pytest.mark.parametrize("inventory", ["per-expert", "stacked", "absent", "partial", "unexpected"])
def test_authenticated_prediction_inventory_matches_native_capture(tmp_path, model_type, inventory):
    model = torch.nn.Linear(2, 2)
    text = {
        "model_type": model_type,
        "num_hidden_layers": 1,
        "mtp_num_hidden_layers": 1,
        "num_experts": 2,
    }
    model.config = SimpleNamespace(text_config=SimpleNamespace(**text))
    names = mtp_weights._configured_names(model, set() if inventory != "stacked" else None)
    prediction = {name: torch.tensor([index, index + 1.0]) for index, name in enumerate(sorted(names))}
    if inventory == "absent":
        prediction.clear()
    elif inventory == "partial":
        prediction.pop("mtp.layers.0.mlp.experts.1.down_proj.weight")
    elif inventory == "unexpected":
        prediction["mtp.unexpected.weight"] = torch.ones(2)
    tensors = {"model.embed_tokens.weight": torch.arange(4).reshape(2, 2).float(), **prediction}
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"text_config": text}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    files = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(tmp_path.iterdir())
    ]
    request = {
        "schema_version": 1,
        "config_id": "prediction-inventory-contract",
        "fixture_locator": str(tmp_path),
        "reference_path": str((tmp_path / "reference").resolve()),
        "reviewed_hash": "0123456789abcdef",
        "fixture_sha256": (
            hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        ),
        "files": files,
        "source_checkpoint": "test/original",
        "num_hidden_layers": 1,
        "vision_depth": None,
        "materialization": "pretrained-prefix-reduction",
    }
    config = SimpleNamespace(name=str(tmp_path), debug=SimpleNamespace(reduced_fixture_mtp=request))
    if inventory in {"per-expert", "stacked"}:
        mtp_weights.capture_mtp_weights(model, tmp_path)
        strict = dict(mtp_weights.iter_preserved_mtp_weights(model))
        assert set(strict) == set(prediction)
        for name, tensor in strict.items():
            torch_assert_equal(tensor, prediction[name])
    if inventory in {"partial", "unexpected"}:
        with pytest.raises(RuntimeError, match="partial or unexpected MTP"):
            maybe_capture_reduced_fixture_mtp(model, tmp_path, config)
        return
    maybe_capture_reduced_fixture_mtp(model, tmp_path, config)
    if inventory == "absent":
        assert not hasattr(model, "_ap_preserved_mtp_weights")
        with pytest.raises(RuntimeError, match="requires captured original MTP"):
            list(mtp_weights.iter_preserved_mtp_weights(model))
        return
    restored = dict(mtp_weights.iter_preserved_mtp_weights(model))
    assert set(restored) == set(prediction)
    for name, tensor in restored.items():
        torch_assert_equal(tensor, prediction[name])
