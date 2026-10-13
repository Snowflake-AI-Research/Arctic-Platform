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

"""GDN construction must honor the installed model's normalization selection."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version
from torch import nn

from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import torch_assert_equal

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "arctic_platform/model/implementations/qwen35/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
)


class FusedNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))


class NativeNorm(FusedNorm):
    pass


def constructor(namespace):
    tree = ast.parse(MODEL_PATH.read_text(), filename=str(MODEL_PATH))
    node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5MoeGatedDeltaNet"
    )
    node.body = [member for member in node.body if isinstance(member, ast.FunctionDef) and member.name == "__init__"]
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(MODEL_PATH), "exec"),
        namespace,
    )
    return namespace["Qwen3_5MoeGatedDeltaNet"]


def config():
    return SimpleNamespace(
        hidden_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )


def installed_text_config(config_class, values):
    text_class = getattr(config_class, "sub_configs", {}).get("text_config", config_class)
    return text_class(**values)


@pytest.mark.parametrize("installed_version", ["5.12.1", "5.17.0"])
@pytest.mark.parametrize("optional_fused_available", [True, False])
def test_gdn_uses_installed_normalization_not_optional_package_presence(installed_version, optional_fused_available):
    namespace = {
        "torch": torch,
        "nn": nn,
        "Qwen3_5MoeConfig": SimpleNamespace,
        "FusedRMSNormGated": FusedNorm if optional_fused_available else None,
        "Qwen3_5MoeRMSNormGated": NativeNorm,
        "HfQwen3_5MoeRMSNormGated": NativeNorm,
        "TRANSFORMERS_VERSION": installed_version,
        "Version": Version,
        "causal_conv1d_fn": None,
        "chunk_gated_delta_rule": None,
        "torch_chunk_gated_delta_rule": None,
    }
    cfg = config()
    layer = constructor(namespace)(cfg)
    installed_norm = (
        FusedNorm if Version(installed_version) < Version("5.17") and optional_fused_available else NativeNorm
    )
    assert type(layer.norm) is installed_norm
    assert layer.norm.eps == cfg.rms_norm_eps
    assert list(layer.norm.state_dict()) == ["weight"]
    assert layer.norm.weight.shape == (cfg.linear_value_head_dim,)


def test_normalization_alias_comes_from_installed_model():
    tree = ast.parse(MODEL_PATH.read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
        and any(
            name.name == "Qwen3_5MoeRMSNormGated" and name.asname == "HfQwen3_5MoeRMSNormGated" for name in node.names
        )
        for node in tree.body
    )


@pytest.mark.parametrize("composite", [False, True])
def test_installed_text_config_preserves_small_gdn_dimensions(composite):
    class TextConfig:
        def __init__(self, **values):
            self.__dict__.update(values)
            self.layer_types = ["linear_attention"]

    class CompositeConfig:
        sub_configs = {"text_config": TextConfig}

        def __init__(self, **values):
            raise AssertionError("GDN requires the text config, not its composite parent")

    values = vars(config())
    result = installed_text_config(CompositeConfig if composite else TextConfig, values)
    assert type(result) is TextConfig
    assert result.layer_types == ["linear_attention"]
    for key, value in values.items():
        assert getattr(result, key) == value


@require_torch_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_selected_normalization_matches_installed_outputs_and_derivatives(dtype):
    from transformers import __version__
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import modeling_qwen3_5_moe as ap

    namespace = {
        **vars(ap),
        "HfQwen3_5MoeRMSNormGated": hf.Qwen3_5MoeRMSNormGated,
        "TRANSFORMERS_VERSION": __version__,
        "Version": Version,
    }
    cfg = config()
    norm = constructor(namespace)(cfg).norm.to(device="cuda", dtype=dtype)
    config_class = inspect.get_annotations(hf.Qwen3_5MoeGatedDeltaNet.__init__, eval_str=True)["config"]
    hf_config = installed_text_config(config_class, vars(cfg))
    reference = hf.Qwen3_5MoeGatedDeltaNet(hf_config, 0).norm
    assert type(norm) is type(reference)
    reference = reference.to(device="cuda", dtype=dtype)
    reference.load_state_dict(norm.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(12345)
    hidden, gate, upstream = [
        torch.randn(256, cfg.linear_value_head_dim, generator=generator).to(device="cuda", dtype=dtype)
        for _ in range(3)
    ]
    left_hidden, left_gate = [value.detach().clone().requires_grad_() for value in (hidden, gate)]
    right_hidden, right_gate = [value.detach().clone().requires_grad_() for value in (hidden, gate)]
    actual, expected = norm(left_hidden, left_gate), reference(right_hidden, right_gate)
    observed = torch.autograd.grad(actual, (left_hidden, left_gate, norm.weight), upstream)
    golden = torch.autograd.grad(expected, (right_hidden, right_gate, reference.weight), upstream)
    torch_assert_equal(actual, expected)
    for left, right in zip(observed, golden):
        torch_assert_equal(left, right)
    print(
        "INSTALLED_NORM_NATIVE_PARITY",
        __version__,
        dtype,
        type(norm).__module__,
        type(norm).__name__,
    )
