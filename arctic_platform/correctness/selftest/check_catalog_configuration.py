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

"""Static correctness selftests: catalog configuration."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers.utils import hub

from arctic_platform.client import CortexConfig
from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.correctness.checks.fwd_bwd import _run_arm
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.config import GPU_WIDTH_MULTIPLE_CAP
from arctic_platform.correctness.harness.config import ArmSpec
from arctic_platform.correctness.harness.config import ModelSpec
from arctic_platform.correctness.harness.config import TestSpec as ConfigSpec
from arctic_platform.correctness.harness.config import TestTolerance as Tolerance
from arctic_platform.correctness.harness.config import arms_for
from arctic_platform.correctness.harness.config import config_checksum
from arctic_platform.correctness.harness.config import correctness_microbatch_tokens
from arctic_platform.correctness.harness.config import hash_directory
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.config import placement_widths
from arctic_platform.correctness.harness.config import scaled_for_placement
from arctic_platform.correctness.harness.config import validate_against_host
from arctic_platform.correctness.harness.names import align
from arctic_platform.correctness.harness.names import canonical_dss_tensor_groups
from arctic_platform.correctness.harness.runner import assert_applicable_tests_registered
from arctic_platform.correctness.models import original
from arctic_platform.correctness.models import validate_local_full_model

from .helpers import _config_factory_native_config

# Applicable tests.


def _applicable_tests_spec(applicable):
    return ConfigSpec(
        config_id="unit",
        config_path="unit.config",
        model=ModelSpec(
            source_checkpoint="model",
            num_hidden_layers=1,
            vision_depth=0,
            seed=0,
            param_count=1,
            cache_path="model",
            content_hash="hash",
            sized_against_gib=1.0,
        ),
        arms=[
            ArmSpec(
                name="fixed",
                global_batch_size=1,
                max_seq_len=8,
                total_tokens=8,
                active_tokens=7,
                dss_microbatches=1,
                pad_fraction=0.125,
            )
        ],
        attn_implementations=["flash_attention_3"],
        applicable_tests=applicable,
    )


def test_applicable_tests_a_registered_id_is_accepted():
    assert_applicable_tests_registered(_applicable_tests_spec(["single-step-grads"]), "unit")


def test_applicable_tests_an_unregistered_id_is_refused_rather_than_filtered_away():
    with pytest.raises(ValueError, match="nothing registers"):
        assert_applicable_tests_registered(_applicable_tests_spec(["t01_grad_norms"]), "unit")


def test_applicable_tests_the_refusal_names_every_unregistered_id():
    """An operator fixing the spec needs all of them, not the first one found."""
    with pytest.raises(ValueError) as failure:
        assert_applicable_tests_registered(_applicable_tests_spec(["t01_grad_norms", "t02_delta_update"]), "unit")
    assert "t01_grad_norms" in str(failure.value)
    assert "t02_delta_update" in str(failure.value)


# Arms.


def test_arms_eight_gpu_arms_cap_gas1_at_64k_total() -> None:
    arms = arms_for(max_row_tokens=262144, gpu_count=8)
    assert [(arm.name, arm.global_batch_size, arm.max_seq_len, arm.total_tokens) for arm in arms] == [
        ("gas1", 8, 8192, 65536),
        ("gas4", 32, 8192, 262144),
    ]


def test_arms_four_gpu_arms_scale_rows_and_sequence_cap() -> None:
    arms = arms_for(max_row_tokens=32768, gpu_count=4)
    assert [(arm.name, arm.global_batch_size, arm.max_seq_len) for arm in arms] == [
        ("gas1", 4, 16384),
        ("gas4", 16, 16384),
    ]


def test_arms_lower_sequence_limit_is_preserved() -> None:
    gas1, gas4 = arms_for(max_row_tokens=2048, gpu_count=8)
    assert (gas1.global_batch_size, gas1.max_seq_len, gas1.total_tokens) == (8, 2048, 16384)
    assert (gas4.global_batch_size, gas4.max_seq_len, gas4.total_tokens) == (32, 2048, 65536)


def test_arms_microbatch_count_uses_the_capped_correctness_budget() -> None:
    gas1, gas4 = arms_for(max_row_tokens=262144, gpu_count=8)
    budget = correctness_microbatch_tokens(262144)
    assert budget == 65536
    assert gas1.microbatches(budget, dp_size=1) == 1
    assert gas4.microbatches(budget, dp_size=1) == 4


def test_arms_smaller_microbatch_budget_is_preserved() -> None:
    assert correctness_microbatch_tokens(10240) == 10240


# Config gpu attention.


def _config_gpu_attention_write_config(tmp_path: Path, gpu_type: str, attention: str | None, n_gpus: int = 8) -> Path:
    path = tmp_path / "configs" / "model" / gpu_type / "workload.config"
    path.parent.mkdir(parents=True)
    training = {"n_gpus": n_gpus, "sp_size": n_gpus}
    if attention is not None:
        training["attn_implementation"] = attention
    path.write_text(json.dumps(_config_factory_native_config(training)))
    return path


@pytest.mark.parametrize(
    ("gpu_type", "expected"),
    [("h200", "flash_attention_3"), ("b200", "flash_attention_4"), ("b300", "flash_attention_4")],
)
def test_config_gpu_attention_attention_is_derived_from_gpu_type(tmp_path: Path, gpu_type: str, expected: str) -> None:
    config = load_config(_config_gpu_attention_write_config(tmp_path, gpu_type, None))
    assert config.attention_implementation == expected


def test_config_gpu_attention_config_rejects_attention_for_another_gpu_family(tmp_path: Path) -> None:
    config = load_config(_config_gpu_attention_write_config(tmp_path, "b200", "flash_attention_3"))
    with pytest.raises(ValueError, match="b200 requires attn_implementation='flash_attention_4'"):
        validate_against_host(config, available_gpus=8, check_gpu_type=False)


def test_config_gpu_attention_config_accepts_gpu_family_attention(tmp_path: Path) -> None:
    config = load_config(_config_gpu_attention_write_config(tmp_path, "b300", "flash_attention_4"))
    validate_against_host(config, available_gpus=8, check_gpu_type=False)


def test_config_gpu_attention_config_uses_fewer_gpus_than_the_host_when_requested(tmp_path: Path) -> None:
    config = load_config(_config_gpu_attention_write_config(tmp_path, "h200", "flash_attention_3", n_gpus=4))
    validate_against_host(config, available_gpus=8, check_gpu_type=False)
    assert config.n_gpus == 4
    assert config.sp_size == 4


def test_config_gpu_attention_effective_training_applies_nested_prime_rl_overrides(tmp_path: Path) -> None:
    path = _config_gpu_attention_write_config(tmp_path, "h200", "flash_attention_3")
    raw = json.loads(path.read_text())
    training = raw["training"]["ds_worker_config"]
    training["fused_lm_head_token_chunk_size"] = 4096
    training["prime_rl"] = {"fused_lm_head_token_chunk_size": 8192, "fused_cross_entropy": False}
    path.write_text(json.dumps(raw))
    effective = load_config(path).effective_training
    assert effective["fused_lm_head_token_chunk_size"] == 8192
    assert effective["fused_cross_entropy"] is False


def test_config_gpu_attention_disabled_lm_head_chunk_resolves_to_none(tmp_path: Path) -> None:
    path = _config_gpu_attention_write_config(tmp_path, "h200", "flash_attention_3")
    raw = json.loads(path.read_text())
    raw["training"]["ds_worker_config"]["prime_rl"] = {"fused_lm_head_token_chunk_size": "disabled"}
    path.write_text(json.dumps(raw))
    assert load_config(path).lm_head_token_chunk_size is None


def test_config_gpu_attention_config_checksum_is_canonical_and_tracks_training_behavior(tmp_path: Path) -> None:
    path = _config_gpu_attention_write_config(tmp_path, "h200", "flash_attention_3", n_gpus=4)
    first = config_checksum(load_config(path))
    raw = json.loads(path.read_text())
    path.write_text(json.dumps(raw, indent=4))
    assert config_checksum(load_config(path)) == first
    raw["training_gpus"] = 8
    path.write_text(json.dumps(raw))
    assert config_checksum(load_config(path)) != first


def test_config_gpu_attention_optimizer_dtype_follows_explicit_deepspeed_state_precision(tmp_path: Path) -> None:
    path = _config_gpu_attention_write_config(tmp_path, "h200", "flash_attention_3")
    raw = json.loads(path.read_text())
    raw["training"]["ds_config"] = {
        "bf16": {"enabled": True, "bf16_master_weights_and_grads": True, "bf16_optimizer_states": True}
    }
    path.write_text(json.dumps(raw))
    assert load_config(path).optimizer_dtype == "bfloat16"


def test_config_gpu_attention_optimizer_dtype_rejects_mixed_master_and_state_precision(tmp_path: Path) -> None:
    path = _config_gpu_attention_write_config(tmp_path, "h200", "flash_attention_3")
    raw = json.loads(path.read_text())
    raw["training"]["ds_config"] = {"bf16": {"enabled": True, "bf16_master_weights_and_grads": True}}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="matching Adam master and state precision"):
        _ = load_config(path).optimizer_dtype


def test_config_gpu_attention_optimizer_dtype_matches_deepspeed_bf16_default_without_override(tmp_path: Path) -> None:
    path = _config_gpu_attention_write_config(tmp_path, "h200", "flash_attention_3")
    assert load_config(path).optimizer_dtype == "bfloat16"


def _config_gpu_attention_write(tmp_path, training_config: dict):
    """Write a minimal single-training-sub-job config and return its path."""
    import json

    path = tmp_path / "h200" / "probe.config"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_config_factory_native_config(training_config)))
    return path


def test_config_gpu_attention_declared_row_width_is_used_as_written(tmp_path) -> None:
    """A config that states max_seq_len is measured at that width."""
    cfg = _config_gpu_attention_write(
        tmp_path, {"max_seq_len": 4096, "n_gpus": 8, "attn_implementation": "flash_attention_3"}
    )
    assert load_config(cfg).max_seq_len == 4096


def test_config_gpu_attention_absent_row_width_follows_the_engine_default(tmp_path) -> None:
    """A config with no max_seq_len uses the Arctic client schema default."""
    cfg = _config_gpu_attention_write(tmp_path, {"n_gpus": 8, "attn_implementation": "flash_attention_3"})
    assert load_config(cfg).max_seq_len == ArcticClientConfig.model_fields["max_seq_len"].default


@pytest.mark.parametrize("attention", [None, "qsa_flex"])
def test_config_gpu_attention_flash_next_requires_model_specific_attention(
    tmp_path: Path, attention: str | None
) -> None:
    path = _config_gpu_attention_write_config(tmp_path, "h200", attention, n_gpus=16)
    raw = json.loads(path.read_text())
    raw["model_name"] = "Qwen/Qwen3.8-Flash-Next"
    path.write_text(json.dumps(raw))
    config = load_config(path)
    validate_against_host(config, available_gpus=16, check_gpu_type=False)
    assert config.attention_implementation == "qsa_flex"


def test_config_gpu_attention_flash_next_rejects_generic_h200_attention(tmp_path: Path) -> None:
    path = _config_gpu_attention_write_config(tmp_path, "h200", "flash_attention_3")
    raw = json.loads(path.read_text())
    raw["model_name"] = "Qwen/Qwen3.8-Flash-Next"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Qwen/Qwen3.8-Flash-Next requires attn_implementation='qsa_flex'"):
        validate_against_host(load_config(path), available_gpus=8, check_gpu_type=False)


def test_config_gpu_attention_flash_next_reviewed_recipe_keeps_training_settings() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "configs/qwen3.8-flash-next/h200/train-sft-full-16gpus-sp16-256k.config"
    )
    config = load_config(path)
    validate_against_host(config, available_gpus=16, check_gpu_type=False)
    assert config.config_id == "qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k"
    assert config.n_gpus == config.sp_size == 16
    assert config.dp_size == 1
    assert config.max_seq_len == config.max_tokens_per_mb == 262144
    assert config.attention_implementation == "qsa_flex"
    assert config.optimizer_dtype == "float32"
    assert config.training["train_batch_size"] == 1
    assert config.training["ep_size"] == 8
    assert config.training["deepep_token_chunk_size"] == 8192
    assert config.training["ac_config"]["offload_config"]["enabled"] is False
    assert config.training["ds_config"]["zero_optimization"]["offload_optimizer"] == {
        "device": "cpu",
        "pin_memory": False,
    }
    assert config.native.backend.type == "remote"
    assert config.native.backend.base_url == "http://127.0.0.1:8003"


def test_config_gpu_attention_flash_next_cortex_wire_preserves_all_training_fields() -> None:
    from arctic_platform.correctness.harness.cortex import CortexTrainingConfig

    path = (
        Path(__file__).resolve().parents[1] / "configs/qwen3.8-flash-next/h200/train-sft-full-16gpus-sp16-256k.config"
    )
    native = load_config(path).native
    wire = CortexTrainingConfig.model_validate(native.model_dump(mode="json")).to_cortex()
    ds = native.training.ds_config
    assert wire == [
        {
            "job_type": "training",
            "model_name": "Qwen/Qwen3.8-Flash-Next",
            "dtype": "bfloat16",
            "seed": 42,
            "training_config": {
                **native.training.ds_worker_config,
                "n_gpus": 16,
                "max_seq_len": 262144,
                "train_batch_size": 1,
                "gradient_clipping": 1.0,
                "optimizer": {"name": ds["optimizer"]["type"], **ds["optimizer"]["params"]},
                "ds_config": ds,
                "debug": {"full_determinism": False},
            },
        }
    ]


# Config tolerance.


def _config_tolerance_context(*, gate: float, dss_norm: float):
    arm = ArmSpec(
        name="fixed",
        global_batch_size=1,
        max_seq_len=8,
        total_tokens=8,
        active_tokens=7,
        dss_microbatches=1,
        pad_fraction=0.125,
    )
    spec = ConfigSpec(
        config_id="unit",
        config_path="unit.config",
        model=ModelSpec(
            source_checkpoint="model",
            num_hidden_layers=1,
            vision_depth=0,
            seed=0,
            param_count=1,
            cache_path="model",
            content_hash="hash",
            sized_against_gib=1.0,
        ),
        arms=[arm],
        attn_implementations=["flash_attention_3"],
        applicable_tests=["single-step-grads"],
        test_tolerances={"single-step-grads": Tolerance(absolute=gate)},
    )
    reference = {"loss": 1.0, "grad_norms": {"weight": 1.0}, "microbatches": 1}
    dss = SimpleNamespace(avg_loss=1.0, grad_norms={"weight": dss_norm}, model_calls=1)
    context = SimpleNamespace(
        config_id="unit", spec=spec, reference_for=lambda _arm: reference, target_for=lambda _arm: dss
    )
    return (context, arm)


def test_config_tolerance_gradient_norm_verdict_uses_the_frozen_config_gate():
    passing_context, arm = _config_tolerance_context(gate=0.01, dss_norm=1.005)
    failing_context, _ = _config_tolerance_context(gate=0.001, dss_norm=1.005)
    assert _run_arm(passing_context, arm).outcome.value == "pass"
    assert _run_arm(failing_context, arm).outcome.value == "fail"


def test_config_tolerance_calibrated_gate_rounds_up_and_has_a_one_millithreshold_floor():
    floor = Tolerance(absolute=0.001, computed_gate=0.0004)
    rounded = Tolerance(absolute=0.002, computed_gate=0.0012)
    assert floor.absolute == 0.001
    assert rounded.absolute == 0.002
    with pytest.raises(ValueError, match="does not equal upward-rounded"):
        Tolerance(absolute=0.001, computed_gate=0.0012)


# Names.


def test_names_qwen35_moe_norms_align_across_fused_and_prime_layouts() -> None:
    dss = {
        "model.layers.0._checkpoint_wrapped_module.mlp.router.gate.weight": 1.0,
        "model.layers.0._checkpoint_wrapped_module.mlp.experts.w1": 3.0,
        "model.layers.0._checkpoint_wrapped_module.mlp.experts.w2": 5.0,
        "model.layers.0._checkpoint_wrapped_module.mlp.experts.w3": 4.0,
        "model.layers.0._checkpoint_wrapped_module.shared_expert.w1.weight": 6.0,
        "model.layers.0._checkpoint_wrapped_module.shared_expert.w2.weight": 7.0,
        "model.layers.0._checkpoint_wrapped_module.shared_expert.w3.weight": 8.0,
        "model.layers.0._checkpoint_wrapped_module.shared_expert_gate.weight": 9.0,
    }
    reference = {
        "model.language_model.layers.0.mlp.gate.weight": 1.0,
        "model.language_model.layers.0.mlp.experts.gate_up_proj": 5.0,
        "model.language_model.layers.0.mlp.experts.down_proj": 5.0,
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": 6.0,
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": 7.0,
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": 8.0,
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": 9.0,
    }
    pairs, only_dss, only_reference = align(dss, reference)
    assert not only_dss
    assert not only_reference
    assert dict(((name, (actual, expected)) for name, actual, expected in pairs))[
        "layers.0.mlp.experts.gate_up_proj"
    ] == (math.hypot(3.0, 4.0), 5.0)


def test_names_equal_checkpoint_aliases_collapse_without_losing_coverage() -> None:
    dss = {
        "model.layers.0.input_layernorm.weight": 2.0,
        "model.language_model.layers.0._checkpoint_wrapped_module.input_layernorm.weight": 2.0,
    }
    reference = {"model.language_model.layers.0.input_layernorm.weight": 2.0}
    pairs, only_dss, only_reference = align(dss, reference)
    assert pairs == [("layers.0.input_layernorm.weight", 2.0, 2.0)]
    assert not only_dss
    assert not only_reference


def test_names_unequal_checkpoint_aliases_fail_coverage() -> None:
    dss = {
        "model.layers.0.input_layernorm.weight": 2.0,
        "model.language_model.layers.0._checkpoint_wrapped_module.input_layernorm.weight": 3.0,
    }
    reference = {"model.language_model.layers.0.input_layernorm.weight": 2.0}
    _, only_dss, _ = align(dss, reference)
    assert only_dss == ["model.language_model.layers.0._checkpoint_wrapped_module.input_layernorm.weight"]


def test_names_unused_zero_vision_gradients_are_excluded() -> None:
    dss = {"model.visual.blocks.0.attn.qkv.weight": 0.0, "model.layers.0.input_layernorm.weight": 2.0}
    reference = {"model.language_model.layers.0.input_layernorm.weight": 2.0}
    pairs, only_dss, only_reference = align(dss, reference)
    assert pairs == [("layers.0.input_layernorm.weight", 2.0, 2.0)]
    assert not only_dss
    assert not only_reference


def test_names_nonzero_vision_gradient_fails_text_only_coverage() -> None:
    dss = {"model.visual.blocks.0.attn.qkv.weight": 1.0}
    _, only_dss, _ = align(dss, {})
    assert only_dss == ["model.visual.blocks.0.attn.qkv.weight"]


def test_names_qwen4_shared_expert_norms_follow_the_weight_conversion() -> None:
    dss = {
        "model.layers.0._checkpoint_wrapped_module.mlp.qwen_shared_expert.w1.weight": 6.0,
        "model.layers.0._checkpoint_wrapped_module.mlp.qwen_shared_expert.w2.weight": 7.0,
        "model.layers.0._checkpoint_wrapped_module.mlp.qwen_shared_expert.w3.weight": 8.0,
    }
    reference = {
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": 6.0,
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": 7.0,
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": 8.0,
    }
    pairs, only_dss, only_reference = align(dss, reference)
    assert not only_dss
    assert not only_reference
    assert len(pairs) == 3
    assert all((actual == expected for _, actual, expected in pairs))


@pytest.mark.parametrize("source,target", [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")])
def test_names_qwen4_shared_expert_tensor_groups_preserve_original_names(source: str, target: str) -> None:
    original = f"model.layers.2.mlp.qwen_shared_expert.{source}.weight"
    groups = canonical_dss_tensor_groups([original])
    group = groups[f"layers.2.mlp.shared_expert.{target}.weight"]
    assert group.originals == (f"layers.2.mlp.qwen_shared_expert.{source}.weight",)
    assert group.concatenate_dim is None


def test_names_qwen4_shared_expert_mapping_preserves_numerical_differences() -> None:
    pairs, only_dss, only_reference = align(
        {"model.layers.0.mlp.qwen_shared_expert.w1.weight": 2.0},
        {"model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": 3.0},
    )
    assert pairs == [("layers.0.mlp.shared_expert.gate_proj.weight", 2.0, 3.0)]
    assert not only_dss
    assert not only_reference


def test_names_qwen4_unequal_shared_expert_aliases_fail_coverage() -> None:
    dss = {"model.layers.0.mlp.qwen_shared_expert.w1.weight": 2.0, "model.layers.0.shared_expert.w1.weight": 3.0}
    reference = {"model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": 2.0}
    pairs, only_dss, only_reference = align(dss, reference)
    assert pairs == [("layers.0.mlp.shared_expert.gate_proj.weight", 2.0, 2.0)]
    assert only_dss == ["model.layers.0.shared_expert.w1.weight"]
    assert not only_reference


# Original models.


_original_models_QWEN8 = "Qwen/Qwen3-8B"

_original_models_DENSE27 = "Qwen/Qwen3.8-27B"

_original_models_FLASH = "Qwen/Qwen3.8-Flash-Next"

_original_models_DENSE_CONFIG = "qwen3.8-27b-b200-train-sft-8gpus-2k"

_original_models_FLASH_CONFIG = "qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k"

_original_models_FLASH_SOURCE = (
    "/checkpoint/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
)


def _original_models_case(model, config_id, source):
    return (
        SimpleNamespace(model_name=model, config_id=config_id),
        SimpleNamespace(model=SimpleNamespace(source_checkpoint=source)),
    )


def test_original_models_runner_reexports_model_library_validator():
    assert runner.validate_local_full_model is validate_local_full_model
    assert validate_local_full_model is original.validate_local_full_model
    assert validate_local_full_model.__module__ == original.__name__


@pytest.mark.parametrize(
    "model,expected",
    [
        (
            _original_models_QWEN8,
            (
                "b968826d9c46dd6066d109eabc6255188de91218",
                "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
                None,
                "model\\.layers\\.(\\d+)\\.",
                None,
                None,
                None,
                None,
                None,
                {},
            ),
        ),
        (
            _original_models_DENSE27,
            (
                "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
                "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
                "model\\.language_model\\.layers\\.(\\d+)\\.",
                1199,
                18,
                55562855904,
                (248320, 5120),
                "9c0eb904ed294a62882adb287b81e36fdb48e5fb8e006ee25b084341ac78fd55",
                {"BF16": 2, "F32": 4},
            ),
        ),
        (
            _original_models_FLASH,
            (
                "de4b8e4d43b917e7706784d8bb445c9af86a3540",
                "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
                "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
                "model\\.language_model\\.layers\\.(\\d+)\\.",
                1658,
                131,
                359999963128,
                (248320, 2560),
                "1d55a6d3f64b1b2b004302864c0e74e11871ddf9f3481b10fc21fc8b0ec53d26",
                {"BF16": 2, "I64": 8},
            ),
        ),
    ],
)
def test_original_models_reviewed_original_metadata_pins_are_unchanged(model, expected):
    assert set(original.ORIGINAL_MODELS) == {_original_models_QWEN8, _original_models_DENSE27, _original_models_FLASH}
    metadata = original.ORIGINAL_MODELS[model]
    assert metadata.model_name == model
    assert (
        metadata.revision,
        metadata.config_sha256,
        metadata.index_sha256,
        metadata.layer_pattern,
        metadata.tensor_count,
        metadata.shard_count,
        metadata.payload_bytes,
        metadata.vocabulary_shape,
        metadata.schema_sha256,
        dict(metadata.dtype_bytes),
    ) == expected


@pytest.mark.parametrize("model", [_original_models_QWEN8, _original_models_DENSE27, _original_models_FLASH])
def test_original_models_registry_and_nested_metadata_are_immutable(model):
    metadata = original.ORIGINAL_MODELS[model]
    with pytest.raises(TypeError):
        original.ORIGINAL_MODELS[model] = metadata
    with pytest.raises(FrozenInstanceError):
        metadata.revision = "0" * 40
    with pytest.raises(TypeError):
        metadata.dtype_bytes["BF16"] = 4
    with pytest.raises(TypeError):
        metadata.source_checkpoints[0] = "unreviewed"
    if metadata.vocabulary_shape is not None:
        with pytest.raises(TypeError):
            metadata.vocabulary_shape[0] = 1


@pytest.mark.parametrize(
    "model,config_id,source",
    [
        (_original_models_QWEN8, "qwen3-8b-h200-train-sft-full-4gpus-64k", _original_models_QWEN8),
        (_original_models_QWEN8, "another-config-using-the-original-qwen3-8b", _original_models_QWEN8),
        (_original_models_DENSE27, _original_models_DENSE_CONFIG, _original_models_DENSE27),
        (_original_models_DENSE27, _original_models_DENSE_CONFIG, "/data-fast/base-models/Qwen/Qwen3.8-27B"),
        (_original_models_FLASH, _original_models_FLASH_CONFIG, _original_models_FLASH),
        (_original_models_FLASH, _original_models_FLASH_CONFIG, _original_models_FLASH_SOURCE),
    ],
)
def test_original_models_original_selection_preserves_exact_accepted_sources(model, config_id, source):
    cfg, spec = _original_models_case(model, config_id, source)
    assert original._original_model(cfg, spec) is original.ORIGINAL_MODELS[model]


@pytest.mark.parametrize(
    "model,config_id,source",
    [
        ("Qwen/unsupported", _original_models_DENSE_CONFIG, _original_models_DENSE27),
        (_original_models_QWEN8, "any-config", "/data-fast/base-models/Qwen/Qwen3-8B"),
        (_original_models_DENSE27, "unreviewed-config", _original_models_DENSE27),
        (_original_models_DENSE27, _original_models_DENSE_CONFIG, "/data-fast/base-models/synthetic/Qwen3.8-27B-4L"),
        (_original_models_FLASH, "unreviewed-config", _original_models_FLASH),
        (
            _original_models_FLASH,
            _original_models_FLASH_CONFIG,
            _original_models_FLASH_SOURCE.replace("de4b8", "00000"),
        ),
        (_original_models_FLASH, _original_models_FLASH_CONFIG, "/data-fast/base-models/Qwen/Qwen3.8-Flash-Next"),
    ],
)
def test_original_models_unreviewed_sources_rejected_before_cache_lookup(monkeypatch, model, config_id, source):
    cfg, spec = _original_models_case(model, config_id, source)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DSS_ALLOW_REMOTE_MODELS", "1")
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: pytest.fail("unreviewed cache lookup"))
    with pytest.raises(ValueError, match="^local full model mode requires a supported pinned original source$"):
        validate_local_full_model(CortexConfig(base_url="http://127.0.0.1:8003"), cfg, spec)


# Placement widths.

_placement_widths_CASES = [
    (8, 4, [4, 8]),
    (8, 8, [8]),
    (8, 16, []),
    (16, 4, [4, 16]),
    (16, 8, [8, 16]),
    (16, 16, [16]),
    (32, 4, [4, 16]),
    (32, 8, [8, 32]),
    (32, 16, [16, 32]),
]


@pytest.mark.parametrize("pool,declared,expected", _placement_widths_CASES)
def test_placement_widths_widths_for_pool_and_declared_width(pool: int, declared: int, expected: list) -> None:
    assert placement_widths(declared, pool) == expected


def test_placement_widths_a_pool_narrower_than_the_config_offers_no_width() -> None:
    """The config is run as written, so a pool it does not fit in yields nothing rather than a narrower run."""
    assert placement_widths(16, 8) == []
    assert placement_widths(16, 15) == []


def test_placement_widths_the_declared_width_is_always_measured_first() -> None:
    """A run cut short after one width has measured the topology the spec was reviewed against."""
    for pool, declared, expected in _placement_widths_CASES:
        if expected:
            assert expected[0] == declared, (pool, declared)


def test_placement_widths_the_multiple_is_capped_however_large_the_pool_is() -> None:
    """A very wide allocation does not turn one config into an arbitrarily wide job."""
    widths = placement_widths(4, 4096)
    assert widths == [4, 4 * GPU_WIDTH_MULTIPLE_CAP]


def test_placement_widths_a_pool_that_holds_the_width_only_once_measures_it_once() -> None:
    """With no room for a second copy there is no wider topology to compare against."""
    assert placement_widths(8, 8) == [8]
    assert placement_widths(8, 15) == [8]


def test_placement_widths_a_nonpositive_declared_width_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        placement_widths(0, 8)


def _placement_widths_config(tmp_path: Path, n_gpus: int = 4, sp_size: int = 2, train_batch_size: int | None = 2):
    ds_config = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1}
    training = {
        "n_gpus": n_gpus,
        "sp_size": sp_size,
        "max_seq_len": 4096,
        "attn_implementation": "flash_attention_3",
        "ds_config": ds_config,
    }
    if train_batch_size is not None:
        ds_config["train_batch_size"] = train_batch_size
        training["train_batch_size"] = train_batch_size
    path = tmp_path / "train-sft-4gpus-64k.config"
    path.write_text(json.dumps(_config_factory_native_config(training, model_name="some/model")))
    return load_config(path)


def test_placement_widths_a_wider_placement_is_not_the_body_the_spec_froze(tmp_path: Path) -> None:
    """``n_gpus`` is part of the hashed training body, so a derived width hashes to something else.

    This is why the spec gate is applied to the config as loaded from the file and never to a config
    returned by ``at_gpu_width``: hashing the derived one would refuse every wider run as drifted, and
    relaxing the hash to ignore ``n_gpus`` would stop the gate noticing a real edit to that field.
    """
    cfg = _placement_widths_config(tmp_path)
    assert config_checksum(cfg.at_gpu_width(16)) != config_checksum(cfg)
    assert config_checksum(cfg) == config_checksum(load_config(cfg.path))


def test_placement_widths_only_the_gpu_count_differs(tmp_path: Path) -> None:
    """Every other training value stays as written, which is what makes the two widths comparable."""
    cfg = _placement_widths_config(tmp_path)
    wider = cfg.at_gpu_width(16)
    moved = {"n_gpus", "ds_config", "train_batch_size"}
    assert (wider.n_gpus, wider.sp_size) == (16, 2)
    assert {k: v for k, v in wider.training.items() if k not in moved} == {
        k: v for k, v in cfg.training.items() if k not in moved
    }
    assert (wider.config_id, wider.path, wider.max_seq_len) == (cfg.config_id, cfg.path, cfg.max_seq_len)


def test_placement_widths_the_config_it_was_derived_from_is_untouched(tmp_path: Path) -> None:
    """The declared width is measured too, so deriving must not overwrite it."""
    cfg = _placement_widths_config(tmp_path)
    cfg.at_gpu_width(16)
    assert cfg.n_gpus == 4
    assert cfg.training["n_gpus"] == 4


def test_placement_widths_the_extra_gpus_widen_data_parallelism(tmp_path: Path) -> None:
    """Sequence-parallel width is a config property and stays put; the extra ranks carry more rows."""
    cfg = _placement_widths_config(tmp_path, n_gpus=4, sp_size=2)
    wider = cfg.at_gpu_width(16)
    assert (wider.sp_size, wider.dp_size) == (2, 8)


def test_placement_widths_the_batch_grows_with_the_width(tmp_path: Path) -> None:
    """DeepSpeed asserts ``train_batch_size == micro * grad_acc * dp_size``, so the batch tracks the width."""
    cfg = _placement_widths_config(tmp_path, n_gpus=4, sp_size=2, train_batch_size=2)
    assert cfg.at_gpu_width(8).training["ds_config"]["train_batch_size"] == 4
    assert cfg.at_gpu_width(16).training["ds_config"]["train_batch_size"] == 8


def test_placement_widths_the_batch_grows_wherever_the_config_states_it(tmp_path: Path) -> None:
    """The multi-step checks size their dataset slice from the training config's own copy of the batch.

    The data plane needs at least one row per data-parallel shard, so a copy left at the declared value
    is short by the placement multiple and the first dispatch refuses the batch.
    """
    cfg = _placement_widths_config(tmp_path, n_gpus=4, sp_size=1, train_batch_size=4)
    assert cfg.at_gpu_width(8).training["train_batch_size"] == 8
    assert cfg.at_gpu_width(8).training["train_batch_size"] >= cfg.at_gpu_width(8).dp_size


def test_placement_widths_the_declared_width_repairs_a_batch_that_does_not_match_deepspeed(tmp_path: Path) -> None:
    """A 16-GPU file written with an 8-GPU batch still places: DeepSpeed needs 16, the file stays 8."""
    cfg = _placement_widths_config(tmp_path, n_gpus=16, sp_size=1, train_batch_size=8)
    placed = cfg.at_gpu_width(16)
    assert placed.training["train_batch_size"] == 16
    assert placed.training["ds_config"]["train_batch_size"] == 16
    assert cfg.training["train_batch_size"] == 8
    assert config_checksum(placed) != config_checksum(cfg)


def test_placement_widths_a_consistent_declared_width_leaves_the_batch_put(tmp_path: Path) -> None:
    """When the file already satisfies the identity, the declared-width copy is a no-op on the batch."""
    cfg = _placement_widths_config(tmp_path, n_gpus=4, sp_size=1, train_batch_size=4)
    placed = cfg.at_gpu_width(4)
    assert placed.training["train_batch_size"] == 4
    assert config_checksum(placed) == config_checksum(cfg)


def test_placement_widths_the_deepspeed_batch_assertion_holds_at_every_width(tmp_path: Path) -> None:
    """The invariant DeepSpeed checks at engine initialization, evaluated for each width this produces."""
    for sp_size in (1, 2, 4):
        cfg = _placement_widths_config(tmp_path, n_gpus=4, sp_size=sp_size, train_batch_size=4 // sp_size)
        for pool in (8, 16, 32):
            for width in placement_widths(cfg.n_gpus, pool):
                wider = cfg.at_gpu_width(width)
                ds_config = wider.training["ds_config"]
                assert (
                    ds_config["train_batch_size"]
                    == ds_config["train_micro_batch_size_per_gpu"]
                    * ds_config["gradient_accumulation_steps"]
                    * wider.dp_size
                ), (sp_size, width)


def test_placement_widths_a_config_stating_no_batch_has_none_written_for_it(tmp_path: Path) -> None:
    """Nothing is invented: a config without ``train_batch_size`` is placed wider without acquiring one."""
    cfg = _placement_widths_config(tmp_path, n_gpus=4, sp_size=2, train_batch_size=None)
    wider = cfg.at_gpu_width(16)
    assert "train_batch_size" not in wider.training.get("ds_config", {})
    assert "train_batch_size" not in wider.training


def test_placement_widths_a_width_that_is_not_a_whole_multiple_is_refused(tmp_path: Path) -> None:
    cfg = _placement_widths_config(tmp_path, n_gpus=4, sp_size=2)
    with pytest.raises(ValueError, match="whole multiple"):
        cfg.at_gpu_width(6)


def test_placement_widths_every_width_this_dimension_produces_is_placeable(tmp_path: Path) -> None:
    """A multiple of the declared width is always divisible by a sequence-parallel size that divides it."""
    cfg = _placement_widths_config(tmp_path, n_gpus=4, sp_size=4)
    for pool in (8, 16, 32, 64):
        for width in placement_widths(cfg.n_gpus, pool):
            assert cfg.at_gpu_width(width).n_gpus == width


def test_placement_widths_a_wider_placement_carries_proportionally_more_rows() -> None:
    """Only the row count moves; the padded width and the case names are the reviewed ones."""
    declared = arms_for(8192, 4)
    wider = scaled_for_placement(declared, 4)
    assert [a.name for a in wider] == [a.name for a in declared]
    assert [a.max_seq_len for a in wider] == [a.max_seq_len for a in declared]
    assert [a.global_batch_size for a in wider] == [a.global_batch_size * 4 for a in declared]


def test_placement_widths_the_declared_width_leaves_the_cases_untouched() -> None:
    """A multiple of one is the config as written, so its cases are the reviewed ones exactly."""
    declared = arms_for(8192, 8)
    assert scaled_for_placement(declared, 1) == declared


def test_placement_widths_every_data_parallel_shard_receives_a_row_at_every_width(tmp_path: Path) -> None:
    """The condition Arctic Platform enforces when it dispatches a batch: rows must cover the data-parallel width.

    Without scaling the cases this fails at any width above the declared one, because the reviewed case
    holds one row per declared GPU and a wider placement has more shards than rows.
    """
    for sp_size in (1, 2, 4):
        cfg = _placement_widths_config(tmp_path, n_gpus=4, sp_size=sp_size, train_batch_size=4 // sp_size)
        declared = arms_for(cfg.max_seq_len, cfg.n_gpus)
        for pool in (8, 16, 32):
            for width in placement_widths(cfg.n_gpus, pool):
                wider = cfg.at_gpu_width(width)
                for arm in scaled_for_placement(declared, width // cfg.n_gpus):
                    assert arm.global_batch_size >= wider.dp_size, (sp_size, width, arm.name)


def test_placement_widths_a_multiple_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        scaled_for_placement(arms_for(8192, 4), 0)


# Spec.


def test_spec_model_hash_ignores_non_model_processor_sidecars(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("model")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    expected = hash_directory(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text("image")
    (tmp_path / "video_preprocessor_config.json").write_text("video")
    assert hash_directory(tmp_path) == expected


def test_spec_model_hash_keeps_tokenizer_metadata_in_identity(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("model")
    expected = hash_directory(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("tokenizer")
    assert hash_directory(tmp_path) != expected
