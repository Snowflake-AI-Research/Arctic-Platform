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
"""One typed model schema for both servers: model_options rewrite, worker-config rejection, legacy conversion."""

import pytest

from arctic_platform.common import model_option_aliases as aliases
from arctic_platform.common.model_settings import ModelSettings
from arctic_platform.common.model_settings import worker_config_model_keys
from arctic_platform.common.training_settings import WorkerTrainingSettings
from arctic_platform.common.utils.server_models import JobConfig
from arctic_platform.rl.config import ArcticRLClientConfig


def test_alias_table_covers_the_registry():
    assert aliases.alias_table_problems() == []


def test_model_options_are_rewritten_onto_the_fields_and_revalidate_unchanged():
    settings = WorkerTrainingSettings.model_validate(
        {"model_options": {"liger": True, "lm_head": {"fp32": True}}, "gradient_checkpointing": False}
    )

    assert settings.use_liger is True
    assert settings.fp32_lm_head is True
    assert settings.activation_checkpointing is False
    assert WorkerTrainingSettings.model_validate(settings.model_dump()).model_dump() == settings.model_dump()


def test_a_field_and_its_model_option_must_agree():
    with pytest.raises(ValueError, match="set the option once, or give both the same value"):
        ModelSettings(model_options={"liger": True}, use_liger=False)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"model_provider": "liger"}, "model_options.liger=true"),
        ({"model_provider": "transformers"}, "must be 'huggingface', 'arctic' or"),
        ({"model_provider": "arctic", "use_liger": True}, "model_options.lm_head.cross_entropy"),
        ({"text_config_overrides": {"a": 1}, "ep_size": 2}, "text_config_overrides is unsupported"),
        ({"tiled_mlp_token_chunk_size": 8, "compile": {"fullgraph": True}}, "fullgraph cannot be combined"),
        ({"prime_rl": {"tiled_mlp_token_chunk_size": 8}}, "use training_config.tiled_mlp_token_chunk_size"),
        ({"fused_lm_head_token_chunk_size": True}, "must be a positive integer"),
        ({"peft_config": {}}, "peft_type"),
    ],
)
def test_invalid_model_settings_are_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        ModelSettings.model_validate(settings)


def test_prime_rl_provider_is_accepted_as_arctic():
    assert ModelSettings(model_provider="prime_rl").model_provider == "arctic"


def test_attn_impl_wins_and_must_agree_with_prime_rl():
    assert ModelSettings(attn_implementation="sdpa", attn_impl="eager").resolved_attn_implementation() == "eager"
    assert ModelSettings(prime_rl={"attn": "sparse_mla"}).resolved_attn_implementation() == "sparse_mla"
    with pytest.raises(ValueError, match="name it once"):
        ModelSettings(attn_implementation="sdpa", prime_rl={"attn": "eager"}).resolved_attn_implementation()


def test_worker_config_model_keys_name_the_replacing_field():
    assert worker_config_model_keys(
        {"enable_gradient_checkpointing": False, "attn_implementation": "sdpa", "zorro_train_enable": True}
    ) == {"enable_gradient_checkpointing": "activation_checkpointing", "attn_implementation": "attn_implementation"}


def test_server_rejects_model_settings_in_ds_worker_config():
    with pytest.raises(ValueError, match="training_config.activation_checkpointing instead of enable_gradient"):
        JobConfig(model_name="m", ds_worker_config={"enable_gradient_checkpointing": False})
    assert JobConfig(model_name="m", ds_worker_config={"zorro_train_enable": True}).ds_worker_config


def test_legacy_client_moves_model_settings_into_training_config():
    legacy = ArcticRLClientConfig(
        model_name="m",
        training_gpus=1,
        ds_worker_config={
            "model_provider": "liger",
            "enable_gradient_checkpointing": False,
            "attn_implementation": "sdpa",
            "zorro_train_enable": True,
        },
    )

    settings = legacy.training_settings()
    assert settings.use_liger is True
    assert settings.model_provider == "huggingface"
    assert settings.activation_checkpointing is False
    assert settings.attn_implementation == "sdpa"
    assert legacy.worker_runtime_config() == {"zorro_train_enable": True}
    job = JobConfig(
        model_name="m",
        training_config=settings.model_dump(exclude_unset=True),
        ds_worker_config=legacy.worker_runtime_config(),
    )
    assert WorkerTrainingSettings.model_validate(job.training_config).use_liger is True


def test_dense_loader_rejects_an_unsupported_model_option_before_loading():
    from arctic_platform.model.option_profile import check_model_options_profile

    settings = ModelSettings(model_options={"checkpointing": {"mode": "full"}})
    with pytest.raises(
        aliases.ModelOptionsUnsupported, match="checkpointing.mode is unsupported with the huggingface"
    ):
        check_model_options_profile(settings.model_dump(), model_path="unused", sp_size=1)
    check_model_options_profile(ModelSettings().model_dump(), model_path="unused", sp_size=1)
