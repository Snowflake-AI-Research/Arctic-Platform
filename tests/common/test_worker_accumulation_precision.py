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

from copy import deepcopy

import pytest


@pytest.fixture
def worker():
    from arctic_platform.common.deepspeed_worker import DeepSpeedWorker

    return object.__new__(DeepSpeedWorker.__ray_metadata__.modified_class)


@pytest.mark.parametrize("job_config", [{}, {"training_config": {}}])
def test_omitted_accumulation_precision_is_not_inserted(worker, job_config):
    config = {}
    assert worker.ds_training_config(job_config, config, {}) is config
    assert "data_types" not in config


@pytest.mark.parametrize("data_types", [{}, {"grad_accum_dtype": "bf16"}, {"grad_accum_dtype": "fp32"}])
def test_explicit_accumulation_precision_is_preserved(worker, data_types):
    config = {"data_types": deepcopy(data_types)}
    worker.ds_training_config({}, config, {})
    assert config["data_types"] == data_types
