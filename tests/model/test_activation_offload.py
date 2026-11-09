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

import torch

from arctic_platform.model.implementations.gpu.activation_offload import ActivationOffloadManager
from arctic_platform.model.implementations.gpu.activation_offload import install_activation_offload


def test_default_tensor_threshold_is_accepted_by_install():
    model = torch.nn.Linear(4, 4)

    manager = install_activation_offload(model)

    assert manager.tensor_size_threshold == 1 << 20


def test_none_tensor_threshold_resets_to_default():
    manager = ActivationOffloadManager()
    manager.configure(tensor_size_threshold=4096)
    assert manager.tensor_size_threshold == 4096

    manager.configure(tensor_size_threshold=None)

    assert manager.tensor_size_threshold == 1 << 20
