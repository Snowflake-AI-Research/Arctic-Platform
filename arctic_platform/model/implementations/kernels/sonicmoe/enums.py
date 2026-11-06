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

# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from enum import Enum

LIBRARY_NAME = "arl_sonicmoe"  # vendored copy: own torch.library namespace
TENSORMAP = "tensormap"


class KernelBackendMoE(Enum):
    scattermoe = "scattermoe"
    torch = "torch"
    sonicmoe = "sonicmoe"


class ActivationType(Enum):
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"

    RELU_SQ = "relu_sq"
    RELU = "relu"
    # Values go straight to QuACK's gemm_act / gemm_dact, and QuACK's only GELU is the
    # tanh approximation (its "geglu" is tanh-approx too). There is no exact-erf option.
    GELU = "gelu_tanh_approx"
    SILU = "silu"


def is_glu(activation_type: ActivationType):
    return activation_type in [ActivationType.SWIGLU, ActivationType.REGLU, ActivationType.GEGLU]
