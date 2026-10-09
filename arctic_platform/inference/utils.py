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

import re
from importlib.metadata import requires

VLLM_VERSION_031 = "0.31.0"
SUPPORTED_VLLM_VERSIONS = (VLLM_VERSION_031,)


def get_compatible_vllm_version():
    reqs = requires("arctic-platform")
    for req in reqs:
        match = re.match("vllm==(.*); extra == \"inference\"", req)
        if match is not None:
            return match.groups()[0]


def get_runtime_vllm_version() -> str:
    import vllm

    version = getattr(vllm, "__version__", None)
    if version is None:
        raise RuntimeError("Unable to determine the runtime vLLM version.")
    return version


def require_supported_vllm_version(
    feature: str = "ArcticInference",
    *,
    version: str | None = None,
) -> str:
    if version is None:
        version = get_runtime_vllm_version()
    if version not in SUPPORTED_VLLM_VERSIONS:
        supported = ", ".join(f"v{item}" for item in SUPPORTED_VLLM_VERSIONS)
        raise RuntimeError(
            f"{feature} supports vLLM {supported} only; found "
            f"vllm=={version}. Revalidate ArcticInference before using a "
            "different vLLM version."
        )
    return version


# For debugging
def print0(*args, **kwargs):
    from vllm.distributed.parallel_state import get_tp_group
    if get_tp_group().is_first_rank:
        print(*args, **kwargs)
