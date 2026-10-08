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

"""Build Arctic Inference native extensions during packaging, when requested."""

import os
import subprocess
import sys

_PRECOMPILED_OPS = "ARCTIC_INFERENCE_PRECOMPILED_OPS"


def precompiled_ops_requested(environ=None):
    env = os.environ if environ is None else environ
    return env.get(_PRECOMPILED_OPS, "").lower() in {"1", "true", "on"}


def get_build_hook():
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface

    class CustomBuildHook(BuildHookInterface):
        def initialize(self, version, build_data):
            if self.target_name == "sdist" or not precompiled_ops_requested():
                return
            try:
                import torch  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    f"{_PRECOMPILED_OPS} is set, but this build cannot import torch. "
                    "Install torch, then re-run pip with --no-build-isolation."
                ) from exc
            setup_py = os.path.join(self.root, "arctic_platform", "inference", "setup.py")
            subprocess.run(
                [sys.executable, setup_py, "build_ext", "--inplace"],
                cwd=self.root,
                check=True,
            )

    return CustomBuildHook
