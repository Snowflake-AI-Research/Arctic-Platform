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
"""Run a tinker-cookbook module with ``arctic_platform.integrations.tinker`` installed.

    python -m arctic_platform.integrations.tinker.run --training-gpus 1 --sampling-gpus 1 \\
        tinker_cookbook.recipes.math_rl.train env=gsm8k model_name=Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import runpy
import sys

_FLAGS = {
    "--training-gpus": "training_gpus",
    "--sampling-gpus": "sampling_gpus",
    "--teacher-sampling-gpus": "teacher_sampling_gpus",
    "--max-prompt-length": "max_prompt_length",
    "--max-response-length": "max_response_length",
}


def main() -> None:
    import arctic_platform.integrations.tinker as tinker

    argv = sys.argv[1:]
    kwargs: dict[str, int] = {}
    index = 0
    while index < len(argv) and argv[index].startswith("--"):
        flag = argv[index]
        if flag not in _FLAGS:
            raise SystemExit(f"unknown flag {flag}; expected one of {', '.join(_FLAGS)}")
        if index + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        kwargs[_FLAGS[flag]] = int(argv[index + 1])
        index += 2
    if "training_gpus" not in kwargs or "sampling_gpus" not in kwargs:
        raise SystemExit(
            "usage: python -m arctic_platform.integrations.tinker.run --training-gpus N --sampling-gpus N <module> [recipe args...]"
        )
    if index >= len(argv):
        raise SystemExit("missing the cookbook module to run")
    tinker.configure(**kwargs)
    module = argv[index]
    sys.argv = [module, *argv[index + 1 :]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
