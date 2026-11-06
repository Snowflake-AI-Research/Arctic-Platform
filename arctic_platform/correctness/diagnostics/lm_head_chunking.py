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

# !/usr/bin/env python3
"""First-two diagnostic that removes the fused LM-head chunk size from a copied config.

The source config and reviewed specification are not edited. This altered workload
still requires compatible prepared reference records; a canonical record must not
be relabelled as an unchunked reference. Diagnostic results are not canonical verdicts.

Usage: ``python -m arctic_platform.correctness.diagnostics.lm_head_chunking
--config <external-path> --test single-step-grads``.
"""

from __future__ import annotations

import copy
import sys
from typing import List
from typing import Optional

from arctic_platform.correctness import api
from arctic_platform.correctness.harness.config import LoadedConfig

KNOB = "fused_lm_head_token_chunk_size"


def without_chunked_head(cfg: LoadedConfig) -> LoadedConfig:
    """A copy of ``cfg`` whose training sub-job has no ``fused_lm_head_token_chunk_size``."""
    stripped = copy.deepcopy(cfg)
    for sub in [stripped.sub_job, *stripped.sub_jobs]:
        training = sub.get("training_config")
        if isinstance(training, dict):
            training.pop(KNOB, None)
            prime_rl = training.get("prime_rl")
            if isinstance(prime_rl, dict):
                prime_rl.pop(KNOB, None)
    return stripped


def main(argv: Optional[List[str]] = None) -> int:
    request, options = api.parse_run_args(sys.argv[1:] if argv is None else argv)
    from arctic_platform.correctness.harness.runner import execute

    def execute_without_chunked_head(cfg, *args, **kwargs):
        print(
            f"diagnostic: {KNOB} removed (was {cfg.training.get(KNOB)!r}) for {cfg.config_id} on {cfg.n_gpus} GPU(s)",
            flush=True,
        )
        return execute(without_chunked_head(cfg), *args, **kwargs)

    return api.run(request, options, executor=execute_without_chunked_head).exit_code


if __name__ == "__main__":
    sys.exit(main())
