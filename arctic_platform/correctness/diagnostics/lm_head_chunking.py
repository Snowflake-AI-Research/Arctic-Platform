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
"""Does the chunked LM head produce a config's last-layer gradient-norm disagreement?

Runs ``python -m arctic_platform.correctness run`` unchanged, except that ``fused_lm_head_token_chunk_size`` is removed from
the training sub-job each placement executes. The key is dropped after the spec's config checksum has been
verified, so the config file, its checksum gate, the cases, and the frozen tolerances are exactly regression's.
Both arms lose the chunked head together: the Arctic Platform payload no longer carries the key, which leaves the head
unchunked, and the single-GPU reference reads the same in-memory config, which selects its ordinary chunked
cross entropy instead of the tiled log-probability path. A verdict that changes between this and a plain
regression run is therefore attributable to that one knob.

Usage: ``python -m arctic_platform.correctness.diagnostics.lm_head_chunking --config <path> --test single-step-grads``; every
argument is passed to ``run``.
"""

from __future__ import annotations

import copy
import sys
from typing import List
from typing import Optional

from arctic_platform.correctness import __main__ as cli
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
    execute = cli.execute

    def execute_without_chunked_head(cfg, *args, **kwargs):
        print(
            f"diagnostic: {KNOB} removed (was {cfg.training.get(KNOB)!r}) for {cfg.config_id} on {cfg.n_gpus} GPU(s)",
            flush=True,
        )
        return execute(without_chunked_head(cfg), *args, **kwargs)

    cli.execute = execute_without_chunked_head
    return cli.main(["run", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
