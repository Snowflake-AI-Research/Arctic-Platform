#!/usr/bin/env python3
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

"""List the throughput runs of a model catalog, one per training file (or per file matching --only), as the lines
run_catalog.sh reads:

    <run name> <nodes> <catalog training json> [run_sft.py arguments]

    plan.py [--catalog <cortex-training>/model-catalog] [--mode full|lora|all] [--only <glob>...] > plan.txt

--repeat / --random list the MoE worst-case probes instead: only the MoE files (ep_size > 1), each with run_sft.py's
--mode repeat rows (max_seq_len - 1 copies of one token: every token routes to the same experts) or --mode throughput
rows (uniform random tokens: every token fans out to as many EP ranks as it can). Run them with STEPS=4; a file passes
when its run completes.
"""

import argparse
import fnmatch
import json
import math
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = os.environ.get("CATALOG") or (
        os.environ["CORTEX"] + "/model-catalog" if "CORTEX" in os.environ else None
    )
    ap.add_argument("--catalog", default=default, required=default is None, help="model-catalog dir ($CORTEX's)")
    ap.add_argument("--mode", choices=["full", "lora", "all"], default="full")
    ap.add_argument("--only", nargs="*", default=["*"], help="globs over <model>/<context>/<mode>")
    ap.add_argument("--prefix", default="suite", help="run names are <prefix>-<model>-<context>-<mode>")
    ap.add_argument("--repeat", action="store_true", help="the MoE files' repeat-token probe (see above)")
    ap.add_argument("--random", action="store_true", help="the MoE files' uniform-random-token probe (see above)")
    args = ap.parse_args()
    for path in sorted((Path(args.catalog) / "models").glob("*/*/h200-training-*.json")):
        model, context = path.parts[-3], path.parts[-2]
        mode = path.stem.rsplit("-", 1)[-1]
        if args.mode != "all" and mode != args.mode:
            continue
        if not any(fnmatch.fnmatch(f"{model}/{context}/{mode}", pattern) for pattern in args.only):
            continue
        tc = json.loads(path.read_text())["sub_job_configs"][0]["training_config"]
        if (args.repeat or args.random) and int(tc.get("ep_size") or 1) <= 1:
            continue
        nodes = max(1, math.ceil(tc["n_gpus"] / 8))
        extra = " --mode repeat" if args.repeat else " --mode throughput" if args.random else ""
        print(f"{args.prefix}-{model}-{context}-{mode} {nodes} {path.resolve()}{extra}")


if __name__ == "__main__":
    main()
