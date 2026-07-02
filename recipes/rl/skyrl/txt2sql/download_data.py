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

"""Thin wrapper around upstream SkyRL's BIRD preprocessor.

Forwards to ``integrations.arctic_rl.envs.preprocess_bird.main`` (lives in
the user's SkyRL clone at ``$SKYRL_HOME/integrations/arctic_rl/envs/``) with
the right ``--bird_dir`` / ``--output_dir`` / ``--max_tokens`` defaults for
this recipe so the user only needs:

    export SKYRL_HOME=<path to SkyRL clone>
    python download_data.py --bird_dir ~/data/bird/raw

The raw BIRD download is *not* automated — BIRD is gated behind a sign-up
form. Follow the "Data preparation" section of this recipe's README to stage
the raw files at ``--bird_dir``, then run this script.

The upstream preprocessor emits one row per BIRD example with the verl-PR-#6
schema that ``integrations.arctic_rl.envs.bird:BirdEnv`` consumes (gold SQL
in ``reward_model.ground_truth``, sqlite path in ``extra_info.db_path``).
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BIRD-SQL into Arctic RL × SkyRL training parquets.",
    )
    parser.add_argument(
        "--bird_dir",
        type=str,
        required=True,
        help=(
            "Raw BIRD root containing `train/train.json`, `train/train_databases/`, "
            "and `dev/dev.json`, `dev/dev_databases/`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/data/bird"),
        help="Where to write {train,val}.parquet. Default: ~/data/bird.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help=(
            "Drop BIRD rows whose tokenized prompt exceeds this many tokens. "
            "16384 keeps the launcher's PROMPT_LEN=16384 happy and drops the "
            "long-tail outlier DBs (works_cycles, movie_3). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HF tokenizer used for the token-length filter. Default: Qwen/Qwen3-8B.",
    )
    args = parser.parse_args()

    skyrl_home = os.environ.get("SKYRL_HOME")
    if not skyrl_home or not os.path.isdir(skyrl_home):
        raise SystemExit(
            "SKYRL_HOME is not set or doesn't exist. Clone SkyRL at the pinned "
            "commit (see this recipe's README) and "
            "`export SKYRL_HOME=<path to clone>`."
        )
    sys.path.insert(0, skyrl_home)

    from integrations.arctic_rl.envs import preprocess_bird as pp

    # Rebuild argv for ``pp.main()`` — only forward BIRD-related args. We hard-
    # code ``--sources bird`` so this script doesn't accidentally pull in
    # Spider/GretelAI data that this recipe isn't validated against.
    sys.argv = [
        "preprocess_bird.py",
        "--sources",
        "bird",
        "--bird_dir",
        args.bird_dir,
        "--output_dir",
        args.output_dir,
        "--max_tokens",
        str(args.max_tokens),
        "--tokenizer",
        args.tokenizer,
    ]
    pp.main()


if __name__ == "__main__":
    main()
