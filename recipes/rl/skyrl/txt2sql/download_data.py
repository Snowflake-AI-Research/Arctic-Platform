#!/usr/bin/env python
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around the vendored BIRD preprocessor.

Just forwards to ``arctic_rl.envs.preprocess_bird.main`` with the right
``--bird_dir`` / ``--output_dir`` / ``--max_tokens`` defaults for this recipe
so the user only needs:

    python download_data.py --bird_dir ~/data/bird/raw

The raw BIRD download is *not* automated — BIRD is gated behind a sign-up
form. Follow the "Data preparation" section of this recipe's README to stage
the raw files at ``--bird_dir``, then run this script.

The vendored preprocessor lives at::

    Arctic-Platform/recipes/rl/skyrl/_lib/arctic_rl/envs/preprocess_bird.py

and emits one row per BIRD example with the verl-PR-#6 schema that the
vendored ``arctic_rl.envs.bird:BirdEnv`` consumes (gold SQL in
``reward_model.ground_truth``, sqlite path in ``extra_info.db_path``).
"""

import argparse
import os
import sys
from pathlib import Path

REPO_LIB = Path(__file__).resolve().parent.parent / "_lib"


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

    if not REPO_LIB.is_dir():
        raise SystemExit(
            f"Vendored arctic_rl library not found at {REPO_LIB}. "
            "Make sure you're running this from the recipe directory in a fresh "
            "Arctic-Platform checkout (recipes/rl/skyrl/txt2sql/)."
        )

    sys.path.insert(0, str(REPO_LIB))

    from arctic_rl.envs import preprocess_bird as pp

    # Rebuild argv for ``pp.main()`` — only forward BIRD-related args. We hard-
    # code ``--sources bird`` so this script doesn't accidentally pull in
    # Spider/GretelAI data that this recipe isn't validated against.
    sys.argv = [
        "preprocess_bird.py",
        "--sources", "bird",
        "--bird_dir", args.bird_dir,
        "--output_dir", args.output_dir,
        "--max_tokens", str(args.max_tokens),
        "--tokenizer", args.tokenizer,
    ]
    pp.main()


if __name__ == "__main__":
    main()
